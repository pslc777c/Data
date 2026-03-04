from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as load_joblib

from common.io import read_parquet, write_parquet
from models.ml1.apply_multitask_nn_ml1 import (
    _apply_pred_factor_peso_caps,
    _compute_post_outputs,
    _expand_ml1_horizon,
    _rebuild_ml1_harvest_chain,
)
from models.ml1.zero_inflated import ZeroInflatedRegressor  # noqa: F401 (needed for joblib loading)
from models.ml1.train_multitask_nn_ml1 import CAT_COLS, NUM_COLS, TARGET_CLIPS, TARGET_COLS


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "gold" / "ml1_nn" / "ds_ml1_nn_v1.parquet"
EVAL_DIR = DATA_DIR / "eval" / "ml1_nn"
TARGET_MODELS_DIR = DATA_DIR / "models" / "ml1_nn_target_models"
OUT_DIR = DATA_DIR / "gold" / "ml1_nn"

FIELD_TARGETS = {
    "target_d_start",
    "target_n_harvest_days",
    "target_factor_tallos_dia",
    "target_share_grado",
    "target_factor_peso_tallo",
}
POST_TARGETS = {
    "target_dh_dias",
    "target_factor_hidr",
    "target_factor_desp",
    "target_factor_ajuste",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("apply_target_model_hybrid_ml1")
    ap.add_argument("--summary", default=None, help="Hybrid summary json path. If omitted, latest is used.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    ap.add_argument("--run-label", default=None, help="Optional suffix label for output run id.")
    ap.add_argument(
        "--focus-tallos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, only overrides target_factor_tallos_dia and keeps other existing pred_* columns from input.",
    )
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _latest_hybrid_summary() -> Path:
    files = sorted(EVAL_DIR.glob("ml1_target_model_hybrid_summary_*.json"))
    if not files:
        raise FileNotFoundError(f"No hybrid summary found in {EVAL_DIR}")
    return files[-1]


def _load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "selected_models" not in obj or not isinstance(obj["selected_models"], dict):
        raise ValueError("Invalid summary. Missing 'selected_models' dict.")
    if not (("field_run" in obj and "post_run" in obj) or ("run_id" in obj)):
        raise ValueError("Summary must have either field_run+post_run or run_id.")
    return obj


def _target_run_id(target: str, summary: dict) -> str:
    if "field_run" in summary and "post_run" in summary:
        if target in FIELD_TARGETS:
            return str(summary["field_run"])
        if target in POST_TARGETS:
            return str(summary["post_run"])
    if "run_id" in summary:
        return str(summary["run_id"])
    raise KeyError(f"Unable to infer model run for target={target}")


def _resolve_target_models(summary: dict) -> dict[str, Path]:
    selected = dict(summary.get("selected_models", {}))
    out: dict[str, Path] = {}
    for t in TARGET_COLS:
        mname = selected.get(t)
        if not mname:
            continue
        rid = _target_run_id(t, summary)
        p = TARGET_MODELS_DIR / rid / f"{t}__{mname}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Target model not found for {t}: {p}")
        out[t] = p
    return out


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        if c not in out.columns:
            out[c] = "UNKNOWN"
        out[c] = _canon_str(out[c])
    for c in NUM_COLS:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _eval_if_possible(df: pd.DataFrame, yhat: np.ndarray, targets: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    maes: list[float] = []
    for i, t in enumerate(targets):
        mcol = f"mask_{t}"
        if t not in df.columns or mcol not in df.columns:
            out[f"mae_{t}"] = np.nan
            continue
        y = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float32)
        m = pd.to_numeric(df[mcol], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.0
        m = m & np.isfinite(y) & np.isfinite(yhat[:, i])
        if m.sum() == 0:
            out[f"mae_{t}"] = np.nan
            continue
        mae = float(np.mean(np.abs(yhat[m, i] - y[m])))
        out[f"mae_{t}"] = mae
        maes.append(mae)
    out["mae_avg"] = float(np.mean(maes)) if maes else np.nan
    return out


def _factor_tallos_day_mae(df: pd.DataFrame, pred: np.ndarray) -> float:
    target_col = "target_factor_tallos_dia"
    mask_col = "mask_target_factor_tallos_dia"
    if not {"ciclo_id", "fecha_evento", target_col, mask_col} <= set(df.columns):
        return np.nan
    v = df[["ciclo_id", "fecha_evento", target_col, mask_col]].copy()
    v["pred"] = pd.to_numeric(pd.Series(pred, index=v.index), errors="coerce")
    v[target_col] = pd.to_numeric(v[target_col], errors="coerce")
    v[mask_col] = pd.to_numeric(v[mask_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    v["fecha_evento"] = pd.to_datetime(v["fecha_evento"], errors="coerce").dt.normalize()
    v = v.dropna(subset=["ciclo_id", "fecha_evento"])
    if v.empty:
        return np.nan

    g = (
        v.groupby(["ciclo_id", "fecha_evento"], dropna=False, as_index=False)
        .agg(
            y_true=(target_col, "median"),
            y_pred=("pred", "median"),
            w=(mask_col, "max"),
        )
    )
    yt = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    yp = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
    ww = pd.to_numeric(g["w"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(ww) & (ww > 0.0)
    if not bool(m.any()):
        return np.nan
    return float(np.sum(np.abs(yp[m] - yt[m]) * ww[m]) / np.sum(ww[m]))


def _make_apply_run_id(summary: dict, run_label: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if "field_run" in summary and "post_run" in summary:
        base = f"{str(summary['field_run'])[:8]}_{str(summary['post_run'])[:8]}"
    else:
        base = str(summary.get("run_id", "hybrid"))[:16]
    if run_label:
        safe = str(run_label).strip().replace(" ", "_")
        return f"{ts}_{safe}_{uuid.uuid4().hex[:6]}"
    return f"{ts}_hyb_{base}_{uuid.uuid4().hex[:6]}"


def main() -> None:
    args = _parse_args()

    summary_path = _latest_hybrid_summary() if args.summary is None else Path(args.summary)
    summary = _load_summary(summary_path)
    model_paths = _resolve_target_models(summary)
    if not model_paths:
        raise ValueError("No target models resolved from summary.")

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")
    df = read_parquet(in_path).copy()
    df = _prepare_df(df)

    targets = list(TARGET_COLS)
    pred_cache: dict[str, np.ndarray] = {}
    pred_cols: list[str] = []
    pred_alias_map: dict[str, str] = {}
    fallback_tallos_to_input = False
    mae_day_tallos_old = np.nan
    mae_day_tallos_new = np.nan

    for i, t in enumerate(targets):
        p = model_paths.get(t)
        if p is None:
            continue
        model = load_joblib(p)
        pred = model.predict(df)
        pred = pd.to_numeric(pd.Series(pred), errors="coerce").to_numpy(dtype=np.float32)
        lo, hi = TARGET_CLIPS[t]
        pred = np.clip(pred, lo, hi)

        col = "pred_" + t.replace("target_", "", 1)
        if bool(args.focus_tallos) and t != "target_factor_tallos_dia" and col in df.columns:
            pred_cache[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        elif bool(args.focus_tallos) and t == "target_factor_tallos_dia" and col in df.columns:
            old_pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            mae_day_tallos_old = _factor_tallos_day_mae(df, old_pred)
            mae_day_tallos_new = _factor_tallos_day_mae(df, pred)
            if np.isfinite(mae_day_tallos_old) and np.isfinite(mae_day_tallos_new) and (mae_day_tallos_new > mae_day_tallos_old):
                pred_cache[col] = old_pred
                fallback_tallos_to_input = True
            else:
                df[col] = pred
                pred_cache[col] = pred
        else:
            df[col] = pred
            pred_cache[col] = pred
        pred_cols.append(col)
        pred_alias_map[t.replace("target_", "", 1) + "_ML1"] = col

    for alias, src in pred_alias_map.items():
        if alias not in df.columns and src in df.columns:
            df[alias] = df[src]

    # Keep stem-weight predictions inside historical agronomic ranges.
    _apply_pred_factor_peso_caps(df)

    yhat = np.full((len(df), len(targets)), np.nan, dtype=np.float32)
    for i, t in enumerate(targets):
        col = "pred_" + t.replace("target_", "", 1)
        if col in pred_cache:
            yhat[:, i] = pred_cache[col]
        elif col in df.columns:
            yhat[:, i] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)

    eval_metrics = _eval_if_possible(df, yhat=yhat, targets=targets)

    # Keep ML1 hybrid output structurally equivalent to multitask apply:
    # expand horizon and rebuild the full harvest->post chain from predicted targets.
    df, n_added = _expand_ml1_horizon(df)
    df = _rebuild_ml1_harvest_chain(df)

    apply_run_id = _make_apply_run_id(summary=summary, run_label=args.run_label)
    df["ml1_multitask_nn_run_id"] = apply_run_id

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (OUT_DIR / f"pred_ml1_target_hybrid_{apply_run_id}.parquet")
    write_parquet(df, out_path)

    post_detail, post_dd, post_dt = _compute_post_outputs(df, run_id=apply_run_id)
    post_detail_path = OUT_DIR / f"pred_ml1_target_hybrid_post_final_{apply_run_id}.parquet"
    post_dd_path = OUT_DIR / f"pred_ml1_target_hybrid_post_dia_destino_{apply_run_id}.parquet"
    post_dt_path = OUT_DIR / f"pred_ml1_target_hybrid_post_dia_total_{apply_run_id}.parquet"
    if not post_detail.empty:
        write_parquet(post_detail, post_detail_path)
    if not post_dd.empty:
        write_parquet(post_dd, post_dd_path)
    if not post_dt.empty:
        write_parquet(post_dt, post_dt_path)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    apply_meta = {
        "apply_run_id": apply_run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary_path": str(summary_path.resolve()),
        "input_path": str(in_path.resolve()),
        "output_path": str(out_path.resolve()),
        "selected_models": summary.get("selected_models", {}),
        "resolved_model_paths": {k: str(v.resolve()) for k, v in model_paths.items()},
        "metrics": eval_metrics,
        "focus_tallos": bool(args.focus_tallos),
        "fallback_tallos_to_input": bool(fallback_tallos_to_input),
        "mae_day_tallos_old_input": float(mae_day_tallos_old) if np.isfinite(mae_day_tallos_old) else None,
        "mae_day_tallos_new_model": float(mae_day_tallos_new) if np.isfinite(mae_day_tallos_new) else None,
    }
    meta_path = EVAL_DIR / f"ml1_target_hybrid_apply_meta_{apply_run_id}.json"
    meta_path.write_text(json.dumps(apply_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Hybrid predictions written: {out_path}")
    print(
        f"     rows={len(df):,} targets_predicted={len(model_paths)} run_id={apply_run_id}"
        f" rows_expanded={n_added:,}"
    )
    if bool(args.focus_tallos):
        print(
            "     focus_tallos="
            f"{bool(args.focus_tallos)} fallback_tallos_to_input={bool(fallback_tallos_to_input)}"
            f" mae_day_old={mae_day_tallos_old if np.isfinite(mae_day_tallos_old) else 'nan'}"
            f" mae_day_new={mae_day_tallos_new if np.isfinite(mae_day_tallos_new) else 'nan'}"
        )
    if np.isfinite(eval_metrics.get("mae_avg", np.nan)):
        print(f"     masked_mae_avg={eval_metrics['mae_avg']:.6f}")
    print(f"[OK] Apply meta saved: {meta_path}")
    if not post_detail.empty:
        print(f"[OK] Post final detail : {post_detail_path}")
        print(f"[OK] Post dia destino : {post_dd_path}")
        print(f"[OK] Post dia total   : {post_dt_path}")


if __name__ == "__main__":
    main()
