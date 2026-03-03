from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet
from models.ml2.ml2_nn_common import final_from_ml1_and_corr


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
SILVER = DATA / "silver"
DEFAULT_INPUT = DATA / "gold" / "ml2_nn" / "ds_ml2_nn_v1.parquet"
MODELS_DIR = DATA / "models" / "ml2_nn"
OUT_DIR = DATA / "gold" / "ml2_nn"
EVAL_DIR = DATA / "eval" / "ml2_nn"
IN_FACT_PESO_REAL = SILVER / "fact_peso_tallo_real_grado_dia.parquet"
IN_DIM_VARIEDAD = SILVER / "dim_variedad_canon.parquet"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("apply_multitask_nn_ml2")
    ap.add_argument("--run-id", default=None, help="Model run_id. If omitted, latest model is used.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None, help="Legacy alias output (defaults to global path).")
    ap.add_argument("--output-global", default=None, help="Path for ML2_global predictions.")
    ap.add_argument("--output-puro", default=None, help="Path for ML2_puro predictions.")
    ap.add_argument("--output-operativo", default=None, help="Path for ML2_Operativo predictions.")
    ap.add_argument(
        "--write-legacy-alias",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, writes legacy pred_ml2_multitask_nn_<run_id>.parquet as alias of operativo.",
    )
    ap.add_argument("--max-iters", type=int, default=3, help="Dynamic coupling iterations (>=1).")
    ap.add_argument("--tol", type=float, default=1e-3, help="Relative tolerance for iterative convergence.")
    ap.add_argument(
        "--anchor-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, when real labels exist they replace projected values for iterative reprojection.",
    )
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _latest_meta() -> Path:
    files = sorted(MODELS_DIR.glob("ml2_multitask_nn_*_meta.json"))
    if not files:
        raise FileNotFoundError(f"No model metadata found in {MODELS_DIR}")
    return files[-1]


def _meta_for_run(run_id: str) -> Path:
    path = MODELS_DIR / f"ml2_multitask_nn_{run_id}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata for run_id={run_id} not found: {path}")
    return path


def _load_artifacts(run_id: str | None) -> tuple[str, dict, dict[str, np.ndarray]]:
    meta_path = _latest_meta() if run_id is None else _meta_for_run(run_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rid = str(meta["run_id"])

    model_path = Path(meta["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    arr = np.load(model_path)
    params = {
        "w1": arr["w1"],
        "b1": arr["b1"],
        "w2": arr["w2"],
        "b2": arr["b2"],
        "w3": arr["w3"],
        "b3": arr["b3"],
        "x_fill": arr["x_fill"] if "x_fill" in arr.files else arr["x_mean"],
        "x_clip_lo": arr["x_clip_lo"] if "x_clip_lo" in arr.files else None,
        "x_clip_hi": arr["x_clip_hi"] if "x_clip_hi" in arr.files else None,
        "x_mean": arr["x_mean"],
        "x_std": arr["x_std"],
        "y_mean": arr["y_mean"],
        "y_std": arr["y_std"],
    }
    return rid, meta, params


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _predict_corr(x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    a1 = _relu(x @ p["w1"] + p["b1"])
    a2 = _relu(a1 @ p["w2"] + p["b2"])
    y_norm = a2 @ p["w3"] + p["b3"]
    y = y_norm * p["y_std"].reshape(1, -1) + p["y_mean"].reshape(1, -1)
    return y.astype(np.float32)


def _build_features(df: pd.DataFrame, meta: dict, p: dict[str, np.ndarray]) -> np.ndarray:
    cat_cols = meta["features"]["cat_cols"]
    num_cols = meta["features"]["num_cols"]
    cat_dummy_cols = meta["features"]["cat_dummy_cols"]

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = _canon_str(df[c])

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x_num = df[num_cols].to_numpy(dtype=np.float32)
    x_fill = p.get("x_fill", p["x_mean"]).astype(np.float32)
    x_clip_lo = p.get("x_clip_lo")
    x_clip_hi = p.get("x_clip_hi")
    x_mean = p["x_mean"].astype(np.float32)
    x_std = p["x_std"].astype(np.float32)

    if x_num.shape[1] != len(x_mean):
        raise ValueError(f"Numeric feature mismatch. got={x_num.shape[1]} expected={len(x_mean)}")

    for j in range(x_num.shape[1]):
        col = x_num[:, j]
        m = ~np.isfinite(col)
        if m.any():
            col[m] = x_fill[j]
            x_num[:, j] = col
    if isinstance(x_clip_lo, np.ndarray) and isinstance(x_clip_hi, np.ndarray):
        if len(x_clip_lo) == x_num.shape[1] and len(x_clip_hi) == x_num.shape[1]:
            x_num = np.clip(x_num, x_clip_lo.reshape(1, -1), x_clip_hi.reshape(1, -1))
    x_num = (x_num - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

    x_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep="=", dtype=np.float32)
    x_cat = x_cat.reindex(columns=cat_dummy_cols, fill_value=0.0)
    x_cat_arr = x_cat.to_numpy(dtype=np.float32)

    return np.concatenate([x_num, x_cat_arr], axis=1).astype(np.float32)


def _final_metrics(df_part: pd.DataFrame, y_corr_pred: np.ndarray, target_specs: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    mae_ml1_all: list[float] = []
    mae_ml2_all: list[float] = []
    r2_ml1_all: list[float] = []
    r2_ml2_all: list[float] = []

    for i, s in enumerate(target_specs):
        orig = str(s["original_target"])
        pred_col = str(s["ml1_pred_col"])
        mode = str(s["mode"])
        lo, hi = float(s["final_clip"][0]), float(s["final_clip"][1])
        key = orig.replace("target_", "", 1)

        mcol = f"mask_{s['corr_target']}"
        wi_full = pd.to_numeric(df_part.get(mcol), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_true = pd.to_numeric(df_part.get(orig), errors="coerce").to_numpy(dtype=np.float32)
        y_ml1 = pd.to_numeric(df_part.get(pred_col), errors="coerce").to_numpy(dtype=np.float32)
        y_ml2 = final_from_ml1_and_corr(y_ml1=y_ml1, corr=y_corr_pred[:, i], mode=mode)
        y_ml2 = np.clip(y_ml2, lo, hi).astype(np.float32)

        mi = (wi_full > 0.0) & np.isfinite(y_true) & np.isfinite(y_ml1) & np.isfinite(y_ml2)
        wi = wi_full[mi]
        wsum = float(wi.sum())
        if wsum <= 0.0:
            out[f"mae_ml1_{key}"] = np.nan
            out[f"mae_ml2_{key}"] = np.nan
            out[f"r2_ml1_{key}"] = np.nan
            out[f"r2_ml2_{key}"] = np.nan
            out[f"n_{key}"] = 0
            out[f"w_{key}"] = 0.0
            continue

        err_ml1 = y_ml1[mi] - y_true[mi]
        err_ml2 = y_ml2[mi] - y_true[mi]
        mae_ml1 = float(np.sum(np.abs(err_ml1) * wi) / wsum)
        mae_ml2 = float(np.sum(np.abs(err_ml2) * wi) / wsum)
        ybar = float(np.sum(y_true[mi] * wi) / wsum)
        sst = float(np.sum(((y_true[mi] - ybar) ** 2) * wi))
        sse_ml1 = float(np.sum((err_ml1**2) * wi))
        sse_ml2 = float(np.sum((err_ml2**2) * wi))
        r2_ml1 = float(1.0 - sse_ml1 / sst) if sst > 1e-12 else np.nan
        r2_ml2 = float(1.0 - sse_ml2 / sst) if sst > 1e-12 else np.nan

        out[f"mae_ml1_{key}"] = mae_ml1
        out[f"mae_ml2_{key}"] = mae_ml2
        out[f"r2_ml1_{key}"] = r2_ml1
        out[f"r2_ml2_{key}"] = r2_ml2
        out[f"improvement_mae_{key}"] = mae_ml1 - mae_ml2
        out[f"n_{key}"] = int(mi.sum())
        out[f"w_{key}"] = wsum

        mae_ml1_all.append(mae_ml1)
        mae_ml2_all.append(mae_ml2)
        if np.isfinite(r2_ml1):
            r2_ml1_all.append(r2_ml1)
        if np.isfinite(r2_ml2):
            r2_ml2_all.append(r2_ml2)

    out["mae_ml1_avg"] = float(np.mean(mae_ml1_all)) if mae_ml1_all else np.nan
    out["mae_ml2_avg"] = float(np.mean(mae_ml2_all)) if mae_ml2_all else np.nan
    out["mae_improvement_avg"] = (
        float(out["mae_ml1_avg"] - out["mae_ml2_avg"])
        if np.isfinite(out["mae_ml1_avg"]) and np.isfinite(out["mae_ml2_avg"])
        else np.nan
    )
    out["r2_ml1_avg"] = float(np.mean(r2_ml1_all)) if r2_ml1_all else np.nan
    out["r2_ml2_avg"] = float(np.mean(r2_ml2_all)) if r2_ml2_all else np.nan
    return out


def _safe_div(a: pd.Series, b: pd.Series, default: float = 1.0) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    out = aa / bb.replace(0, np.nan)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _stage_series(df: pd.DataFrame) -> pd.Series:
    if "stage" not in df.columns:
        return pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")
    return _canon_str(df["stage"])


def _num_series(df: pd.DataFrame, col: str, default: float | None = np.nan) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series(np.nan, index=df.index, dtype="float64")
    if default is not None:
        s = s.fillna(default)
    return s


def _hash_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.arange(len(df), dtype=np.uint64), index=df.index)
    d = pd.DataFrame(index=df.index)
    for c in cols:
        if c not in df.columns:
            d[c] = "NA"
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            d[c] = pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d").fillna("NA")
        else:
            d[c] = s.astype("string").fillna("NA")
    return pd.util.hash_pandas_object(d, index=False).astype(np.uint64)


def _prepare_dynamic_ref(df: pd.DataFrame) -> dict:
    st = _stage_series(df)
    is_hg = st.eq("HARVEST_GRADE").to_numpy()
    is_post = st.eq("POST").to_numpy()

    out: dict[str, object] = {
        "is_hg": is_hg,
        "is_post": is_post,
        "orig_tallos_pred_ml1_dia": _num_series(df, "tallos_pred_ml1_dia", default=0.0).to_numpy(dtype=np.float64),
        "orig_pred_factor_tallos_dia": _num_series(df, "pred_factor_tallos_dia", default=1.0).to_numpy(dtype=np.float64),
        "orig_tallos_post_proy": _num_series(df, "tallos_post_proy", default=0.0).to_numpy(dtype=np.float64),
        "orig_kg_verde_ref": _num_series(df, "kg_verde_ref", default=0.0).to_numpy(dtype=np.float64),
        "orig_gramos_verde_ref": _num_series(df, "gramos_verde_ref", default=0.0).to_numpy(dtype=np.float64),
        "orig_cajas_split": _num_series(df, "cajas_split_grado_dia", default=0.0).to_numpy(dtype=np.float64),
        "orig_cajas_ml1": _num_series(df, "cajas_ml1_grado_dia", default=0.0).to_numpy(dtype=np.float64),
        "orig_cajas_post_seed": _num_series(df, "cajas_post_seed", default=0.0).to_numpy(dtype=np.float64),
    }

    key_cols = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado"] if c in df.columns]
    day_key_cols = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon"] if c in df.columns]
    key_hash = _hash_key(df, key_cols)
    day_hash = _hash_key(df, day_key_cols)
    out["key_hash"] = key_hash
    out["day_hash"] = day_hash

    tallos_hg = _num_series(df, "tallos_pred_ml1_grado_dia", default=0.0)
    peso_hg = _num_series(df, "peso_tallo_baseline_g", default=0.0)
    fpeso_hg = _num_series(df, "pred_factor_peso_tallo", default=1.0)
    kg_hg_calc = tallos_hg * peso_hg * fpeso_hg / 1000.0
    kg_hg_base = _num_series(df, "kg_verde_ref", default=np.nan)
    kg_hg = kg_hg_base.where(np.isfinite(kg_hg_base), kg_hg_calc).fillna(0.0)

    hg_orig = pd.DataFrame(
        {
            "key_hash": key_hash[is_hg].to_numpy(),
            "tallos_hg_orig": tallos_hg[is_hg].to_numpy(dtype=np.float64),
            "kg_hg_orig": kg_hg[is_hg].to_numpy(dtype=np.float64),
        }
    )
    if not hg_orig.empty:
        hg_orig = hg_orig.groupby("key_hash", as_index=True).mean(numeric_only=True)
    else:
        hg_orig = pd.DataFrame(columns=["tallos_hg_orig", "kg_hg_orig"]).set_index(pd.Index([], name="key_hash"))
    out["hg_orig"] = hg_orig
    return out


def _predict_ml2_targets(
    df_state: pd.DataFrame,
    meta: dict,
    params: dict[str, np.ndarray],
    specs: list[dict],
    shrinkage: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    x = _build_features(df_state, meta=meta, p=params)
    ycorr_raw = _predict_corr(x, params)
    ycorr_adj = ycorr_raw.copy()
    pred_final: dict[str, np.ndarray] = {}
    corr_raw_map: dict[str, np.ndarray] = {}
    corr_adj_map: dict[str, np.ndarray] = {}

    for i, s in enumerate(specs):
        corr_target = str(s["corr_target"])
        orig = str(s["original_target"])
        pred_col = str(s["ml1_pred_col"])
        mode = str(s["mode"])
        lo, hi = float(s["final_clip"][0]), float(s["final_clip"][1])
        key = orig.replace("target_", "", 1)

        gamma = float(shrinkage.get(corr_target, 1.0))
        corr_raw = ycorr_raw[:, i].astype(np.float32)
        corr_adj = (corr_raw * gamma).astype(np.float32)
        y_base = _num_series(df_state, pred_col, default=np.nan).to_numpy(dtype=np.float32)
        y_final = final_from_ml1_and_corr(y_ml1=y_base, corr=corr_adj, mode=mode)
        y_final = np.clip(y_final, lo, hi).astype(np.float32)

        ycorr_adj[:, i] = corr_adj
        corr_raw_map[corr_target] = corr_raw
        corr_adj_map[corr_target] = corr_adj
        pred_final[key] = y_final

    return ycorr_raw, ycorr_adj, pred_final, corr_raw_map, corr_adj_map


def _anchor_predictions_with_real(
    df_state: pd.DataFrame,
    specs: list[dict],
    pred_final: dict[str, np.ndarray],
    anchor_real: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s in specs:
        key = str(s["original_target"]).replace("target_", "", 1)
        y = pred_final[key].copy()
        if anchor_real:
            real_col = str(s["original_target"])
            mask_col = f"mask_{real_col}"
            y_real = _num_series(df_state, real_col, default=np.nan).to_numpy(dtype=np.float32)
            w = _num_series(df_state, mask_col, default=0.0).to_numpy(dtype=np.float32)
            m = (w > 0.0) & np.isfinite(y_real)
            if m.any():
                y[m] = y_real[m]
        out[key] = y
    return out


def _inject_predictions_into_features(
    df_state: pd.DataFrame,
    specs: list[dict],
    pred_final: dict[str, np.ndarray],
) -> None:
    for s in specs:
        key = str(s["original_target"]).replace("target_", "", 1)
        pred_col = str(s["ml1_pred_col"])
        df_state[pred_col] = pred_final[key]


def _apply_dynamic_links(df_state: pd.DataFrame, ref: dict) -> None:
    st = _stage_series(df_state)
    is_hg = pd.Series(ref["is_hg"], index=df_state.index)
    is_post = pd.Series(ref["is_post"], index=df_state.index)
    is_hg_or_post = is_hg | is_post

    if "fecha_evento" in df_state.columns:
        fev = _to_date(df_state["fecha_evento"])
        df_state["fecha_evento"] = fev
    else:
        fev = pd.Series(pd.NaT, index=df_state.index, dtype="datetime64[ns]")

    # Rebuild harvest calendar from VEG prediction and propagate day_in_harvest/rel_pos.
    if {"ciclo_id", "pred_d_start", "pred_n_harvest_days", "fecha_evento"} <= set(df_state.columns):
        veg = df_state.loc[st.eq("VEG"), ["ciclo_id", "fecha_evento", "pred_d_start", "pred_n_harvest_days"]].copy()
        veg["ciclo_id"] = veg["ciclo_id"].astype("string")
        veg["fecha_sp"] = _to_date(veg["fecha_evento"])
        veg = veg.dropna(subset=["ciclo_id", "fecha_sp"])
        if not veg.empty:
            veg = veg.sort_values("fecha_sp").drop_duplicates(subset=["ciclo_id"], keep="last")
            d_start = np.rint(pd.to_numeric(veg["pred_d_start"], errors="coerce").fillna(0.0)).astype(int)
            n_days = np.rint(pd.to_numeric(veg["pred_n_harvest_days"], errors="coerce").fillna(1.0)).clip(lower=1).astype(int)
            h_start = veg["fecha_sp"] + pd.to_timedelta(d_start, unit="D")
            h_end = h_start + pd.to_timedelta(n_days - 1, unit="D")

            map_start = pd.Series(h_start.to_numpy(), index=veg["ciclo_id"])
            map_end = pd.Series(h_end.to_numpy(), index=veg["ciclo_id"])
            map_n = pd.Series(n_days.to_numpy(dtype=np.float64), index=veg["ciclo_id"])

            cid = df_state["ciclo_id"].astype("string")
            df_state["pred_harvest_start_ml2"] = pd.to_datetime(cid.map(map_start), errors="coerce").dt.normalize()
            df_state["pred_harvest_end_ml2"] = pd.to_datetime(cid.map(map_end), errors="coerce").dt.normalize()
            df_state["pred_n_harvest_days_ml2"] = pd.to_numeric(cid.map(map_n), errors="coerce")

            day = (fev - pd.to_datetime(df_state["pred_harvest_start_ml2"], errors="coerce")).dt.days + 1
            if "day_in_harvest" in df_state.columns:
                m = is_hg_or_post & day.notna()
                df_state.loc[m, "day_in_harvest"] = pd.to_numeric(day.loc[m], errors="coerce")
            if "n_harvest_days" in df_state.columns:
                nd = pd.to_numeric(df_state["pred_n_harvest_days_ml2"], errors="coerce")
                m = is_hg_or_post & nd.notna()
                df_state.loc[m, "n_harvest_days"] = nd.loc[m]
            if "rel_pos" in df_state.columns:
                nd = pd.to_numeric(df_state.get("n_harvest_days"), errors="coerce").replace(0, np.nan)
                rel = (pd.to_numeric(df_state.get("day_in_harvest"), errors="coerce") / nd).clip(lower=0.0, upper=1.0)
                m = is_hg_or_post & rel.notna()
                df_state.loc[m, "rel_pos"] = rel.loc[m]

    # Rebuild tallos day as relative adjustment over ML1 reference.
    # This keeps continuity: if factor_tallos_dia doesn't change, tallos_dia stays at ML1.
    fac_cur = _num_series(df_state, "pred_factor_tallos_dia", default=1.0)
    fac_ref = pd.Series(ref["orig_pred_factor_tallos_dia"], index=df_state.index, dtype="float64")
    tallos_ref = pd.Series(ref["orig_tallos_pred_ml1_dia"], index=df_state.index, dtype="float64")
    ratio_day = _safe_div(fac_cur, fac_ref, default=1.0).clip(lower=0.0)
    tallos_day = (tallos_ref * ratio_day).clip(lower=0.0)

    if "tallos_pred_ml1_dia" in df_state.columns:
        df_state.loc[is_hg_or_post, "tallos_pred_ml1_dia"] = tallos_day.loc[is_hg_or_post]

    share_norm = pd.Series(np.nan, index=df_state.index, dtype="float64")
    if is_hg.any():
        s_use = _num_series(df_state.loc[is_hg], "pred_share_grado", default=np.nan).fillna(0.0)
        grp_cols = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon"] if c in df_state.columns]
        if grp_cols:
            grp_idx = [df_state.loc[is_hg, c] for c in grp_cols]
            den = s_use.groupby(grp_idx, dropna=False).transform("sum")
            ngrp = s_use.groupby(grp_idx, dropna=False).transform("size").clip(lower=1)
            sn = np.where(den > 0.0, s_use / den, 1.0 / ngrp.astype(float))
        else:
            den = float(s_use.sum())
            sn = (s_use / den).to_numpy() if den > 0 else np.full(len(s_use), 1.0 / max(len(s_use), 1))
        share_norm.loc[is_hg] = pd.to_numeric(pd.Series(sn, index=df_state.loc[is_hg].index), errors="coerce").fillna(0.0)

    df_state["share_grado_ml2_norm"] = share_norm

    if "tallos_pred_ml1_grado_dia" in df_state.columns and is_hg.any():
        tallos_hg = (
            _num_series(df_state.loc[is_hg], "tallos_pred_ml1_dia", default=0.0)
            * _num_series(df_state.loc[is_hg], "share_grado_ml2_norm", default=0.0)
        ).clip(lower=0.0)
        df_state.loc[is_hg, "tallos_pred_ml1_grado_dia"] = tallos_hg

    # Derive kg verde at HG level from tallos+peso factor, then propagate ratios to POST.
    key_hash = ref["key_hash"]
    day_hash = ref["day_hash"]
    hg_orig = ref["hg_orig"]

    ratio_t = pd.Series(1.0, index=df_state.index, dtype="float64")
    ratio_k = pd.Series(1.0, index=df_state.index, dtype="float64")
    map_hg_tallos = pd.Series(dtype="float64")
    map_hg_day = pd.Series(dtype="float64")

    if is_hg.any():
        tallos_hg_new = _num_series(df_state.loc[is_hg], "tallos_pred_ml1_grado_dia", default=0.0).clip(lower=0.0)
        peso_hg = _num_series(df_state.loc[is_hg], "peso_tallo_baseline_g", default=0.0).clip(lower=0.0)
        fp_hg = _num_series(df_state.loc[is_hg], "pred_factor_peso_tallo", default=1.0).clip(lower=0.0)
        kg_hg_new = (tallos_hg_new * peso_hg * fp_hg / 1000.0).clip(lower=0.0)
        g_hg_new = kg_hg_new * 1000.0

        if "kg_verde_ref" in df_state.columns:
            df_state.loc[is_hg, "kg_verde_ref"] = kg_hg_new
        if "gramos_verde_ref" in df_state.columns:
            df_state.loc[is_hg, "gramos_verde_ref"] = g_hg_new

        hg_new = pd.DataFrame(
            {
                "key_hash": key_hash[is_hg].to_numpy(),
                "tallos_hg_new": tallos_hg_new.to_numpy(dtype=np.float64),
                "kg_hg_new": kg_hg_new.to_numpy(dtype=np.float64),
            }
        )
        hg_new = hg_new.groupby("key_hash", as_index=True).mean(numeric_only=True)
        map_hg_tallos = hg_new["tallos_hg_new"]

        day_new = pd.DataFrame(
            {
                "day_hash": day_hash[is_hg].to_numpy(),
                "tallos_day_new": _num_series(df_state.loc[is_hg], "tallos_pred_ml1_dia", default=0.0).to_numpy(dtype=np.float64),
            }
        )
        map_hg_day = day_new.groupby("day_hash", as_index=True)["tallos_day_new"].mean()

        if not hg_orig.empty:
            merge = hg_orig.join(hg_new, how="left")
            rt = _safe_div(merge["tallos_hg_new"], merge["tallos_hg_orig"], default=1.0).clip(lower=0.0)
            rk = _safe_div(merge["kg_hg_new"], merge["kg_hg_orig"], default=1.0).clip(lower=0.0)
            ratio_t = key_hash.map(rt).fillna(1.0)
            ratio_k = key_hash.map(rk).fillna(1.0)

    if is_post.any():
        ratio_t_arr = ratio_t.to_numpy(dtype=np.float64)
        ratio_k_arr = ratio_k.to_numpy(dtype=np.float64)

        if "tallos_post_proy" in df_state.columns:
            df_state.loc[is_post, "tallos_post_proy"] = ref["orig_tallos_post_proy"][is_post.to_numpy()] * ratio_t_arr[is_post.to_numpy()]
        if "kg_verde_ref" in df_state.columns:
            df_state.loc[is_post, "kg_verde_ref"] = ref["orig_kg_verde_ref"][is_post.to_numpy()] * ratio_k_arr[is_post.to_numpy()]
        if "gramos_verde_ref" in df_state.columns:
            df_state.loc[is_post, "gramos_verde_ref"] = ref["orig_gramos_verde_ref"][is_post.to_numpy()] * ratio_k_arr[is_post.to_numpy()]
        if "cajas_split_grado_dia" in df_state.columns:
            df_state.loc[is_post, "cajas_split_grado_dia"] = ref["orig_cajas_split"][is_post.to_numpy()] * ratio_k_arr[is_post.to_numpy()]
        if "cajas_ml1_grado_dia" in df_state.columns:
            df_state.loc[is_post, "cajas_ml1_grado_dia"] = ref["orig_cajas_ml1"][is_post.to_numpy()] * ratio_k_arr[is_post.to_numpy()]
        if "cajas_post_seed" in df_state.columns:
            df_state.loc[is_post, "cajas_post_seed"] = ref["orig_cajas_post_seed"][is_post.to_numpy()] * ratio_k_arr[is_post.to_numpy()]

        if "tallos_pred_ml1_grado_dia" in df_state.columns and not map_hg_tallos.empty:
            df_state.loc[is_post, "tallos_pred_ml1_grado_dia"] = key_hash[is_post].map(map_hg_tallos).fillna(
                _num_series(df_state.loc[is_post], "tallos_pred_ml1_grado_dia", default=0.0)
            )
        if "tallos_pred_ml1_dia" in df_state.columns and not map_hg_day.empty:
            df_state.loc[is_post, "tallos_pred_ml1_dia"] = day_hash[is_post].map(map_hg_day).fillna(
                _num_series(df_state.loc[is_post], "tallos_pred_ml1_dia", default=0.0)
            )

    # Rebuild fecha_post from dh and refresh calendar features used by NN.
    if "pred_dh_dias" in df_state.columns and "fecha_evento" in df_state.columns:
        dh_int = np.rint(_num_series(df_state, "pred_dh_dias", default=0.0)).astype(int)
        fecha_post_new = (fev + pd.to_timedelta(dh_int, unit="D")).dt.normalize()
        if "fecha_post" in df_state.columns:
            df_state.loc[is_post, "fecha_post"] = fecha_post_new.loc[is_post]
        if "dow_post" in df_state.columns:
            df_state.loc[is_post, "dow_post"] = fecha_post_new.loc[is_post].dt.dayofweek.astype(float)
        if "month_post" in df_state.columns:
            df_state.loc[is_post, "month_post"] = fecha_post_new.loc[is_post].dt.month.astype(float)
        if "weekofyear_post" in df_state.columns:
            df_state.loc[is_post, "weekofyear_post"] = fecha_post_new.loc[is_post].dt.isocalendar().week.astype(float)


def _final_metrics_from_pred_cols(df_part: pd.DataFrame, target_specs: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    mae_ml1_all: list[float] = []
    mae_ml2_all: list[float] = []
    r2_ml1_all: list[float] = []
    r2_ml2_all: list[float] = []

    for s in target_specs:
        orig = str(s["original_target"])
        pred_col = str(s["ml1_pred_col"])
        key = orig.replace("target_", "", 1)
        pred_ml2_col = f"pred_ml2_{key}"

        mcol = f"mask_{s['corr_target']}"
        wi_full = pd.to_numeric(df_part.get(mcol), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        y_true = pd.to_numeric(df_part.get(orig), errors="coerce").to_numpy(dtype=np.float32)
        y_ml1 = pd.to_numeric(df_part.get(pred_col), errors="coerce").to_numpy(dtype=np.float32)
        y_ml2 = pd.to_numeric(df_part.get(pred_ml2_col), errors="coerce").to_numpy(dtype=np.float32)

        mi = (wi_full > 0.0) & np.isfinite(y_true) & np.isfinite(y_ml1) & np.isfinite(y_ml2)
        wi = wi_full[mi]
        wsum = float(wi.sum())
        if wsum <= 0.0:
            out[f"mae_ml1_{key}"] = np.nan
            out[f"mae_ml2_{key}"] = np.nan
            out[f"r2_ml1_{key}"] = np.nan
            out[f"r2_ml2_{key}"] = np.nan
            out[f"n_{key}"] = 0
            out[f"w_{key}"] = 0.0
            continue

        err_ml1 = y_ml1[mi] - y_true[mi]
        err_ml2 = y_ml2[mi] - y_true[mi]
        mae_ml1 = float(np.sum(np.abs(err_ml1) * wi) / wsum)
        mae_ml2 = float(np.sum(np.abs(err_ml2) * wi) / wsum)
        ybar = float(np.sum(y_true[mi] * wi) / wsum)
        sst = float(np.sum(((y_true[mi] - ybar) ** 2) * wi))
        sse_ml1 = float(np.sum((err_ml1**2) * wi))
        sse_ml2 = float(np.sum((err_ml2**2) * wi))
        r2_ml1 = float(1.0 - sse_ml1 / sst) if sst > 1e-12 else np.nan
        r2_ml2 = float(1.0 - sse_ml2 / sst) if sst > 1e-12 else np.nan

        out[f"mae_ml1_{key}"] = mae_ml1
        out[f"mae_ml2_{key}"] = mae_ml2
        out[f"r2_ml1_{key}"] = r2_ml1
        out[f"r2_ml2_{key}"] = r2_ml2
        out[f"improvement_mae_{key}"] = mae_ml1 - mae_ml2
        out[f"n_{key}"] = int(mi.sum())
        out[f"w_{key}"] = wsum

        mae_ml1_all.append(mae_ml1)
        mae_ml2_all.append(mae_ml2)
        if np.isfinite(r2_ml1):
            r2_ml1_all.append(r2_ml1)
        if np.isfinite(r2_ml2):
            r2_ml2_all.append(r2_ml2)

    out["mae_ml1_avg"] = float(np.mean(mae_ml1_all)) if mae_ml1_all else np.nan
    out["mae_ml2_avg"] = float(np.mean(mae_ml2_all)) if mae_ml2_all else np.nan
    out["mae_improvement_avg"] = (
        float(out["mae_ml1_avg"] - out["mae_ml2_avg"])
        if np.isfinite(out["mae_ml1_avg"]) and np.isfinite(out["mae_ml2_avg"])
        else np.nan
    )
    out["r2_ml1_avg"] = float(np.mean(r2_ml1_all)) if r2_ml1_all else np.nan
    out["r2_ml2_avg"] = float(np.mean(r2_ml2_all)) if r2_ml2_all else np.nan
    return out


def _run_dynamic_inference(
    df: pd.DataFrame,
    meta: dict,
    params: dict[str, np.ndarray],
    specs: list[dict],
    shrinkage: dict[str, float],
    max_iters: int,
    tol: float,
    anchor_real: bool,
) -> dict:
    n_iters = max(int(max_iters), 1)
    df_state = df.copy()
    if "fecha_evento" in df_state.columns:
        df_state["fecha_evento"] = _to_date(df_state["fecha_evento"])
    if "fecha_post" in df_state.columns:
        df_state["fecha_post"] = _to_date(df_state["fecha_post"])

    ref = _prepare_dynamic_ref(df)
    prev_pred: dict[str, np.ndarray] = {}
    max_rel_change = np.nan
    converged = False

    last_ycorr_raw = None
    last_ycorr_adj = None
    last_pred: dict[str, np.ndarray] = {}
    last_corr_raw_map: dict[str, np.ndarray] = {}
    last_corr_adj_map: dict[str, np.ndarray] = {}

    for it in range(1, n_iters + 1):
        ycorr_raw, ycorr_adj, pred, corr_raw_map, corr_adj_map = _predict_ml2_targets(
            df_state=df_state,
            meta=meta,
            params=params,
            specs=specs,
            shrinkage=shrinkage,
        )
        pred = _anchor_predictions_with_real(df_state=df_state, specs=specs, pred_final=pred, anchor_real=anchor_real)
        _inject_predictions_into_features(df_state=df_state, specs=specs, pred_final=pred)
        _apply_dynamic_links(df_state=df_state, ref=ref)

        if prev_pred:
            rels: list[float] = []
            for s in specs:
                key = str(s["original_target"]).replace("target_", "", 1)
                cur = pred[key]
                prv = prev_pred[key]
                den = np.maximum(np.abs(prv), 1e-6)
                rel = np.abs(cur - prv) / den
                rels.append(float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0)
            max_rel_change = float(np.nanmax(rels)) if rels else np.nan
            if np.isfinite(max_rel_change) and max_rel_change <= float(tol):
                converged = True
                last_ycorr_raw = ycorr_raw
                last_ycorr_adj = ycorr_adj
                last_pred = {k: v.copy() for k, v in pred.items()}
                last_corr_raw_map = {k: v.copy() for k, v in corr_raw_map.items()}
                last_corr_adj_map = {k: v.copy() for k, v in corr_adj_map.items()}
                return {
                    "df_state": df_state,
                    "ycorr_raw": last_ycorr_raw,
                    "ycorr_adj": last_ycorr_adj,
                    "pred": last_pred,
                    "corr_raw_map": last_corr_raw_map,
                    "corr_adj_map": last_corr_adj_map,
                    "iterations": it,
                    "converged": converged,
                    "max_rel_change": max_rel_change,
                }

        prev_pred = {k: v.copy() for k, v in pred.items()}
        last_ycorr_raw = ycorr_raw
        last_ycorr_adj = ycorr_adj
        last_pred = {k: v.copy() for k, v in pred.items()}
        last_corr_raw_map = {k: v.copy() for k, v in corr_raw_map.items()}
        last_corr_adj_map = {k: v.copy() for k, v in corr_adj_map.items()}

    return {
        "df_state": df_state,
        "ycorr_raw": last_ycorr_raw,
        "ycorr_adj": last_ycorr_adj,
        "pred": last_pred,
        "corr_raw_map": last_corr_raw_map,
        "corr_adj_map": last_corr_adj_map,
        "iterations": n_iters,
        "converged": converged,
        "max_rel_change": max_rel_change,
    }


def _build_dynamic_output(
    df: pd.DataFrame,
    dyn: dict,
    specs: list[dict],
    run_id: str,
    layer_name: str,
    anchor_real: bool,
) -> pd.DataFrame:
    df_state = dyn["df_state"]
    df_out = df.copy()
    for s in specs:
        corr_target = str(s["corr_target"])
        orig = str(s["original_target"])
        key = orig.replace("target_", "", 1)
        df_out[f"pred_{corr_target}_raw"] = dyn["corr_raw_map"][corr_target]
        df_out[f"pred_{corr_target}"] = dyn["corr_adj_map"][corr_target]
        df_out[f"pred_ml2_{key}_corr"] = dyn["corr_adj_map"][corr_target]
        df_out[f"pred_ml2_{key}"] = dyn["pred"][key]

    if "fecha_post" in df_out.columns:
        df_out["fecha_post_ml1"] = _to_date(df_out["fecha_post"])
    if "pred_harvest_start_ml2" in df_state.columns:
        df_out["pred_harvest_start_ml2"] = _to_date(df_state["pred_harvest_start_ml2"])
    if "pred_harvest_end_ml2" in df_state.columns:
        df_out["pred_harvest_end_ml2"] = _to_date(df_state["pred_harvest_end_ml2"])
    if "pred_n_harvest_days_ml2" in df_state.columns:
        df_out["pred_n_harvest_days_ml2"] = _num_series(df_state, "pred_n_harvest_days_ml2", default=np.nan)

    if "day_in_harvest" in df_state.columns:
        df_out["day_in_harvest_ml2"] = _num_series(df_state, "day_in_harvest", default=np.nan)
    if "n_harvest_days" in df_state.columns:
        df_out["n_harvest_days_ml2"] = _num_series(df_state, "n_harvest_days", default=np.nan)
    if "rel_pos" in df_state.columns:
        df_out["rel_pos_ml2"] = _num_series(df_state, "rel_pos", default=np.nan)

    if "share_grado_ml2_norm" in df_state.columns:
        df_out["share_grado_ml2_norm"] = _num_series(df_state, "share_grado_ml2_norm", default=np.nan)
    if "tallos_pred_ml1_dia" in df_state.columns:
        df_out["tallos_pred_ml2_dia"] = _num_series(df_state, "tallos_pred_ml1_dia", default=0.0)
    if "tallos_pred_ml1_grado_dia" in df_state.columns:
        df_out["tallos_pred_ml2_grado_dia"] = _num_series(df_state, "tallos_pred_ml1_grado_dia", default=0.0)
    if "tallos_post_proy" in df_state.columns:
        df_out["tallos_post_ml2_proy"] = _num_series(df_state, "tallos_post_proy", default=0.0)

    if "kg_verde_ref" in df_state.columns:
        df_out["kg_verde_ml2"] = _num_series(df_state, "kg_verde_ref", default=0.0)
    if "gramos_verde_ref" in df_state.columns:
        df_out["gramos_verde_ml2"] = _num_series(df_state, "gramos_verde_ref", default=0.0)
    if "cajas_split_grado_dia" in df_state.columns:
        df_out["cajas_split_grado_dia_ml2"] = _num_series(df_state, "cajas_split_grado_dia", default=0.0)
    if "cajas_ml1_grado_dia" in df_state.columns:
        df_out["cajas_ml1_grado_dia_ml2"] = _num_series(df_state, "cajas_ml1_grado_dia", default=0.0)
    if "cajas_post_seed" in df_state.columns:
        df_out["cajas_post_seed_ml2"] = _num_series(df_state, "cajas_post_seed", default=0.0)
    if "fecha_post" in df_state.columns:
        df_out["fecha_post_ml2"] = _to_date(df_state["fecha_post"])

    # ------------------------------------------------------------------
    # ML2 temporal reprojection axis for downstream planning.
    # ------------------------------------------------------------------
    stage_u = _stage_series(df_out)
    is_hg = stage_u.eq("HARVEST_GRADE")
    is_post = stage_u.eq("POST")
    is_chain = is_hg | is_post

    fev_orig = pd.to_datetime(df_out.get("fecha_evento"), errors="coerce").dt.normalize()
    day_ml2 = _num_series(df_out, "day_in_harvest_ml2", default=np.nan)
    day_base = _num_series(df_out, "day_in_harvest", default=np.nan)
    day_ml2 = day_ml2.where(day_ml2 >= 1.0, day_base)
    # Robust fallback: derive day index from observed calendar order when raw day_in_harvest is missing/broken.
    if bool(is_chain.any()) and {"ciclo_id", "fecha_evento"} <= set(df_out.columns):
        ord_tbl = (
            df_out.loc[is_chain & fev_orig.notna(), ["ciclo_id", "fecha_evento"]]
            .copy()
            .rename(columns={"fecha_evento": "fev"})
        )
        if not ord_tbl.empty:
            ord_tbl["ciclo_id"] = ord_tbl["ciclo_id"].astype("string")
            ord_tbl["fev"] = pd.to_datetime(ord_tbl["fev"], errors="coerce").dt.normalize()
            ord_tbl = ord_tbl.dropna(subset=["ciclo_id", "fev"]).drop_duplicates(subset=["ciclo_id", "fev"])
            ord_tbl = ord_tbl.sort_values(["ciclo_id", "fev"], kind="mergesort")
            ord_tbl["day_ord"] = ord_tbl.groupby("ciclo_id", dropna=False).cumcount() + 1
            ord_idx = pd.MultiIndex.from_frame(ord_tbl[["ciclo_id", "fev"]])
            ord_map = pd.Series(ord_tbl["day_ord"].to_numpy(dtype=np.float64), index=ord_idx)
            row_idx = pd.MultiIndex.from_arrays(
                [df_out["ciclo_id"].astype("string"), fev_orig],
                names=["ciclo_id", "fev"],
            )
            day_ord_row = pd.Series(row_idx.map(ord_map), index=df_out.index, dtype="float64")
            day_ml2 = day_ml2.where(day_ml2 >= 1.0, day_ord_row)
    day_ml2 = day_ml2.where(day_ml2 >= 1.0, np.nan)
    df_out["day_in_harvest_ml2"] = day_ml2

    hs = pd.Series(dtype="datetime64[ns]")
    if bool(anchor_real):
        if {"ciclo_id", "pred_harvest_start_ml2"} <= set(df_out.columns):
            hs_map = (
                df_out.loc[stage_u.eq("VEG"), ["ciclo_id", "pred_harvest_start_ml2"]]
                .dropna(subset=["ciclo_id"])
                .drop_duplicates(subset=["ciclo_id"], keep="last")
            )
            hs_map["ciclo_id"] = hs_map["ciclo_id"].astype("string")
            hs = pd.Series(
                pd.to_datetime(hs_map["pred_harvest_start_ml2"], errors="coerce").dt.normalize().to_numpy(),
                index=hs_map["ciclo_id"],
            )
    else:
        # Global/Puro timeline: keep ML1 calendar anchor (first HG date) and only adjust horizon forward.
        hs_src = (
            df_out.loc[is_hg & fev_orig.notna(), ["ciclo_id", "fecha_evento"]]
            .copy()
            .rename(columns={"fecha_evento": "fev"})
        )
        if hs_src.empty:
            hs_src = (
                df_out.loc[is_chain & fev_orig.notna(), ["ciclo_id", "fecha_evento"]]
                .copy()
                .rename(columns={"fecha_evento": "fev"})
            )
        if not hs_src.empty:
            hs_src["ciclo_id"] = hs_src["ciclo_id"].astype("string")
            hs_src["fev"] = pd.to_datetime(hs_src["fev"], errors="coerce").dt.normalize()
            hs_src = (
                hs_src.dropna(subset=["ciclo_id", "fev"])
                .groupby("ciclo_id", dropna=False, as_index=False)
                .agg(h_start=("fev", "min"))
            )
            hs = pd.Series(pd.to_datetime(hs_src["h_start"], errors="coerce").dt.normalize().to_numpy(), index=hs_src["ciclo_id"])
        if {"ciclo_id", "pred_harvest_start_ml2"} <= set(df_out.columns):
            hs_fallback = (
                df_out.loc[stage_u.eq("VEG"), ["ciclo_id", "pred_harvest_start_ml2"]]
                .dropna(subset=["ciclo_id"])
                .drop_duplicates(subset=["ciclo_id"], keep="last")
            )
            if not hs_fallback.empty:
                hs_fallback["ciclo_id"] = hs_fallback["ciclo_id"].astype("string")
                hs_pred = pd.Series(
                    pd.to_datetime(hs_fallback["pred_harvest_start_ml2"], errors="coerce").dt.normalize().to_numpy(),
                    index=hs_fallback["ciclo_id"],
                )
                if hs.empty:
                    hs = hs_pred
                else:
                    hs = hs.combine_first(hs_pred)

    hs_row = df_out["ciclo_id"].astype("string").map(hs) if not hs.empty else pd.Series(pd.NaT, index=df_out.index)
    fev_ml2 = fev_orig.copy()
    m_retime = is_chain & hs_row.notna() & day_ml2.notna()
    if m_retime.any():
        day_int = np.rint(day_ml2.loc[m_retime]).astype(int)
        fev_ml2.loc[m_retime] = (
            pd.to_datetime(hs_row.loc[m_retime], errors="coerce").dt.normalize()
            + pd.to_timedelta(day_int - 1, unit="D")
        ).dt.normalize()
    df_out["fecha_evento_ml2"] = fev_ml2

    # Rebuild fecha_post_ml2 from reprojected fecha_evento_ml2 + dh when available.
    dh_ml2 = _num_series(df_out, "pred_ml2_dh_dias", default=np.nan)
    m_post_fp = is_post & pd.to_datetime(df_out["fecha_evento_ml2"], errors="coerce").notna() & dh_ml2.notna() & (dh_ml2 >= 0.0)
    if m_post_fp.any():
        dh_int = np.rint(dh_ml2.loc[m_post_fp]).astype(int)
        fev_ref = pd.to_datetime(df_out.loc[m_post_fp, "fecha_evento_ml2"], errors="coerce").dt.normalize()
        df_out.loc[m_post_fp, "fecha_post_ml2"] = (fev_ref + pd.to_timedelta(dh_int, unit="D")).dt.normalize()

    # Keep observed-real dates fixed when using operativo anchoring.
    if anchor_real:
        fpost_orig = pd.to_datetime(df_out.get("fecha_post"), errors="coerce").dt.normalize()
        m_real_hg = (
            pd.to_numeric(df_out.get("mask_target_factor_tallos_dia"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(df_out.get("mask_target_share_grado"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(df_out.get("mask_target_factor_peso_tallo"), errors="coerce").fillna(0.0).gt(0.0)
        ) & is_hg
        m_real_post = (
            pd.to_numeric(df_out.get("mask_target_dh_dias"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(df_out.get("mask_target_factor_hidr"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(df_out.get("mask_target_factor_desp"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(df_out.get("mask_target_factor_ajuste"), errors="coerce").fillna(0.0).gt(0.0)
        ) & is_post
        m_real_evt = (m_real_hg | m_real_post) & fev_orig.notna()
        if m_real_evt.any():
            df_out.loc[m_real_evt, "fecha_evento_ml2"] = fev_orig.loc[m_real_evt]
        m_real_fp = m_real_post & fpost_orig.notna()
        if m_real_fp.any():
            df_out.loc[m_real_fp, "fecha_post_ml2"] = fpost_orig.loc[m_real_fp]

    # ------------------------------------------------------------------
    # Mass balance: ML2 harvest/post totals should close vs tallos_proy.
    # ------------------------------------------------------------------
    if {"ciclo_id", "tallos_proy", "tallos_pred_ml2_grado_dia"} <= set(df_out.columns):
        cyc_bal = (
            df_out.loc[is_hg, ["ciclo_id", "tallos_pred_ml2_grado_dia", "tallos_proy"]]
            .groupby("ciclo_id", dropna=False, as_index=False)
            .agg(
                tallos_ml2_total=("tallos_pred_ml2_grado_dia", "sum"),
                tallos_target=("tallos_proy", "max"),
            )
        )
        cyc_bal["tallos_ml2_total"] = pd.to_numeric(cyc_bal["tallos_ml2_total"], errors="coerce")
        cyc_bal["tallos_target"] = pd.to_numeric(cyc_bal["tallos_target"], errors="coerce")
        cyc_bal["tallos_target_eff"] = cyc_bal["tallos_target"]

        # ML2 global keeps model-only rows, but can use observed signal as lower bound
        # to avoid collapsing totals when real progress is already above original target.
        if layer_name == "ML2_GLOBAL":
            m_obs = (
                pd.to_numeric(df_out.get("mask_target_factor_tallos_dia"), errors="coerce").fillna(0.0).gt(0.0)
                & pd.to_numeric(df_out.get("mask_target_share_grado"), errors="coerce").fillna(0.0).gt(0.0)
            ) & is_hg
            if bool(m_obs.any()):
                obs_day = (
                    pd.to_numeric(df_out.loc[m_obs, "target_factor_tallos_dia"], errors="coerce")
                    * pd.to_numeric(df_out.loc[m_obs, "tallos_pred_baseline_dia"], errors="coerce")
                )
                obs_share = pd.to_numeric(df_out.loc[m_obs, "target_share_grado"], errors="coerce")
                obs_grade = (obs_day * obs_share).clip(lower=0.0)
                obs_tbl = (
                    pd.DataFrame(
                        {
                            "ciclo_id": df_out.loc[m_obs, "ciclo_id"].astype("string").to_numpy(),
                            "tallos_obs_real": obs_grade.to_numpy(dtype=np.float64),
                        }
                    )
                    .groupby("ciclo_id", dropna=False, as_index=False)
                    .agg(tallos_obs_real=("tallos_obs_real", "sum"))
                )
                cyc_bal = cyc_bal.merge(obs_tbl, on="ciclo_id", how="left")
                cyc_bal["tallos_obs_real"] = pd.to_numeric(cyc_bal["tallos_obs_real"], errors="coerce").fillna(0.0)
                cyc_bal["tallos_target_eff"] = np.maximum(
                    pd.to_numeric(cyc_bal["tallos_target"], errors="coerce").fillna(0.0),
                    cyc_bal["tallos_obs_real"],
                )

        cyc_bal["scale_ml2_mass"] = np.where(
            cyc_bal["tallos_target_eff"].notna()
            & (cyc_bal["tallos_target_eff"] > 0.0)
            & cyc_bal["tallos_ml2_total"].notna()
            & (cyc_bal["tallos_ml2_total"] > 0.0),
            cyc_bal["tallos_target_eff"] / cyc_bal["tallos_ml2_total"],
            1.0,
        )
        scale_map = pd.Series(
            cyc_bal["scale_ml2_mass"].to_numpy(dtype=np.float64),
            index=cyc_bal["ciclo_id"].astype("string"),
        )
        srow = df_out["ciclo_id"].astype("string").map(scale_map).fillna(1.0)
        df_out["scale_ml2_mass"] = srow

        cols_hg = ["tallos_pred_ml2_dia", "tallos_pred_ml2_grado_dia"]
        for c in cols_hg:
            if c in df_out.columns:
                df_out.loc[is_hg, c] = pd.to_numeric(df_out.loc[is_hg, c], errors="coerce") * srow.loc[is_hg]

        cols_post = [
            "tallos_post_ml2_proy",
            "kg_verde_ml2",
            "gramos_verde_ml2",
            "cajas_split_grado_dia_ml2",
            "cajas_ml1_grado_dia_ml2",
            "cajas_post_seed_ml2",
        ]
        for c in cols_post:
            if c in df_out.columns:
                df_out.loc[is_post, c] = pd.to_numeric(df_out.loc[is_post, c], errors="coerce") * srow.loc[is_post]
    else:
        df_out["scale_ml2_mass"] = 1.0

    # Keep day totals coherent with grade totals after all adjustments.
    if bool(is_hg.any()) and {"tallos_pred_ml2_grado_dia", "tallos_pred_ml2_dia"} <= set(df_out.columns):
        day_col = "fecha_evento_ml2" if "fecha_evento_ml2" in df_out.columns else ("fecha_evento" if "fecha_evento" in df_out.columns else None)
        day_key = [c for c in ["ciclo_id", day_col, "bloque_base", "variedad_canon"] if c and c in df_out.columns]
        if day_key:
            hg_day = (
                df_out.loc[is_hg, day_key + ["tallos_pred_ml2_grado_dia"]]
                .groupby(day_key, dropna=False, as_index=False)
                .agg(tallos_dia_new=("tallos_pred_ml2_grado_dia", "sum"))
            )
            # If tail is unnaturally flat, apply a light decay while preserving cycle mass.
            date_col_local = day_key[1] if len(day_key) >= 2 else None
            grp_shape_cols = [c for c in day_key if c != date_col_local]
            if date_col_local and grp_shape_cols:
                adj_parts: list[pd.DataFrame] = []
                for _, g in hg_day.groupby(grp_shape_cols, dropna=False, sort=False):
                    gg = g.sort_values(date_col_local, kind="mergesort").copy()
                    vals = pd.to_numeric(gg["tallos_dia_new"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
                    n = len(vals)
                    if n >= 5:
                        tail_n = int(min(6, n))
                        tail = vals[-tail_n:]
                        t_mean = float(np.nanmean(tail))
                        t_std = float(np.nanstd(tail))
                        if t_mean > 1e-9 and (t_std / t_mean) < 0.01:
                            adj = vals.copy()
                            decay = np.linspace(1.00, 0.90, tail_n, dtype=np.float64)
                            adj[-tail_n:] = tail * decay
                            s0 = float(np.nansum(vals))
                            s1 = float(np.nansum(adj))
                            if s0 > 0.0 and s1 > 0.0:
                                adj = adj * (s0 / s1)
                            vals = np.clip(adj, 0.0, np.inf)
                    gg["tallos_dia_new"] = vals
                    adj_parts.append(gg)
                if adj_parts:
                    hg_day = pd.concat(adj_parts, ignore_index=True)

            hg_day_idx = pd.MultiIndex.from_frame(hg_day[day_key])
            hg_day_map = pd.Series(
                pd.to_numeric(hg_day["tallos_dia_new"], errors="coerce").to_numpy(dtype=np.float64),
                index=hg_day_idx,
            )
            hg_rows = pd.MultiIndex.from_frame(df_out.loc[is_hg, day_key])
            df_out.loc[is_hg, "tallos_pred_ml2_dia"] = hg_rows.map(hg_day_map).to_numpy(dtype=np.float64)
            if bool(is_post.any()):
                post_rows = pd.MultiIndex.from_frame(df_out.loc[is_post, day_key])
                post_map = post_rows.map(hg_day_map)
                post_old = pd.to_numeric(df_out.loc[is_post, "tallos_pred_ml2_dia"], errors="coerce")
                df_out.loc[is_post, "tallos_pred_ml2_dia"] = pd.Series(post_map, index=df_out.loc[is_post].index).where(
                    pd.Series(post_map, index=df_out.loc[is_post].index).notna(),
                    post_old,
                )

    kg_green = _num_series(df_out, "kg_verde_ml2", default=np.nan)
    if not np.isfinite(kg_green).any():
        kg_green = _num_series(df_out, "kg_verde_ref", default=0.0)
    prod_ml2 = (
        _num_series(df_out, "pred_ml2_factor_hidr", default=1.0)
        * _num_series(df_out, "pred_ml2_factor_desp", default=1.0)
        * _num_series(df_out, "pred_ml2_factor_ajuste", default=1.0)
    )
    df_out["kg_post_ml2"] = kg_green * prod_ml2

    cajas_green = _num_series(df_out, "cajas_split_grado_dia_ml2", default=np.nan)
    if not np.isfinite(cajas_green).any():
        cajas_green = _num_series(df_out, "cajas_split_grado_dia", default=0.0)
    df_out["cajas_post_ml2"] = cajas_green * prod_ml2
    df_out["aprovechamiento_ml2"] = np.where(kg_green > 0, df_out["kg_post_ml2"] / kg_green, np.nan)

    df_out["ml2_multitask_nn_run_id"] = run_id
    df_out["ml2_layer"] = layer_name
    df_out["ml2_dynamic_iters"] = int(dyn["iterations"])
    df_out["ml2_dynamic_converged"] = bool(dyn["converged"])
    df_out["ml2_dynamic_max_rel_change"] = float(dyn["max_rel_change"]) if np.isfinite(dyn["max_rel_change"]) else np.nan
    df_out["ml2_dynamic_anchor_real"] = bool(anchor_real)
    if "ml2_anchor_real" not in df_out.columns:
        df_out["ml2_anchor_real"] = False
    return df_out


def _dedup_harvest_grade_rows(df_out: pd.DataFrame, prefer_real: bool) -> pd.DataFrame:
    if df_out.empty or "stage" not in df_out.columns:
        return df_out
    st = _stage_series(df_out)
    is_hg = st.eq("HARVEST_GRADE")
    if not bool(is_hg.any()):
        return df_out

    date_col = "fecha_evento_ml2" if "fecha_evento_ml2" in df_out.columns else ("fecha_evento" if "fecha_evento" in df_out.columns else None)
    key = [c for c in ["ciclo_id", date_col, "bloque_base", "variedad_canon", "grado"] if c and c in df_out.columns]
    if not key:
        return df_out

    hg = df_out.loc[is_hg].copy()
    score = pd.Series(0.0, index=hg.index, dtype="float64")
    if prefer_real:
        for c in [
            "mask_target_factor_tallos_dia",
            "mask_target_share_grado",
            "mask_target_factor_peso_tallo",
        ]:
            if c in hg.columns:
                score = score + pd.to_numeric(hg[c], errors="coerce").fillna(0.0)
    hg["__score"] = score

    sort_cols = ["__score"]
    asc = [False]
    if "row_id" in hg.columns:
        sort_cols.append("row_id")
        asc.append(True)
    hg = hg.sort_values(sort_cols, ascending=asc, kind="mergesort")
    hg = hg.drop_duplicates(subset=key, keep="first").drop(columns=["__score"])

    out = pd.concat([df_out.loc[~is_hg], hg], ignore_index=True)
    order_cols = [c for c in ["ciclo_id", date_col, "stage", "bloque_base", "variedad_canon", "grado", "destino"] if c in out.columns]
    if order_cols:
        out = out.sort_values(order_cols, kind="mergesort").reset_index(drop=True)
    return out


def _overlay_operational_real_harvest(df_out: pd.DataFrame) -> pd.DataFrame:
    if df_out.empty or not IN_FACT_PESO_REAL.exists():
        return df_out

    out = df_out.copy()
    st = _stage_series(out)
    is_hg = st.eq("HARVEST_GRADE")
    if not bool(is_hg.any()):
        return out

    date_col = "fecha_evento_ml2" if "fecha_evento_ml2" in out.columns else ("fecha_evento" if "fecha_evento" in out.columns else None)
    key_cols = [c for c in [date_col, "bloque_base", "variedad_canon", "grado"] if c and c in out.columns]
    day_cols = [c for c in [date_col, "bloque_base", "variedad_canon"] if c and c in out.columns]
    if len(key_cols) < 4:
        return out

    hg = out.loc[is_hg].copy()
    hg["__idx"] = hg.index
    hg[date_col] = _to_date(hg[date_col])
    hg["bloque_base_int"] = pd.to_numeric(hg["bloque_base"], errors="coerce").astype("Int64")
    hg["grado_int"] = pd.to_numeric(hg["grado"], errors="coerce").astype("Int64")
    hg["variedad_canon"] = _canon_str(hg["variedad_canon"])

    fr = read_parquet(IN_FACT_PESO_REAL).copy()
    fr.columns = [str(c).strip() for c in fr.columns]
    fr["fecha_evento"] = _to_date(fr["fecha"])
    if "bloque_base" not in fr.columns:
        if "bloque_padre" in fr.columns:
            fr["bloque_base"] = fr["bloque_padre"]
        elif "bloque" in fr.columns:
            fr["bloque_base"] = fr["bloque"]
        else:
            fr["bloque_base"] = pd.NA
    fr["bloque_base_int"] = pd.to_numeric(fr["bloque_base"], errors="coerce").astype("Int64")
    fr["grado_int"] = pd.to_numeric(fr["grado"], errors="coerce").astype("Int64")

    if "variedad_canon" not in fr.columns:
        if "variedad" in fr.columns and IN_DIM_VARIEDAD.exists():
            dim = read_parquet(IN_DIM_VARIEDAD).copy()
            dim.columns = [str(c).strip() for c in dim.columns]
            dim["variedad_raw_norm"] = _canon_str(dim["variedad_raw"])
            dim["variedad_canon"] = _canon_str(dim["variedad_canon"])
            fr["variedad_raw_norm"] = _canon_str(fr["variedad"])
            fr = fr.merge(dim[["variedad_raw_norm", "variedad_canon"]].drop_duplicates(), on="variedad_raw_norm", how="left")
            fr["variedad_canon"] = fr["variedad_canon"].fillna(fr["variedad_raw_norm"])
        else:
            fr["variedad_canon"] = "UNKNOWN"
    fr["variedad_canon"] = _canon_str(fr["variedad_canon"])
    fr["tallos_real"] = _num_series(fr, "tallos_real", default=0.0)
    fr["peso_real_g"] = _num_series(fr, "peso_real_g", default=np.nan)
    fr["peso_tallo_real_g"] = _num_series(fr, "peso_tallo_real_g", default=np.nan)
    fr["kg_verde_real"] = fr["peso_real_g"] / 1000.0
    fr["kg_verde_real"] = fr["kg_verde_real"].fillna(fr["tallos_real"] * fr["peso_tallo_real_g"] / 1000.0)
    fr["peso_tallo_real_g"] = fr["peso_tallo_real_g"].fillna(_safe_div(fr["peso_real_g"], fr["tallos_real"], default=np.nan))

    frg = (
        fr[["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int", "tallos_real", "kg_verde_real", "peso_tallo_real_g"]]
        .dropna(subset=["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"])
        .groupby(["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"], dropna=False, as_index=False)
        .agg(
            tallos_grado_real=("tallos_real", "sum"),
            kg_verde_grado_real=("kg_verde_real", "sum"),
            peso_tallo_real_g=("peso_tallo_real_g", "mean"),
        )
    )
    if frg.empty:
        return out

    day = (
        frg.groupby(["fecha_evento", "bloque_base_int", "variedad_canon"], dropna=False, as_index=False)
        .agg(tallos_dia_real=("tallos_grado_real", "sum"))
    )
    frg = frg.merge(day, on=["fecha_evento", "bloque_base_int", "variedad_canon"], how="left")
    frg["share_grado_real"] = _safe_div(frg["tallos_grado_real"], frg["tallos_dia_real"], default=np.nan)

    hg = hg.merge(
        frg,
        left_on=[date_col, "bloque_base_int", "variedad_canon", "grado_int"],
        right_on=["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"],
        how="left",
    )
    hg = hg.merge(
        day.rename(columns={"tallos_dia_real": "tallos_dia_real_day"}),
        left_on=[date_col, "bloque_base_int", "variedad_canon"],
        right_on=["fecha_evento", "bloque_base_int", "variedad_canon"],
        how="left",
    )

    # Fill explicit zero-harvest days up to the last observed real date:
    # if a day exists in the model grid but there is no real entry before/at
    # the latest real date for that block-variety, treat it as real zero.
    obs_last = (
        day.groupby(["bloque_base_int", "variedad_canon"], dropna=False, as_index=False)
        .agg(last_real_fecha=("fecha_evento", "max"))
    )
    hg = hg.merge(obs_last, on=["bloque_base_int", "variedad_canon"], how="left")

    day_real = pd.to_numeric(hg.get("tallos_dia_real_day"), errors="coerce")
    fev = _to_date(hg[date_col])
    m_zero_obs = day_real.isna() & fev.notna() & hg["last_real_fecha"].notna() & (fev <= hg["last_real_fecha"])
    day_real = day_real.where(~m_zero_obs, 0.0)
    hg["tallos_dia_real_day"] = day_real

    m_day = day_real.notna()
    if not bool(m_day.any()):
        return out

    grade_real = pd.to_numeric(hg.get("tallos_grado_real"), errors="coerce").fillna(0.0)
    share_real = _safe_div(grade_real, day_real, default=0.0)
    kg_grade_real = pd.to_numeric(hg.get("kg_verde_grado_real"), errors="coerce").fillna(0.0)
    m_grade = m_day & grade_real.gt(0.0)

    idx = pd.to_numeric(hg.loc[m_day, "__idx"], errors="coerce").astype("Int64")
    m_idx = idx.notna()
    idx = idx.loc[m_idx].astype(int)
    hgm = hg.loc[m_day].loc[m_idx]

    out.loc[idx, "tallos_pred_ml2_grado_dia"] = pd.to_numeric(hgm["tallos_grado_real"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    out.loc[idx, "tallos_pred_ml2_dia"] = pd.to_numeric(hgm["tallos_dia_real_day"], errors="coerce").to_numpy(dtype=np.float32)
    out.loc[idx, "share_grado_ml2_norm"] = _safe_div(
        pd.to_numeric(hgm["tallos_grado_real"], errors="coerce").fillna(0.0),
        pd.to_numeric(hgm["tallos_dia_real_day"], errors="coerce"),
        default=0.0,
    ).to_numpy(dtype=np.float32)
    if "pred_ml2_share_grado" in out.columns:
        out.loc[idx, "pred_ml2_share_grado"] = _safe_div(
            pd.to_numeric(hgm["tallos_grado_real"], errors="coerce").fillna(0.0),
            pd.to_numeric(hgm["tallos_dia_real_day"], errors="coerce"),
            default=0.0,
        ).to_numpy(dtype=np.float32)
    if "pred_ml2_factor_tallos_dia" in out.columns:
        fac = _safe_div(
            pd.to_numeric(hgm["tallos_dia_real_day"], errors="coerce"),
            pd.to_numeric(hgm["tallos_pred_baseline_dia"], errors="coerce"),
            default=np.nan,
        )
        out.loc[idx, "pred_ml2_factor_tallos_dia"] = fac.to_numpy(dtype=np.float32)
    if "pred_ml2_factor_peso_tallo" in out.columns:
        idx_g = pd.to_numeric(hg.loc[m_grade, "__idx"], errors="coerce").astype("Int64")
        mg_idx = idx_g.notna()
        idx_g = idx_g.loc[mg_idx].astype(int)
        hgg = hg.loc[m_grade].loc[mg_idx]
        facp = _safe_div(
            pd.to_numeric(hgg["peso_tallo_real_g"], errors="coerce"),
            pd.to_numeric(hgg["peso_tallo_baseline_g"], errors="coerce"),
            default=np.nan,
        )
        out.loc[idx_g, "pred_ml2_factor_peso_tallo"] = facp.to_numpy(dtype=np.float32)
    if "kg_verde_ml2" in out.columns:
        out.loc[idx, "kg_verde_ml2"] = pd.to_numeric(hgm["kg_verde_grado_real"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if "gramos_verde_ml2" in out.columns:
        out.loc[idx, "gramos_verde_ml2"] = pd.to_numeric(hgm["kg_verde_grado_real"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) * 1000.0
    if "ml2_anchor_real" not in out.columns:
        out["ml2_anchor_real"] = False
    out.loc[idx, "ml2_anchor_real"] = True

    # Propagate corrected HG day/grade tallos to POST rows of same key.
    if bool(st.eq("POST").any()):
        post = out.loc[st.eq("POST")].copy()
        post["__idx"] = post.index
        post[date_col] = _to_date(post[date_col]) if date_col in post.columns else pd.NaT
        post["bloque_base_int"] = pd.to_numeric(post.get("bloque_base"), errors="coerce").astype("Int64")
        post["grado_int"] = pd.to_numeric(post.get("grado"), errors="coerce").astype("Int64")
        post["variedad_canon"] = _canon_str(post.get("variedad_canon", "UNKNOWN"))

        hg_map = hg.loc[m_day, [date_col, "bloque_base_int", "variedad_canon", "grado_int", "tallos_grado_real", "tallos_dia_real_day"]].copy()
        p2 = post.merge(
            hg_map,
            left_on=[date_col, "bloque_base_int", "variedad_canon", "grado_int"],
            right_on=[date_col, "bloque_base_int", "variedad_canon", "grado_int"],
            how="left",
        )
        mp = pd.to_numeric(p2["tallos_dia_real_day"], errors="coerce").notna()
        if bool(mp.any()):
            pidx = pd.to_numeric(p2.loc[mp, "__idx"], errors="coerce").astype("Int64")
            mp_idx = pidx.notna()
            pidx = pidx.loc[mp_idx].astype(int)
            p2m = p2.loc[mp].loc[mp_idx]
            out.loc[pidx, "tallos_pred_ml2_grado_dia"] = pd.to_numeric(p2m["tallos_grado_real"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            out.loc[pidx, "tallos_pred_ml2_dia"] = pd.to_numeric(p2m["tallos_dia_real_day"], errors="coerce").to_numpy(dtype=np.float32)

    return out


def _compute_metrics_bundle(
    df: pd.DataFrame,
    df_out: pd.DataFrame,
    specs: list[dict],
    dyn: dict,
) -> dict[str, dict]:
    ycorr_adj = dyn["ycorr_adj"]
    global_metrics = _final_metrics(df_part=df, y_corr_pred=ycorr_adj, target_specs=specs)

    active_metrics: dict[str, float]
    if "is_active_cycle" in df.columns:
        m_active = df["is_active_cycle"].fillna(False).astype(bool).to_numpy()
        if m_active.any():
            active_metrics = _final_metrics(
                df_part=df.loc[m_active].reset_index(drop=True),
                y_corr_pred=ycorr_adj[m_active],
                target_specs=specs,
            )
        else:
            active_metrics = {"mae_ml2_avg": np.nan, "r2_ml2_avg": np.nan}
        active_metrics["n_rows"] = int(m_active.sum())
    else:
        active_metrics = {"mae_ml2_avg": np.nan, "r2_ml2_avg": np.nan, "n_rows": 0}

    global_dyn_metrics = _final_metrics_from_pred_cols(df_part=df_out, target_specs=specs)
    if "is_active_cycle" in df_out.columns:
        m_active_s = df_out["is_active_cycle"].fillna(False).astype(bool)
        if m_active_s.any():
            active_dyn_metrics = _final_metrics_from_pred_cols(
                df_part=df_out.loc[m_active_s].reset_index(drop=True),
                target_specs=specs,
            )
        else:
            active_dyn_metrics = {"mae_ml2_avg": np.nan, "r2_ml2_avg": np.nan}
        active_dyn_metrics["n_rows"] = int(m_active_s.sum())
    else:
        active_dyn_metrics = {"mae_ml2_avg": np.nan, "r2_ml2_avg": np.nan, "n_rows": 0}

    return {
        "global_metrics": global_metrics,
        "active_metrics": active_metrics,
        "global_dyn_metrics": global_dyn_metrics,
        "active_dyn_metrics": active_dyn_metrics,
    }


def _metrics_frame(
    run_id: str,
    layer_name: str,
    in_path: Path,
    out_path: Path,
    dyn: dict,
    bundle: dict[str, dict],
    n_rows: int,
    anchor_real: bool,
) -> pd.DataFrame:
    global_metrics = bundle["global_metrics"]
    active_metrics = bundle["active_metrics"]
    global_dyn_metrics = bundle["global_dyn_metrics"]
    active_dyn_metrics = bundle["active_dyn_metrics"]
    metrics_row = {
        "run_id": run_id,
        "ml2_layer": layer_name,
        "input_path": str(in_path).replace("\\", "/"),
        "output_path": str(out_path).replace("\\", "/"),
        "dynamic_iters": int(dyn["iterations"]),
        "dynamic_converged": bool(dyn["converged"]),
        "dynamic_max_rel_change": float(dyn["max_rel_change"]) if np.isfinite(dyn["max_rel_change"]) else np.nan,
        "dynamic_anchor_real": bool(anchor_real),
        "r2_global_ml1_avg": global_metrics.get("r2_ml1_avg"),
        "r2_global_ml2_avg": global_metrics.get("r2_ml2_avg"),
        "mae_global_ml1_avg": global_metrics.get("mae_ml1_avg"),
        "mae_global_ml2_avg": global_metrics.get("mae_ml2_avg"),
        "r2_global_ml2_avg_dynamic": global_dyn_metrics.get("r2_ml2_avg"),
        "mae_global_ml2_avg_dynamic": global_dyn_metrics.get("mae_ml2_avg"),
        "r2_active_ml2_avg": active_metrics.get("r2_ml2_avg"),
        "mae_active_ml2_avg": active_metrics.get("mae_ml2_avg"),
        "r2_active_ml2_avg_dynamic": active_dyn_metrics.get("r2_ml2_avg"),
        "mae_active_ml2_avg_dynamic": active_dyn_metrics.get("mae_ml2_avg"),
        "n_rows": int(n_rows),
        "n_active_rows": int(active_metrics.get("n_rows", 0)),
        "created_at_utc": pd.Timestamp.now("UTC"),
    }
    return pd.DataFrame(
        [
            {
                **metrics_row,
                **{f"global_{k}": v for k, v in global_metrics.items()},
                **{f"active_{k}": v for k, v in active_metrics.items()},
                **{f"global_dynamic_{k}": v for k, v in global_dyn_metrics.items()},
                **{f"active_dynamic_{k}": v for k, v in active_dyn_metrics.items()},
            }
        ]
    )


def _expand_ml2_horizon_rows(df_out: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df_out.empty or "stage" not in df_out.columns or "ciclo_id" not in df_out.columns:
        return df_out, 0

    work = df_out.copy()
    work["stage"] = _stage_series(work)
    work["ciclo_id"] = work["ciclo_id"].astype("string").fillna("UNKNOWN")
    if "fecha_evento_ml2" in work.columns:
        work["fecha_evento_ml2"] = _to_date(work["fecha_evento_ml2"])
    if "fecha_evento" in work.columns:
        work["fecha_evento"] = _to_date(work["fecha_evento"])
    if "fecha_post_ml2" in work.columns:
        work["fecha_post_ml2"] = _to_date(work["fecha_post_ml2"])

    max_row_id = pd.to_numeric(work.get("row_id"), errors="coerce").max()
    next_row_id = int(max_row_id) + 1 if np.isfinite(max_row_id) else 1

    target_cols = [c for c in work.columns if c.startswith("target_")]
    mask_cols = [c for c in work.columns if c.startswith("mask_")]

    rows_new: list[pd.DataFrame] = []

    veg = work.loc[work["stage"].eq("VEG"), ["ciclo_id", "pred_harvest_start_ml2", "pred_ml2_n_harvest_days"]].copy()
    if veg.empty:
        return work, 0
    veg = veg.sort_values(["ciclo_id"]).drop_duplicates(subset=["ciclo_id"], keep="last")
    veg["pred_harvest_start_ml2"] = _to_date(veg["pred_harvest_start_ml2"])
    veg["n_days_i"] = np.rint(pd.to_numeric(veg["pred_ml2_n_harvest_days"], errors="coerce")).astype("Int64").clip(lower=1)
    veg = veg.dropna(subset=["pred_harvest_start_ml2", "n_days_i"])
    if veg.empty:
        return work, 0

    for r in veg.itertuples(index=False):
        cid = str(r.ciclo_id)
        n_days = int(r.n_days_i)
        h_start = pd.Timestamp(r.pred_harvest_start_ml2).normalize()

        cyc = work.loc[work["ciclo_id"].eq(cid)].copy()
        hg = cyc.loc[cyc["stage"].eq("HARVEST_GRADE")].copy()
        if hg.empty:
            continue

        fe_hg = pd.to_datetime(hg.get("fecha_evento_ml2", hg.get("fecha_evento")), errors="coerce").dt.normalize()
        day_hg = pd.to_numeric(hg.get("day_in_harvest_ml2", hg.get("day_in_harvest")), errors="coerce").round()
        if fe_hg.notna().any():
            m_day1 = day_hg.eq(1.0) & fe_hg.notna()
            if bool(m_day1.any()):
                h_start = pd.to_datetime(fe_hg.loc[m_day1], errors="coerce").min().normalize()
            else:
                h_start = pd.to_datetime(fe_hg, errors="coerce").min().normalize()
        existing_dates = set(fe_hg.dropna().tolist())
        if not existing_dates:
            continue

        missing_dates = [
            (h_start + pd.Timedelta(days=d - 1)).normalize()
            for d in range(1, n_days + 1)
            if (h_start + pd.Timedelta(days=d - 1)).normalize() not in existing_dates
        ]
        if not missing_dates:
            continue

        last_date = max(existing_dates)
        hg_tpl = hg.loc[fe_hg.eq(last_date)].copy()
        if hg_tpl.empty:
            hg_tpl = hg.tail(1).copy()
            if hg_tpl.empty:
                continue

        post = cyc.loc[cyc["stage"].eq("POST")].copy()
        post_tpl = pd.DataFrame()
        if not post.empty:
            fe_post = pd.to_datetime(post.get("fecha_evento_ml2", post.get("fecha_evento")), errors="coerce").dt.normalize()
            post_tpl = post.loc[fe_post.eq(last_date)].copy()
            if post_tpl.empty:
                post_tpl = post.tail(min(len(post), len(hg_tpl))).copy()

        for fev in missing_dates:
            d = int((pd.Timestamp(fev) - h_start).days) + 1
            rel = float(d) / float(max(n_days, 1))
            tail_step = int((pd.Timestamp(fev) - pd.Timestamp(last_date)).days)
            tail_decay = 1.0
            if tail_step > 0:
                tail_decay = float(max(0.82, 1.0 - 0.03 * tail_step))

            add_hg = hg_tpl.copy()
            add_hg["fecha_evento_ml2"] = fev
            if "fecha_evento" in add_hg.columns:
                add_hg["fecha_evento"] = fev
            add_hg["day_in_harvest_ml2"] = float(d)
            if "day_in_harvest" in add_hg.columns:
                add_hg["day_in_harvest"] = float(d)
            add_hg["pred_ml2_n_harvest_days"] = float(n_days)
            if "n_harvest_days_ml2" in add_hg.columns:
                add_hg["n_harvest_days_ml2"] = float(n_days)
            if "n_harvest_days" in add_hg.columns:
                add_hg["n_harvest_days"] = float(n_days)
            if "rel_pos_ml2" in add_hg.columns:
                add_hg["rel_pos_ml2"] = rel
            if "rel_pos" in add_hg.columns:
                add_hg["rel_pos"] = rel
            if tail_step > 0:
                for c in [
                    "tallos_pred_ml2_dia",
                    "tallos_pred_ml2_grado_dia",
                    "kg_verde_ml2",
                    "gramos_verde_ml2",
                    "cajas_split_grado_dia_ml2",
                    "cajas_ml1_grado_dia_ml2",
                    "cajas_post_seed_ml2",
                ]:
                    if c in add_hg.columns:
                        add_hg[c] = pd.to_numeric(add_hg[c], errors="coerce") * tail_decay
            for c in target_cols:
                add_hg[c] = np.nan
            for c in mask_cols:
                add_hg[c] = 0.0
            add_hg["ml2_row_generated"] = 1
            if "row_id" in add_hg.columns:
                add_hg["row_id"] = np.arange(next_row_id, next_row_id + len(add_hg), dtype=np.int64)
                next_row_id += len(add_hg)
            rows_new.append(add_hg)

            if not post_tpl.empty:
                add_p = post_tpl.copy()
                add_p["fecha_evento_ml2"] = fev
                if "fecha_evento" in add_p.columns:
                    add_p["fecha_evento"] = fev
                add_p["day_in_harvest_ml2"] = float(d)
                if "day_in_harvest" in add_p.columns:
                    add_p["day_in_harvest"] = float(d)
                add_p["pred_ml2_n_harvest_days"] = float(n_days)
                if "n_harvest_days_ml2" in add_p.columns:
                    add_p["n_harvest_days_ml2"] = float(n_days)
                if "n_harvest_days" in add_p.columns:
                    add_p["n_harvest_days"] = float(n_days)
                if "rel_pos_ml2" in add_p.columns:
                    add_p["rel_pos_ml2"] = rel
                if "rel_pos" in add_p.columns:
                    add_p["rel_pos"] = rel
                if tail_step > 0:
                    for c in [
                        "tallos_pred_ml2_dia",
                        "tallos_pred_ml2_grado_dia",
                        "tallos_post_ml2_proy",
                        "kg_verde_ml2",
                        "gramos_verde_ml2",
                        "cajas_split_grado_dia_ml2",
                        "cajas_ml1_grado_dia_ml2",
                        "cajas_post_seed_ml2",
                    ]:
                        if c in add_p.columns:
                            add_p[c] = pd.to_numeric(add_p[c], errors="coerce") * tail_decay
                if "pred_ml2_dh_dias" in add_p.columns:
                    dh = np.rint(pd.to_numeric(add_p["pred_ml2_dh_dias"], errors="coerce")).astype("Int64")
                    add_p["fecha_post_ml2"] = _to_date(pd.to_datetime(fev) + pd.to_timedelta(dh.fillna(0).astype(int), unit="D"))
                for c in target_cols:
                    add_p[c] = np.nan
                for c in mask_cols:
                    add_p[c] = 0.0
                add_p["ml2_row_generated"] = 1
                if "row_id" in add_p.columns:
                    add_p["row_id"] = np.arange(next_row_id, next_row_id + len(add_p), dtype=np.int64)
                    next_row_id += len(add_p)
                rows_new.append(add_p)

    if not rows_new:
        if "ml2_row_generated" not in work.columns:
            work["ml2_row_generated"] = 0
        return work, 0

    ext = pd.concat([work] + rows_new, ignore_index=True)
    if "ml2_row_generated" not in ext.columns:
        ext["ml2_row_generated"] = 0
    ext["ml2_row_generated"] = pd.to_numeric(ext["ml2_row_generated"], errors="coerce").fillna(0).astype("int8")
    ext = ext.sort_values(["ciclo_id", "fecha_evento_ml2", "stage", "bloque_base", "variedad_canon", "grado", "destino"], kind="mergesort").reset_index(drop=True)
    return ext, int(len(ext) - len(work))


def main() -> None:
    args = _parse_args()
    run_id, meta, params = _load_artifacts(args.run_id)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")
    df = read_parquet(in_path).copy()
    shrinkage = {str(k): float(v) for k, v in dict(meta.get("correction_shrinkage", {})).items()}
    specs: list[dict] = list(meta["target_specs"])

    # ML2 puro: sin anclaje de reales (comparable 1:1 contra ML1).
    dyn_pure = _run_dynamic_inference(
        df=df,
        meta=meta,
        params=params,
        specs=specs,
        shrinkage=shrinkage,
        max_iters=args.max_iters,
        tol=args.tol,
        anchor_real=False,
    )
    df_out_pure = _build_dynamic_output(
        df=df,
        dyn=dyn_pure,
        specs=specs,
        run_id=run_id,
        layer_name="ML2_PURO",
        anchor_real=False,
    )
    df_out_pure, n_added_pure = _expand_ml2_horizon_rows(df_out_pure)
    df_out_pure = _dedup_harvest_grade_rows(df_out_pure, prefer_real=False)
    bundle_pure = _compute_metrics_bundle(df=df, df_out=df_out_pure, specs=specs, dyn=dyn_pure)

    # ML2 global: no reemplaza real fila-a-fila, pero sí reproyecta la cadena completa.
    # Usa la misma inferencia base y una salida temporal/mass-balance específica.
    dyn_global = dyn_pure
    df_out_global = _build_dynamic_output(
        df=df,
        dyn=dyn_global,
        specs=specs,
        run_id=run_id,
        layer_name="ML2_GLOBAL",
        anchor_real=False,
    )
    df_out_global, n_added_global = _expand_ml2_horizon_rows(df_out_global)
    df_out_global = _dedup_harvest_grade_rows(df_out_global, prefer_real=False)
    bundle_global = _compute_metrics_bundle(df=df, df_out=df_out_global, specs=specs, dyn=dyn_global)

    # ML2 operativo: ancla a reales observados y reproyecta.
    dyn_oper = _run_dynamic_inference(
        df=df,
        meta=meta,
        params=params,
        specs=specs,
        shrinkage=shrinkage,
        max_iters=args.max_iters,
        tol=args.tol,
        anchor_real=bool(args.anchor_real),
    )
    df_out_oper = _build_dynamic_output(
        df=df,
        dyn=dyn_oper,
        specs=specs,
        run_id=run_id,
        layer_name="ML2_OPERATIVO",
        anchor_real=bool(args.anchor_real),
    )
    df_out_oper, n_added_oper = _expand_ml2_horizon_rows(df_out_oper)
    df_out_oper = _dedup_harvest_grade_rows(df_out_oper, prefer_real=bool(args.anchor_real))
    if bool(args.anchor_real):
        df_out_oper = _overlay_operational_real_harvest(df_out_oper)
    bundle_oper = _compute_metrics_bundle(df=df, df_out=df_out_oper, specs=specs, dyn=dyn_oper)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path_global = Path(args.output_global) if args.output_global else (
        Path(args.output) if args.output else (OUT_DIR / f"pred_ml2_multitask_nn_global_{run_id}.parquet")
    )
    out_path_pure = Path(args.output_puro) if args.output_puro else (OUT_DIR / f"pred_ml2_multitask_nn_puro_{run_id}.parquet")
    out_path_oper = Path(args.output_operativo) if args.output_operativo else (OUT_DIR / f"pred_ml2_multitask_nn_operativo_{run_id}.parquet")
    legacy_path = OUT_DIR / f"pred_ml2_multitask_nn_{run_id}.parquet"

    write_parquet(df_out_global, out_path_global)
    write_parquet(df_out_pure, out_path_pure)
    write_parquet(df_out_oper, out_path_oper)
    if bool(args.write_legacy_alias):
        if out_path_global.resolve() != legacy_path.resolve():
            write_parquet(df_out_global, legacy_path)
        else:
            legacy_path = out_path_global

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics_global = _metrics_frame(
        run_id=run_id,
        layer_name="ML2_GLOBAL",
        in_path=in_path,
        out_path=out_path_global,
        dyn=dyn_global,
        bundle=bundle_global,
        n_rows=len(df_out_global),
        anchor_real=False,
    )
    metrics_pure = _metrics_frame(
        run_id=run_id,
        layer_name="ML2_PURO",
        in_path=in_path,
        out_path=out_path_pure,
        dyn=dyn_pure,
        bundle=bundle_pure,
        n_rows=len(df_out_pure),
        anchor_real=False,
    )
    metrics_oper = _metrics_frame(
        run_id=run_id,
        layer_name="ML2_OPERATIVO",
        in_path=in_path,
        out_path=out_path_oper,
        dyn=dyn_oper,
        bundle=bundle_oper,
        n_rows=len(df_out_oper),
        anchor_real=bool(args.anchor_real),
    )

    metrics_global_path = EVAL_DIR / f"ml2_multitask_nn_apply_metrics_global_{run_id}.parquet"
    metrics_pure_path = EVAL_DIR / f"ml2_multitask_nn_apply_metrics_puro_{run_id}.parquet"
    metrics_oper_path = EVAL_DIR / f"ml2_multitask_nn_apply_metrics_operativo_{run_id}.parquet"
    metrics_layers_path = EVAL_DIR / f"ml2_multitask_nn_apply_metrics_layers_{run_id}.parquet"
    metrics_legacy_path = EVAL_DIR / f"ml2_multitask_nn_apply_metrics_{run_id}.parquet"

    write_parquet(metrics_global, metrics_global_path)
    write_parquet(metrics_pure, metrics_pure_path)
    write_parquet(metrics_oper, metrics_oper_path)
    write_parquet(pd.concat([metrics_global, metrics_pure, metrics_oper], ignore_index=True), metrics_layers_path)
    # legacy metrics alias keeps global semantics.
    write_parquet(metrics_global, metrics_legacy_path)

    g_global = bundle_global["global_metrics"]
    g_pure = bundle_pure["global_metrics"]
    g_oper = bundle_oper["global_metrics"]
    gd_global = bundle_global["global_dyn_metrics"]
    gd_pure = bundle_pure["global_dyn_metrics"]
    gd_oper = bundle_oper["global_dyn_metrics"]
    ad_global = bundle_global["active_dyn_metrics"]
    ad_pure = bundle_pure["active_dyn_metrics"]
    ad_oper = bundle_oper["active_dyn_metrics"]

    print("[OK] Predictions written:")
    print(f"     ML2_global    : {out_path_global}")
    print(f"     ML2_puro      : {out_path_pure}")
    print(f"     ML2_operativo : {out_path_oper}")
    if bool(args.write_legacy_alias):
        print(f"     legacy alias  : {legacy_path}")
    print("[OK] Metrics written:")
    print(f"     global    : {metrics_global_path}")
    print(f"     puro      : {metrics_pure_path}")
    print(f"     operativo : {metrics_oper_path}")
    print(f"     layers    : {metrics_layers_path}")
    print(f"     legacy    : {metrics_legacy_path}")
    print(
        "     R2 avg global model "
        f"global={g_global.get('r2_ml2_avg', np.nan):.6f} "
        f"puro={g_pure.get('r2_ml2_avg', np.nan):.6f} "
        f"operativo={g_oper.get('r2_ml2_avg', np.nan):.6f}"
    )
    print(
        "     R2 avg global dynamic "
        f"global={gd_global.get('r2_ml2_avg', np.nan):.6f} "
        f"puro={gd_pure.get('r2_ml2_avg', np.nan):.6f} "
        f"operativo={gd_oper.get('r2_ml2_avg', np.nan):.6f}"
    )
    print(
        "     R2 avg active dynamic "
        f"global={ad_global.get('r2_ml2_avg', np.nan):.6f} "
        f"puro={ad_pure.get('r2_ml2_avg', np.nan):.6f} "
        f"operativo={ad_oper.get('r2_ml2_avg', np.nan):.6f}"
    )
    print(
        "     MAE avg active dynamic "
        f"global={ad_global.get('mae_ml2_avg', np.nan):.6f} "
        f"puro={ad_pure.get('mae_ml2_avg', np.nan):.6f} "
        f"operativo={ad_oper.get('mae_ml2_avg', np.nan):.6f}"
    )
    print(
        "     Dynamic iters "
        f"global={int(dyn_global['iterations'])} "
        f"puro={int(dyn_pure['iterations'])} "
        f"operativo={int(dyn_oper['iterations'])} "
        f"oper_converged={bool(dyn_oper['converged'])}"
    )
    print(
        "     rows_expanded "
        f"global={n_added_global:,} "
        f"puro={n_added_pure:,} "
        f"operativo={n_added_oper:,}"
    )


if __name__ == "__main__":
    main()
