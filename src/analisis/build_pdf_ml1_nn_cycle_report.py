from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common.io import read_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models" / "ml1_nn"
GOLD_DIR = DATA_DIR / "gold" / "ml1_nn"
EVAL_DIR = DATA_DIR / "eval" / "ml1_nn"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_pdf_ml1_nn_cycle_report")
    ap.add_argument("--cycle-id", required=True)
    ap.add_argument("--run-id", default=None, help="If omitted, latest run is used")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _latest_meta() -> Path:
    files = sorted(MODELS_DIR.glob("ml1_multitask_nn_*_meta.json"))
    if not files:
        raise FileNotFoundError(f"No model metadata found in {MODELS_DIR}")
    return files[-1]


def _meta_for_run(run_id: str) -> Path:
    path = MODELS_DIR / f"ml1_multitask_nn_{run_id}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata for run_id={run_id} not found: {path}")
    return path


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _draw_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    txt = "\n".join(lines)
    fig.text(0.03, 0.95, txt, va="top", ha="left", family="monospace", fontsize=9)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pick_first(series: pd.Series) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else None


def _build_features(df: pd.DataFrame, meta: dict, params: dict[str, np.ndarray]) -> np.ndarray:
    cat_cols = meta["features"]["cat_cols"]
    num_cols = meta["features"]["num_cols"]
    cat_dummy_cols = meta["features"]["cat_dummy_cols"]

    d = df.copy()
    for c in cat_cols:
        if c not in d.columns:
            d[c] = "UNKNOWN"
        d[c] = _canon_str(d[c])

    for c in num_cols:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")

    x_num = d[num_cols].to_numpy(dtype=np.float32)
    x_fill = params.get("x_fill", params["x_mean"]).astype(np.float32)
    x_clip_lo = params.get("x_clip_lo")
    x_clip_hi = params.get("x_clip_hi")
    x_mean = params["x_mean"].astype(np.float32)
    x_std = params["x_std"].astype(np.float32)

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

    x_cat = pd.get_dummies(d[cat_cols], prefix=cat_cols, prefix_sep="=", dtype=np.float32)
    x_cat = x_cat.reindex(columns=cat_dummy_cols, fill_value=0.0)
    x_cat_arr = x_cat.to_numpy(dtype=np.float32)
    return np.concatenate([x_num, x_cat_arr], axis=1).astype(np.float32)


def _local_gradient_one(
    x_row: np.ndarray,
    params: dict[str, np.ndarray],
    target_idx: int,
    n_num: int,
) -> np.ndarray:
    w1, b1 = params["w1"], params["b1"]
    w2, b2 = params["w2"], params["b2"]
    w3, b3 = params["w3"], params["b3"]
    x_std = params["x_std"]
    y_std = params["y_std"]

    z1 = x_row @ w1 + b1
    g1 = (z1 > 0.0).astype(np.float32)
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ w2 + b2
    g2 = (z2 > 0.0).astype(np.float32)

    s2 = g2 * w3[:, target_idx]
    s1 = g1 * (w2 @ s2)
    grad = w1 @ s1
    grad = grad * y_std[target_idx]

    # Numeric gradients back to raw units (because x_num was standardized)
    grad_num = grad[:n_num] / np.where(np.abs(x_std) > 1e-12, x_std, 1.0)
    grad_full = grad.copy()
    grad_full[:n_num] = grad_num
    return grad_full


def _importance_for_target(
    x: np.ndarray,
    feature_names: list[str],
    params: dict[str, np.ndarray],
    target_idx: int,
    n_num: int,
    feature_active_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    if len(x) == 0:
        return pd.DataFrame(columns=["feature", "w_abs", "w_norm"])

    grads = np.vstack([_local_gradient_one(x[i], params, target_idx, n_num=n_num) for i in range(len(x))])
    w_abs = np.mean(np.abs(grads), axis=0)
    if feature_active_mask is not None and len(feature_active_mask) == len(w_abs):
        w_abs = np.where(feature_active_mask.astype(bool), w_abs, 0.0)
    # If we have enough local rows, also suppress effectively constant features.
    if len(x) >= 4:
        x_var = np.var(x.astype(np.float64), axis=0)
        active_var = np.isfinite(x_var) & (x_var > 1e-12)
        w_abs = np.where(active_var, w_abs, 0.0)
    s = float(np.sum(w_abs))
    w_norm = w_abs / s if s > 0 else w_abs
    out = pd.DataFrame({"feature": feature_names, "w_abs": w_abs, "w_norm": w_norm})
    out = out.sort_values("w_norm", ascending=False).reset_index(drop=True)
    return out


def _load_params(meta: dict) -> dict[str, np.ndarray]:
    model_path = Path(meta["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    arr = np.load(model_path)
    return {
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


def _bar_page(pdf: PdfPages, df: pd.DataFrame, title: str, topn: int = 20) -> None:
    sub = df.head(topn).iloc[::-1]
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    ax.barh(sub["feature"], sub["w_norm"])
    ax.set_title(title)
    ax.set_xlabel("w (normalized absolute local gradient)")
    ax.grid(True, axis="x", alpha=0.3)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    meta_path = _latest_meta() if args.run_id is None else _meta_for_run(args.run_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    run_id = str(meta["run_id"])
    cycle_id = str(args.cycle_id)

    pred_path = GOLD_DIR / f"pred_ml1_multitask_nn_{run_id}.parquet"
    post_path = GOLD_DIR / f"pred_ml1_multitask_nn_post_final_{run_id}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    df = read_parquet(pred_path).copy()
    cyc = df[df["ciclo_id"].astype("string") == cycle_id].copy()
    if cyc.empty:
        raise ValueError(f"cycle_id not found in predictions: {cycle_id}")

    params = _load_params(meta)
    feature_names = list(meta["features"]["feature_names"])
    num_cols = list(meta["features"]["num_cols"])
    targets = list(meta["targets"])
    n_num = len(meta["features"]["num_cols"])
    blocked_prefix = ("bloque_base=", "area=")

    x_cyc = _build_features(cyc, meta=meta, params=params)

    stage = cyc["stage"].astype("string").str.upper().str.strip()
    veg = cyc[stage.eq("VEG")].copy()
    hg = cyc[stage.eq("HARVEST_GRADE")].copy()
    post = cyc[stage.eq("POST")].copy()

    d_start = _pick_first(veg.get("d_start_ML1", veg.get("pred_d_start")))
    n_days = _pick_first(veg.get("n_harvest_days_ML1", veg.get("pred_n_harvest_days")))
    fecha_sp = pd.to_datetime(veg["fecha_evento"], errors="coerce").dropna().min() if len(veg) else None
    h_start = None
    h_end = None
    if fecha_sp is not None and d_start is not None and n_days is not None:
        h_start = fecha_sp + pd.Timedelta(days=int(round(d_start)))
        h_end = h_start + pd.Timedelta(days=int(round(max(n_days, 1))) - 1)

    tallos_ml1_total = float(pd.to_numeric(hg.get("tallos_grado_dia_ML1"), errors="coerce").sum()) if len(hg) else np.nan
    tallos_base_total = float(pd.to_numeric(hg.get("tallos_grado_dia_baseline"), errors="coerce").sum()) if len(hg) else np.nan

    post_tot = {}
    if post_path.exists():
        p = read_parquet(post_path).copy()
        p = p[p["ciclo_id"].astype("string") == cycle_id].copy()
        if len(p):
            post_tot = {
                "kg_verde_ML1_total": float(pd.to_numeric(p.get("kg_verde_ML1"), errors="coerce").sum()),
                "kg_post_ML1_total": float(pd.to_numeric(p.get("kg_post_ML1"), errors="coerce").sum()),
                "cajas_verde_ML1_total": float(pd.to_numeric(p.get("cajas_verde_ML1"), errors="coerce").sum()),
                "cajas_post_ML1_total": float(pd.to_numeric(p.get("cajas_post_ML1"), errors="coerce").sum()),
            }
            post_tot["aprovechamiento_ML1_total"] = (
                post_tot["kg_post_ML1_total"] / post_tot["kg_verde_ML1_total"]
                if post_tot["kg_verde_ML1_total"] > 0
                else np.nan
            )

    # Local importances
    imp_pages: list[tuple[str, pd.DataFrame]] = []
    for t in ["target_factor_ajuste", "target_d_start", "target_n_harvest_days"]:
        if t not in targets:
            continue
        tidx = targets.index(t)
        if t == "target_factor_ajuste":
            mask = stage.eq("POST")
        elif t in {"target_d_start", "target_n_harvest_days"}:
            mask = stage.eq("VEG")
        else:
            mask = pd.Series([True] * len(cyc), index=cyc.index)
        if mask.sum() == 0:
            continue
        sub = cyc.loc[mask].copy()
        active_num = np.array(
            [
                pd.to_numeric(sub.get(c), errors="coerce").notna().any()
                for c in num_cols
            ],
            dtype=bool,
        )
        x_sub = x_cyc[mask.to_numpy()]
        if x_sub.shape[1] > n_num:
            active_cat = (np.abs(x_sub[:, n_num:]) > 1e-12).any(axis=0)
            cat_names = feature_names[n_num:]
            for k, nm in enumerate(cat_names):
                if str(nm).startswith(blocked_prefix):
                    active_cat[k] = False
        else:
            active_cat = np.array([], dtype=bool)
        active_mask = np.concatenate([active_num, active_cat]).astype(bool)
        imp = _importance_for_target(
            x=x_sub,
            feature_names=feature_names,
            params=params,
            target_idx=tidx,
            n_num=n_num,
            feature_active_mask=active_mask,
        )
        imp_pages.append((t, imp))

    safe_cycle = cycle_id[:18]
    out_path = Path(args.out) if args.out else (EVAL_DIR / f"ml1_nn_cycle_report_{safe_cycle}_{run_id}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        prep = dict(meta.get("preprocess", {}))
        lines = [
            f"run_id: {run_id}",
            f"cycle_id: {cycle_id}",
            "",
            f"rows_total_cycle: {len(cyc):,}",
            f"rows_veg: {int(stage.eq('VEG').sum()):,}",
            f"rows_harvest_grade: {int(stage.eq('HARVEST_GRADE').sum()):,}",
            f"rows_post: {int(stage.eq('POST').sum()):,}",
            "",
            f"d_start_ML1 (days): {d_start}",
            f"n_harvest_days_ML1: {n_days}",
            f"fecha_sp: {fecha_sp}",
            f"harvest_start_ML1: {h_start}",
            f"harvest_end_ML1: {h_end}",
            "",
            f"tallos_harvest_ML1_total: {tallos_ml1_total:,.2f}",
            f"tallos_harvest_baseline_total: {tallos_base_total:,.2f}",
        ]
        if post_tot:
            lines.extend(
                [
                    "",
                    f"kg_verde_ML1_total: {post_tot['kg_verde_ML1_total']:,.2f}",
                    f"kg_post_ML1_total: {post_tot['kg_post_ML1_total']:,.2f}",
                    f"cajas_verde_ML1_total: {post_tot['cajas_verde_ML1_total']:,.2f}",
                    f"cajas_post_ML1_total: {post_tot['cajas_post_ML1_total']:,.2f}",
                    f"aprovechamiento_ML1_total: {post_tot['aprovechamiento_ML1_total']:.4f}",
                ]
            )
        lines.extend(
            [
                "",
                "R2 (val) by target:",
            ]
        )
        if prep:
            lines.extend(
                [
                    "",
                    f"feature_clip_q: [{prep.get('feature_clip_q_low')}, {prep.get('feature_clip_q_high')}]",
                    f"target_outlier_q: [{prep.get('target_outlier_q_low')}, {prep.get('target_outlier_q_high')}]",
                    f"dropped_location_cats: {prep.get('dropped_location_cats')}",
                ]
            )
        for t in targets:
            r2k = f"r2_{t}"
            lines.append(f"  {t}: {meta.get('metrics_val', {}).get(r2k)}")
        _draw_text_page(pdf, "ML1 Cycle Summary", lines)

        for t, imp in imp_pages:
            _bar_page(pdf, imp, title=f"Local Feature Weights (w) - {t}", topn=20)

        if post_tot and post_path.exists():
            p = read_parquet(post_path).copy()
            p = p[p["ciclo_id"].astype("string") == cycle_id].copy()
            if len(p):
                p["fecha_post"] = pd.to_datetime(p["fecha_post"], errors="coerce")
                agg = (
                    p.groupby("fecha_post", dropna=False, as_index=False)
                    .agg(
                        kg_verde_ML1=("kg_verde_ML1", "sum"),
                        kg_post_ML1=("kg_post_ML1", "sum"),
                        cajas_verde_ML1=("cajas_verde_ML1", "sum"),
                        cajas_post_ML1=("cajas_post_ML1", "sum"),
                    )
                    .sort_values("fecha_post")
                )

                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.plot(agg["fecha_post"], agg["kg_verde_ML1"], label="kg_verde_ML1")
                ax.plot(agg["fecha_post"], agg["kg_post_ML1"], label="kg_post_ML1")
                ax.set_title("Post Timeline (kg) - Cycle")
                ax.set_xlabel("fecha_post")
                ax.set_ylabel("kg")
                ax.grid(True, alpha=0.3)
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_subplot(111)
                ax.plot(agg["fecha_post"], agg["cajas_verde_ML1"], label="cajas_verde_ML1")
                ax.plot(agg["fecha_post"], agg["cajas_post_ML1"], label="cajas_post_ML1")
                ax.set_title("Post Timeline (cajas) - Cycle")
                ax.set_xlabel("fecha_post")
                ax.set_ylabel("cajas")
                ax.grid(True, alpha=0.3)
                ax.legend()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"[OK] Cycle report PDF written: {out_path}")


if __name__ == "__main__":
    main()
