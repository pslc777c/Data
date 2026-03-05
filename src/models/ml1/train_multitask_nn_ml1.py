from __future__ import annotations

import argparse
import json
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
IN_DS = DATA_DIR / "gold" / "ml1_nn" / "ds_ml1_nn_v1.parquet"
MODELS_DIR = DATA_DIR / "models" / "ml1_nn"
EVAL_DIR = DATA_DIR / "eval" / "ml1_nn"

TARGET_COLS = [
    "target_d_start",
    "target_n_harvest_days",
    "target_factor_tallos_dia",
    "target_share_grado",
    "target_factor_peso_tallo",
    "target_dh_dias",
    "target_factor_hidr",
    "target_factor_desp",
    "target_factor_ajuste",
]

TARGET_CLIPS: dict[str, tuple[float, float]] = {
    "target_d_start": (0.0, 180.0),
    "target_n_harvest_days": (1.0, 180.0),
    "target_factor_tallos_dia": (0.03, 5.00),
    "target_share_grado": (0.00, 1.00),
    "target_factor_peso_tallo": (0.60, 1.60),
    "target_dh_dias": (0.0, 30.0),
    "target_factor_hidr": (0.80, 3.00),
    "target_factor_desp": (0.05, 1.00),
    "target_factor_ajuste": (0.50, 2.00),
}

CAT_COLS = [
    "stage",
    "variedad_canon",
    "tipo_sp",
    "grado",
    "destino",
    "fallback_level_b2",
]

NUM_COLS = [
    "tallos_proy",
    "tallos_pred_baseline_grado_dia",
    "tallos_pred_ml1_grado_dia",
    "tallos_pred_ml1_dia",
    "sp_month",
    "sp_weekofyear",
    "sp_doy",
    "sp_dow",
    "day_in_harvest",
    "rel_pos",
    "n_harvest_days",
    "pct_avance_real",
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    "share_grado_baseline",
    "peso_tallo_baseline_g",
    "tallos_pred_baseline_dia",
    "tallos_post",
    "tallos_post_proy",
    "peso_base_g",
    "peso_post_g",
    "gramos_verde_ref",
    "gramos_post_real_ref",
    "kg_verde_ref",
    "kg_post_real_ref",
    "w2_kg",
    "w2a_kg",
    "wideal_kg",
    "desp_pct",
    "share_block_post",
    "share_dest_real_b2",
    "b2_peso_dest_real_kg",
    "cajas_ml1_grado_dia",
    "cajas_split_grado_dia",
    "cajas_post_seed",
    "dow",
    "month",
    "weekofyear",
    "dow_post",
    "month_post",
    "weekofyear_post",
    "ar_tallos_real_dia_lag1",
    "ar_tallos_real_dia_roll3",
    "ar_tallos_real_dia_roll7",
    "ar_ratio_real_vs_base_lag1",
    "ar_ratio_real_vs_base_roll3",
    "ar_pct_avance_real_lag1",
    "ar_gdc_dia_roll3",
    "ar_temp_avg_dia_roll3",
    "ar_rainfall_mm_roll3",
]


@dataclass
class Params:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("train_multitask_nn_ml1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--val-quantile", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feature-clip-q-low", type=float, default=0.005)
    ap.add_argument("--feature-clip-q-high", type=float, default=0.995)
    ap.add_argument("--target-outlier-q-low", type=float, default=0.01)
    ap.add_argument("--target-outlier-q-high", type=float, default=0.99)
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _prepare_dataframe() -> pd.DataFrame:
    if not IN_DS.exists():
        raise FileNotFoundError(f"Dataset not found: {IN_DS}")

    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha_evento", "has_any_target"} | set(TARGET_COLS)
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Dataset missing required columns: {sorted(miss)}")

    df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce").dt.normalize()
    df = df[df["has_any_target"] == 1].copy()
    df = df[df["fecha_evento"].notna()].copy()

    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = _canon_str(df[c])

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for t in TARGET_COLS:
        df[t] = pd.to_numeric(df[t], errors="coerce")
        mcol = f"mask_{t}"
        if mcol not in df.columns:
            df[mcol] = df[t].notna().astype("float32")
        df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)

    return df.reset_index(drop=True)


def _build_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    feature_clip_q_low: float,
    feature_clip_q_high: float,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    # Numeric: fill train median, then z-score with train stats
    x_num = df[NUM_COLS].copy()
    x_num = x_num.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    train_med = x_num.iloc[train_idx].median(numeric_only=True)
    fill_arr = train_med.to_numpy(dtype=np.float32)
    fill_arr = np.where(np.isfinite(fill_arr), fill_arr, 0.0).astype(np.float32)
    x_num = x_num.fillna(pd.Series(fill_arr, index=x_num.columns))

    ql = float(np.clip(feature_clip_q_low, 0.0, 0.49))
    qh = float(np.clip(feature_clip_q_high, ql + 1e-6, 1.0))
    clip_lo = x_num.iloc[train_idx].quantile(ql, numeric_only=True).to_numpy(dtype=np.float32)
    clip_hi = x_num.iloc[train_idx].quantile(qh, numeric_only=True).to_numpy(dtype=np.float32)
    clip_lo = np.where(np.isfinite(clip_lo), clip_lo, fill_arr).astype(np.float32)
    clip_hi = np.where(np.isfinite(clip_hi), clip_hi, fill_arr).astype(np.float32)
    bad = clip_lo > clip_hi
    if bad.any():
        clip_lo[bad] = fill_arr[bad]
        clip_hi[bad] = fill_arr[bad]

    lo_s = pd.Series(clip_lo, index=x_num.columns)
    hi_s = pd.Series(clip_hi, index=x_num.columns)
    x_num = x_num.clip(lower=lo_s, upper=hi_s, axis=1)

    x_mean = x_num.iloc[train_idx].mean(numeric_only=True).to_numpy(dtype=np.float32)
    x_std = x_num.iloc[train_idx].std(numeric_only=True, ddof=0).to_numpy(dtype=np.float32)
    x_mean = np.where(np.isfinite(x_mean), x_mean, fill_arr).astype(np.float32)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-8), x_std, 1.0).astype(np.float32)

    x_num_arr = x_num.to_numpy(dtype=np.float32)
    x_num_arr = (x_num_arr - x_mean) / x_std

    # Categorical: one-hot
    x_cat = pd.get_dummies(df[CAT_COLS], prefix=CAT_COLS, prefix_sep="=", dtype=np.float32)
    cat_dummy_cols = list(x_cat.columns)
    x_cat_arr = x_cat.to_numpy(dtype=np.float32)

    x = np.concatenate([x_num_arr, x_cat_arr], axis=1).astype(np.float32)
    feature_names = NUM_COLS + cat_dummy_cols
    return x, feature_names, x_mean, x_std, cat_dummy_cols, fill_arr, clip_lo, clip_hi


def _build_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = df[TARGET_COLS].to_numpy(dtype=np.float32)
    m = df[[f"mask_{t}" for t in TARGET_COLS]].to_numpy(dtype=np.float32)
    y = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
    m = np.where(np.isfinite(y), m, 0.0).astype(np.float32)
    return y, m, df[[f"mask_{t}" for t in TARGET_COLS]].to_numpy(dtype=np.float32)


def _apply_target_outlier_filter(
    y: np.ndarray,
    m: np.ndarray,
    train_idx: np.ndarray,
    q_low: float,
    q_high: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    m2 = m.copy().astype(np.float32)
    ql = float(np.clip(q_low, 0.0, 0.49))
    qh = float(np.clip(q_high, ql + 1e-6, 1.0))

    rows: list[dict] = []
    for j, t in enumerate(TARGET_COLS):
        in_train = (m2[train_idx, j] > 0.0) & np.isfinite(y[train_idx, j])
        vals = y[train_idx, j][in_train]
        row = {
            "target": t,
            "q_low": ql,
            "q_high": qh,
            "bound_low": np.nan,
            "bound_high": np.nan,
            "n_train_masked": int(in_train.sum()),
            "n_removed_train": 0,
            "n_masked_total_before": int(((m2[:, j] > 0.0) & np.isfinite(y[:, j])).sum()),
            "n_removed_total": 0,
        }
        if vals.size >= 80:
            lo = float(np.nanquantile(vals, ql))
            hi = float(np.nanquantile(vals, qh))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                before = (m2[:, j] > 0.0) & np.isfinite(y[:, j])
                is_out = before & ((y[:, j] < lo) | (y[:, j] > hi))
                m2[is_out, j] = 0.0
                rem_train = int(is_out[train_idx].sum())
                rem_all = int(is_out.sum())
                row["bound_low"] = lo
                row["bound_high"] = hi
                row["n_removed_train"] = rem_train
                row["n_removed_total"] = rem_all
        rows.append(row)

    audit = pd.DataFrame(rows)
    return m2, audit


def _split_indices(df: pd.DataFrame, q: float) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    cutoff = df["fecha_evento"].quantile(q)
    is_val = (df["fecha_evento"] >= cutoff).to_numpy()

    if is_val.mean() <= 0.05 or is_val.mean() >= 0.95:
        n = len(df)
        cut = int(n * 0.8)
        order = np.argsort(df["fecha_evento"].to_numpy())
        train_idx = order[:cut]
        val_idx = order[cut:]
    else:
        train_idx = np.where(~is_val)[0]
        val_idx = np.where(is_val)[0]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Unable to build train/val split.")

    return train_idx.astype(np.int64), val_idx.astype(np.int64), pd.to_datetime(cutoff).normalize()


def _init_params(in_dim: int, h1: int, h2: int, out_dim: int, seed: int) -> Params:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, np.sqrt(2.0 / max(in_dim, 1)), size=(in_dim, h1)).astype(np.float32)
    b1 = np.zeros((h1,), dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(2.0 / max(h1, 1)), size=(h1, h2)).astype(np.float32)
    b2 = np.zeros((h2,), dtype=np.float32)
    w3 = rng.normal(0.0, np.sqrt(2.0 / max(h2, 1)), size=(h2, out_dim)).astype(np.float32)
    b3 = np.zeros((out_dim,), dtype=np.float32)
    return Params(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)


def _forward(x: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = x @ p.w1 + p.b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ p.w2 + p.b2
    a2 = np.maximum(z2, 0.0)
    yhat = a2 @ p.w3 + p.b3
    return z1, a1, z2, a2, yhat


def _loss_and_grads(
    x: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    p: Params,
    weight_decay: float,
) -> tuple[float, Params]:
    z1, a1, z2, a2, yhat = _forward(x, p)

    err = yhat - y
    denom = m.sum(axis=0, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0).astype(np.float32)

    per_target = ((err**2) * m).sum(axis=0, keepdims=True) / denom
    active = (m.sum(axis=0, keepdims=True) > 0.0).astype(np.float32)
    n_active = float(np.maximum(active.sum(), 1.0))

    data_loss = float((per_target * active).sum() / n_active)
    reg_loss = 0.5 * weight_decay * float((p.w1**2).sum() + (p.w2**2).sum() + (p.w3**2).sum())
    loss = data_loss + reg_loss

    dy = (2.0 * err * m / denom) / n_active

    gw3 = a2.T @ dy + weight_decay * p.w3
    gb3 = dy.sum(axis=0)

    da2 = dy @ p.w3.T
    dz2 = da2 * (z2 > 0.0)

    gw2 = a1.T @ dz2 + weight_decay * p.w2
    gb2 = dz2.sum(axis=0)

    da1 = dz2 @ p.w2.T
    dz1 = da1 * (z1 > 0.0)

    gw1 = x.T @ dz1 + weight_decay * p.w1
    gb1 = dz1.sum(axis=0)

    grads = Params(
        w1=gw1.astype(np.float32),
        b1=gb1.astype(np.float32),
        w2=gw2.astype(np.float32),
        b2=gb2.astype(np.float32),
        w3=gw3.astype(np.float32),
        b3=gb3.astype(np.float32),
    )
    return loss, grads


def _predict(x: np.ndarray, p: Params) -> np.ndarray:
    _, _, _, _, yhat = _forward(x, p)
    return yhat


def _masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, m: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    maes: list[float] = []
    rmses: list[float] = []
    for i, t in enumerate(TARGET_COLS):
        wi_full = m[:, i].astype(np.float32)
        yi = y_true[:, i]
        pi = y_pred[:, i]
        mi = (wi_full > 0.0) & np.isfinite(yi) & np.isfinite(pi)
        wi = wi_full[mi]
        wsum = float(wi.sum())
        if wsum <= 0.0:
            out[f"mae_{t}"] = np.nan
            out[f"rmse_{t}"] = np.nan
            out[f"n_{t}"] = 0
            out[f"w_{t}"] = 0.0
            continue
        err = pi[mi] - yi[mi]
        mae = float(np.sum(np.abs(err) * wi) / wsum)
        rmse = float(np.sqrt(np.sum((err**2) * wi) / wsum))
        ybar = float(np.sum(yi[mi] * wi) / wsum)
        sse = float(np.sum((err**2) * wi))
        sst = float(np.sum(((yi[mi] - ybar) ** 2) * wi))
        r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
        out[f"mae_{t}"] = mae
        out[f"rmse_{t}"] = rmse
        out[f"r2_{t}"] = r2
        out[f"n_{t}"] = int(mi.sum())
        out[f"w_{t}"] = wsum
        maes.append(mae)
        rmses.append(rmse)
    out["mae_avg"] = float(np.mean(maes)) if maes else np.nan
    out["rmse_avg"] = float(np.mean(rmses)) if rmses else np.nan
    r2_vals = [out.get(f"r2_{t}") for t in TARGET_COLS if np.isfinite(out.get(f"r2_{t}", np.nan))]
    out["r2_avg"] = float(np.mean(r2_vals)) if r2_vals else np.nan
    return out


def _compute_target_norm_stats(y: np.ndarray, m: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.zeros((len(TARGET_COLS),), dtype=np.float32)
    sd = np.ones((len(TARGET_COLS),), dtype=np.float32)
    for j in range(len(TARGET_COLS)):
        mj = m[idx, j] > 0.0
        if mj.sum() == 0:
            continue
        v = y[idx, j][mj].astype(np.float32)
        mu[j] = float(np.mean(v))
        s = float(np.std(v, ddof=0))
        sd[j] = s if np.isfinite(s) and s > 1e-8 else 1.0
    return mu, sd


def _normalize_targets(y: np.ndarray, m: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    y_norm = (y - mu.reshape(1, -1)) / sd.reshape(1, -1)
    y_norm = np.where(m > 0.0, y_norm, 0.0)
    return y_norm.astype(np.float32)


def _denormalize_targets(y_norm: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return y_norm * sd.reshape(1, -1) + mu.reshape(1, -1)


def _save_feature_pvalues(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    m_train: np.ndarray,
    feature_names: list[str],
    run_id: str,
) -> tuple[Path, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    for j, t in enumerate(TARGET_COLS):
        mj = m_train[:, j] > 0.0
        if mj.sum() < 30:
            continue
        xj = x_train[mj]
        yj = y_train_raw[mj, j]
        keep = np.var(xj, axis=0) > 1e-12
        if keep.sum() == 0:
            continue
        xj2 = xj[:, keep]
        feat2 = [f for f, k in zip(feature_names, keep.tolist()) if k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fval, pval = f_regression(xj2, yj, center=True)
        part = pd.DataFrame(
            {
                "run_id": run_id,
                "target": t,
                "feature": feat2,
                "f_value": fval,
                "p_value": pval,
            }
        )
        part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["f_value", "p_value"])
        rows.append(part)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["run_id", "target", "feature", "f_value", "p_value"]
    )
    out = out.sort_values(["target", "p_value", "feature"], kind="mergesort").reset_index(drop=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"ml1_nn_feature_pvalues_{run_id}.parquet"
    write_parquet(out, out_path)
    return out_path, out


def _save_feature_diagnostics(
    x_train: np.ndarray,
    feature_names: list[str],
    pvals: pd.DataFrame,
    run_id: str,
) -> Path:
    var = np.var(x_train.astype(np.float64), axis=0)
    d = pd.DataFrame(
        {
            "run_id": run_id,
            "feature": feature_names,
            "train_variance": var,
        }
    )
    if not pvals.empty and {"feature", "p_value"} <= set(pvals.columns):
        min_p = pvals.groupby("feature", dropna=False)["p_value"].min()
        d["min_p_value"] = d["feature"].map(min_p).astype(float)
    else:
        d["min_p_value"] = np.nan

    d["feature_group"] = np.where(
        d["feature"].str.contains("=", regex=False),
        d["feature"].str.split("=", n=1).str[0],
        "NUMERIC",
    )
    d["is_autoregressive"] = d["feature"].str.startswith("ar_")
    d["is_near_constant"] = d["train_variance"] <= 1e-10
    d["is_weak_signal"] = d["min_p_value"].fillna(1.0) > 0.20
    d = d.sort_values(["is_near_constant", "is_weak_signal", "min_p_value", "feature"], ascending=[False, False, True, True])

    out_path = EVAL_DIR / f"ml1_nn_feature_diagnostics_{run_id}.parquet"
    write_parquet(d.reset_index(drop=True), out_path)
    return out_path


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    df = _prepare_dataframe()
    train_idx, val_idx, cutoff = _split_indices(df, args.val_quantile)

    y_raw, m, _ = _build_targets(df)
    m, outlier_audit = _apply_target_outlier_filter(
        y=y_raw,
        m=m,
        train_idx=train_idx,
        q_low=args.target_outlier_q_low,
        q_high=args.target_outlier_q_high,
    )
    x, feature_names, x_mean, x_std, cat_dummy_cols, x_fill, x_clip_lo, x_clip_hi = _build_features(
        df,
        train_idx=train_idx,
        feature_clip_q_low=args.feature_clip_q_low,
        feature_clip_q_high=args.feature_clip_q_high,
    )

    y_mu, y_sd = _compute_target_norm_stats(y_raw, m, train_idx)
    y_norm = _normalize_targets(y_raw, m, y_mu, y_sd)

    x_train = x[train_idx]
    y_train = y_norm[train_idx]
    m_train = m[train_idx]
    x_val = x[val_idx]
    y_val = y_norm[val_idx]
    m_val = m[val_idx]

    p = _init_params(
        in_dim=x_train.shape[1],
        h1=args.hidden1,
        h2=args.hidden2,
        out_dim=len(TARGET_COLS),
        seed=args.seed,
    )

    # Adam buffers
    mw1 = np.zeros_like(p.w1)
    vw1 = np.zeros_like(p.w1)
    mb1 = np.zeros_like(p.b1)
    vb1 = np.zeros_like(p.b1)
    mw2 = np.zeros_like(p.w2)
    vw2 = np.zeros_like(p.w2)
    mb2 = np.zeros_like(p.b2)
    vb2 = np.zeros_like(p.b2)
    mw3 = np.zeros_like(p.w3)
    vw3 = np.zeros_like(p.w3)
    mb3 = np.zeros_like(p.b3)
    vb3 = np.zeros_like(p.b3)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    tstep = 0

    best_p = p
    best_val = np.inf
    wait = 0
    hist: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x_train))
        batch_losses: list[float] = []

        for i0 in range(0, len(order), args.batch_size):
            ib = order[i0 : i0 + args.batch_size]
            xb = x_train[ib]
            yb = y_train[ib]
            mb = m_train[ib]

            loss, g = _loss_and_grads(xb, yb, mb, p, args.weight_decay)
            batch_losses.append(loss)

            tstep += 1

            # w1
            mw1 = beta1 * mw1 + (1.0 - beta1) * g.w1
            vw1 = beta2 * vw1 + (1.0 - beta2) * (g.w1**2)
            mw1h = mw1 / (1.0 - beta1**tstep)
            vw1h = vw1 / (1.0 - beta2**tstep)
            p.w1 -= args.lr * mw1h / (np.sqrt(vw1h) + eps)

            # b1
            mb1 = beta1 * mb1 + (1.0 - beta1) * g.b1
            vb1 = beta2 * vb1 + (1.0 - beta2) * (g.b1**2)
            mb1h = mb1 / (1.0 - beta1**tstep)
            vb1h = vb1 / (1.0 - beta2**tstep)
            p.b1 -= args.lr * mb1h / (np.sqrt(vb1h) + eps)

            # w2
            mw2 = beta1 * mw2 + (1.0 - beta1) * g.w2
            vw2 = beta2 * vw2 + (1.0 - beta2) * (g.w2**2)
            mw2h = mw2 / (1.0 - beta1**tstep)
            vw2h = vw2 / (1.0 - beta2**tstep)
            p.w2 -= args.lr * mw2h / (np.sqrt(vw2h) + eps)

            # b2
            mb2 = beta1 * mb2 + (1.0 - beta1) * g.b2
            vb2 = beta2 * vb2 + (1.0 - beta2) * (g.b2**2)
            mb2h = mb2 / (1.0 - beta1**tstep)
            vb2h = vb2 / (1.0 - beta2**tstep)
            p.b2 -= args.lr * mb2h / (np.sqrt(vb2h) + eps)

            # w3
            mw3 = beta1 * mw3 + (1.0 - beta1) * g.w3
            vw3 = beta2 * vw3 + (1.0 - beta2) * (g.w3**2)
            mw3h = mw3 / (1.0 - beta1**tstep)
            vw3h = vw3 / (1.0 - beta2**tstep)
            p.w3 -= args.lr * mw3h / (np.sqrt(vw3h) + eps)

            # b3
            mb3 = beta1 * mb3 + (1.0 - beta1) * g.b3
            vb3 = beta2 * vb3 + (1.0 - beta2) * (g.b3**2)
            mb3h = mb3 / (1.0 - beta1**tstep)
            vb3h = vb3 / (1.0 - beta2**tstep)
            p.b3 -= args.lr * mb3h / (np.sqrt(vb3h) + eps)

        # Metrics in original target scale
        yhat_train = _denormalize_targets(_predict(x_train, p), y_mu, y_sd)
        yhat_val = _denormalize_targets(_predict(x_val, p), y_mu, y_sd)
        train_metrics = _masked_metrics(y_true=y_raw[train_idx], y_pred=yhat_train, m=m_train)
        val_metrics = _masked_metrics(y_true=y_raw[val_idx], y_pred=yhat_val, m=m_val)

        row = {
            "epoch": epoch,
            "loss_batch_mean": float(np.mean(batch_losses)) if batch_losses else np.nan,
            "mae_train_avg": train_metrics["mae_avg"],
            "mae_val_avg": val_metrics["mae_avg"],
        }
        hist.append(row)
        print(
            f"[EPOCH {epoch:03d}] "
            f"loss={row['loss_batch_mean']:.6f} "
            f"mae_train={row['mae_train_avg']:.5f} "
            f"mae_val={row['mae_val_avg']:.5f}"
        )

        cur_val = float(val_metrics["mae_avg"]) if np.isfinite(val_metrics["mae_avg"]) else np.inf
        if cur_val + 1e-8 < best_val:
            best_val = cur_val
            best_p = Params(
                w1=p.w1.copy(),
                b1=p.b1.copy(),
                w2=p.w2.copy(),
                b2=p.b2.copy(),
                w3=p.w3.copy(),
                b3=p.b3.copy(),
            )
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[EARLY STOP] patience={args.patience} reached at epoch={epoch}.")
                break

    # Final metrics with best params
    yhat_train_best = _denormalize_targets(_predict(x_train, best_p), y_mu, y_sd)
    yhat_val_best = _denormalize_targets(_predict(x_val, best_p), y_mu, y_sd)
    train_metrics = _masked_metrics(y_true=y_raw[train_idx], y_pred=yhat_train_best, m=m_train)
    val_metrics = _masked_metrics(y_true=y_raw[val_idx], y_pred=yhat_val_best, m=m_val)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"ml1_multitask_nn_{run_id}.npz"
    np.savez_compressed(
        model_path,
        w1=best_p.w1,
        b1=best_p.b1,
        w2=best_p.w2,
        b2=best_p.b2,
        w3=best_p.w3,
        b3=best_p.b3,
        x_fill=x_fill,
        x_clip_lo=x_clip_lo,
        x_clip_hi=x_clip_hi,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mu,
        y_std=y_sd,
    )

    pval_path, pvals_df = _save_feature_pvalues(
        x_train=x_train,
        y_train_raw=y_raw[train_idx],
        m_train=m_train,
        feature_names=feature_names,
        run_id=run_id,
    )
    feat_diag_path = _save_feature_diagnostics(
        x_train=x_train,
        feature_names=feature_names,
        pvals=pvals_df,
        run_id=run_id,
    )
    outlier_path = EVAL_DIR / f"ml1_nn_target_outlier_filter_{run_id}.parquet"
    write_parquet(outlier_audit, outlier_path)

    hist_df = pd.DataFrame(hist)
    hist_path = EVAL_DIR / f"ml1_multitask_nn_train_history_{run_id}.parquet"
    write_parquet(hist_df, hist_path)

    meta = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": str(IN_DS).replace("\\", "/"),
        "model_path": str(model_path).replace("\\", "/"),
        "history_path": str(hist_path).replace("\\", "/"),
        "pvalues_path": str(pval_path).replace("\\", "/"),
        "feature_diagnostics_path": str(feat_diag_path).replace("\\", "/"),
        "target_outlier_filter_path": str(outlier_path).replace("\\", "/"),
        "split": {
            "type": "temporal_quantile",
            "val_quantile": args.val_quantile,
            "cutoff_fecha_evento": str(cutoff.date()),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
        },
        "network": {
            "input_dim": int(x.shape[1]),
            "hidden1": int(args.hidden1),
            "hidden2": int(args.hidden2),
            "output_dim": len(TARGET_COLS),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "seed": int(args.seed),
        },
        "features": {
            "cat_cols": CAT_COLS,
            "num_cols": NUM_COLS,
            "cat_dummy_cols": cat_dummy_cols,
            "feature_names": feature_names,
        },
        "preprocess": {
            "feature_clip_q_low": float(args.feature_clip_q_low),
            "feature_clip_q_high": float(args.feature_clip_q_high),
            "target_outlier_q_low": float(args.target_outlier_q_low),
            "target_outlier_q_high": float(args.target_outlier_q_high),
            "dropped_location_cats": ["bloque_base", "area"],
            "n_outliers_removed_total": int(pd.to_numeric(outlier_audit["n_removed_total"], errors="coerce").fillna(0).sum()),
        },
        "targets": TARGET_COLS,
        "target_clips": TARGET_CLIPS,
        "metrics_train": train_metrics,
        "metrics_val": val_metrics,
    }

    meta_path = MODELS_DIR / f"ml1_multitask_nn_{run_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Meta saved : {meta_path}")
    print(f"[OK] History    : {hist_path}")
    print(f"[OK] P-values   : {pval_path}")
    print(f"[OK] Feat diag  : {feat_diag_path}")
    print(f"[OK] Outliers   : {outlier_path}")
    print(f"     mae_val_avg={val_metrics['mae_avg']:.6f}")


if __name__ == "__main__":
    main()
