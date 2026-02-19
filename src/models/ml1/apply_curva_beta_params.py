# src/models/ml1/apply_curva_beta_params.py
from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet

# =============================================================================
# Paths
# =============================================================================
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")

REG_BETA_PARAMS = Path("models_registry/ml1/curva_beta_params")
REG_MULT_DIA = Path("models_registry/ml1/curva_beta_multiplier_dia")

CAP_PATH = Path("data/gold/dim_cap_tallos_real_dia.parquet")

OUT_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")

# =============================================================================
# Daily model columns (multiplier)
# =============================================================================
NUM_COLS_DAILY = [
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
    "dow",
    "month",
    "weekofyear",
]
CAT_COLS = ["variedad_canon", "area", "tipo_sp"]

# =============================================================================
# Config
# =============================================================================
EPS = 1e-12
REL_CLIP = 1e-4

# Small mixing with baseline share (robustness)
LAMBDA_BASE = 0.05

# Multiplier stability clip: exp(log_mult) in [0.37, 2.72]
LOG_MULT_CLIP = (-1.0, 1.0)

# Mild smoothing after combining (keeps unimodal-ish)
SMOOTH_WIN = 3

# Factor caps (statistical, not capacity)
FACTOR_MIN, FACTOR_MAX = 0.01, 5.0

# Cap table fallback
CAP_FALLBACK = 4000.0

# Cap redistribution
MAX_REDIST_ITERS = 20

# =============================================================================
# Helpers
# =============================================================================
def _latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index(df.columns.astype(str))
    if cols.is_unique:
        return df
    out = df.copy()
    seen: dict[str, list[int]] = {}
    for i, c in enumerate(out.columns.astype(str)):
        seen.setdefault(c, []).append(i)
    keep: dict[str, pd.Series] = {}
    for c, idxs in seen.items():
        s = out.iloc[:, idxs[0]]
        for j in idxs[1:]:
            s2 = out.iloc[:, j]
            s = s.where(s.notna(), s2)
        keep[c] = s
    ordered: list[str] = []
    for c in out.columns.astype(str):
        if c not in ordered:
            ordered.append(c)
    return pd.DataFrame({c: keep[c] for c in ordered})


def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    if out_col in df.columns:
        base = df[out_col]
    else:
        base = pd.Series([pd.NA] * len(df), index=df.index)
    for c in candidates:
        if c in df.columns:
            base = base.where(base.notna(), df[c])
    df[out_col] = base


def _log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - logB


def _beta_share(rel: np.ndarray, a: float, b: float) -> np.ndarray:
    rel = np.clip(rel, REL_CLIP, 1.0 - REL_CLIP)
    lp = _log_beta_pdf(rel, float(a), float(b))
    lp = lp - np.max(lp)
    p = np.exp(lp)
    s = float(np.sum(p))
    return p / s if s > 0 else np.zeros_like(p)


def _smooth_share_centered(df: pd.DataFrame, share_col: str, is_h_col: str) -> np.ndarray:
    if SMOOTH_WIN <= 1:
        return df[share_col].to_numpy(dtype=float)

    tmp = df[["ciclo_id", share_col, is_h_col]].copy()
    tmp[share_col] = pd.to_numeric(tmp[share_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    tmp.loc[~tmp[is_h_col], share_col] = 0.0

    sm = (
        tmp.groupby("ciclo_id", dropna=False)[share_col]
        .rolling(window=SMOOTH_WIN, center=True, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    ).to_numpy(dtype=float)

    is_h = tmp[is_h_col].to_numpy(dtype=bool)
    sm = np.where(is_h, sm, 0.0)

    s = pd.Series(sm).groupby(tmp["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)
    sm = np.where(s > 0, sm / s, sm)
    return sm


def _first_nonnull(s: pd.Series) -> float:
    s2 = s.dropna()
    return float(s2.iloc[0]) if len(s2) else float("nan")


def _apply_cap_and_redistribute(t: np.ndarray, cap: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Apply per-day cap within mask and redistribute excess to days with slack.
    Returns (t_capped, uncovered_excess) where uncovered_excess>0 implies not enough slack.
    """
    out = t.copy()
    out[~mask] = 0.0

    cap2 = cap.copy()
    cap2[~mask] = 0.0

    target_total = float(np.sum(t[mask]))
    out = np.minimum(out, cap2)
    cur_total = float(np.sum(out[mask]))
    excess = target_total - cur_total

    if excess <= 1e-6:
        return out, 0.0

    for _ in range(MAX_REDIST_ITERS):
        slack = np.clip(cap2 - out, 0.0, None)
        slack_sum = float(np.sum(slack[mask]))
        if slack_sum <= 1e-6:
            break
        add = slack / slack_sum * excess
        add[~mask] = 0.0
        out2 = np.minimum(out + add, cap2)
        new_total = float(np.sum(out2[mask]))
        new_excess = target_total - new_total
        out = out2
        if new_excess <= 1e-6:
            return out, 0.0
        excess = new_excess

    return out, float(max(excess, 0.0))


def _compute_beta_cycle_features(panel: pd.DataFrame, cyc_key: list[str], needed: list[str]) -> pd.DataFrame:
    """
    Build cycle-level feature frame to satisfy beta model's expected feature names.
    Handles patterns:
      - col__mean, col__sum, col__last
      - real_total (filled NaN; beta model may tolerate missing)
      - any base col present (mean over harvest)
    """
    cols = cyc_key + ["fecha"] + NUM_COLS_DAILY + CAT_COLS
    cols = list(dict.fromkeys(cols))  # dedupe preserve order
    df = panel.loc[panel["is_harvest"], cols].copy()


    # Ensure numeric for daily cols
    for c in NUM_COLS_DAILY:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort for __last
    df = df.sort_values(cyc_key + ["fecha"], kind="mergesort")

    g = df.groupby(cyc_key, dropna=False)

    out = g[[]].size().rename("_n").reset_index().drop(columns=["_n"])

    # Build each needed numeric feature
    for f in needed:
        if f in out.columns:
            continue

        if f == "real_total":
            out[f] = np.nan
            continue

        if "__" in f:
            base, agg = f.split("__", 1)
            if base not in df.columns:
                out[f] = np.nan
                continue
            if agg == "mean":
                out[f] = g[base].mean().reset_index(drop=True).to_numpy(dtype=float)
            elif agg == "sum":
                out[f] = g[base].sum().reset_index(drop=True).to_numpy(dtype=float)
            elif agg == "last":
                out[f] = (
                    g[base]
                    .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1]) if pd.to_numeric(s, errors="coerce").dropna().shape[0] > 0 else np.nan)
                    .reset_index(drop=True)
                    .to_numpy(dtype=float)
                )

            else:
                out[f] = np.nan
        else:
            # base numeric col: use mean over harvest
            if f in df.columns:
                out[f] = g[f].mean().reset_index(drop=True).to_numpy(dtype=float)
            else:
                out[f] = np.nan

    # Attach cats (take first non-null per group)
    for c in CAT_COLS:
        if c not in out.columns:
            out[c] = (
                g[c].apply(lambda s: _canon_str(s.dropna()).iloc[0] if s.notna().any() else "UNKNOWN")
                .reset_index(drop=True)
            )
        out[c] = _canon_str(out[c].fillna("UNKNOWN"))

    return out


# =============================================================================
# Main
# =============================================================================
def main(beta_version: str | None = None, mult_version: str | None = None) -> None:
    # ---- Load beta params models
    beta_dir = _latest_version_dir(REG_BETA_PARAMS) if beta_version is None else (REG_BETA_PARAMS / beta_version)
    if not beta_dir.exists():
        raise FileNotFoundError(f"No existe beta version: {beta_dir}")

    metrics_beta = beta_dir / "metrics.json"
    if not metrics_beta.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {beta_dir}")

    with open(metrics_beta, "r", encoding="utf-8") as f:
        meta_beta = json.load(f)

    model_a = load(beta_dir / "model_alpha.joblib")
    model_b = load(beta_dir / "model_beta.joblib")

    # ---- Load multiplier model
    mult_dir = _latest_version_dir(REG_MULT_DIA) if mult_version is None else (REG_MULT_DIA / mult_version)
    if not mult_dir.exists():
        raise FileNotFoundError(f"No existe mult version: {mult_dir}")

    metrics_mult = mult_dir / "metrics.json"
    if not metrics_mult.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {mult_dir}")

    with open(metrics_mult, "r", encoding="utf-8") as f:
        meta_mult = json.load(f)

    model_mult = load(mult_dir / "model_log_mult.joblib")

    # ---- Load cap table
    if not CAP_PATH.exists():
        raise FileNotFoundError(f"No existe cap table: {CAP_PATH}")
    cap = read_parquet(CAP_PATH).copy()
    for c in ["area", "tipo_sp", "variedad_canon"]:
        if c not in cap.columns:
            raise ValueError(f"cap table: falta columna {c}")
        cap[c] = _canon_str(cap[c].fillna("UNKNOWN"))
    if "cap_dia" not in cap.columns:
        raise ValueError("cap table: falta cap_dia")
    cap["cap_dia"] = pd.to_numeric(cap["cap_dia"], errors="coerce").fillna(CAP_FALLBACK).astype(float)

    # ---- Load inputs
    feat = _dedupe_columns(read_parquet(FEATURES_PATH).copy())
    uni = read_parquet(UNIVERSE_PATH).copy()

    # Canon keys
    for df in (feat, uni):
        df["ciclo_id"] = df["ciclo_id"].astype(str)
        df["fecha"] = _to_date(df["fecha"])
        df["bloque_base"] = _canon_int(df["bloque_base"])
        df["variedad_canon"] = _canon_str(df["variedad_canon"])

    for c in ["area", "tipo_sp"]:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"
        feat[c] = _canon_str(feat[c].fillna("UNKNOWN"))

    # Coalesce harvest position columns
    _coalesce_cols(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce_cols(feat, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce_cols(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])

    for c in ["day_in_harvest", "rel_pos", "n_harvest_days"]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    if "tallos_pred_baseline_dia" not in feat.columns:
        raise ValueError("features_curva: falta tallos_pred_baseline_dia")
    feat["tallos_pred_baseline_dia"] = pd.to_numeric(feat["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0)

    if "tallos_proy" in feat.columns:
        feat["tallos_proy"] = pd.to_numeric(feat["tallos_proy"], errors="coerce").fillna(0.0)
    else:
        feat["tallos_proy"] = 0.0

    # Ensure daily cols exist
    for c in NUM_COLS_DAILY:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in CAT_COLS:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    # Universe panel
    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()
    take = list(dict.fromkeys(key + ["tallos_pred_baseline_dia", "tallos_proy"] + NUM_COLS_DAILY + CAT_COLS))
    panel = uni_k.merge(feat[take], on=key, how="left")


    # Canon cats
    panel["variedad_canon"] = _canon_str(panel["variedad_canon"])
    panel["area"] = _canon_str(panel["area"].fillna("UNKNOWN"))
    panel["tipo_sp"] = _canon_str(panel["tipo_sp"].fillna("UNKNOWN"))

    # Harvest mask
    dih = pd.to_numeric(panel["day_in_harvest"], errors="coerce")
    nh = pd.to_numeric(panel["n_harvest_days"], errors="coerce")
    is_h = dih.notna() & nh.notna() & (dih >= 1) & (nh >= 1) & (dih <= nh)
    panel["is_harvest"] = is_h
    is_h_np = is_h.to_numpy(dtype=bool)

    # rel_pos fallback if missing
    if panel["rel_pos"].isna().any():
        dih2 = pd.to_numeric(panel["day_in_harvest"], errors="coerce").astype(float)
        nh2 = pd.to_numeric(panel["n_harvest_days"], errors="coerce").astype(float)
        panel["rel_pos"] = np.clip((dih2 - 0.5) / (nh2 + EPS), REL_CLIP, 1.0 - REL_CLIP)

    # =============================================================================
    # 1) Predict alpha/beta (cycle-level) using the beta model's expected features
    # =============================================================================
    feat_names_beta = meta_beta.get("feature_names", [])
    num_cols_beta = meta_beta.get("feature_cols_numeric", [])
    if not feat_names_beta:
        raise ValueError("beta metrics.json no contiene feature_names")
    if not isinstance(num_cols_beta, list):
        num_cols_beta = []

    cyc_key = ["ciclo_id", "bloque_base", "variedad_canon", "area", "tipo_sp"]
    cyc = _compute_beta_cycle_features(panel, cyc_key=cyc_key, needed=num_cols_beta)

    # Build X for beta (dummies + align)
    Xb = cyc[num_cols_beta + CAT_COLS].copy() if num_cols_beta else cyc[CAT_COLS].copy()
    for c in num_cols_beta:
        if c not in Xb.columns:
            Xb[c] = np.nan
        Xb[c] = pd.to_numeric(Xb[c], errors="coerce")
    for c in CAT_COLS:
        if c not in Xb.columns:
            Xb[c] = "UNKNOWN"
        Xb[c] = _canon_str(Xb[c].fillna("UNKNOWN"))

    Xb = pd.get_dummies(Xb, columns=CAT_COLS, dummy_na=True)
    for c in feat_names_beta:
        if c not in Xb.columns:
            Xb[c] = 0.0
    Xb = Xb[feat_names_beta]

    z_a = model_a.predict(Xb)
    z_b = model_b.predict(Xb)

    alpha = 1.0 + np.exp(np.clip(pd.to_numeric(pd.Series(z_a), errors="coerce").fillna(0.0).to_numpy(dtype=float), -6, 6))
    beta = 1.0 + np.exp(np.clip(pd.to_numeric(pd.Series(z_b), errors="coerce").fillna(0.0).to_numpy(dtype=float), -6, 6))

    cyc["alpha_pred"] = np.clip(alpha, 1.05, 100.0)
    cyc["beta_pred"] = np.clip(beta, 1.05, 100.0)

    panel = panel.merge(
        cyc[cyc_key + ["alpha_pred", "beta_pred"]],
        on=cyc_key,
        how="left",
    )

    # =============================================================================
    # 2) share_beta per cycle (shape prior)
    # =============================================================================
    panel = panel.sort_values(["ciclo_id", "fecha"], kind="mergesort").reset_index(drop=True)

    share_beta = np.zeros(len(panel), dtype=float)
    for cid, idx in panel.groupby("ciclo_id", dropna=False).indices.items():
        ii = np.array(list(idx), dtype=int)
        m = panel.loc[ii, "is_harvest"].to_numpy(dtype=bool)
        if not m.any():
            continue
        rel = panel.loc[ii[m], "rel_pos"].to_numpy(dtype=float)
        a = float(pd.to_numeric(panel.loc[ii[m], "alpha_pred"], errors="coerce").dropna().iloc[0]) if panel.loc[ii[m], "alpha_pred"].notna().any() else 2.0
        b = float(pd.to_numeric(panel.loc[ii[m], "beta_pred"], errors="coerce").dropna().iloc[0]) if panel.loc[ii[m], "beta_pred"].notna().any() else 2.0
        a = max(a, 1.05)
        b = max(b, 1.05)
        sh = _beta_share(rel, a, b)
        share_beta[ii[m]] = sh

    # =============================================================================
    # 3) baseline share (harvest) for robust mixing
    # =============================================================================
    baseline = pd.to_numeric(panel["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    b_h = np.where(is_h_np, baseline, 0.0)
    sb = pd.Series(b_h).groupby(panel["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)
    base_share = np.where(sb > 0, b_h / sb, 0.0)

    # =============================================================================
    # 4) Predict daily multiplier (clima/GDC/etc)
    # =============================================================================
    feat_names_mult = meta_mult.get("feature_names", [])
    if not feat_names_mult:
        raise ValueError("mult metrics.json no contiene feature_names")

    Xd = panel[NUM_COLS_DAILY + CAT_COLS].copy()
    for c in NUM_COLS_DAILY:
        Xd[c] = pd.to_numeric(Xd[c], errors="coerce")
    for c in CAT_COLS:
        Xd[c] = _canon_str(Xd[c].fillna("UNKNOWN"))

    Xd = pd.get_dummies(Xd, columns=CAT_COLS, dummy_na=True)
    for c in feat_names_mult:
        if c not in Xd.columns:
            Xd[c] = 0.0
    Xd = Xd[feat_names_mult]

    log_mult = model_mult.predict(Xd)
    log_mult = pd.to_numeric(pd.Series(log_mult), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    log_mult = np.clip(log_mult, LOG_MULT_CLIP[0], LOG_MULT_CLIP[1])
    mult = np.exp(log_mult)
    mult = np.where(is_h_np, mult, 0.0)

    # =============================================================================
    # 5) Combine shares: share ∝ share_beta * mult, plus small baseline mixing
    # =============================================================================
    share_raw = share_beta * mult
    share_raw = np.where(is_h_np, share_raw, 0.0)

    lam = float(LAMBDA_BASE)
    share_raw = np.where(is_h_np, (1.0 - lam) * share_raw + lam * base_share, 0.0)

    s0 = pd.Series(share_raw).groupby(panel["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)
    share = np.where(s0 > 0, share_raw / s0, base_share)
    panel["share_curva_ml1"] = share

    panel["share_smooth"] = _smooth_share_centered(panel, "share_curva_ml1", "is_harvest")

    # =============================================================================
    # 6) Convert to tallos/day using cyc_total (tallos_proy preferred)
    # =============================================================================
    tproy = pd.to_numeric(panel["tallos_proy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cyc_tproy = pd.Series(tproy).groupby(panel["ciclo_id"], dropna=False).transform("max").to_numpy(dtype=float)
    cyc_total = np.where(cyc_tproy > 0, cyc_tproy, sb)

    tallos_pre_cap = cyc_total * panel["share_smooth"].to_numpy(dtype=float)

    # =============================================================================
    # 7) Join cap and apply cap + redistribute within each cycle
    # =============================================================================
    panel = panel.merge(
        cap[["area", "tipo_sp", "variedad_canon", "cap_dia"]].drop_duplicates(),
        on=["area", "tipo_sp", "variedad_canon"],
        how="left",
    )
    cap_dia = pd.to_numeric(panel["cap_dia"], errors="coerce").fillna(CAP_FALLBACK).to_numpy(dtype=float)

    tallos_cap = np.zeros(len(panel), dtype=float)
    uncovered = np.zeros(len(panel), dtype=float)  # store per row as 0; aggregate per cycle later

    for cid, idx in panel.groupby("ciclo_id", dropna=False).indices.items():
        ii = np.array(list(idx), dtype=int)
        m = panel.loc[ii, "is_harvest"].to_numpy(dtype=bool)
        if not m.any():
            continue
        tc, exc = _apply_cap_and_redistribute(tallos_pre_cap[ii], cap_dia[ii], m)
        tallos_cap[ii] = tc
        if exc > 0:
            uncovered[ii[m]] = exc  # mark on harvest rows

    # Post-cap share (for audit)
    share_post_cap = np.zeros(len(panel), dtype=float)
    for cid, idx in panel.groupby("ciclo_id", dropna=False).indices.items():
        ii = np.array(list(idx), dtype=int)
        m = panel.loc[ii, "is_harvest"].to_numpy(dtype=bool)
        if not m.any():
            continue
        tot2 = float(np.sum(tallos_cap[ii][m]))
        if tot2 > 0:
            share_post_cap[ii[m]] = tallos_cap[ii][m] / tot2

    # =============================================================================
    # 8) Factor vs baseline (harvest) using baseline_h_dia = cyc_total * base_share
    # =============================================================================
    # factor = peso diario final (share), ya cappeado
    factor_raw = np.where(is_h_np, share_post_cap, 0.0).astype(float)
    factor = factor_raw.copy()  # ya está en [0,1] y suma 1 por ciclo (harvest)

    # flags opcionales
    was_capped_factor = np.zeros(len(panel), dtype="int8")

    factor = np.where(np.isfinite(factor), factor, 1.0)

    was_capped_factor = ((factor_raw < FACTOR_MIN) | (factor_raw > FACTOR_MAX)).astype("int8")

    # =============================================================================
    # Output
    # =============================================================================
    out = panel[key].copy()
    out["factor_curva_ml1_raw"] = factor_raw
    out["factor_curva_ml1"] = factor
    out["ml1_version"] = f"beta={beta_dir.name}|mult={mult_dir.name}"
    out["created_at"] = pd.Timestamp.utcnow()

    # Audit fields
    out["alpha_pred"] = pd.to_numeric(panel["alpha_pred"], errors="coerce")
    out["beta_pred"] = pd.to_numeric(panel["beta_pred"], errors="coerce")
    out["log_mult_pred"] = log_mult
    out["mult_pred"] = mult

    out["share_beta"] = share_beta
    out["share_base"] = base_share
    out["share_pre_cap"] = panel["share_smooth"].to_numpy(dtype=float)
    out["tallos_pred_ml1_dia_pre_cap"] = tallos_pre_cap
    out["cap_dia"] = cap_dia
    out["tallos_pred_ml1_dia_cap"] = tallos_cap
    out["share_post_cap"] = share_post_cap
    out["uncovered_excess_cycle"] = uncovered  # non-zero if not enough slack in cycle
    out["was_capped_factor"] = was_capped_factor
    out["factor_curva_ml1_raw"] = factor_raw
    out["factor_curva_ml1"] = factor

    out = out.sort_values(["ciclo_id", "bloque_base", "variedad_canon", "fecha"]).reset_index(drop=True)

    # Duplicate-column hard check before writing
    cols = pd.Index(out.columns.astype(str))
    if not cols.is_unique:
        dup = cols[cols.duplicated()].unique().tolist()
        raise ValueError(f"[FATAL] columnas duplicadas en OUT: {dup}")

    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={out['ml1_version'].iloc[0] if len(out) else ''}")


if __name__ == "__main__":
    main(beta_version=None, mult_version=None)
