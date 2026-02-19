from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet

# =============================================================================
# Paths
# =============================================================================
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/curva_cdf_dia")
OUT_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")

# =============================================================================
# Model columns (must match training)
# =============================================================================
NUM_COLS = [
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
CAT_COLS_MERGE = ["area", "tipo_sp"]

# =============================================================================
# Statistical adjustments (NOT capacity)
# =============================================================================
SMOOTH_WIN = 5   # rolling mean centered on share (per cycle)
EPS = 1e-12
FACTOR_MIN, FACTOR_MAX = 0.2, 5.0


# =============================================================================
# Helpers
# =============================================================================
def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    if out_col in df.columns:
        base = df[out_col]
    else:
        base = pd.Series([pd.NA] * len(df), index=df.index)

    for c in candidates:
        if c in df.columns:
            base = base.where(base.notna(), df[c])
    df[out_col] = base


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index(df.columns.astype(str))
    if cols.is_unique:
        return df

    out = df.copy()
    seen: dict[str, list[int]] = {}
    for i, c in enumerate(out.columns.astype(str)):
        seen.setdefault(c, []).append(i)

    keep_series: dict[str, pd.Series] = {}
    for c, idxs in seen.items():
        if len(idxs) == 1:
            keep_series[c] = out.iloc[:, idxs[0]]
        else:
            s = out.iloc[:, idxs[0]]
            for j in idxs[1:]:
                s2 = out.iloc[:, j]
                s = s.where(s.notna(), s2)
            keep_series[c] = s

    ordered: list[str] = []
    for c in out.columns.astype(str):
        if c not in ordered:
            ordered.append(c)

    return pd.DataFrame({c: keep_series[c] for c in ordered})


def _baseline_share_and_cdf(tmp: pd.DataFrame, is_h_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    baseline_share: b_h / sum(b_h) per cycle (harvest only)
    baseline_cdf: cumsum(baseline_share) per cycle (already sorted)
    sb: sum baseline harvest per cycle (transform)
    """
    baseline = pd.to_numeric(tmp["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    b_h = np.where(is_h_np, baseline, 0.0)

    sb = pd.Series(b_h).groupby(tmp["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)
    base_share = np.where(sb > 0, b_h / sb, 0.0)

    base_cdf = pd.Series(base_share).groupby(tmp["ciclo_id"], dropna=False).cumsum().to_numpy(dtype=float)
    base_cdf = np.clip(base_cdf, 0.0, 1.0)
    return base_share, base_cdf, sb


def _smooth_share_centered(tmp: pd.DataFrame, share_col: str, is_h_col: str) -> np.ndarray:
    """
    Smooth share per cycle (harvest only) via centered rolling mean.
    Then renormalize to sum=1 per cycle (harvest only).
    """
    if SMOOTH_WIN <= 1:
        return tmp[share_col].to_numpy(dtype=float)

    df = tmp[["ciclo_id", share_col, is_h_col]].copy()
    df[share_col] = pd.to_numeric(df[share_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    df.loc[~df[is_h_col], share_col] = 0.0

    sm = (
        df.groupby("ciclo_id", dropna=False)[share_col]
        .rolling(window=SMOOTH_WIN, center=True, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    ).to_numpy(dtype=float)

    is_h = df[is_h_col].to_numpy(dtype=bool)
    sm = np.where(is_h, sm, 0.0)

    s = pd.Series(sm).groupby(df["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)
    sm = np.where(s > 0, sm / s, sm)
    return sm


def _first_nonnull(s: pd.Series) -> float:
    s2 = s.dropna()
    return float(s2.iloc[0]) if len(s2) else np.nan


def _last_nonnull(s: pd.Series) -> float:
    s2 = s.dropna()
    return float(s2.iloc[-1]) if len(s2) else np.nan


# =============================================================================
# Main
# =============================================================================
def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir() if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {ver_dir}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = ver_dir / "model_curva_cdf_dia.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No encontré modelo: {model_path}")
    model = load(model_path)

    feat = _dedupe_columns(read_parquet(FEATURES_PATH).copy())
    uni = read_parquet(UNIVERSE_PATH).copy()

    # -------------------------
    # Canon keys
    # -------------------------
    feat["ciclo_id"] = feat["ciclo_id"].astype(str)
    feat["fecha"] = _to_date(feat["fecha"])
    feat["bloque_base"] = _canon_int(feat["bloque_base"])
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    for c in ["area", "tipo_sp"]:
        if c in feat.columns:
            feat[c] = _canon_str(feat[c])

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

    # Ensure model cols exist
    for c in NUM_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in CAT_COLS:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    # Universe
    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    feat_take = key + ["tallos_pred_baseline_dia", "tallos_proy"] + NUM_COLS + CAT_COLS_MERGE
    feat_take = list(dict.fromkeys(feat_take))
    panel = uni_k.merge(feat[feat_take], on=key, how="left")

    # Fill defaults for missing matches
    for c in NUM_COLS:
        if c not in panel.columns:
            panel[c] = np.nan
    for c in CAT_COLS:
        if c not in panel.columns:
            panel[c] = "UNKNOWN"

    panel["variedad_canon"] = _canon_str(panel["variedad_canon"])
    if "area" in panel.columns:
        panel["area"] = _canon_str(panel["area"].fillna("UNKNOWN"))
    if "tipo_sp" in panel.columns:
        panel["tipo_sp"] = _canon_str(panel["tipo_sp"].fillna("UNKNOWN"))

    # Harvest mask
    dih = pd.to_numeric(panel["day_in_harvest"], errors="coerce")
    nh = pd.to_numeric(panel["n_harvest_days"], errors="coerce")
    is_h = dih.notna() & nh.notna() & (dih >= 1) & (nh >= 1) & (dih <= nh)
    is_h_np = is_h.to_numpy(dtype=bool)

    # One-hot aligned with training
    X = panel[NUM_COLS + CAT_COLS].copy()
    X = pd.get_dummies(X, columns=CAT_COLS, dummy_na=True)

    feat_names = meta.get("feature_names", [])
    if not feat_names:
        raise ValueError("metrics.json no contiene feature_names (necesario para alinear dummies).")

    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]

    # Predict CDF
    cdf_pred = model.predict(X)
    cdf_pred = pd.to_numeric(pd.Series(cdf_pred), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cdf_pred = np.clip(cdf_pred, 0.0, 1.0)
    cdf_pred = np.where(is_h_np, cdf_pred, 0.0)

    tmp = panel[key + ["tallos_pred_baseline_dia", "tallos_proy", "day_in_harvest", "rel_pos", "n_harvest_days"]].copy()
    tmp["ml1_version"] = ver_dir.name
    tmp["cdf_pred_raw"] = cdf_pred
    tmp["is_harvest"] = is_h_np

    # Sort within cycle
    tmp["_dih_sort"] = pd.to_numeric(tmp["day_in_harvest"], errors="coerce")
    tmp["_sort_key"] = np.where(
        tmp["_dih_sort"].notna(),
        tmp["_dih_sort"].astype(float),
        tmp["fecha"].astype("int64").astype(float),
    )
    tmp = tmp.sort_values(["ciclo_id", "_sort_key"], kind="mergesort").reset_index(drop=True)

    # Monotone
    tmp["cdf_pred_mono"] = tmp.groupby("ciclo_id", dropna=False)["cdf_pred_raw"].cummax().clip(0.0, 1.0)

    # Baseline fallback (already sorted)
    base_share, base_cdf, sb = _baseline_share_and_cdf(tmp, tmp["is_harvest"].to_numpy(dtype=bool))

    # Anchor within harvest: (cdf - start)/(end-start)
    cdf_h = tmp["cdf_pred_mono"].where(tmp["is_harvest"])
    cdf_start = cdf_h.groupby(tmp["ciclo_id"], dropna=False).transform(_first_nonnull).to_numpy(dtype=float)
    cdf_end = cdf_h.groupby(tmp["ciclo_id"], dropna=False).transform(_last_nonnull).to_numpy(dtype=float)
    denom = cdf_end - cdf_start

    cdf_adj = (tmp["cdf_pred_mono"].to_numpy(dtype=float) - cdf_start) / (denom + EPS)
    cdf_adj = np.clip(cdf_adj, 0.0, 1.0)

    use_model = tmp["is_harvest"].to_numpy(dtype=bool) & np.isfinite(denom) & (denom > 1e-6)
    cdf_final = np.where(use_model, cdf_adj, base_cdf)
    cdf_final = np.where(tmp["is_harvest"].to_numpy(dtype=bool), cdf_final, 0.0)

    tmp["cdf_pred_adj"] = cdf_final
    tmp["cdf_pred_adj"] = tmp.groupby("ciclo_id", dropna=False)["cdf_pred_adj"].cummax().clip(0.0, 1.0)

    # Force end=1 inside harvest (if harvest exists)
    cdf_end2 = tmp["cdf_pred_adj"].where(tmp["is_harvest"]).groupby(tmp["ciclo_id"], dropna=False).transform(_last_nonnull).to_numpy(dtype=float)
    scale = np.where(np.isfinite(cdf_end2) & (cdf_end2 > 1e-6), cdf_end2, 1.0)
    tmp["cdf_pred_adj"] = np.where(tmp["is_harvest"], tmp["cdf_pred_adj"] / scale, 0.0)
    tmp["cdf_pred_adj"] = tmp["cdf_pred_adj"].clip(0.0, 1.0)

    # Share = diff(CDF)
    tmp["share_pred_in"] = tmp.groupby("ciclo_id", dropna=False)["cdf_pred_adj"].diff()
    tmp["share_pred_in"] = tmp["share_pred_in"].fillna(tmp["cdf_pred_adj"]).astype(float)
    tmp["share_pred_in"] = tmp["share_pred_in"].clip(lower=0.0)
    tmp["share_pred_in"] = np.where(tmp["is_harvest"].to_numpy(dtype=bool), tmp["share_pred_in"].to_numpy(dtype=float), 0.0)

    # Renormalize per cycle (harvest); if sum=0 fallback baseline share
    s = tmp.groupby("ciclo_id", dropna=False)["share_pred_in"].transform("sum").astype(float).to_numpy()
    share = np.where(s > 0, tmp["share_pred_in"].to_numpy(dtype=float) / s, base_share)
    tmp["share_curva_ml1"] = share

    # Smooth
    tmp["share_smooth"] = _smooth_share_centered(tmp, "share_curva_ml1", "is_harvest")

    # Total per cycle: prefer tallos_proy if >0 else sum baseline harvest (sb)
    tproy = pd.to_numeric(tmp["tallos_proy"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cyc_tproy = pd.Series(tproy).groupby(tmp["ciclo_id"], dropna=False).transform("max").to_numpy(dtype=float)
    cyc_total = np.where(cyc_tproy > 0, cyc_tproy, sb)

    tallos_ml1_dia = cyc_total * tmp["share_smooth"].to_numpy(dtype=float)
    tmp["tallos_pred_ml1_dia_from_cdf"] = tallos_ml1_dia

    baseline = pd.to_numeric(tmp["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    factor_raw = np.where(tmp["is_harvest"].to_numpy(dtype=bool), tallos_ml1_dia / (baseline + 1e-9), 1.0)
    factor = np.clip(factor_raw, FACTOR_MIN, FACTOR_MAX)
    factor = np.where(np.isfinite(factor), factor, 1.0)

    # Flags (FIX: numpy dtype, NOT "Int64")
    was_capped_pre = np.where((factor_raw < FACTOR_MIN) | (factor_raw > FACTOR_MAX), 1, 0).astype("int8")
    was_capped_post = np.where((factor < FACTOR_MIN) | (factor > FACTOR_MAX), 1, 0).astype("int8")

    # Output
    out = tmp[key].copy()
    out["factor_curva_ml1_raw"] = factor_raw
    out["factor_curva_ml1"] = factor
    out["ml1_version"] = ver_dir.name
    out["created_at"] = pd.Timestamp.utcnow()

    # Audit columns (keep names compatible with tus auditorías)
    out["_share_source"] = np.where(use_model, "cdf_adj", np.where(sb > 0, "baseline", "zero"))
    out["cap_share"] = np.nan  # placeholder (si luego metes cap/floor por bins)
    out["share_pred_in"] = tmp["share_pred_in"].to_numpy(dtype=float)
    out["share_smooth"] = tmp["share_smooth"].to_numpy(dtype=float)
    out["share_curva_ml1"] = tmp["share_curva_ml1"].to_numpy(dtype=float)
    out["tallos_pred_ml1_dia_smooth"] = tmp["tallos_pred_ml1_dia_from_cdf"].to_numpy(dtype=float)
    out["factor_curva_ml1_raw_smooth"] = factor_raw  # aquí factor_raw ya viene del share_smooth
    out["was_capped_pre"] = was_capped_pre
    out["was_capped_post"] = was_capped_post

    out = out.sort_values(["bloque_base", "variedad_canon", "fecha"]).reset_index(drop=True)
    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={ver_dir.name}")


if __name__ == "__main__":
    main(version=None)
