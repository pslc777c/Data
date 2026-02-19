from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet

# =============================================================================
# Paths
# =============================================================================
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")
DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")

# Usamos los alpha/beta fiteados
BETA_TRAINSET_PATH = Path("data/features/trainset_curva_beta_params.parquet")

OUT_PATH = Path("data/features/trainset_curva_beta_multiplier_dia.parquet")

# =============================================================================
# Columns (match your previous intent)
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

# =============================================================================
# Config
# =============================================================================
EPS = 1e-12
REL_CLIP = 1e-4
MIN_REAL_TOTAL_CYCLE = 50.0
INCLUDE_ZERO_REAL_DAYS = True
TARGET_CLIP = (-4.0, 4.0)

# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")

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

def _load_var_map(dim_var: pd.DataFrame) -> dict[str, str]:
    _require(dim_var, ["variedad_raw", "variedad_canon"], "dim_variedad_canon")
    dv = dim_var.copy()
    dv["variedad_raw"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    dv = dv.dropna(subset=["variedad_raw", "variedad_canon"]).drop_duplicates(subset=["variedad_raw"])
    return dict(zip(dv["variedad_raw"], dv["variedad_canon"]))

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

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    for p in [FEATURES_PATH, UNIVERSE_PATH, PROG_PATH, DIM_VAR_PATH, BETA_TRAINSET_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    feat = _dedupe_columns(read_parquet(FEATURES_PATH).copy())
    uni = read_parquet(UNIVERSE_PATH).copy()
    prog = read_parquet(PROG_PATH).copy()
    dim_var = read_parquet(DIM_VAR_PATH).copy()
    beta_ts = read_parquet(BETA_TRAINSET_PATH).copy()

    var_map = _load_var_map(dim_var)

    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe")
    _require(feat, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "features")
    _require(prog, ["ciclo_id", "fecha", "bloque_base"], "prog")
    _require(beta_ts, ["ciclo_id", "bloque_base", "variedad_canon", "alpha", "beta"], "beta_trainset")

    # Canon
    for df in (feat, uni, prog, beta_ts):
        df["ciclo_id"] = df["ciclo_id"].astype(str)
        if "fecha" in df.columns:
            df["fecha"] = _to_date(df["fecha"])
        df["bloque_base"] = _canon_int(df["bloque_base"])

    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])
    beta_ts["variedad_canon"] = _canon_str(beta_ts["variedad_canon"])

    # Canon prog variedad
    if "variedad" in prog.columns:
        prog["variedad_raw"] = _canon_str(prog["variedad"])
        prog["variedad_canon"] = prog["variedad_raw"].map(var_map).fillna(prog["variedad_raw"])
    else:
        prog["variedad_canon"] = _canon_str(prog["variedad_canon"])

    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Coalesce harvest cols
    _coalesce_cols(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce_cols(feat, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce_cols(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])

    for c in ["day_in_harvest", "rel_pos", "n_harvest_days"]:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # Ensure feature cols exist
    for c in NUM_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in CAT_COLS:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    # Universe panel
    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    feat_take = key + NUM_COLS + CAT_COLS
    feat_take = list(dict.fromkeys(feat_take))
    panel = (
        uni_k
        .merge(feat[feat_take], on=key, how="left")
        .merge(prog[key + ["tallos_real_dia"]].drop_duplicates(subset=key), on=key, how="left")
    )
    panel["tallos_real_dia"] = pd.to_numeric(panel["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Harvest mask
    dih = pd.to_numeric(panel["day_in_harvest"], errors="coerce")
    nh = pd.to_numeric(panel["n_harvest_days"], errors="coerce")
    is_h = dih.notna() & nh.notna() & (dih >= 1) & (nh >= 1) & (dih <= nh)
    panel = panel[is_h].copy()

    # rel_pos fallback if needed
    if panel["rel_pos"].isna().any():
        dih2 = pd.to_numeric(panel["day_in_harvest"], errors="coerce").astype(float)
        nh2 = pd.to_numeric(panel["n_harvest_days"], errors="coerce").astype(float)
        panel["rel_pos"] = np.clip((dih2 - 0.5) / (nh2 + EPS), REL_CLIP, 1.0 - REL_CLIP)

    # Canon cats
    panel["variedad_canon"] = _canon_str(panel["variedad_canon"])
    panel["area"] = _canon_str(panel["area"].fillna("UNKNOWN"))
    panel["tipo_sp"] = _canon_str(panel["tipo_sp"].fillna("UNKNOWN"))

    # Filtrar ciclos con señal real
    grp = ["ciclo_id", "bloque_base", "variedad_canon"]
    real_total_grp = panel.groupby(grp, dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    panel = panel[real_total_grp >= MIN_REAL_TOTAL_CYCLE].copy()
    if len(panel) == 0:
        raise ValueError("No hay ciclos con real_total suficiente para entrenar multiplicador. Baja MIN_REAL_TOTAL_CYCLE o revisa PROG.")

    # Share real por ciclo
    tot = panel.groupby("ciclo_id", dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    panel["share_real"] = np.where(tot > 0, panel["tallos_real_dia"].astype(float) / tot, 0.0)

    # Adjuntar alpha/beta fiteados (por grupo)
    beta_take = beta_ts[["ciclo_id", "bloque_base", "variedad_canon", "alpha", "beta"]].drop_duplicates()
    panel = panel.merge(beta_take, on=["ciclo_id", "bloque_base", "variedad_canon"], how="left")

    panel = panel[panel["alpha"].notna() & panel["beta"].notna()].copy()
    if len(panel) == 0:
        raise ValueError("No quedaron filas con alpha/beta (beta_trainset no alinea).")

    # Share beta por ciclo
    panel = panel.sort_values(["ciclo_id", "fecha"], kind="mergesort").reset_index(drop=True)
    share_beta = np.zeros(len(panel), dtype=float)

    for cid, idx in panel.groupby("ciclo_id", dropna=False).indices.items():
        ii = np.array(list(idx), dtype=int)
        rel = panel.loc[ii, "rel_pos"].to_numpy(dtype=float)
        a = float(pd.to_numeric(panel.loc[ii, "alpha"], errors="coerce").dropna().iloc[0])
        b = float(pd.to_numeric(panel.loc[ii, "beta"], errors="coerce").dropna().iloc[0])
        a = max(a, 1.05)
        b = max(b, 1.05)
        sh = _beta_share(rel, a, b)
        share_beta[ii] = sh

    panel["share_beta"] = share_beta

    # Target log-mult
    mult = panel["share_real"].to_numpy(dtype=float) / (panel["share_beta"].to_numpy(dtype=float) + EPS)
    y = np.log(np.clip(mult, EPS, 1e6))
    y = np.clip(y, TARGET_CLIP[0], TARGET_CLIP[1])
    panel["y_log_mult"] = y

    if not INCLUDE_ZERO_REAL_DAYS:
        panel = panel[panel["tallos_real_dia"] > 0].copy()

    # Keep only needed columns (DEDUP!)
    keep = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"] + NUM_COLS + CAT_COLS + [
        "tallos_real_dia", "share_real", "share_beta", "alpha", "beta", "y_log_mult"
    ]
    keep = [c for c in keep if c in panel.columns]
    keep = list(dict.fromkeys(keep))  # <-- FIX: remove duplicates preserving order

    out = panel[keep].copy()
    out["created_at"] = created_at

    # Hard check before write
    cols = pd.Index(out.columns.astype(str))
    if not cols.is_unique:
        dup = cols[cols.duplicated()].unique().tolist()
        raise ValueError(f"[FATAL] columnas duplicadas antes de escribir: {dup}")

    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | cycles={out['ciclo_id'].nunique():,}")

if __name__ == "__main__":
    main()
