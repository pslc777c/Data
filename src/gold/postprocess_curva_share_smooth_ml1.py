from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet

# =============================================================================
# Paths
# =============================================================================
PRED_FACTOR_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")
FEATURES_CURVA_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")
DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")

OUT_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")  # overwrite downstream-safe

# =============================================================================
# Hyperparams (estadísticos + smoothing)
# =============================================================================
# cap por segmento = quantil alto del share_real (por defecto P99.5)
CAP_Q = 0.995

# buckets de duración (días) para estabilizar caps
NDAYS_BINS = [0, 25, 35, 45, 55, 65, 80, 120, 10_000]
NDAYS_LABELS = ["<=25", "26-35", "36-45", "46-55", "56-65", "66-80", "81-120", "120+"]

# smoothing (share-space)
SMOOTH_WIN = 5          # 3 o 5 recomendado
SMOOTH_CENTER = True    # centered rolling
SMOOTH_MINP = 1

# factor safety (mantener compat)
FACTOR_MIN, FACTOR_MAX = 0.2, 5.0
EPS = 1e-9


# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int64(s: pd.Series) -> pd.Series:
    # usamos Int64 nullable para evitar merges raros con NaNs
    x = pd.to_numeric(s, errors="coerce")
    return x.astype("Int64")


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Cols={list(df.columns)}")


def _load_var_map(dim_var: pd.DataFrame) -> dict[str, str]:
    _require(dim_var, ["variedad_raw", "variedad_canon"], "dim_variedad_canon")
    dv = dim_var.copy()
    dv["variedad_raw"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    dv = dv.dropna(subset=["variedad_raw", "variedad_canon"]).drop_duplicates(subset=["variedad_raw"])
    return dict(zip(dv["variedad_raw"], dv["variedad_canon"]))


def _ndays_bucket(n: pd.Series) -> pd.Series:
    n2 = pd.to_numeric(n, errors="coerce")
    return pd.cut(n2, bins=NDAYS_BINS, labels=NDAYS_LABELS, right=True, include_lowest=True).astype(str)


def _rolling_smooth_share(df: pd.DataFrame, share_col: str, out_col: str) -> pd.DataFrame:
    """
    Suaviza share por ciclo en el orden temporal de fecha.
    """
    out = df.copy()
    out = out.sort_values(["ciclo_id", "fecha"], kind="mergesort")
    out[out_col] = (
        out.groupby("ciclo_id", dropna=False)[share_col]
        .transform(lambda s: s.rolling(SMOOTH_WIN, center=SMOOTH_CENTER, min_periods=SMOOTH_MINP).mean())
    )
    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.now("UTC")

    # -------------------------
    # Read inputs
    # -------------------------
    for p in [PRED_FACTOR_PATH, FEATURES_CURVA_PATH, UNIVERSE_PATH, PROG_PATH, DIM_VAR_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    pred = read_parquet(PRED_FACTOR_PATH).copy()
    feat = read_parquet(FEATURES_CURVA_PATH).copy()
    uni = read_parquet(UNIVERSE_PATH).copy()
    prog = read_parquet(PROG_PATH).copy()
    dim_var = read_parquet(DIM_VAR_PATH).copy()

    var_map = _load_var_map(dim_var)

    # -------------------------
    # Canon llaves
    # -------------------------
    _require(pred, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "pred_factor_curva_ml1")
    pred["ciclo_id"] = pred["ciclo_id"].astype(str)
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_int64(pred["bloque_base"])
    pred["variedad_canon"] = _canon_str(pred["variedad_canon"])

    _require(
        feat,
        ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "tallos_pred_baseline_dia", "tallos_proy"],
        "features_curva",
    )
    feat["ciclo_id"] = feat["ciclo_id"].astype(str)
    feat["fecha"] = _to_date(feat["fecha"])
    feat["bloque_base"] = _canon_int64(feat["bloque_base"])
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    for c in ["area", "tipo_sp"]:
        if c in feat.columns:
            feat[c] = _canon_str(feat[c])

    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")
    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int64(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    _require(prog, ["ciclo_id", "fecha", "bloque_base", "variedad", "tallos_real_dia"], "dim_cosecha_progress_bloque_fecha")
    prog["ciclo_id"] = prog["ciclo_id"].astype(str)
    prog["fecha"] = _to_date(prog["fecha"])
    prog["bloque_base"] = _canon_int64(prog["bloque_base"])
    prog["variedad_raw"] = _canon_str(prog["variedad"])
    prog["variedad_canon"] = prog["variedad_raw"].map(var_map).fillna(prog["variedad_raw"])
    prog["variedad_canon"] = _canon_str(prog["variedad_canon"])
    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0)

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]

    # -------------------------
    # Universe day-count (n_harvest_days_pred) por ciclo (para buckets)
    # -------------------------
    if "n_harvest_days_pred" in uni.columns:
        n_days = (
            uni.groupby("ciclo_id", dropna=False)["n_harvest_days_pred"]
            .max()
            .rename("n_days")
            .reset_index()
        )
    else:
        n_days = uni.groupby("ciclo_id", dropna=False)["fecha"].count().rename("n_days").reset_index()

    # -------------------------
    # Build share_real panelizado en universe (missing days = 0)
    # -------------------------
    uni_k = uni[key].drop_duplicates()
    prog_k = prog[key + ["tallos_real_dia"]].drop_duplicates(subset=key)

    real_panel = uni_k.merge(prog_k, on=key, how="left")
    real_panel["tallos_real_dia"] = pd.to_numeric(real_panel["tallos_real_dia"], errors="coerce").fillna(0.0)

    cyc_sum = real_panel.groupby("ciclo_id", dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    real_panel["has_real"] = cyc_sum > 0
    real_panel["share_real"] = np.where(
        cyc_sum > 0,
        real_panel["tallos_real_dia"].astype(float) / cyc_sum,
        np.nan,
    )

    # -------------------------
    # Segment info (variedad/area/tipo_sp + bucket)
    # -------------------------
    seg_cols = ["ciclo_id", "bloque_base", "variedad_canon"]
    for c in ["area", "tipo_sp"]:
        if c in feat.columns:
            seg_cols.append(c)

    seg = feat[seg_cols].drop_duplicates(subset=["ciclo_id", "bloque_base", "variedad_canon"])
    seg = seg.merge(n_days, on="ciclo_id", how="left")
    seg["ndays_bucket"] = _ndays_bucket(seg["n_days"])
    for c in ["area", "tipo_sp"]:
        if c in seg.columns:
            seg[c] = _canon_str(seg[c].fillna("UNKNOWN"))

    # -------------------------
    # Caps estadísticos (por segmento) usando share_real histórico
    # -------------------------
    real_for_caps = real_panel.merge(
        seg,
        on=["ciclo_id", "bloque_base", "variedad_canon"],
        how="left",
        suffixes=("", "_seg"),
    )

    seg_key = ["variedad_canon", "ndays_bucket"]
    if "area" in real_for_caps.columns:
        seg_key.append("area")
    if "tipo_sp" in real_for_caps.columns:
        seg_key.append("tipo_sp")

    caps_base = real_for_caps[real_for_caps["has_real"] & real_for_caps["share_real"].notna()].copy()

    # fallback global cap
    global_cap = float(caps_base["share_real"].quantile(CAP_Q)) if len(caps_base) else 1.0

    caps = (
        caps_base.groupby(seg_key, dropna=False)["share_real"]
        .quantile(CAP_Q)
        .rename("cap_share")
        .reset_index()
    )
    caps["cap_share"] = pd.to_numeric(caps["cap_share"], errors="coerce").fillna(global_cap)
    caps["cap_share"] = caps["cap_share"].clip(lower=0.0, upper=0.30)

    # -------------------------
    # Construir share_pred desde pred_factor (preferir share_curva_ml1 si existe)
    # -------------------------
    take_feat = key + ["tallos_pred_baseline_dia", "tallos_proy"]
    for c in ["area", "tipo_sp"]:
        if c in feat.columns:
            take_feat.append(c)
    take_feat = list(dict.fromkeys(take_feat))

    pred2 = pred.merge(feat[take_feat].drop_duplicates(subset=key), on=key, how="left", suffixes=("", "_f"))
    pred2 = pred2.merge(n_days, on="ciclo_id", how="left")
    pred2["ndays_bucket"] = _ndays_bucket(pred2["n_days"])

    for c in ["area", "tipo_sp"]:
        if c in pred2.columns:
            pred2[c] = _canon_str(pred2[c].fillna("UNKNOWN"))

    # robust: si faltan columnas numéricas, crear con 0
    if "tallos_pred_baseline_dia" in pred2.columns:
        pred2["tallos_pred_baseline_dia"] = pd.to_numeric(pred2["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0)
    else:
        pred2["tallos_pred_baseline_dia"] = 0.0

    if "tallos_proy" in pred2.columns:
        pred2["tallos_proy"] = pd.to_numeric(pred2["tallos_proy"], errors="coerce").fillna(0.0)
    else:
        pred2["tallos_proy"] = 0.0

    if "share_curva_ml1" in pred2.columns:
        share_pred = pd.to_numeric(pred2["share_curva_ml1"], errors="coerce").fillna(0.0)
        share_pred = share_pred.clip(lower=0.0)
        pred2["share_pred_in"] = share_pred
        pred2["_share_source"] = "share_curva_ml1"
    else:
        if "factor_curva_ml1" not in pred2.columns:
            raise ValueError("pred_factor_curva_ml1 no tiene share_curva_ml1 ni factor_curva_ml1. No puedo reconstruir share.")
        factor = pd.to_numeric(pred2["factor_curva_ml1"], errors="coerce").fillna(1.0)
        tallos_ml1_dia = pred2["tallos_pred_baseline_dia"].astype(float) * factor.astype(float)
        denom = tallos_ml1_dia.groupby(pred2["ciclo_id"]).transform("sum").astype(float)
        pred2["share_pred_in"] = np.where(denom > 0, tallos_ml1_dia / denom, 0.0)
        pred2["_share_source"] = "factor_curva_ml1"

    # -------------------------
    # Aplicar cap estadístico + smoothing + renorm por ciclo
    # -------------------------
    # IMPORTANTÍSIMO: pred ya puede traer cap_share; si no lo botamos, el merge crea cap_share_x/cap_share_y y luego falla.
    if "cap_share" in pred2.columns:
        pred2 = pred2.drop(columns=["cap_share"], errors="ignore")

    pred2 = pred2.merge(caps, on=seg_key, how="left")

    # si por algo no vino cap_share tras el merge, fallback a global_cap
    if "cap_share" not in pred2.columns:
        pred2["cap_share"] = float(global_cap)
    else:
        pred2["cap_share"] = pd.to_numeric(pred2["cap_share"], errors="coerce").fillna(global_cap)

    pred2["cap_share"] = pred2["cap_share"].clip(lower=0.0, upper=0.30)

    # cap pre-smooth
    pred2["share_cap_pre"] = pd.to_numeric(pred2["share_pred_in"], errors="coerce").fillna(0.0).clip(lower=0.0)
    pred2["was_capped_pre"] = pred2["share_cap_pre"] > pred2["cap_share"]
    pred2["share_cap_pre"] = np.minimum(pred2["share_cap_pre"], pred2["cap_share"])

    # smooth
    pred2 = _rolling_smooth_share(pred2, "share_cap_pre", "share_smooth")
    pred2["share_smooth"] = pd.to_numeric(pred2["share_smooth"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # cap post-smooth
    pred2["was_capped_post"] = pred2["share_smooth"] > pred2["cap_share"]
    pred2["share_cap_post"] = np.minimum(pred2["share_smooth"], pred2["cap_share"])

    # renormalize per cycle
    s = pred2.groupby("ciclo_id", dropna=False)["share_cap_post"].transform("sum").astype(float)
    pred2["share_final"] = np.where(s > 0, pred2["share_cap_post"] / s, 0.0)

    # -------------------------
    # Reconstrucción tallos_ml1_dia y factor compatible downstream
    # -------------------------
    pred2["tallos_pred_ml1_dia_smooth"] = pred2["tallos_proy"].astype(float) * pred2["share_final"].astype(float)

    base = pred2["tallos_pred_baseline_dia"].astype(float)
    pred2["factor_curva_ml1_raw_smooth"] = np.where(base > 0, pred2["tallos_pred_ml1_dia_smooth"] / (base + EPS), 1.0)
    pred2["factor_curva_ml1_smooth"] = pred2["factor_curva_ml1_raw_smooth"].clip(lower=FACTOR_MIN, upper=FACTOR_MAX)
    pred2["factor_curva_ml1_smooth"] = np.where(np.isfinite(pred2["factor_curva_ml1_smooth"]), pred2["factor_curva_ml1_smooth"], 1.0)

    # -------------------------
    # Overwrite principal factor columns (downstream)
    # -------------------------
    if "ml1_version" not in pred2.columns:
        pred2["ml1_version"] = "UNKNOWN"

    pred2["factor_curva_ml1_raw"] = pred2["factor_curva_ml1_raw_smooth"]
    pred2["factor_curva_ml1"] = pred2["factor_curva_ml1_smooth"]

    # mantener share para auditoría
    pred2["share_curva_ml1_raw"] = pred2.get("share_curva_ml1_raw", pred2["share_pred_in"])
    pred2["share_curva_ml1"] = pred2["share_final"]

    pred2["created_at"] = created_at

    # -------------------------
    # Checks
    # -------------------------
    cyc = pred2.groupby("ciclo_id", dropna=False).agg(
        proy=("tallos_proy", "max"),
        sum_ml1=("tallos_pred_ml1_dia_smooth", "sum"),
    ).reset_index()
    cyc["abs_diff"] = (cyc["proy"] - cyc["sum_ml1"]).abs()
    max_abs = float(cyc["abs_diff"].max()) if len(cyc) else float("nan")

    ss = pred2.groupby("ciclo_id", dropna=False)["share_final"].sum()
    smin, smax = (float(ss.min()), float(ss.max())) if len(ss) else (float("nan"), float("nan"))

    cap_pre = float(pred2["was_capped_pre"].mean()) if len(pred2) else float("nan")
    cap_post = float(pred2["was_capped_post"].mean()) if len(pred2) else float("nan")

    # -------------------------
    # Write output (compatible)
    # -------------------------
    keep = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon",
        "factor_curva_ml1", "factor_curva_ml1_raw", "ml1_version", "created_at",
        # auditoría (no rompe downstream)
        "_share_source", "cap_share",
        "share_pred_in", "share_smooth", "share_curva_ml1",
        "tallos_pred_ml1_dia_smooth",
        "factor_curva_ml1_raw_smooth",
        "was_capped_pre", "was_capped_post",
    ]
    keep = [c for c in keep if c in pred2.columns]

    out = pred2[keep].sort_values(["bloque_base", "variedad_canon", "fecha", "ciclo_id"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_PATH)

    print(f"OK -> {OUT_PATH} | rows={len(out):,}")
    print(f"[CHECK] mass balance vs tallos_proy | max abs diff: {max_abs:.12f}")
    print(f"[CHECK] share sum per ciclo | min={smin:.8f} max={smax:.8f}")
    print(f"[CHECK] capped rate pre={cap_pre:.4f} post={cap_post:.4f}")
    print(f"[CAP] global_cap(Q={CAP_Q}) = {global_cap:.6f}")
    print(f"[SMOOTH] win={SMOOTH_WIN} centered={SMOOTH_CENTER}")


if __name__ == "__main__":
    main()
