from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/gold/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"
IN_GRID_ML1 = GOLD_DIR / "universe_harvest_grid_ml1.parquet"
IN_CLIMA = SILVER_DIR / "dim_clima_bloque_dia.parquet"
IN_FACTOR_SOH = GOLD_DIR / "factors" / "factor_ml2_harvest_start.parquet"  # ML2 Inicio (prod)

OUT_DS = GOLD_DIR / "ml2_datasets" / "ds_harvest_horizon_ml2_v2.parquet"

# As-of sampling (V2)
ASOF_FREQ_DAYS = 7
MIN_DAYS_AFTER_SP = 14
MAX_ASOF_POINTS_PER_CICLO = 10

# Real duration sanity
MIN_REAL_DAYS = 1
MAX_REAL_DAYS = 120


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return (a / b).fillna(0.0)


def _build_cycle_header(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce universe_harvest_grid_ml1 (diario) a 1 fila por ciclo_id.
    """
    g = grid.copy()
    g["harvest_start_pred"] = _to_date(g["harvest_start_pred"])
    g["harvest_end_pred"] = _to_date(g["harvest_end_pred"])
    g["fecha_sp"] = _to_date(g["fecha_sp"])

    head = (
        g.groupby("ciclo_id", as_index=False)
        .agg(
            bloque_base=("bloque_base", "first"),
            variedad_canon=("variedad_canon", "first"),
            area=("area", "first"),
            tipo_sp=("tipo_sp", "first"),
            estado=("estado", "first"),
            tallos_proy=("tallos_proy", "first"),
            fecha_sp=("fecha_sp", "first"),
            harvest_start_pred=("harvest_start_pred", "min"),
            harvest_end_pred=("harvest_end_pred", "max"),
            n_harvest_days_pred=("n_harvest_days_pred", "max"),
            ml1_version=("ml1_version", "max"),
        )
    )
    head["days_sp_to_start_pred"] = (head["harvest_start_pred"] - head["fecha_sp"]).dt.days
    return head


def _make_asof_dates(fecha_sp: pd.Timestamp, inicio_real: pd.Timestamp) -> list[pd.Timestamp]:
    """
    V2: as_of cada ASOF_FREQ_DAYS desde (sp + MIN_DAYS_AFTER_SP) hasta (inicio_real - 1).
    Cap MAX_ASOF_POINTS_PER_CICLO y asegura incluir el último as_of.
    """
    if pd.isna(fecha_sp) or pd.isna(inicio_real):
        return []

    last = inicio_real - pd.Timedelta(days=1)
    start = fecha_sp + pd.Timedelta(days=MIN_DAYS_AFTER_SP)

    if last < start:
        return [max(fecha_sp, last)]

    dates = list(pd.date_range(start=start, end=last, freq=f"{ASOF_FREQ_DAYS}D"))
    if not dates or dates[-1] != last:
        dates.append(last)

    if len(dates) > MAX_ASOF_POINTS_PER_CICLO:
        dates = dates[-MAX_ASOF_POINTS_PER_CICLO:]

    return [pd.Timestamp(d).normalize() for d in dates]


def _features_asof(clima: pd.DataFrame, ciclo_chunk: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Features clima por ciclo_id usando rango [fecha_sp, as_of_date].
    Si un ciclo no tiene clima en el rango, queda con ceros (no se pierde).
    """
    base = ciclo_chunk[["ciclo_id", "bloque_base", "fecha_sp"]].drop_duplicates().copy()
    base["bloque_base"] = _canon_str(base["bloque_base"])
    base["fecha_sp"] = _to_date(base["fecha_sp"])

    cl = clima.copy()
    cl["fecha"] = _to_date(cl["fecha"])
    cl["bloque_base"] = _canon_str(cl["bloque_base"])
    cl = cl.loc[cl["fecha"] <= as_of_date, :].copy()

    m = base.merge(cl, on="bloque_base", how="left")
    m = m.loc[(m["fecha"] >= m["fecha_sp"]) & (m["fecha"] <= as_of_date), :].copy()

    # no hay datos -> ceros
    if m.empty:
        out = base[["ciclo_id"]].copy()
        for c in [
            "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
            "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
            "solar_cum_sp", "solar_7d",
            "temp_avg_7d",
        ]:
            out[c] = 0.0
        return out

    # usar gdc_dia acumulado (ignorar gdc_base)
    m["gdc_dia"] = pd.to_numeric(m["gdc_dia"], errors="coerce").fillna(0.0)
    m["rainfall_mm_dia"] = pd.to_numeric(m["rainfall_mm_dia"], errors="coerce").fillna(0.0)
    m["solar_energy_j_m2_dia"] = pd.to_numeric(m["solar_energy_j_m2_dia"], errors="coerce").fillna(0.0)
    m["temp_avg_dia"] = pd.to_numeric(m["temp_avg_dia"], errors="coerce")
    m["en_lluvia_dia"] = pd.to_numeric(m["en_lluvia_dia"], errors="coerce").fillna(0.0)

    m = m.sort_values(["ciclo_id", "fecha"])

    def _roll_sum(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(window=w, min_periods=1).sum()

    def _roll_mean(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(window=w, min_periods=1).mean()

    m["gdc_7d"] = m.groupby("ciclo_id")["gdc_dia"].transform(lambda s: _roll_sum(s, 7))
    m["gdc_14d"] = m.groupby("ciclo_id")["gdc_dia"].transform(lambda s: _roll_sum(s, 14))
    m["rain_7d"] = m.groupby("ciclo_id")["rainfall_mm_dia"].transform(lambda s: _roll_sum(s, 7))
    m["solar_7d"] = m.groupby("ciclo_id")["solar_energy_j_m2_dia"].transform(lambda s: _roll_sum(s, 7))
    m["enlluvia_days_7d"] = m.groupby("ciclo_id")["en_lluvia_dia"].transform(lambda s: _roll_sum(s, 7))
    m["temp_avg_7d"] = m.groupby("ciclo_id")["temp_avg_dia"].transform(lambda s: _roll_mean(s, 7))

    m["gdc_cum_sp"] = m.groupby("ciclo_id")["gdc_dia"].cumsum()
    m["rain_cum_sp"] = m.groupby("ciclo_id")["rainfall_mm_dia"].cumsum()
    m["solar_cum_sp"] = m.groupby("ciclo_id")["solar_energy_j_m2_dia"].cumsum()

    last = m.groupby("ciclo_id", as_index=False).tail(1).copy()
    last["days_from_sp"] = (as_of_date - last["fecha_sp"]).dt.days
    last["days_from_sp"] = pd.to_numeric(last["days_from_sp"], errors="coerce").fillna(0).clip(lower=0)
    last["gdc_per_day"] = _safe_div(last["gdc_cum_sp"], last["days_from_sp"].replace(0, np.nan))

    feat_cols = [
        "ciclo_id",
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
        "temp_avg_7d",
    ]
    last = last[feat_cols].copy()

    out = base[["ciclo_id"]].merge(last, on="ciclo_id", how="left")

    for c in [
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["temp_avg_7d"] = pd.to_numeric(out["temp_avg_7d"], errors="coerce")
    med = out["temp_avg_7d"].median()
    if pd.isna(med):
        med = 0.0
    out["temp_avg_7d"] = out["temp_avg_7d"].fillna(med)

    return out


def main() -> None:
    ciclo = read_parquet(IN_CICLO).copy()
    grid = read_parquet(IN_GRID_ML1).copy()
    clima = read_parquet(IN_CLIMA).copy()

    # ML2 inicio (factor) es opcional: si existe lo usamos; si no, fallback a 0
    factor_soh = None
    if IN_FACTOR_SOH.exists():
        factor_soh = read_parquet(IN_FACTOR_SOH).copy()

    # canon y fechas
    ciclo["bloque_base"] = _canon_str(ciclo["bloque_base"])
    ciclo["fecha_sp"] = _to_date(ciclo["fecha_sp"])
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])
    ciclo["fecha_fin_cosecha"] = _to_date(ciclo["fecha_fin_cosecha"])
    ciclo["estado"] = _canon_str(ciclo["estado"])

    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    head = _build_cycle_header(grid)

    # Base ciclo
    df0 = ciclo.merge(head, on="ciclo_id", how="inner", suffixes=("", "_ml1"))

    # Requiere reales válidos de inicio/fin + pred n_harvest_days
    df0 = df0.loc[
        df0["fecha_inicio_cosecha"].notna()
        & df0["fecha_fin_cosecha"].notna()
        & df0["n_harvest_days_pred"].notna(),
        :
    ].copy()

    # Real duration
    df0["n_harvest_days_real"] = (df0["fecha_fin_cosecha"] - df0["fecha_inicio_cosecha"]).dt.days + 1
    df0["n_harvest_days_real"] = pd.to_numeric(df0["n_harvest_days_real"], errors="coerce")

    df0 = df0.loc[
        df0["n_harvest_days_real"].between(MIN_REAL_DAYS, MAX_REAL_DAYS, inclusive="both"),
        :
    ].copy()

    df0["n_harvest_days_pred"] = pd.to_numeric(df0["n_harvest_days_pred"], errors="coerce")
    df0 = df0.loc[df0["n_harvest_days_pred"].notna(), :].copy()

    # Target
    df0["error_horizon_days"] = (df0["n_harvest_days_real"] - df0["n_harvest_days_pred"]).astype(int)

    # Panel as_of por ciclo (V2)
    rows = []
    for r in df0.itertuples(index=False):
        for a in _make_asof_dates(r.fecha_sp, r.fecha_inicio_cosecha):
            rows.append((r.ciclo_id, a))
    asof_df = pd.DataFrame(rows, columns=["ciclo_id", "as_of_date"])

    df = df0.merge(asof_df, on="ciclo_id", how="inner")
    df = df.loc[df["as_of_date"] < df["fecha_inicio_cosecha"], :].copy()

    # Calendario
    df["dow"] = df["as_of_date"].dt.dayofweek
    df["month"] = df["as_of_date"].dt.month
    df["weekofyear"] = df["as_of_date"].dt.isocalendar().week.astype(int)
    df["days_from_sp"] = (df["as_of_date"] - df["fecha_sp"]).dt.days

    # Agregar pred_error_start_days (ML2 Inicio) si existe
    df["pred_error_start_days"] = 0.0
    if factor_soh is not None and len(factor_soh):
        factor_soh = factor_soh.copy()
        # si factor tiene as_of_date o no: en prod normalmente es único; aquí usamos el valor por ciclo
        factor_soh["pred_error_start_days"] = pd.to_numeric(factor_soh["pred_error_start_days"], errors="coerce")
        soh = factor_soh[["ciclo_id", "pred_error_start_days"]].drop_duplicates("ciclo_id")
        df = df.drop(columns=["pred_error_start_days"]).merge(soh, on="ciclo_id", how="left")
        df["pred_error_start_days"] = pd.to_numeric(df["pred_error_start_days"], errors="coerce").fillna(0.0)

    # Features clima por as_of_date (batch)
    feats = []
    for as_of_date, chunk in df.groupby("as_of_date"):
        feats.append(_features_asof(clima, chunk[["ciclo_id", "bloque_base", "fecha_sp"]], pd.Timestamp(as_of_date)))
    feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(columns=["ciclo_id"])

    df = df.merge(feat, on="ciclo_id", how="left")

    # Final imputaciones numéricas
    for c in [
        "days_sp_to_start_pred",
        "n_harvest_days_pred",
        "tallos_proy",
        "days_from_sp",
        "pred_error_start_days",
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
        "temp_avg_7d",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(df):,} cycles={df['ciclo_id'].nunique():,} as_of_dates={df['as_of_date'].nunique():,}")


if __name__ == "__main__":
    main()
