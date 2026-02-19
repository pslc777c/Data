from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"

IN_GRID = GOLD / "universe_harvest_grid_ml2.parquet"
IN_PRED = GOLD / "pred_tallos_grado_dia_ml1_full.parquet"
IN_REAL = SILVER / "fact_cosecha_real_grado_dia.parquet"
IN_CLIMA = SILVER / "dim_clima_bloque_dia.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_share_grado_ml2_v1.parquet"

EPS = 1e-6
CLIP_LOG_ERR = 1.2  # ratio ∈ [0.301..3.320]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_block_base(df: pd.DataFrame) -> pd.DataFrame:
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
        return df
    if "bloque_padre" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_padre"])
        return df
    if "bloque" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque"])
        return df
    raise KeyError("No encuentro columna de bloque: bloque_base / bloque_padre / bloque")


def main() -> None:
    grid = read_parquet(IN_GRID).copy()
    pred = read_parquet(IN_PRED).copy()
    real = read_parquet(IN_REAL).copy()
    clima = read_parquet(IN_CLIMA).copy()

    # ---- Canon grid ----
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    if "variedad_canon" in grid.columns:
        grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    # base features from grid (sin duplicar por grado)
    base_cols = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon",
        "area", "tipo_sp", "estado",
        "rel_pos_final", "day_in_harvest_final", "n_harvest_days_final",
        "harvest_start_final", "harvest_end_final",
        "dow", "month", "weekofyear",
    ]
    base_cols = [c for c in base_cols if c in grid.columns]
    base = grid[base_cols].drop_duplicates(["ciclo_id", "fecha"]).copy()

    # ---- Canon pred ML1 ----
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_str(pred["bloque_base"])
    pred["grado"] = _canon_str(pred["grado"])
    if "variedad_canon" in pred.columns:
        pred["variedad_canon"] = _canon_str(pred["variedad_canon"])

    # columnas necesarias
    need_pred = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
                 "tallos_pred_ml1_grado_dia", "share_grado_ml1"]
    for c in need_pred:
        if c not in pred.columns:
            raise KeyError(f"Falta columna en pred_tallos_grado_dia_ml1_full: {c}")

    pred_small = pred[need_pred].copy()
    pred_small["tallos_pred_ml1_grado_dia"] = pd.to_numeric(pred_small["tallos_pred_ml1_grado_dia"], errors="coerce").fillna(0.0)
    pred_small["share_grado_ml1"] = pd.to_numeric(pred_small["share_grado_ml1"], errors="coerce")

    # ---- Canon real ----
    real["fecha"] = _to_date(real["fecha"])
    real = _resolve_block_base(real)
    real["grado"] = _canon_str(real["grado"])
    if "variedad" in real.columns:
        real["variedad"] = _canon_str(real["variedad"])

    real["tallos_real"] = pd.to_numeric(real["tallos_real"], errors="coerce").fillna(0.0)
    real_g = real.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))
    tot = real_g.groupby(["fecha", "bloque_base"], as_index=False).agg(tallos_real_dia=("tallos_real", "sum"))
    real_g = real_g.merge(tot, on=["fecha", "bloque_base"], how="left")
    real_g["share_real"] = np.where(real_g["tallos_real_dia"] > 0, real_g["tallos_real"] / real_g["tallos_real_dia"], np.nan)

    # ---- Merge base (grid) con pred (por grado) ----
    df = base.merge(
        pred_small,
        on=["ciclo_id", "fecha", "bloque_base", "variedad_canon"],
        how="inner"
    )

    # ---- Merge share real por (fecha,bloque_base,grado) ----
    df = df.merge(real_g[["fecha", "bloque_base", "grado", "tallos_real", "tallos_real_dia", "share_real"]],
                  on=["fecha", "bloque_base", "grado"], how="left")

    # ---- Merge clima por (fecha,bloque_base) ----
    clima["fecha"] = _to_date(clima["fecha"])
    clima["bloque_base"] = _canon_str(clima["bloque_base"])
    clima_cols = [
        "fecha", "bloque_base",
        "gdc_dia",
        "rainfall_mm_dia", "en_lluvia_dia",
        "temp_avg_dia", "solar_energy_j_m2_dia",
        "wind_speed_avg_dia", "wind_run_dia",
    ]
    clima_cols = [c for c in clima_cols if c in clima.columns]
    df = df.merge(clima[clima_cols].copy(), on=["fecha", "bloque_base"], how="left")

    # ---- Target residual ----
    df["share_grado_ml1"] = pd.to_numeric(df["share_grado_ml1"], errors="coerce")
    df["share_real"] = pd.to_numeric(df["share_real"], errors="coerce")

    ratio = (df["share_real"] + EPS) / (df["share_grado_ml1"] + EPS)
    df["log_error_share"] = np.log(ratio).clip(-CLIP_LOG_ERR, CLIP_LOG_ERR)
    df["share_ratio"] = np.exp(df["log_error_share"])

    # ---- Limpieza ----
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(df):,} cycles={df['ciclo_id'].nunique():,} fecha_range=[{df['fecha'].min().date()}..{df['fecha'].max().date()}]")
    print("     pred_col_used=tallos_pred_ml1_grado_dia  share_col_used=share_grado_ml1")
    print("     target=log_error_share  clip=±1.2  ratio_range=[0.301..3.320]")


if __name__ == "__main__":
    main()
