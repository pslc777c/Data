from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

IN_GRID_ML2 = GOLD_DIR / "universe_harvest_grid_ml2.parquet"
IN_REAL = SILVER_DIR / "fact_cosecha_real_grado_dia.parquet"
IN_PRED_ML1 = GOLD_DIR / "pred_tallos_grado_dia_ml1_full.parquet"
IN_CLIMA = SILVER_DIR / "dim_clima_bloque_dia.parquet"

OUT_DS = GOLD_DIR / "ml2_datasets" / "ds_tallos_curve_ml2_v2.parquet"

EPS = 1.0
CLIP_LOG_ERR = 1.5  # exp(±1.5) ≈ factor [0.22 .. 4.48]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    grid = read_parquet(IN_GRID_ML2).copy()
    real = read_parquet(IN_REAL).copy()
    pred = read_parquet(IN_PRED_ML1).copy()
    clima = read_parquet(IN_CLIMA).copy()

    # --- GRID ML2 ---
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])
    if "estado" in grid.columns:
        grid["estado"] = _canon_str(grid["estado"])

    base_cols = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "area", "tipo_sp", "estado",
        "rel_pos_final", "day_in_harvest_final", "n_harvest_days_final",
        "harvest_start_final", "harvest_end_final",
        "ml1_version",
    ]
    base_cols = [c for c in base_cols if c in grid.columns]
    grid0 = grid[base_cols].drop_duplicates(["ciclo_id", "fecha"]).copy()

    # --- REAL tallos (sum grades) ---
    real["fecha"] = _to_date(real["fecha"])

    # Resolver bloque_base desde columnas posibles
    if "bloque_base" in real.columns:
        real["bloque_base"] = _canon_str(real["bloque_base"])
    elif "bloque_padre" in real.columns:
        real["bloque_base"] = _canon_str(real["bloque_padre"])
    elif "bloque" in real.columns:
        real["bloque_base"] = _canon_str(real["bloque"])
    else:
        raise KeyError(
            "No encuentro columna de bloque en fact_cosecha_real_grado_dia. "
            "Espero una de: bloque_base / bloque_padre / bloque"
        )

    if "variedad" in real.columns:
        real["variedad"] = _canon_str(real["variedad"])

    # real: (fecha, bloque_base, variedad, grado, tallos_real)
    real_agg = (
        real.groupby(["fecha", "bloque_base"], as_index=False)
            .agg(tallos_real_dia=("tallos_real", "sum"))
    )

    # --- PRED ML1 tallos (sum grades) ---
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_str(pred["bloque_base"])
    if "variedad_canon" in pred.columns:
        pred["variedad_canon"] = _canon_str(pred["variedad_canon"])

    # pred expected: (fecha, bloque_base, variedad_canon, grado, tallos_pred_*)
    # Detect the predicted column (fallback order)
    # En este proyecto, la columna canónica ML1 a nivel grado-día es:
    # tallos_pred_ml1_grado_dia (sumar por grado -> total día)
    if "tallos_pred_ml1_grado_dia" in pred.columns:
        pred_col = "tallos_pred_ml1_grado_dia"
    elif "tallos_pred_ml1_dia" in pred.columns:
        # fallback (menos preferido)
        pred_col = "tallos_pred_ml1_dia"
    else:
        raise KeyError(
            "No encuentro columna de tallos predicha en pred_tallos_grado_dia_ml1_full.parquet. "
            "Espero: tallos_pred_ml1_grado_dia (preferida) o tallos_pred_ml1_dia (fallback)."
        )


    pred_agg = (
        pred.groupby(["fecha", "bloque_base", "variedad_canon"], as_index=False)
            .agg(tallos_pred_ml1_dia=(pred_col, "sum"))
    )

    # --- CLIMA diario (ya por bloque_base + fecha) ---
    clima["fecha"] = _to_date(clima["fecha"])
    clima["bloque_base"] = _canon_str(clima["bloque_base"])

    clima_cols = [
        "fecha", "bloque_base",
        "gdc_dia", "gdc_base",
        "rainfall_mm_dia", "horas_lluvia", "en_lluvia_dia",
        "temp_avg_dia", "solar_energy_j_m2_dia",
        "wind_speed_avg_dia", "wind_run_dia",
    ]
    clima_cols = [c for c in clima_cols if c in clima.columns]
    clima0 = clima[clima_cols].copy()


    # --- JOIN ---
    df = (
        grid0
        .merge(pred_agg, on=["fecha", "bloque_base", "variedad_canon"], how="left")
        .merge(real_agg, on=["fecha", "bloque_base"], how="left")
        .merge(clima0, on=["fecha", "bloque_base"], how="left")
    )

    # pyarrow no permite columnas duplicadas (ej: gdc_base repetida)
    df = df.loc[:, ~df.columns.duplicated()].copy()


    df["tallos_pred_ml1_dia"] = pd.to_numeric(df["tallos_pred_ml1_dia"], errors="coerce").fillna(0.0)
    df["tallos_real_dia"] = pd.to_numeric(df["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Target: log error ratio
    ratio = (df["tallos_real_dia"] + EPS) / (df["tallos_pred_ml1_dia"] + EPS)
    df["log_error"] = np.log(ratio).clip(-CLIP_LOG_ERR, CLIP_LOG_ERR)

    # convenience columns
    df["error_ratio"] = np.exp(df["log_error"])

    # Filter: para entrenamiento, necesitamos pred > 0 o real > 0 (para no meter basura total)
    df = df.loc[(df["tallos_pred_ml1_dia"] > 0) | (df["tallos_real_dia"] > 0), :].copy()

    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, OUT_DS)

    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(df):,} cycles={df['ciclo_id'].nunique():,} "
          f"fecha_range=[{df['fecha'].min().date()}..{df['fecha'].max().date()}]")
    print(f"     pred_col_used={pred_col}")


if __name__ == "__main__":
    main()
