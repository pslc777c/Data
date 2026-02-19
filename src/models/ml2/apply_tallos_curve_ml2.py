from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models" / "ml2"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

# Inputs
IN_GRID_ML2 = GOLD_DIR / "universe_harvest_grid_ml2.parquet"
IN_PRED_ML1 = GOLD_DIR / "pred_tallos_grado_dia_ml1_full.parquet"
IN_REAL = SILVER_DIR / "fact_cosecha_real_grado_dia.parquet"
IN_CLIMA = SILVER_DIR / "dim_clima_bloque_dia.parquet"

# Outputs (PROD)
OUT_FACTOR_PROD = GOLD_DIR / "factors" / "factor_ml2_tallos_curve_dia.parquet"
OUT_FINAL_PROD = GOLD_DIR / "pred_tallos_dia_ml2_final.parquet"

# Outputs (BACKTEST)
OUT_FACTOR_BT = EVAL_DIR / "backtest_factor_ml2_tallos_curve_dia.parquet"
OUT_FINAL_BT = EVAL_DIR / "backtest_pred_tallos_dia_ml2_final.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _load_latest_model() -> tuple[object, dict]:
    metas = sorted(MODELS_DIR.glob("tallos_curve_ml2_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No ML2 meta found in {MODELS_DIR}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = MODELS_DIR / f"tallos_curve_ml2_{meta['run_id']}.pkl"
    import joblib
    model = joblib.load(model_path)
    return model, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="backtest")
    return p.parse_args()


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
    args = _parse_args()
    model, meta = _load_latest_model()

    grid = read_parquet(IN_GRID_ML2).copy()
    pred = read_parquet(IN_PRED_ML1).copy()
    clima = read_parquet(IN_CLIMA).copy()

    # --- GRID ---
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    if "variedad_canon" in grid.columns:
        grid["variedad_canon"] = _canon_str(grid["variedad_canon"])
    if "tipo_sp" in grid.columns:
        grid["tipo_sp"] = _canon_str(grid["tipo_sp"])
    if "estado" in grid.columns:
        grid["estado"] = _canon_str(grid["estado"])

    base_cols = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "area", "tipo_sp", "estado",
        "rel_pos_final", "day_in_harvest_final", "n_harvest_days_final",
        "harvest_start_final", "harvest_end_final",
        "dow", "month", "weekofyear",
        "ml1_version",
    ]
    base_cols = [c for c in base_cols if c in grid.columns]
    df = grid[base_cols].drop_duplicates(["ciclo_id", "fecha"]).copy()

    # --- PRED ML1 tallos día (sum grades) ---
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_str(pred["bloque_base"])
    pred["variedad_canon"] = _canon_str(pred["variedad_canon"])

    # Canon predicted col
    if "tallos_pred_ml1_grado_dia" in pred.columns:
        pred_col = "tallos_pred_ml1_grado_dia"
    elif "tallos_pred_ml1_dia" in pred.columns:
        pred_col = "tallos_pred_ml1_dia"
    else:
        raise KeyError("No encuentro tallos_pred_ml1_grado_dia ni tallos_pred_ml1_dia en pred ML1.")

    pred_agg = (
        pred.groupby(["fecha", "bloque_base", "variedad_canon"], as_index=False)
            .agg(tallos_pred_ml1_dia=(pred_col, "sum"))
    )
    df = df.merge(pred_agg, on=["fecha", "bloque_base", "variedad_canon"], how="left")
    df["tallos_pred_ml1_dia"] = pd.to_numeric(df["tallos_pred_ml1_dia"], errors="coerce").fillna(0.0)

    # --- REAL (solo en backtest; para KPIs posteriores) ---
    if args.mode == "backtest":
        real = read_parquet(IN_REAL).copy()
        real["fecha"] = _to_date(real["fecha"])
        real = _resolve_block_base(real)
        real_agg = (
            real.groupby(["fecha", "bloque_base"], as_index=False)
                .agg(tallos_real_dia=("tallos_real", "sum"))
        )
        df = df.merge(real_agg, on=["fecha", "bloque_base"], how="left")
        df["tallos_real_dia"] = pd.to_numeric(df["tallos_real_dia"], errors="coerce").fillna(0.0)
    else:
        df["tallos_real_dia"] = 0.0  # placeholder

    # --- CLIMA diario ---
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
    df = df.merge(clima0, on=["fecha", "bloque_base"], how="left")

    # Evitar duplicadas por seguridad pyarrow
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # --- Features cleaning as meta ---
    for c in meta.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    for c in meta.get("cat_cols", []):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    feature_cols = meta["num_cols"] + meta["cat_cols"]
    X = df[feature_cols].copy()

    pred_log = model.predict(X).astype(float)

    lo, hi = meta["guardrails"]["clip_log_error"]
    pred_log = np.clip(pred_log, lo, hi)

    df["pred_log_error"] = pred_log
    df["pred_ratio"] = np.exp(df["pred_log_error"])

    # Final prediction
    df["tallos_final_ml2_dia"] = df["tallos_pred_ml1_dia"] * df["pred_ratio"]

    # Versioning
    df["ml2_run_id"] = meta["run_id"]
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    cols_factor = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon",
        "tallos_pred_ml1_dia", "tallos_real_dia",
        "pred_log_error", "pred_ratio", "tallos_final_ml2_dia",
        "ml2_run_id", "created_at",
    ]
    if "ml1_version" in df.columns:
        cols_factor.insert(-2, "ml1_version")

    out_factor = df[cols_factor].copy()

    cols_final = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon",
        "tallos_pred_ml1_dia", "tallos_final_ml2_dia",
        "ml2_run_id", "created_at",
    ]
    if "ml1_version" in df.columns:
        cols_final.insert(-2, "ml1_version")

    out_final = df[cols_final].copy()

    if "ml1_version" not in df.columns:
        print("[WARN] ml1_version not found in universe grid; outputs will omit ml1_version.")

    if args.mode == "prod":
        OUT_FACTOR_PROD.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_PROD)
        write_parquet(out_final, OUT_FINAL_PROD)
        print(f"[OK] PROD factor: {OUT_FACTOR_PROD} rows={len(out_factor):,}")
        print(f"[OK] PROD final : {OUT_FINAL_PROD} rows={len(out_final):,}")
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_BT)
        write_parquet(out_final, OUT_FINAL_BT)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR_BT} rows={len(out_factor):,}")
        print(f"[OK] BACKTEST final : {OUT_FINAL_BT} rows={len(out_final):,}")


if __name__ == "__main__":
    main()
