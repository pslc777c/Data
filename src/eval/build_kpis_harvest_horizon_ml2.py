from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"

IN_FINAL_PROD = GOLD_DIR / "pred_harvest_horizon_final_ml2.parquet"
IN_FINAL_BT = EVAL_DIR / "backtest_pred_harvest_horizon_final_ml2.parquet"

OUT_PROD = EVAL_DIR / "ml2_harvest_horizon_eval_prod.parquet"
OUT_BT = EVAL_DIR / "ml2_harvest_horizon_eval_backtest.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else float("nan")


def _bias(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x)) if len(x) else float("nan")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="backtest")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ciclo = read_parquet(IN_CICLO).copy()
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])
    ciclo["fecha_fin_cosecha"] = _to_date(ciclo["fecha_fin_cosecha"])

    if args.mode == "prod":
        pred = read_parquet(IN_FINAL_PROD).copy()
        out_path = OUT_PROD
    else:
        pred = read_parquet(IN_FINAL_BT).copy()
        out_path = OUT_BT

    # Canon dates
    pred["harvest_end_pred"] = _to_date(pred["harvest_end_pred"])
    pred["harvest_end_final"] = _to_date(pred["harvest_end_final"])

    pred["harvest_start_pred"] = _to_date(pred["harvest_start_pred"])
    pred["harvest_start_final"] = _to_date(pred["harvest_start_final"])

    # Merge
    df = ciclo.merge(pred, on="ciclo_id", how="inner")

    # Real duration
    df["n_harvest_days_real"] = (df["fecha_fin_cosecha"] - df["fecha_inicio_cosecha"]).dt.days + 1

    # Filter to valid rows
    df = df.loc[
        df["n_harvest_days_real"].notna()
        & df["n_harvest_days_pred"].notna()
        & df["n_harvest_days_final"].notna()
        & df["harvest_end_pred"].notna()
        & df["harvest_end_final"].notna(),
        :
    ].copy()

    df["n_harvest_days_real"] = pd.to_numeric(df["n_harvest_days_real"], errors="coerce")
    df["n_harvest_days_pred"] = pd.to_numeric(df["n_harvest_days_pred"], errors="coerce")
    df["n_harvest_days_final"] = pd.to_numeric(df["n_harvest_days_final"], errors="coerce")

    # Errors on duration
    df["err_n_days_ml1"] = df["n_harvest_days_real"] - df["n_harvest_days_pred"]
    df["err_n_days_ml2"] = df["n_harvest_days_real"] - df["n_harvest_days_final"]

    # Errors on end-date (in days)
    df["err_end_ml1_days"] = (df["fecha_fin_cosecha"] - df["harvest_end_pred"]).dt.days
    df["err_end_ml2_days"] = (df["fecha_fin_cosecha"] - df["harvest_end_final"]).dt.days

    # KPIs
    mae_ml1_n = _mae(df["err_n_days_ml1"])
    mae_ml2_n = _mae(df["err_n_days_ml2"])

    mae_ml1_end = _mae(df["err_end_ml1_days"])
    mae_ml2_end = _mae(df["err_end_ml2_days"])

    out = pd.DataFrame([{
        "mode": args.mode,
        "n": int(len(df)),
        "mae_n_days_ml1": mae_ml1_n,
        "mae_n_days_ml2": mae_ml2_n,
        "bias_n_days_ml1": _bias(df["err_n_days_ml1"]),
        "bias_n_days_ml2": _bias(df["err_n_days_ml2"]),
        "improvement_abs_n_days": (mae_ml1_n - mae_ml2_n) if (pd.notna(mae_ml1_n) and pd.notna(mae_ml2_n)) else np.nan,
        "mae_end_days_ml1": mae_ml1_end,
        "mae_end_days_ml2": mae_ml2_end,
        "bias_end_days_ml1": _bias(df["err_end_ml1_days"]),
        "bias_end_days_ml2": _bias(df["err_end_ml2_days"]),
        "improvement_abs_end_days": (mae_ml1_end - mae_ml2_end) if (pd.notna(mae_ml1_end) and pd.notna(mae_ml2_end)) else np.nan,
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(out, out_path)

    print(f"[OK] Wrote eval: {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
