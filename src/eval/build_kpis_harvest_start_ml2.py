from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/eval/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"
IN_FINAL = EVAL_DIR / "backtest_pred_harvest_start_final_ml2.parquet"

OUT_EVAL = EVAL_DIR / "ml2_harvest_start_eval.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def main() -> None:
    ciclo = read_parquet(IN_CICLO).copy()
    pred = read_parquet(IN_FINAL).copy()

    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])
    pred["harvest_start_pred"] = _to_date(pred["harvest_start_pred"])
    pred["harvest_start_final"] = _to_date(pred["harvest_start_final"])

    df = ciclo.merge(pred, on="ciclo_id", how="inner")
    df = df.loc[
        df["fecha_inicio_cosecha"].notna()
        & df["harvest_start_pred"].notna()
        & df["harvest_start_final"].notna(),
        :
    ].copy()

    df["err_ml1_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_pred"]).dt.days.astype(int)
    df["err_ml2_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_final"]).dt.days.astype(int)

    def _mae(x: pd.Series) -> float:
        return float(np.mean(np.abs(x))) if len(x) else float("nan")

    def _bias(x: pd.Series) -> float:
        return float(np.mean(x)) if len(x) else float("nan")

    mae_ml1 = _mae(df["err_ml1_days"])
    mae_ml2 = _mae(df["err_ml2_days"])

    out = pd.DataFrame([{
        "n": int(len(df)),
        "mae_ml1_days": mae_ml1,
        "mae_ml2_days": mae_ml2,
        "bias_ml1_days": _bias(df["err_ml1_days"]),
        "bias_ml2_days": _bias(df["err_ml2_days"]),
        "improvement_abs_days": (mae_ml1 - mae_ml2) if (pd.notna(mae_ml1) and pd.notna(mae_ml2)) else float("nan"),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_EVAL)
    print(f"[OK] Wrote eval: {OUT_EVAL}")


if __name__ == "__main__":
    main()
