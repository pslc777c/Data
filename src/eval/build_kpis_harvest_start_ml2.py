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

# (A) Backtest histórico (si existe)
IN_FINAL_BT = EVAL_DIR / "backtest_pred_harvest_start_final_ml2.parquet"

# (B) Producción (lo que hoy sí te está saliendo del pipeline)
# Ajusta el nombre si tu script de apply escribe otro parquet.
IN_FINAL_PROD = GOLD_DIR / "pred_harvest_start_final_ml2.parquet"

OUT_EVAL = EVAL_DIR / "ml2_harvest_start_eval.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _pick_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    # si ninguno existe, dispara error con mensaje claro
    raise FileNotFoundError(
        "No encuentro input para KPI harvest_start. Probé:\n"
        + "\n".join([f" - {str(p)}" for p in paths])
    )


def main() -> None:
    ciclo = read_parquet(IN_CICLO).copy()

    # ✅ FIX: usa backtest si existe; si no, usa prod (gold)
    in_final = _pick_existing(IN_FINAL_BT, IN_FINAL_PROD)
    pred = read_parquet(in_final).copy()

    ciclo.columns = [str(c).strip() for c in ciclo.columns]
    pred.columns = [str(c).strip() for c in pred.columns]

    # Normalización de fechas
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])

    # tolerante a nombres alternativos
    if "harvest_start_pred" not in pred.columns:
        # si tu baseline se llama distinto, agrega candidates aquí
        c = next((x for x in ["harvest_start_pred_ml1", "harvest_start_ml1", "harvest_start_pred"] if x in pred.columns), None)
        if c is None:
            raise KeyError("No encuentro columna baseline harvest_start_pred (ML1) en pred.")
        pred = pred.rename(columns={c: "harvest_start_pred"})

    if "harvest_start_final" not in pred.columns:
        c = next((x for x in ["harvest_start_final", "harvest_start_ml2_final", "harvest_start_ml2"] if x in pred.columns), None)
        if c is None:
            raise KeyError("No encuentro columna harvest_start_final (ML2) en pred.")
        pred = pred.rename(columns={c: "harvest_start_final"})

    pred["harvest_start_pred"] = _to_date(pred["harvest_start_pred"])
    pred["harvest_start_final"] = _to_date(pred["harvest_start_final"])

    # merge
    if "ciclo_id" not in ciclo.columns or "ciclo_id" not in pred.columns:
        raise KeyError("Falta ciclo_id en ciclo o pred para hacer merge.")

    df = ciclo.merge(pred, on="ciclo_id", how="inner")
    df = df.loc[
        df["fecha_inicio_cosecha"].notna()
        & df["harvest_start_pred"].notna()
        & df["harvest_start_final"].notna(),
        :
    ].copy()

    # errores
    df["err_ml1_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_pred"]).dt.days.astype(int)
    df["err_ml2_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_final"]).dt.days.astype(int)

    def _mae(x: pd.Series) -> float:
        return float(np.mean(np.abs(x))) if len(x) else float("nan")

    def _bias(x: pd.Series) -> float:
        return float(np.mean(x)) if len(x) else float("nan")

    mae_ml1 = _mae(df["err_ml1_days"])
    mae_ml2 = _mae(df["err_ml2_days"])

    out = pd.DataFrame([{
        "input_file": str(in_final),
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
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()