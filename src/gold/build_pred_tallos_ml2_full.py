from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
EVAL = DATA / "eval" / "ml2"

# Backtest source (lo que ya tienes)
IN_TALLOS_ML2_BT = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"

# Output GOLD (vista final)
OUT_TALLOS_GRADO = GOLD / "pred_tallos_grado_dia_ml2_full.parquet"
OUT_TALLOS_DIA = GOLD / "pred_tallos_dia_ml2_full.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    df = read_parquet(IN_TALLOS_ML2_BT).copy()

    # Canon
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_str(df["bloque_base"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["grado"] = _canon_str(df["grado"])

    # Resolver columna de tallos final
    if "tallos_final_grado_dia" not in df.columns:
        raise KeyError(f"No encuentro tallos_final_grado_dia en {IN_TALLOS_ML2_BT}. Columnas: {list(df.columns)}")

    out = df[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "tallos_final_grado_dia"]].copy()
    out = out.rename(columns={"tallos_final_grado_dia": "tallos_pred_ml2_grado_dia"})
    out["created_at"] = pd.Timestamp(datetime.now()).normalize()

    # Dia total
    keys_day = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    out_day = out.groupby(keys_day, as_index=False).agg(tallos_pred_ml2_dia=("tallos_pred_ml2_grado_dia", "sum"))
    out_day["created_at"] = out["created_at"].iloc[0]

    GOLD.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_TALLOS_GRADO)
    write_parquet(out_day, OUT_TALLOS_DIA)

    print(f"[OK] Wrote: {OUT_TALLOS_GRADO} rows={len(out)}")
    print(f"[OK] Wrote: {OUT_TALLOS_DIA} rows={len(out_day)}")


if __name__ == "__main__":
    main()
