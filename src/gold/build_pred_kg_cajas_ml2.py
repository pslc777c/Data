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
EVAL = DATA / "eval" / "ml2"

# ---- INPUTS ML2 (backtest) ----
IN_TALLOS_ML2_BT = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"
IN_PESO_ML2_BT = EVAL / "backtest_pred_peso_tallo_grado_dia_ml2_final.parquet"

# ---- INPUTS ML2 (prod - si luego los materializas en gold) ----
IN_TALLOS_ML2_PROD = GOLD / "pred_tallos_grado_dia_ml2_full.parquet"          # opcional futuro
IN_PESO_ML2_PROD = GOLD / "pred_peso_tallo_grado_dia_ml2_full.parquet"       # opcional futuro

# ---- INPUTS ML1 (para “heredar” conversión a cajas) ----
IN_KG_ML1 = GOLD / "pred_kg_grado_dia_ml1_full.parquet"
IN_CAJAS_ML1 = GOLD / "pred_cajas_grado_dia_ml1_full.parquet"

# ---- OUTPUTS ML2 ----
OUT_KG_GRADO = GOLD / "pred_kg_grado_dia_ml2_full.parquet"
OUT_KG_DIA = GOLD / "pred_kg_dia_ml2_full.parquet"
OUT_CAJAS_GRADO = GOLD / "pred_cajas_grado_dia_ml2_full.parquet"
OUT_CAJAS_DIA = GOLD / "pred_cajas_dia_ml2_full.parquet"


KEYS = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"No encuentro ninguna de estas columnas: {candidates}. Tengo: {list(df.columns)}")


def main(mode: str = "backtest") -> None:
    mode = (mode or "backtest").strip().lower()
    if mode not in ("backtest", "prod"):
        raise ValueError("--mode debe ser backtest o prod")

    # ---- Load ML2 ----
    if mode == "backtest":
        tallos = read_parquet(IN_TALLOS_ML2_BT).copy()
        peso = read_parquet(IN_PESO_ML2_BT).copy()
    else:
        tallos = read_parquet(IN_TALLOS_ML2_PROD).copy()
        peso = read_parquet(IN_PESO_ML2_PROD).copy()

    # Canon keys
    for df in (tallos, peso):
        df["fecha"] = _to_date(df["fecha"])
        df["bloque_base"] = _canon_str(df["bloque_base"])
        df["variedad_canon"] = _canon_str(df["variedad_canon"])
        df["grado"] = _canon_str(df["grado"])

    # ---- resolve ML2 columns ----
    col_tallos_ml2 = _pick_col(
        tallos,
        ["tallos_final_grado_dia", "tallos_ml2_grado_dia", "tallos_pred_ml2_grado_dia"]
    )
    # peso backtest normalmente trae algo tipo peso_tallo_final_g
    col_peso_ml2 = _pick_col(
        peso,
        ["peso_tallo_final_g", "peso_tallo_ml2_g", "peso_tallo_pred_ml2_g"]
    )

    tallos_use = tallos[KEYS + [col_tallos_ml2]].copy().rename(columns={col_tallos_ml2: "tallos_ml2_grado_dia"})
    peso_use = peso[KEYS + [col_peso_ml2]].copy().rename(columns={col_peso_ml2: "peso_tallo_ml2_g"})

    # ---- Merge ML2 tallos + peso ----
    df = tallos_use.merge(peso_use, on=KEYS, how="left")

    df["tallos_ml2_grado_dia"] = pd.to_numeric(df["tallos_ml2_grado_dia"], errors="coerce").fillna(0.0)
    df["peso_tallo_ml2_g"] = pd.to_numeric(df["peso_tallo_ml2_g"], errors="coerce")

    # KG ML2
    df["kg_ml2_grado_dia"] = df["tallos_ml2_grado_dia"] * df["peso_tallo_ml2_g"] / 1000.0
    df.loc[df["kg_ml2_grado_dia"] < 0, "kg_ml2_grado_dia"] = 0.0

    # ---- Load ML1 kg/cajas for conversion ----
    kg1 = read_parquet(IN_KG_ML1).copy()
    cajas1 = read_parquet(IN_CAJAS_ML1).copy()

    for df1 in (kg1, cajas1):
        df1["fecha"] = _to_date(df1["fecha"])
        df1["bloque_base"] = _canon_str(df1["bloque_base"])
        df1["variedad_canon"] = _canon_str(df1["variedad_canon"])
        df1["grado"] = _canon_str(df1["grado"])

    col_kg_ml1 = _pick_col(
        kg1,
        ["kg_pred_ml1_grado_dia", "kg_ml1_grado_dia", "kg_pred_grado_dia", "kg_pred_ml1"]
    )
    col_cajas_ml1 = _pick_col(
        cajas1,
        ["cajas_pred_ml1_grado_dia", "cajas_ml1_grado_dia", "cajas_pred_grado_dia", "cajas_pred_ml1"]
    )

    kg1_use = kg1[KEYS + [col_kg_ml1]].copy().rename(columns={col_kg_ml1: "kg_ml1_grado_dia"})
    cajas1_use = cajas1[KEYS + [col_cajas_ml1]].copy().rename(columns={col_cajas_ml1: "cajas_ml1_grado_dia"})

    df = df.merge(kg1_use, on=KEYS, how="left").merge(cajas1_use, on=KEYS, how="left")

    df["kg_ml1_grado_dia"] = pd.to_numeric(df["kg_ml1_grado_dia"], errors="coerce").fillna(0.0)
    df["cajas_ml1_grado_dia"] = pd.to_numeric(df["cajas_ml1_grado_dia"], errors="coerce").fillna(0.0)

    # Cajas ML2 heredando conversión ML1
    eps = 1e-9
    ratio = np.where(df["kg_ml1_grado_dia"] > 0, df["kg_ml2_grado_dia"] / (df["kg_ml1_grado_dia"] + eps), 0.0)
    df["cajas_ml2_grado_dia"] = df["cajas_ml1_grado_dia"] * ratio
    df.loc[df["cajas_ml2_grado_dia"] < 0, "cajas_ml2_grado_dia"] = 0.0

    # ---- Build outputs (grado/día) ----
    out_kg_grado = df[KEYS + ["kg_ml2_grado_dia"]].copy()
    out_cajas_grado = df[KEYS + ["cajas_ml2_grado_dia"]].copy()

    created_at = pd.Timestamp(datetime.now()).normalize()
    out_kg_grado["created_at"] = created_at
    out_cajas_grado["created_at"] = created_at

    # ---- Aggregate to día (sum grados) ----
    keys_day = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    out_kg_day = out_kg_grado.groupby(keys_day, as_index=False).agg(kg_ml2_dia=("kg_ml2_grado_dia", "sum"))
    out_cajas_day = out_cajas_grado.groupby(keys_day, as_index=False).agg(cajas_ml2_dia=("cajas_ml2_grado_dia", "sum"))
    out_kg_day["created_at"] = created_at
    out_cajas_day["created_at"] = created_at

    GOLD.mkdir(parents=True, exist_ok=True)
    write_parquet(out_kg_grado, OUT_KG_GRADO)
    write_parquet(out_kg_day, OUT_KG_DIA)
    write_parquet(out_cajas_grado, OUT_CAJAS_GRADO)
    write_parquet(out_cajas_day, OUT_CAJAS_DIA)

    print(f"[OK] Wrote: {OUT_KG_GRADO} rows={len(out_kg_grado)}")
    print(f"[OK] Wrote: {OUT_KG_DIA} rows={len(out_kg_day)}")
    print(f"[OK] Wrote: {OUT_CAJAS_GRADO} rows={len(out_cajas_grado)}")
    print(f"[OK] Wrote: {OUT_CAJAS_DIA} rows={len(out_cajas_day)}")
    print(f"     mode={mode}  tallos_col={col_tallos_ml2}  peso_col={col_peso_ml2}")
    print(f"     kg_ml1_col={col_kg_ml1}  cajas_ml1_col={col_cajas_ml1}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="backtest", choices=["backtest", "prod"])
    args = p.parse_args()
    main(mode=args.mode)
