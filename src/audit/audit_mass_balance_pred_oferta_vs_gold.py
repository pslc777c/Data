from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet


PRED_OFER_DIA = Path("data/preds/pred_oferta_dia.parquet")
GOLD_TALLOS_GR = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    if not PRED_OFER_DIA.exists():
        raise FileNotFoundError(PRED_OFER_DIA)
    if not GOLD_TALLOS_GR.exists():
        raise FileNotFoundError(GOLD_TALLOS_GR)

    ofer = read_parquet(PRED_OFER_DIA).copy()
    gold = read_parquet(GOLD_TALLOS_GR).copy()

    # -------------------------
    # Canon
    # -------------------------
    ofer["fecha"] = _to_date(ofer["fecha"])
    ofer["ciclo_id"] = ofer["ciclo_id"].astype(str)

    if "bloque_base" in ofer.columns:
        ofer["bloque_base"] = _canon_int(ofer["bloque_base"])
    elif "bloque_padre" in ofer.columns:
        ofer["bloque_base"] = _canon_int(ofer["bloque_padre"])
    else:
        raise ValueError("pred_oferta_dia: falta bloque_base/bloque_padre")

    if "variedad_canon" in ofer.columns:
        ofer["variedad_canon"] = _canon_str(ofer["variedad_canon"])
    elif "variedad" in ofer.columns:
        ofer["variedad_canon"] = _canon_str(ofer["variedad"])
    else:
        raise ValueError("pred_oferta_dia: falta variedad/variedad_canon")

    ofer["tallos_pred"] = pd.to_numeric(ofer["tallos_pred"], errors="coerce")
    ofer["tallos_proy"] = pd.to_numeric(ofer["tallos_proy"], errors="coerce")

    gold["fecha"] = _to_date(gold["fecha"])
    gold["ciclo_id"] = gold["ciclo_id"].astype(str)
    gold["bloque_base"] = _canon_int(gold["bloque_base"])
    gold["variedad_canon"] = _canon_str(gold["variedad_canon"])
    gold["grado"] = _canon_int(gold["grado"])
    gold["tallos_pred_baseline_grado_dia"] = pd.to_numeric(gold["tallos_pred_baseline_grado_dia"], errors="coerce")
    gold["tallos_pred_ml1_grado_dia"] = pd.to_numeric(gold["tallos_pred_ml1_grado_dia"], errors="coerce")

    # -------------------------
    # A) Duplicados en pred_oferta_dia
    # -------------------------
    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    dup_n = int(ofer.duplicated(subset=key).sum())
    print("================================================================================")
    print("[A] DUPLICADOS pred_oferta_dia")
    print("================================================================================")
    print(f"rows={len(ofer):,} | dup_rows_by_key={dup_n:,} | dup_rate={dup_n/max(len(ofer),1):.4f}")

    if dup_n > 0:
        top = (
            ofer[ofer.duplicated(subset=key, keep=False)]
            .groupby(key, dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\nTop duplicated keys (count):")
        print(top.to_string())

    # -------------------------
    # B) Consistencia de tallos_proy dentro de ciclo_id
    # -------------------------
    print("\n================================================================================")
    print("[B] CONSISTENCIA tallos_proy")
    print("================================================================================")
    cyc_proy = ofer.groupby("ciclo_id", dropna=False).agg(
        n_rows=("tallos_proy", "size"),
        n_unique_proy=("tallos_proy", lambda x: x.dropna().nunique()),
        proy_min=("tallos_proy", "min"),
        proy_max=("tallos_proy", "max"),
    ).reset_index()

    bad = cyc_proy[cyc_proy["n_unique_proy"] > 1].copy()
    print(f"ciclos total={len(cyc_proy):,} | ciclos con tallos_proy variable={len(bad):,} ({len(bad)/max(len(cyc_proy),1):.2%})")
    if len(bad):
        print("\nEjemplos (top 20 por rango proy_max-proy_min):")
        bad["range"] = bad["proy_max"] - bad["proy_min"]
        print(bad.sort_values("range", ascending=False).head(20).to_string(index=False))

    # -------------------------
    # C) Mass balance baseline vs tallos_proy (en pred_oferta_dia)
    # -------------------------
    print("\n================================================================================")
    print("[C] MASS BALANCE en pred_oferta_dia (baseline vs tallos_proy)")
    print("================================================================================")
    cyc = ofer.groupby("ciclo_id", dropna=False).agg(
        proy_any=("tallos_proy", "max"),       # max es más robusto si viene repetido
        base_sum=("tallos_pred", "sum"),
    ).reset_index()
    cyc["abs_diff"] = (cyc["base_sum"] - cyc["proy_any"]).abs()
    print(cyc["abs_diff"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print(f"max_abs_diff={float(cyc['abs_diff'].max()):.3f}")
    worst = cyc.sort_values("abs_diff", ascending=False).head(20)
    print("\nTop 20 ciclos peor diff (pred_oferta_dia):")
    print(worst.to_string(index=False))

    # -------------------------
    # D) Mass balance en GOLD (ML1 sum grado/dia)
    # -------------------------
    print("\n================================================================================")
    print("[D] MASS BALANCE en GOLD (sum ML1 grado/día vs target)")
    print("================================================================================")
    gold_cyc = gold.groupby("ciclo_id", dropna=False).agg(
        ml1_sum=("tallos_pred_ml1_grado_dia", "sum"),
        base_sum=("tallos_pred_baseline_grado_dia", "sum"),
    ).reset_index()

    gold_join = gold_cyc.merge(cyc[["ciclo_id", "proy_any"]], on="ciclo_id", how="left")
    gold_join["abs_diff_ml1_vs_proy"] = (gold_join["ml1_sum"] - gold_join["proy_any"]).abs()
    print(gold_join["abs_diff_ml1_vs_proy"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())
    print(f"max_abs_diff_ml1_vs_proy={float(gold_join['abs_diff_ml1_vs_proy'].max()):.3f}")

    print("\nTop 20 ciclos peor diff (GOLD vs proy_any):")
    print(gold_join.sort_values("abs_diff_ml1_vs_proy", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
