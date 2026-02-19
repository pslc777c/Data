from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =========================
# INPUTS / OUTPUTS
# =========================
IN_TALLOS = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")
IN_PESO   = Path("data/gold/pred_peso_tallo_grado_ml1.parquet")

OUT_KG_GRADO      = Path("data/gold/pred_kg_grado_dia_ml1_full.parquet")
OUT_CAJAS_GRADO   = Path("data/gold/pred_cajas_grado_dia_ml1_full.parquet")
OUT_CAJAS_DIA     = Path("data/gold/pred_cajas_dia_ml1_full.parquet")

# NUEVO: agregados operacionales (sin ciclo_id)
OUT_KG_GRADO_AGG    = Path("data/gold/pred_kg_grado_dia_ml1_agg.parquet")
OUT_CAJAS_GRADO_AGG = Path("data/gold/pred_cajas_grado_dia_ml1_agg.parquet")


# =========================
# HELPERS
# =========================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")


def _detect_tallos_cols(tallos: pd.DataFrame) -> tuple[str, str]:
    cand_baseline = [c for c in ["tallos_pred_baseline_grado_dia", "tallos_baseline_grado_dia", "tallos_pred_baseline"] if c in tallos.columns]
    cand_ml1      = [c for c in ["tallos_pred_ml1_grado_dia", "tallos_ml1_grado_dia", "tallos_pred_ml1"] if c in tallos.columns]

    if not cand_baseline:
        cand_baseline = [c for c in tallos.columns if ("tallos" in c and "baseline" in c)]
    if not cand_ml1:
        cand_ml1 = [c for c in tallos.columns if ("tallos" in c and "ml1" in c)]

    if not cand_baseline or not cand_ml1:
        raise ValueError(
            "No pude detectar columnas tallos baseline/ml1 en pred_tallos_grado_dia_ml1_full. "
            f"Cols={list(tallos.columns)}"
        )
    return cand_baseline[0], cand_ml1[0]


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    tallos = read_parquet(IN_TALLOS).copy()
    peso   = read_parquet(IN_PESO).copy()

    tallos.columns = [str(c).strip() for c in tallos.columns]
    peso.columns   = [str(c).strip() for c in peso.columns]

    # ciclo_id es opcional pero recomendado
    has_ciclo = "ciclo_id" in tallos.columns

    keys = ["fecha", "bloque_base", "variedad_canon", "grado"]
    if has_ciclo:
        keys_ciclo = ["ciclo_id"] + keys
    else:
        keys_ciclo = keys

    _require(tallos, keys_ciclo, "pred_tallos_grado_dia_ml1_full")
    _require(peso, keys + ["peso_tallo_baseline_g", "peso_tallo_ml1_g"], "pred_peso_tallo_grado_ml1")

    # Normalización
    tallos["fecha"] = _to_date(tallos["fecha"])
    tallos["bloque_base"] = _canon_int(tallos["bloque_base"])
    tallos["grado"] = _canon_int(tallos["grado"])
    tallos["variedad_canon"] = tallos["variedad_canon"].astype(str).str.upper().str.strip()
    if has_ciclo:
        tallos["ciclo_id"] = tallos["ciclo_id"].astype(str)

    peso["fecha"] = _to_date(peso["fecha"])
    peso["bloque_base"] = _canon_int(peso["bloque_base"])
    peso["grado"] = _canon_int(peso["grado"])
    peso["variedad_canon"] = peso["variedad_canon"].astype(str).str.upper().str.strip()

    col_tallos_baseline, col_tallos_ml1 = _detect_tallos_cols(tallos)

    # Peso: dedup por keys (sin ciclo_id, porque peso se modela a ese grano)
    dup = int(peso.duplicated(subset=keys).sum())
    if dup > 0:
        print(f"[WARN] peso duplicado por {keys}; colapso mean.")
        agg = {"peso_tallo_baseline_g": "mean", "peso_tallo_ml1_g": "mean"}
        if "factor_peso_tallo_ml1" in peso.columns:
            agg["factor_peso_tallo_ml1"] = "mean"
        if "ml1_version" in peso.columns:
            agg["ml1_version"] = "first"
        peso = peso.groupby(keys, dropna=False, as_index=False).agg(agg)

    # Merge
    df = tallos.merge(peso, on=keys, how="left", validate="m:1")

    # Num
    for c in [col_tallos_baseline, col_tallos_ml1, "peso_tallo_baseline_g", "peso_tallo_ml1_g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # KG
    df["kg_baseline_grado_dia"] = (df[col_tallos_baseline] * df["peso_tallo_baseline_g"]) / 1000.0
    df["kg_ml1_grado_dia"]      = (df[col_tallos_ml1]      * df["peso_tallo_ml1_g"])      / 1000.0

    # Cajas (10 kg/caja)
    df["cajas_baseline_grado_dia"] = df["kg_baseline_grado_dia"] / 10.0
    df["cajas_ml1_grado_dia"]      = df["kg_ml1_grado_dia"] / 10.0

    df["created_at"] = created_at

    # =========================
    # OUTPUT (con ciclo_id si existe)
    # =========================
    out_cols = []
    if has_ciclo:
        out_cols.append("ciclo_id")
    out_cols += [
        "fecha", "bloque_base", "variedad_canon", "grado",
        col_tallos_baseline, col_tallos_ml1,
        "peso_tallo_baseline_g", "peso_tallo_ml1_g",
        "kg_baseline_grado_dia", "kg_ml1_grado_dia",
        "cajas_baseline_grado_dia", "cajas_ml1_grado_dia",
        "created_at",
    ]

    out = df[out_cols].sort_values(
        (["ciclo_id"] if has_ciclo else []) + ["bloque_base", "variedad_canon", "fecha", "grado"]
    ).reset_index(drop=True)

    # Split outputs
    out_kg = out[[c for c in out.columns if not c.startswith("cajas_")]].copy()
    out_cajas = out[[c for c in out.columns if not c.startswith("kg_") and c not in [col_tallos_baseline, col_tallos_ml1]]].copy()

    write_parquet(out_kg, OUT_KG_GRADO)
    print(f"OK -> {OUT_KG_GRADO} | rows={len(out_kg):,}")

    write_parquet(out_cajas, OUT_CAJAS_GRADO)
    print(f"OK -> {OUT_CAJAS_GRADO} | rows={len(out_cajas):,}")

    # =========================
    # OUTPUT agregados operacionales (sin ciclo_id)
    # =========================
    grp = ["fecha", "bloque_base", "variedad_canon", "grado"]

    out_kg_agg = (
        out.groupby(grp, dropna=False, as_index=False)
           .agg(
               kg_baseline_grado_dia=("kg_baseline_grado_dia", "sum"),
               kg_ml1_grado_dia=("kg_ml1_grado_dia", "sum"),
               cajas_baseline_grado_dia=("cajas_baseline_grado_dia", "sum"),
               cajas_ml1_grado_dia=("cajas_ml1_grado_dia", "sum"),
           )
    )
    out_kg_agg["created_at"] = created_at

    write_parquet(out_kg_agg[["fecha","bloque_base","variedad_canon","grado","kg_baseline_grado_dia","kg_ml1_grado_dia","created_at"]], OUT_KG_GRADO_AGG)
    print(f"OK -> {OUT_KG_GRADO_AGG} | rows={len(out_kg_agg):,}")

    write_parquet(out_kg_agg[["fecha","bloque_base","variedad_canon","grado","cajas_baseline_grado_dia","cajas_ml1_grado_dia","created_at"]], OUT_CAJAS_GRADO_AGG)
    print(f"OK -> {OUT_CAJAS_GRADO_AGG} | rows={len(out_kg_agg):,}")

    # =========================
    # OUTPUT día (sin ciclo_id, como antes)
    # =========================
    grp_day = ["fecha", "bloque_base", "variedad_canon"]
    out_day = (
        out_kg_agg.groupby(grp_day, dropna=False, as_index=False)
                  .agg(
                      cajas_baseline_dia=("cajas_baseline_grado_dia", "sum"),
                      cajas_ml1_dia=("cajas_ml1_grado_dia", "sum"),
                      kg_baseline_dia=("kg_baseline_grado_dia", "sum"),
                      kg_ml1_dia=("kg_ml1_grado_dia", "sum"),
                  )
    )
    out_day["created_at"] = created_at

    out_day = out_day.sort_values(["bloque_base","variedad_canon","fecha"]).reset_index(drop=True)
    write_parquet(out_day, OUT_CAJAS_DIA)
    print(f"OK -> {OUT_CAJAS_DIA} | rows={len(out_day):,}")

    cov_peso = float(df["peso_tallo_ml1_g"].notna().mean())
    cov_tallos = float(df[col_tallos_ml1].notna().mean())
    print(f"[CHECK] coverage tallos_ml1 notna: {cov_tallos:.4f}")
    print(f"[CHECK] coverage peso_tallo_ml1_g notna: {cov_peso:.4f}")
    if has_ciclo:
        dup_oper = int(out_kg_agg.duplicated(subset=grp).sum())
        print(f"[CHECK] duplicates en agregado operacional (debe ser 0): {dup_oper:,}")


if __name__ == "__main__":
    main()
