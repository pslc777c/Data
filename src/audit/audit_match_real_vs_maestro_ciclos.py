from __future__ import annotations

import pandas as pd
import numpy as np

from src.common.io import read_parquet

REAL_PATH = "data/silver/fact_cosecha_real_grado_dia.parquet"
MAESTRO_PATH = "data/silver/fact_ciclo_maestro.parquet"

def canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()

def main() -> None:
    real = read_parquet(REAL_PATH)
    m = read_parquet(MAESTRO_PATH)

    # columnas
    print("REAL cols:", real.columns.tolist())
    print("MAESTRO cols:", m.columns.tolist())

    # real keys
    real_bloque = canon(real["bloque_padre"])
    real_var = canon(real["variedad"])
    real_fecha = pd.to_datetime(real["fecha"], errors="coerce").dt.normalize()

    # maestro candidates de bloque
    cand_bloques = [c for c in ["bloque_padre", "bloque_base", "bloque", "bloque_id"] if c in m.columns]
    print("\nMaestro bloque candidates:", cand_bloques)

    # stats de fechas
    m_sp = pd.to_datetime(m["fecha_sp"], errors="coerce").dt.normalize()
    print("\nREAL fecha min/max:", real_fecha.min(), real_fecha.max())
    print("MAESTRO fecha_sp min/max:", m_sp.min(), m_sp.max())

    # prueba de intersección por bloque (ignorando variedad)
    real_b_set = set(real_bloque.dropna().unique().tolist())
    for c in cand_bloques:
        mb = canon(m[c])
        inter = len(real_b_set.intersection(set(mb.dropna().unique().tolist())))
        print(f"Intersección bloques REAL vs MAESTRO.{c}: {inter:,} (real_unique={len(real_b_set):,}, maestro_unique={mb.nunique():,})")

    # prueba de variedad (solo para ver si coincide algo)
    inter_var = len(set(real_var.dropna().unique()).intersection(set(canon(m["variedad"]).dropna().unique())))
    print(f"\nIntersección variedades REAL vs MAESTRO.variedad: {inter_var:,} (real_unique={real_var.nunique():,}, maestro_unique={canon(m['variedad']).nunique():,})")

    # muestra algunos valores para inspección visual
    print("\nEjemplos REAL bloque_padre:", real_bloque.dropna().unique()[:10])
    for c in cand_bloques:
        print(f"Ejemplos MAESTRO {c}:", canon(m[c]).dropna().unique()[:10])

if __name__ == "__main__":
    main()
