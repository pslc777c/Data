from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    df_maestro = read_parquet(Path("data/silver/fact_ciclo_maestro.parquet"))
    df_clima = read_parquet(Path("data/silver/dim_clima_bloque_dia.parquet"))
    df_grid = read_parquet(Path("data/silver/grid_ciclo_fecha.parquet"))

    # -------------------------
    # Validaciones
    # -------------------------
    need_m = {"ciclo_id", "bloque_base", "fecha_sp"}
    miss = need_m - set(df_maestro.columns)
    if miss:
        raise ValueError(f"fact_ciclo_maestro.parquet sin columnas: {sorted(miss)}")

    need_c = {"fecha", "bloque_base", "gdc_dia"}
    miss = need_c - set(df_clima.columns)
    if miss:
        raise ValueError(f"dim_clima_bloque_dia.parquet sin columnas: {sorted(miss)}")

    if "fecha" not in df_grid.columns or "ciclo_id" not in df_grid.columns:
        raise ValueError("grid_ciclo_fecha.parquet debe tener al menos columnas: fecha, ciclo_id")

    # -------------------------
    # Normalización de fechas
    # -------------------------
    m = df_maestro[["ciclo_id", "bloque_base", "fecha_sp"]].copy()
    m["fecha_sp"] = pd.to_datetime(m["fecha_sp"], errors="coerce").dt.normalize()

    # puede haber duplicados por bloque/ciclo: nos quedamos con 1 fecha_sp consistente
    m = (
        m.dropna(subset=["fecha_sp"])
         .groupby(["ciclo_id", "bloque_base"], as_index=False)
         .agg(fecha_sp=("fecha_sp", "min"))
    )

    g = df_grid[["ciclo_id", "fecha"]].copy()
    g["fecha"] = pd.to_datetime(g["fecha"], errors="coerce").dt.normalize()
    g = g.dropna(subset=["fecha"]).drop_duplicates()

    # Armamos calendario ciclo-fecha y lo cruzamos con bloques del ciclo
    # (para evitar inventar fechas fuera del ciclo)
    blocks = m[["ciclo_id", "bloque_base", "fecha_sp"]].copy()

    base = blocks.merge(g, on="ciclo_id", how="left")

    # Mantener solo fechas >= fecha_sp (térmicamente tiene sentido)
    base = base[base["fecha"] >= base["fecha_sp"]].copy()

    # -------------------------
    # Join clima (gdc_dia)
    # -------------------------
    clima = df_clima[["fecha", "bloque_base", "gdc_dia"]].copy()
    clima["fecha"] = pd.to_datetime(clima["fecha"], errors="coerce").dt.normalize()

    out = base.merge(clima, on=["fecha", "bloque_base"], how="left")

    # Si no hay clima para algún día, asumimos 0 GDC (o puedes dejar NaN)
    out["gdc_dia"] = out["gdc_dia"].fillna(0.0)

    # -------------------------
    # Cálculos desde S/P
    # -------------------------
    out["dias_desde_sp"] = (out["fecha"] - out["fecha_sp"]).dt.days

    out = out.sort_values(["ciclo_id", "bloque_base", "fecha"]).reset_index(drop=True)

    out["gdc_acum_desde_sp"] = (
        out.groupby(["ciclo_id", "bloque_base"])["gdc_dia"].cumsum()
    )

    out["created_at"] = created_at

    out = out[
        [
            "ciclo_id",
            "bloque_base",
            "fecha",
            "fecha_sp",
            "dias_desde_sp",
            "gdc_dia",
            "gdc_acum_desde_sp",
            "created_at",
        ]
    ].sort_values(["ciclo_id", "bloque_base", "fecha"]).reset_index(drop=True)

    write_parquet(out, Path("data/silver/dim_estado_termico_cultivo_bloque_fecha.parquet"))
    print(f"OK -> data/silver/dim_estado_termico_cultivo_bloque_fecha.parquet | rows={len(out):,}")


if __name__ == "__main__":
    main()
