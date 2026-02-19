from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    # -------------------------
    # Inputs (SILVER)
    # -------------------------
    df_real = read_parquet(Path("data/silver/fact_cosecha_real_grado_dia.parquet"))
    df_maestro = read_parquet(Path("data/silver/fact_ciclo_maestro.parquet"))
    df_clima = read_parquet(Path("data/silver/dim_clima_bloque_dia.parquet"))

    # grid_ciclo_fecha lo generas desde ciclo_maestro_from_fenograma
    df_grid = read_parquet(Path("data/silver/grid_ciclo_fecha.parquet"))

    # milestones: lo usamos después para validaciones / futura etapa (no es obligatorio para calcular progreso real)
    # df_milestones = read_parquet(Path("data/silver/milestones_ciclo_final.parquet"))

    # -------------------------
    # Validaciones mínimas de columnas
    # -------------------------
    need_real = {"fecha", "bloque_padre", "variedad", "grado", "tallos_real"}
    miss = need_real - set(df_real.columns)
    if miss:
        raise ValueError(f"fact_cosecha_real_grado_dia.parquet sin columnas: {sorted(miss)}")

    need_maestro = {"bloque", "bloque_base", "ciclo_id", "variedad", "area"}
    miss = need_maestro - set(df_maestro.columns)
    if miss:
        raise ValueError(f"fact_ciclo_maestro.parquet sin columnas: {sorted(miss)}")

    need_clima = {"fecha", "bloque_base", "gdc_dia"}
    miss = need_clima - set(df_clima.columns)
    if miss:
        raise ValueError(f"dim_clima_bloque_dia.parquet sin columnas: {sorted(miss)}")

    # grid: no sé tu schema exacto; validamos lo mínimo para mapear fecha -> ciclo
    # Ideal: que tenga (fecha, ciclo_id) y/o (fecha, bloque_base, ciclo_id)
    if "fecha" not in df_grid.columns or "ciclo_id" not in df_grid.columns:
        raise ValueError("grid_ciclo_fecha.parquet debe tener al menos columnas: fecha, ciclo_id")

    # -------------------------
    # 1) Normalizar fecha y definir bloque_base
    # -------------------------
    r = df_real.copy()
    r["fecha"] = pd.to_datetime(r["fecha"]).dt.normalize()

    # En este dataset, bloque_padre YA ES el bloque_base
    r = r.rename(columns={"bloque_padre": "bloque_base"})

    # -------------------------
    # 2) Real a día-bloque_base-variedad
    # -------------------------
    real_dia = (
        r.groupby(["fecha", "bloque_base", "variedad"], as_index=False)
         .agg(tallos_real_dia=("tallos_real", "sum"))
    )

    # -------------------------
    # 3) Asignar ciclo_id (preferir grid si tiene granularidad por bloque_base)
    # -------------------------
    g = df_grid.copy()
    g["fecha"] = pd.to_datetime(g["fecha"]).dt.normalize()

    join_cols = set(g.columns)

    if "bloque_base" in join_cols:
        # Mejor caso: grid ya está por bloque_base-fecha
        real_dia = real_dia.merge(
            g[["fecha", "bloque_base", "ciclo_id"]].drop_duplicates(),
            on=["fecha", "bloque_base"],
            how="left",
        )
    else:
        # Caso mínimo: grid por fecha-ciclo (menos preciso si hay ciclos simultáneos)
        # En ese caso usamos ciclo_id del maestro como fallback para filtrar.
        real_dia = real_dia.merge(
            g[["fecha", "ciclo_id"]].drop_duplicates(),
            on=["fecha"],
            how="left",
            suffixes=("", "_grid"),
        )
        # Si hay múltiples ciclos por fecha, esto puede duplicar filas.
        # Para no reventar, intentamos quedarnos con el ciclo del maestro.
        # Nota: para esto necesitamos ciclo_id por bloque_base en maestro.
        m_ciclo = df_maestro[["bloque_base", "ciclo_id"]].drop_duplicates()
        real_dia = real_dia.merge(m_ciclo, on="bloque_base", how="left", suffixes=("", "_maestro"))

        # Resolver ciclo_id final:
        # - si ciclo_id_maestro existe, úsalo
        # - si no, usa el del grid (si quedó único)
        # Primero, renombramos para no confundir
        if "ciclo_id_maestro" not in real_dia.columns:
            # si merge no lo creó por colisiones, lo armamos
            pass

        # En este branch, hay potencial de duplicación. Una forma segura:
        # nos quedamos con filas donde ciclo_id (del grid) == ciclo_id_maestro.
        if "ciclo_id_maestro" in real_dia.columns:
            real_dia = real_dia[real_dia["ciclo_id"] == real_dia["ciclo_id_maestro"]].copy()

        # Si aún queda nulo, fallback a ciclo_id_maestro
        real_dia["ciclo_id"] = real_dia["ciclo_id"].fillna(real_dia.get("ciclo_id_maestro"))

        # Limpiar columnas auxiliares
        drop_cols = [c for c in ["ciclo_id_maestro"] if c in real_dia.columns]
        if drop_cols:
            real_dia = real_dia.drop(columns=drop_cols)

    if real_dia["ciclo_id"].isna().any():
        bad = real_dia[real_dia["ciclo_id"].isna()][["fecha", "bloque_base"]].drop_duplicates().head(30)
        raise ValueError(
            "No pude asignar ciclo_id a algunas filas (fecha, bloque_base). "
            f"Ejemplos: {bad.to_dict('records')}"
        )

    # -------------------------
    # 4) Calcular inicio/fin real por ciclo-bloque-variedad
    # -------------------------
    # Inicio real: primera fecha con tallos_real_dia > 0
    real_pos = real_dia[real_dia["tallos_real_dia"] > 0].copy()

    anchors = (
        real_pos.groupby(["ciclo_id", "bloque_base", "variedad"], as_index=False)
                .agg(
                    fecha_inicio_real=("fecha", "min"),
                    fecha_fin_real=("fecha", "max"),
                    tallos_total_real=("tallos_real_dia", "sum"),
                )
    )

    out = real_dia.merge(anchors, on=["ciclo_id", "bloque_base", "variedad"], how="left")

    # dia_rel y acumulados (solo donde hay inicio_real)
    out = out.sort_values(["ciclo_id", "bloque_base", "variedad", "fecha"]).reset_index(drop=True)

    # tallos_acum_real: cumsum sobre días con inicio
    out["tallos_acum_real"] = (
        out.groupby(["ciclo_id", "bloque_base", "variedad"])["tallos_real_dia"].cumsum()
    )

    # pct_avance_real: solo si tallos_total_real > 0
    out["pct_avance_real"] = np.where(
        out["tallos_total_real"].fillna(0) > 0,
        out["tallos_acum_real"] / out["tallos_total_real"],
        np.nan,
    )

    # dia_rel_cosecha_real
    out["dia_rel_cosecha_real"] = np.where(
        out["fecha_inicio_real"].notna(),
        (out["fecha"] - out["fecha_inicio_real"]).dt.days,
        np.nan,
    )

    # en_ventana_cosecha_real (según real)
    out["en_ventana_cosecha_real"] = np.where(
        out["fecha_inicio_real"].notna() & out["fecha_fin_real"].notna(),
        ((out["fecha"] >= out["fecha_inicio_real"]) & (out["fecha"] <= out["fecha_fin_real"])).astype(int),
        0,
    )

    # -------------------------
    # 5) Join clima (gdc_dia) y calcular gdc_acum_real
    # -------------------------
    clima = df_clima[["fecha", "bloque_base", "gdc_dia"]].copy()
    clima["fecha"] = pd.to_datetime(clima["fecha"]).dt.normalize()

    out = out.merge(clima, on=["fecha", "bloque_base"], how="left")

    # gdc_acum_real: acumular gdc desde fecha_inicio_real
    out["gdc_dia_eff"] = out["gdc_dia"].fillna(0.0)

    # Marcador: solo acumular desde inicio_real (antes de inicio, 0)
    out["iniciado"] = np.where(
        out["fecha_inicio_real"].notna() & (out["fecha"] >= out["fecha_inicio_real"]),
        1,
        0,
    )
    out["gdc_dia_iniciado"] = out["gdc_dia_eff"] * out["iniciado"]

    out["gdc_acum_real"] = (
        out.groupby(["ciclo_id", "bloque_base", "variedad"])["gdc_dia_iniciado"].cumsum()
    )

    out = out.drop(columns=["gdc_dia_eff", "iniciado", "gdc_dia_iniciado"])

    # -------------------------
    # 6) Final
    # -------------------------
    out["created_at"] = created_at

    cols = [
        "ciclo_id",
        "fecha",
        "bloque_base",
        "variedad",
        "tallos_real_dia",
        "tallos_acum_real",
        "tallos_total_real",
        "pct_avance_real",
        "fecha_inicio_real",
        "fecha_fin_real",
        "dia_rel_cosecha_real",
        "en_ventana_cosecha_real",
        "gdc_dia",
        "gdc_acum_real",
        "created_at",
    ]

    out = out[cols].sort_values(["ciclo_id", "bloque_base", "variedad", "fecha"]).reset_index(drop=True)

    write_parquet(out, Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet"))
    print(f"OK -> data/silver/dim_cosecha_progress_bloque_fecha.parquet | rows={len(out):,}")


if __name__ == "__main__":
    main()
