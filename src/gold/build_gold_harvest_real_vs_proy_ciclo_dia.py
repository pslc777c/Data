from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from src.common.io import read_parquet, write_parquet


FACT_REAL_PATH = Path("data/silver/fact_cosecha_real_grado_dia.parquet")
MASTER_PATH = Path("data/silver/fact_ciclo_maestro.parquet")

OUT_PATH = Path("data/gold/harvest_real_vs_totalproy_ciclo_dia.parquet")

MIN_DATE = None  # e.g. "2024-01-01"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Falta columna '{label}' esperada: '{col}'. Disponibles: {sorted(df.columns.tolist())}")


def main() -> None:
    # -------------------------
    # 1) REAL (grado-día -> bloque + fecha)  [variedad queda como atributo]
    # -------------------------
    real = read_parquet(FACT_REAL_PATH)

    _require(real, "bloque_padre", "bloque_padre (REAL)")
    _require(real, "fecha", "fecha (REAL)")
    _require(real, "tallos_real", "tallos_real (REAL)")
    _require(real, "variedad", "variedad (REAL)")

    real = real.copy()
    real["fecha"] = _to_date(real["fecha"])
    real = real[real["fecha"].notna()].copy()
    real["tallos_real"] = pd.to_numeric(real["tallos_real"], errors="coerce").fillna(0.0)

    # llave de bloque: REAL usa bloque_padre y calza con MAESTRO.bloque_padre/bloque_base
    real["bloque_key"] = real["bloque_padre"].astype(str).str.strip()

    # variedad solo como atributo (no para match)
    real["variedad_real"] = _canon_str(real["variedad"])

    if MIN_DATE:
        real = real[real["fecha"] >= pd.to_datetime(MIN_DATE)].copy()

    # Sumamos sobre grado -> REAL diario por bloque
    # Nota: si quieres conservar variedad en el output, la llevamos como "first" (en tu real solo hay 2 variedades)
    real_daily = (
        real.groupby(["bloque_key", "fecha"], as_index=False)
        .agg(
            real_dia=("tallos_real", "sum"),
            variedad_real=("variedad_real", "first"),
        )
        .sort_values(["bloque_key", "fecha"])
        .reset_index(drop=True)
    )

    # -------------------------
    # 2) MAESTRO (ciclos) y ventanas
    # -------------------------
    m = read_parquet(MASTER_PATH)

    for c in ["ciclo_id", "tallos_proy", "fecha_sp", "fecha_fin_cosecha", "bloque_padre", "bloque_base"]:
        _require(m, c, f"{c} (MAESTRO)")

    cycles = m.copy()
    # Usamos bloque_padre del maestro porque es el que interseca 127/127 con el real
    cycles["bloque_key"] = cycles["bloque_padre"].astype(str).str.strip()

    cycles["fecha_sp"] = _to_date(cycles["fecha_sp"])
    cycles["fecha_fin_cosecha"] = _to_date(cycles["fecha_fin_cosecha"])

    cycles["total_proy_final"] = pd.to_numeric(cycles["tallos_proy"], errors="coerce").astype(float)

    keep_cols = ["ciclo_id", "bloque_key", "fecha_sp", "fecha_fin_cosecha", "total_proy_final"]
    if "estado" in cycles.columns:
        keep_cols.append("estado")

    cycles = cycles[keep_cols].copy()
    cycles = cycles.sort_values(["bloque_key", "fecha_sp"]).reset_index(drop=True)

    # -------------------------
    # 3) Asignación de ciclo_id por BLOQUE + fecha_sp (merge_asof)
    # -------------------------
    # merge_asof exige orden exacto
    real_daily = real_daily.sort_values(["bloque_key", "fecha"]).reset_index(drop=True)
    cycles = cycles.sort_values(["bloque_key", "fecha_sp"]).reset_index(drop=True)

    # Intento directo + fallback por grupo (robusto)
    try:
        assigned = pd.merge_asof(
            real_daily,
            cycles,
            left_on="fecha",
            right_on="fecha_sp",
            by=["bloque_key"],
            direction="backward",
            allow_exact_matches=True,
        )
    except ValueError as e:
        if "keys must be sorted" not in str(e).lower():
            raise

        parts = []
        right_groups = {k: g.sort_values("fecha_sp").reset_index(drop=True) for k, g in cycles.groupby("bloque_key", sort=False)}
        for bk, gleft in real_daily.groupby("bloque_key", sort=False):
            gright = right_groups.get(bk)
            gleft2 = gleft.sort_values("fecha").reset_index(drop=True)

            if gright is None or gright.empty:
                tmp = gleft2.copy()
                tmp["ciclo_id"] = pd.NA
                tmp["fecha_sp"] = pd.NaT
                tmp["fecha_fin_cosecha"] = pd.NaT
                tmp["total_proy_final"] = np.nan
                if "estado" in cycles.columns:
                    tmp["estado"] = pd.NA
                parts.append(tmp)
                continue

            tmp = pd.merge_asof(
                gleft2,
                gright,
                left_on="fecha",
                right_on="fecha_sp",
                direction="backward",
                allow_exact_matches=True,
            )
            # Asegurar que la llave del grupo exista siempre
            tmp["bloque_key"] = bk
            parts.append(tmp)


        assigned = pd.concat(parts, ignore_index=True)

    # Validación de fin (si hay fin, la fecha real debe estar <= fin)
    fin = assigned["fecha_fin_cosecha"]
    mask_ok = fin.isna() | (assigned["fecha"] <= fin)

    assigned.loc[~mask_ok, "ciclo_id"] = pd.NA
    assigned.loc[~mask_ok, "total_proy_final"] = np.nan
    if "estado" in assigned.columns:
        assigned.loc[~mask_ok, "estado"] = pd.NA

    # -------------------------
    # 4) Métricas por ciclo_id
    # -------------------------
    assigned["has_ciclo"] = assigned["ciclo_id"].notna()

    assigned["real_acum"] = np.nan
    assigned.loc[assigned["has_ciclo"], "real_acum"] = (
        assigned.loc[assigned["has_ciclo"]]
        .groupby("ciclo_id")["real_dia"]
        .cumsum()
    )

    assigned["pct_vs_total_proy"] = np.nan
    mask = assigned["has_ciclo"] & assigned["total_proy_final"].notna() & (assigned["total_proy_final"] > 0)
    assigned.loc[mask, "pct_vs_total_proy"] = assigned.loc[mask, "real_acum"] / assigned.loc[mask, "total_proy_final"]

    assigned["is_closed"] = assigned["has_ciclo"] & assigned["fecha_fin_cosecha"].notna()

    tot_real_closed = (
        assigned.loc[assigned["is_closed"]]
        .groupby("ciclo_id", as_index=False)["real_dia"]
        .sum()
        .rename(columns={"real_dia": "total_real_final_closed"})
    )
    assigned = assigned.merge(tot_real_closed, on="ciclo_id", how="left")

    assigned["pct_vs_total_real_closed"] = np.nan
    mask2 = assigned["is_closed"] & assigned["total_real_final_closed"].notna() & (assigned["total_real_final_closed"] > 0)
    assigned.loc[mask2, "pct_vs_total_real_closed"] = assigned.loc[mask2, "real_acum"] / assigned.loc[mask2, "total_real_final_closed"]

    assigned["estado_vs_totalproy"] = np.select(
        [
            ~assigned["has_ciclo"],
            assigned["total_proy_final"].isna(),
            assigned["pct_vs_total_proy"] < 0.8,
            assigned["pct_vs_total_proy"] <= 1.05,
        ],
        [
            "SIN_CICLO_ASIGNADO",
            "SIN_TOTAL_PROY",
            "ATRASADO_<80%",
            "EN_RANGO_80_105%",
        ],
        default="ADELANTADO_>105%",
    )

    # Output
    cols = [
        "ciclo_id",
        "bloque_key",
        "variedad_real",
        "fecha",
        "real_dia",
        "real_acum",
        "total_proy_final",
        "pct_vs_total_proy",
        "fecha_sp",
        "fecha_fin_cosecha",
        "is_closed",
        "total_real_final_closed",
        "pct_vs_total_real_closed",
        "estado_vs_totalproy",
    ]
    if "estado" in assigned.columns:
        cols.insert(cols.index("estado_vs_totalproy"), "estado")

    final = (
        assigned[cols]
        .rename(columns={"bloque_key": "bloque_base"})
        .sort_values(["ciclo_id", "fecha"], na_position="last")
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(final, OUT_PATH)

    n_rows = len(final)
    n_ciclos = final["ciclo_id"].nunique(dropna=True)
    n_sin = final["ciclo_id"].isna().sum()
    print(f"[OK] Wrote: {OUT_PATH} | rows={n_rows:,} | ciclos={n_ciclos:,} | sin_ciclo_asignado={n_sin:,}")


if __name__ == "__main__":
    main()
