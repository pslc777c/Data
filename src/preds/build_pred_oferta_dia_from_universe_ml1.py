from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")
IN_MAESTRO = Path("data/silver/fact_ciclo_maestro.parquet")

OUT = Path("data/preds/pred_oferta_dia.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Cols={list(df.columns)}")


def main() -> None:
    created_at = datetime.now().isoformat(timespec="seconds")

    if not IN_GRID.exists():
        raise FileNotFoundError(f"No existe: {IN_GRID}")
    if not IN_MAESTRO.exists():
        raise FileNotFoundError(f"No existe: {IN_MAESTRO}")

    grid = read_parquet(IN_GRID).copy()
    maestro = read_parquet(IN_MAESTRO).copy()

    # -------------------------
    # Requisitos mínimos
    # -------------------------
    _require(grid, ["ciclo_id", "fecha"], "universe_harvest_grid_ml1")
    _require(maestro, ["ciclo_id", "tallos_proy"], "fact_ciclo_maestro")

    # -------------------------
    # Canon
    # -------------------------
    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])

    # bloque_base / variedad_canon desde grid (preferido)
    if "bloque_base" in grid.columns:
        grid["bloque_base"] = _canon_int(grid["bloque_base"])
    elif "bloque_padre" in grid.columns:
        grid["bloque_base"] = _canon_int(grid["bloque_padre"])

    if "variedad_canon" in grid.columns:
        grid["variedad_canon"] = _canon_str(grid["variedad_canon"])
    elif "variedad" in grid.columns:
        grid["variedad_canon"] = _canon_str(grid["variedad"])

    maestro["ciclo_id"] = maestro["ciclo_id"].astype(str)
    maestro["tallos_proy"] = pd.to_numeric(maestro["tallos_proy"], errors="coerce").astype(float)

    # opcionales de maestro
    for c in ["bloque_base", "bloque", "bloque_padre"]:
        if c in maestro.columns:
            maestro[c] = _canon_int(maestro[c])
    for c in ["variedad_canon", "variedad"]:
        if c in maestro.columns:
            maestro[c] = _canon_str(maestro[c])
    if "fecha_sp" in maestro.columns:
        maestro["fecha_sp"] = _to_date(maestro["fecha_sp"])
    for c in ["fecha_inicio_cosecha", "fecha_fin_cosecha"]:
        if c in maestro.columns:
            maestro[c] = _to_date(maestro[c])

    # -------------------------
    # Dedupe hard del grid
    # -------------------------
    key = ["ciclo_id", "fecha"]
    # Si tu grid trae bloque_base/variedad_canon, las metemos al grano final
    if "bloque_base" in grid.columns:
        key.append("bloque_base")
    if "variedad_canon" in grid.columns:
        key.append("variedad_canon")

    dup = int(grid.duplicated(subset=key).sum())
    if dup > 0:
        # en grid, duplicados deberían ser imposibles; los colapsamos por seguridad
        print(f"[WARN] universe_harvest_grid_ml1 duplicado por {key}; colapso. dup={dup:,}")
        agg = {}
        for c in ["harvest_start_pred", "harvest_end_pred", "n_harvest_days_pred"]:
            if c in grid.columns:
                agg[c] = "first"
        for c in ["area", "tipo_sp", "estado", "bloque", "bloque_padre"]:
            if c in grid.columns and c not in key:
                agg[c] = "first"
        grid = grid.groupby(key, as_index=False).agg(agg) if agg else grid.drop_duplicates(subset=key)

    # -------------------------
    # Join maestro (tallos_proy y metadatos)
    # -------------------------
    m_take = ["ciclo_id", "tallos_proy"]
    for c in ["bloque", "bloque_padre", "bloque_base", "variedad", "variedad_canon", "tipo_sp", "area", "estado"]:
        if c in maestro.columns:
            m_take.append(c)
    for c in ["fecha_sp", "fecha_inicio_cosecha", "fecha_fin_cosecha"]:
        if c in maestro.columns:
            m_take.append(c)

    m2 = maestro[m_take].drop_duplicates(subset=["ciclo_id"])
    out = grid.merge(m2, on="ciclo_id", how="left", suffixes=("", "_m"))

    # asegurar bloque_base/variedad_canon
    if "bloque_base" not in out.columns and "bloque_base_m" in out.columns:
        out["bloque_base"] = out["bloque_base_m"]
    if "variedad_canon" not in out.columns:
        if "variedad_canon_m" in out.columns:
            out["variedad_canon"] = out["variedad_canon_m"]
        elif "variedad_m" in out.columns:
            out["variedad_canon"] = out["variedad_m"]

    # Ventana: preferimos la predicha del grid
    if "harvest_start_pred" in out.columns:
        out["harvest_start"] = _to_date(out["harvest_start_pred"])
    elif "fecha_inicio_cosecha" in out.columns:
        out["harvest_start"] = _to_date(out["fecha_inicio_cosecha"])
    else:
        out["harvest_start"] = pd.NaT

    if "harvest_end_pred" in out.columns:
        out["harvest_end_eff"] = _to_date(out["harvest_end_pred"])
    elif "fecha_fin_cosecha" in out.columns:
        out["harvest_end_eff"] = _to_date(out["fecha_fin_cosecha"])
    else:
        out["harvest_end_eff"] = pd.NaT

    if "n_harvest_days_pred" in out.columns:
        out["n_harvest_days"] = pd.to_numeric(out["n_harvest_days_pred"], errors="coerce").astype("Int64")
    else:
        # fallback por diferencia
        out["n_harvest_days"] = (out["harvest_end_eff"] - out["harvest_start"]).dt.days.add(1)
        out["n_harvest_days"] = pd.to_numeric(out["n_harvest_days"], errors="coerce").astype("Int64")

    # -------------------------
    # Oferta baseline: uniform + ajuste de residuo último día
    # -------------------------
    out["tallos_proy"] = pd.to_numeric(out["tallos_proy"], errors="coerce")
    out["n_harvest_days"] = pd.to_numeric(out["n_harvest_days"], errors="coerce").astype("Int64")

    bad = out["tallos_proy"].isna() | out["n_harvest_days"].isna() | (out["n_harvest_days"].astype(float) <= 0)
    if bad.any():
        nbad = int(bad.sum())
        print(f"[WARN] filas sin tallos_proy o n_harvest_days inválido: {nbad:,}. tallos_pred=0 en esas filas.")
        out.loc[bad, "tallos_pred"] = 0.0

    ok = ~bad
    out.loc[ok, "tallos_pred"] = out.loc[ok, "tallos_proy"] / out.loc[ok, "n_harvest_days"].astype(float)

    # ajuste de residuo para garantizar sum exacto por ciclo
    grp = ["ciclo_id"]
    out = out.sort_values(["ciclo_id", "fecha"]).reset_index(drop=True)

    # marca último día por ciclo dentro del grid
    out["_is_last"] = out["fecha"].eq(out.groupby("ciclo_id")["fecha"].transform("max"))

    sums = out.groupby("ciclo_id", dropna=False)["tallos_pred"].transform("sum")
    target = out.groupby("ciclo_id", dropna=False)["tallos_proy"].transform("max")

    resid = (target - sums)
    out.loc[out["_is_last"] & ok, "tallos_pred"] = out.loc[out["_is_last"] & ok, "tallos_pred"] + resid[out["_is_last"] & ok]
    out = out.drop(columns=["_is_last"], errors="ignore")

    # stage fijo (este dataset ES harvest-grid)
    out["stage"] = "HARVEST"

    # -------------------------
    # Salida final
    # -------------------------
    out["created_at"] = created_at

    cols = [
        "ciclo_id", "fecha",
        "bloque" if "bloque" in out.columns else None,
        "bloque_padre" if "bloque_padre" in out.columns else None,
        "bloque_base",
        "variedad" if "variedad" in out.columns else None,
        "variedad_canon",
        "tipo_sp" if "tipo_sp" in out.columns else None,
        "area" if "area" in out.columns else None,
        "estado" if "estado" in out.columns else None,
        "stage",
        "harvest_start",
        "harvest_end_eff",
        "n_harvest_days",
        "tallos_proy",
        "tallos_pred",
        "created_at",
    ]
    cols = [c for c in cols if c is not None and c in out.columns]
    out = out[cols].sort_values(["ciclo_id", "fecha", "bloque_base", "variedad_canon"]).reset_index(drop=True)

    # Checks finales
    key2 = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    dup2 = int(out.duplicated(subset=key2).sum())
    if dup2:
        raise ValueError(f"[FATAL] Salida tiene duplicados por {key2}: {dup2}")

    cyc = out.groupby("ciclo_id", dropna=False).agg(
        proy=("tallos_proy", "max"),
        sum_pred=("tallos_pred", "sum"),
    ).reset_index()
    cyc["abs_diff"] = (cyc["proy"] - cyc["sum_pred"]).abs()
    print(f"[CHECK] ciclo mass-balance | max abs diff: {float(cyc['abs_diff'].max()):.12f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT)
    print(f"OK -> {OUT} | rows={len(out):,} | fecha_min={out['fecha'].min().date()} fecha_max={out['fecha'].max().date()}")


if __name__ == "__main__":
    main()
