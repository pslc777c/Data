from __future__ import annotations

import pandas as pd

def build_grid_ciclo_fecha(fact_ciclo_maestro: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    # Espera columnas: ciclo_id, fecha_sp (date/timestamp)
    base = fact_ciclo_maestro[["ciclo_id", "fecha_sp"]].copy()
    base["fecha_sp"] = pd.to_datetime(base["fecha_sp"]).dt.normalize()

    rows = []
    for ciclo_id, fecha_sp in base.itertuples(index=False):
        start = fecha_sp
        end = start + pd.Timedelta(days=horizon_days)
        fechas = pd.date_range(start=start, end=end, freq="D")
        tmp = pd.DataFrame({"ciclo_id": ciclo_id, "fecha": fechas})
        rows.append(tmp)

    grid = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ciclo_id", "fecha"])
    # calendarios útiles
    grid["fecha"] = pd.to_datetime(grid["fecha"]).dt.normalize()
    # Semana_ (ISO o Sunday-based; ajustar a tu estándar)
    # Aquí uso ISO week por defecto:
    isocal = grid["fecha"].dt.isocalendar()
    grid["anio"] = isocal["year"].astype(int)
    grid["semana_iso"] = isocal["week"].astype(int)
    grid["mes"] = grid["fecha"].dt.month.astype(int)
    return grid
