from __future__ import annotations

from pathlib import Path
import pandas as pd

from common.io import write_parquet


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
BRONZE = DATA / "bronze"

OUT_PATH = BRONZE / "calendario.parquet"


def main() -> None:
    BRONZE.mkdir(parents=True, exist_ok=True)

    fecha_inicio = pd.Timestamp("2023-01-01")
    # equivalente a Date.AddYears(Date.From(DateTime.LocalNow()), 2)
    fecha_fin = pd.Timestamp.today().normalize() + pd.DateOffset(years=2)

    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    cal = pd.DataFrame({"Fecha": fechas.normalize()})

    cal["Año"] = cal["Fecha"].dt.year.astype("Int64")
    cal["Mes"] = cal["Fecha"].dt.month.astype("Int64")
    cal["Mes_Nombre"] = cal["Fecha"].dt.month_name()
    cal["Dia"] = cal["Fecha"].dt.day.astype("Int64")

    # Day.Sunday = 0 en PowerQuery; pandas dayofweek: Monday=0
    # Para "Dia_Semana" con domingo=0:
    cal["Dia_Semana"] = ((cal["Fecha"].dt.dayofweek + 1) % 7).astype("Int64")
    cal["Nombre_del_dia"] = cal["Fecha"].dt.day_name()

    # Semana_Año (lunes a domingo) -> ISO week
    iso = cal["Fecha"].dt.isocalendar()
    cal["Semana"] = iso.week.astype("Int64")
    cal["Semana_Año"] = (iso.year % 100).astype(int).astype(str).str.zfill(2) + iso.week.astype(int).astype(str).str.zfill(2)

    write_parquet(cal, OUT_PATH)
    print(f"[OK] Calendario guardado en {OUT_PATH} | rows={len(cal):,} | {cal['Fecha'].min().date()}..{cal['Fecha'].max().date()}")


if __name__ == "__main__":
    main()
