from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hour_floor(s: pd.Series) -> pd.Series:
    return _to_dt(s).dt.floor("h")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _standardize_bronze_weather(df: pd.DataFrame, station_expected: str | None = None) -> pd.DataFrame:
    """
    Normaliza BRONZE weather_hour_{main,a4}.parquet a un esquema común:
      - fecha -> fecha_hora (floor hora)
      - station upper
      - asegura presence de columnas esperadas (con NaN si faltan)
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    if "fecha" not in d.columns:
        raise ValueError(f"Bronze weather: falta columna 'fecha'. Columnas={list(d.columns)}")

    d["fecha_hora"] = _hour_floor(d["fecha"])
    d = d[d["fecha_hora"].notna()].copy()

    if "station" not in d.columns:
        d["station"] = station_expected if station_expected else "UNKNOWN"

    d["station"] = (
        d["station"].astype(str).str.upper().str.strip()
        .replace({"A-4": "A4", "SJP": "A4"})
    )
    if station_expected:
        d = d[d["station"].eq(station_expected.upper())].copy()

    # Columnas posibles (las que ya traes en BRONZE)
    wanted = [
        "rainfall_mm", "et", "temp_avg", "wind_dir_of_avg", "wind_run",
        "solar_rad_avg", "solar_rad_hi", "solar_energy",
        "hum_last", "uv_index_avg", "wind_speed_avg",
    ]
    for c in wanted:
        if c not in d.columns:
            d[c] = np.nan

    # Tipos numéricos
    for c in wanted:
        d[c] = _to_num(d[c])

    # Deduplicación técnica por station+hora
    d = d.sort_values("fecha_hora").drop_duplicates(subset=["station", "fecha_hora"], keep="last")
    return d


def add_solar_joules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # fallback: si no hay solar_rad_avg pero sí solar_rad_hi, usar hi como proxy
    if out["solar_rad_avg"].isna().all() and not out["solar_rad_hi"].isna().all():
        _warn("solar_rad_avg vacío; usando solar_rad_hi como proxy.")
        out["solar_rad_avg"] = out["solar_rad_hi"]

    # Joules por hora: W/m2 * 3600 s
    out["solar_energy_j_m2"] = out["solar_rad_avg"].astype(float) * 3600.0
    return out


def main() -> None:
    cfg = load_settings()
    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Inputs BRONZE
    p_main = bronze_dir / "weather_hour_main.parquet"
    p_a4 = bronze_dir / "weather_hour_a4.parquet"
    if not p_main.exists():
        raise FileNotFoundError(f"No existe BRONZE: {p_main}. Ejecuta build_weather_hour_main.py")
    if not p_a4.exists():
        raise FileNotFoundError(f"No existe BRONZE: {p_a4}. Ejecuta build_weather_hour_a4.py")

    # Input SILVER (estado ya calculado)
    p_estado = silver_dir / "weather_hour_estado.parquet"
    if not p_estado.exists():
        raise FileNotFoundError(
            f"No existe SILVER: {p_estado}. Ejecuta primero src/silver/build_weather_hour_estado.py"
        )

    df_main = pd.read_parquet(p_main)
    df_a4 = pd.read_parquet(p_a4)
    est = pd.read_parquet(p_estado)

    main_std = _standardize_bronze_weather(df_main, station_expected="MAIN")
    a4_std = _standardize_bronze_weather(df_a4, station_expected="A4")
    raw = pd.concat([main_std, a4_std], ignore_index=True)

    # Normalizar/validar estado
    est.columns = [str(c).strip() for c in est.columns]
    # En el script de estado que te dejé: fecha_hora + station + Estado_Kardex + En_Lluvia + kardex_mm...
    required_estado = {"fecha_hora", "station", "Estado_Kardex", "En_Lluvia", "kardex_mm", "kardex_prev_mm"}
    missing = required_estado - set(est.columns)
    if missing:
        raise ValueError(
            f"weather_hour_estado.parquet no tiene columnas requeridas: {sorted(missing)}. "
            "Alinea build_weather_hour_estado.py con ese esquema."
        )

    est["fecha_hora"] = _hour_floor(est["fecha_hora"])
    est["station"] = est["station"].astype(str).str.upper().str.strip()

    # Join raw + estado
    wide = raw.merge(
        est[["fecha_hora", "station", "En_Lluvia", "kardex_prev_mm", "kardex_mm", "Estado_Kardex"]],
        on=["fecha_hora", "station"],
        how="left",
    )

    # Imputación conservadora si faltan horas de estado (no debería, pero cubrimos gaps)
    miss = int(wide["Estado_Kardex"].isna().sum())
    if miss > 0:
        _warn(f"Hay {miss} filas sin estado (join). Se imputa SECO/0 y kardex=0. Revisa cobertura.")
        wide["Estado_Kardex"] = wide["Estado_Kardex"].fillna("SECO")
        wide["En_Lluvia"] = _to_num(wide["En_Lluvia"]).fillna(0).astype(int)
        wide["kardex_prev_mm"] = _to_num(wide["kardex_prev_mm"]).fillna(0.0)
        wide["kardex_mm"] = _to_num(wide["kardex_mm"]).fillna(0.0)

    # Joules
    wide = add_solar_joules(wide)

    wide["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Output final: mantener nombres esperados por tus downstream (dt_hora/fecha si necesitas)
    # En tus otros scripts tú usas weather_hour_wide con columnas tipo dt_hora/fecha.
    # Aquí dejamos dt_hora = fecha_hora para que el join sea más directo.
    wide = wide.rename(columns={"fecha_hora": "dt_hora"})

    out_path = silver_dir / "weather_hour_wide.parquet"
    write_parquet(wide, out_path)

    _info(f"OK: weather_hour_wide={len(wide)} filas -> {out_path}")
    _info(f"Stations: {wide['station'].value_counts(dropna=False).to_dict()}")
    _info(f"Rango: {wide['dt_hora'].min()} -> {wide['dt_hora'].max()}")


if __name__ == "__main__":
    main()
