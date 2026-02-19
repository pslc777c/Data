# src/bronze/build_weather_hour_main.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pyodbc
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _read_weather_table(
    conn: pyodbc.Connection,
    schema: str,
    table: str,
    dt_min: pd.Timestamp,
) -> pd.DataFrame:
    wanted = [
        "fecha",
        "rainfall_mm",
        "et",
        "temp_avg",
        "wind_dir_of_avg",
        "wind_run",
        "solar_rad_avg",
        "solar_rad_hi",
        "solar_energy",
        "hum_last",
        "uv_index_avg",
        "wind_speed_avg",
    ]

    cols_df = pd.read_sql(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        conn,
        params=[schema, table],
    )
    existing = set(cols_df["COLUMN_NAME"].astype(str).tolist())
    if "fecha" not in existing:
        raise ValueError(f"{schema}.{table} no tiene columna 'fecha' (obligatoria).")

    selected = [c for c in wanted if c in existing]
    missing = [c for c in wanted if c not in existing]
    if missing:
        _warn(f"{schema}.{table}: columnas ausentes (se crearán NaN): {missing}")

    select_sql = ", ".join(selected) if selected else "fecha"
    q = f"""
        SELECT {select_sql}
        FROM {schema}.{table}
        WHERE fecha >= ?
        ORDER BY fecha
    """
    df = pd.read_sql(q, conn, params=[dt_min])
    df.columns = [str(c).strip() for c in df.columns]

    # Asegurar presencia de todas las wanted
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    df["fecha"] = _to_dt(df["fecha"])
    df = df[df["fecha"].notna()].copy()

    for c in [
        "rainfall_mm", "et", "temp_avg", "wind_run",
        "solar_rad_avg", "solar_rad_hi", "solar_energy",
        "hum_last", "uv_index_avg", "wind_speed_avg",
    ]:
        df[c] = _to_num(df[c])

    return df


def main() -> None:
    cfg = load_settings()
    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("sources", {})
    server = src.get("sql_server", "")
    driver = src.get("odbc_driver", "ODBC Driver 17 for SQL Server")
    user = src.get("sql_user", "")
    password = src.get("sql_password", "")

    db_weather = src.get("sql_db_weather", "WeatherStation")
    schema = src.get("sql_schema_weather", "dbo")
    table_main = src.get("sql_table_weather_main", "Weather_Station_Hour")

    if not server:
        raise ValueError("Config: define sources.sql_server en settings.yaml")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    dt_min = pd.to_datetime(cfg.get("bronze", {}).get("weather_fecha_min", "2024-01-01 01:00:00"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_weather};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    _info(f"Leyendo clima MAIN desde {db_weather}.{schema}.{table_main} >= {dt_min} ...")
    with pyodbc.connect(conn_str) as conn:
        df = _read_weather_table(conn, schema, table_main, dt_min)

    df["station"] = "MAIN"
    df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = bronze_dir / "weather_hour_main.parquet"
    write_parquet(df, out_path)

    _info(f"OK: bronze weather_hour_main={len(df)} filas -> {out_path}")
    _info(f"Rango fechas: {df['fecha'].min()} -> {df['fecha'].max()}")


if __name__ == "__main__":
    main()
