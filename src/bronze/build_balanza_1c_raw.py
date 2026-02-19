# src/bronze/build_balanza_1c_raw.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import pyodbc
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("sources", {})
    server = src.get("sql_server", "")
    db = src.get("sql_db", "")
    schema = src.get("sql_schema", "dbo")
    driver = src.get("odbc_driver", "ODBC Driver 17 for SQL Server")
    user = src.get("sql_user", "")
    password = src.get("sql_password", "")

    view_1c = src.get("sql_view_balanza_1c", "BAL_View_Balanza_1C")

    if not (server and db and view_1c):
        raise ValueError("Config: define sources.sql_server, sources.sql_db y sources.sql_view_balanza_1c")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    dt_min = pd.to_datetime(cfg.get("bronze", {}).get("balanza_fecha_min", "2024-01-01"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    q_1c = f"""
        SELECT *
        FROM {schema}.{view_1c}
        WHERE Fecha >= ?
    """

    _info(f"Leyendo {schema}.{view_1c} >= {dt_min.date()} ...")
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(q_1c, conn, params=[dt_min])

    df.columns = [str(c).strip() for c in df.columns]

    # Tipado técnico mínimo permitido en Bronze
    if "Fecha" in df.columns:
        df["Fecha"] = _to_dt(df["Fecha"])
        bad = int(df["Fecha"].isna().sum())
        if bad > 0:
            _warn(f"Hay {bad} filas con Fecha no parseable. Se eliminan en Bronze (imposibilidad técnica).")
            df = df[df["Fecha"].notna()].copy()

    df["bronze_source"] = view_1c
    df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out = bronze_dir / "balanza_1c_raw.parquet"
    write_parquet(df, out)

    _info(f"OK: balanza_1c_raw={len(df)} -> {out}")
    if "Fecha" in df.columns:
        _info(f"Rango fechas: {df['Fecha'].min()} -> {df['Fecha'].max()}")


if __name__ == "__main__":
    main()
