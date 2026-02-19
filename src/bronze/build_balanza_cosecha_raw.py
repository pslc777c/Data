# src/bronze/build_balanza_cosecha_raw.py
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


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _get_existing_columns(conn: pyodbc.Connection, schema: str, table: str) -> set[str]:
    cols = pd.read_sql(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        """,
        conn,
        params=[schema, table],
    )
    return set(cols["COLUMN_NAME"].astype(str).tolist())


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("sources", {})
    server = src.get("sql_server", "")
    db = src.get("sql_db", "")
    schema = src.get("sql_schema", "dbo")
    table = src.get("sql_table_balanza", "")
    driver = src.get("odbc_driver", "ODBC Driver 17 for SQL Server")
    user = src.get("sql_user", "")
    password = src.get("sql_password", "")

    if not (server and db and table):
        raise ValueError("Config: define sources.sql_server, sources.sql_db y sources.sql_table_balanza")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    dist_cfg = cfg.get("dist_grado", {})
    fecha_min = pd.to_datetime(dist_cfg.get("fecha_min", "2024-01-01"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    wanted = [
        "Fecha",
        "Variedad",
        "Bloque",
        "Grado",
        "Destino",
        "Tallos",
        "peso_menos_vegetativo",   # <-- requerido para peso_tallo
    ]

    _info(f"Leyendo BALANZA cosecha raw desde {db}.{schema}.{table} >= {fecha_min.date()} ...")
    with pyodbc.connect(conn_str) as conn:
        existing = _get_existing_columns(conn, schema, table)

        if "Fecha" not in existing:
            raise ValueError(f"{schema}.{table} no tiene columna 'Fecha' (obligatoria).")

        selected = [c for c in wanted if c in existing]
        missing = [c for c in wanted if c not in existing]
        if missing:
            _warn(f"{schema}.{table}: columnas ausentes (se crearán como NaN en Bronze): {missing}")

        select_sql = ", ".join(selected) if selected else "Fecha"
        query = f"""
            SELECT {select_sql}
            FROM {schema}.{table}
            WHERE Fecha >= ?
        """
        df = pd.read_sql(query, conn, params=[fecha_min])

    df.columns = [str(c).strip() for c in df.columns]

    # Asegurar presencia de todas las columnas wanted (si faltaron en SQL)
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    # Normalizaciones técnicas permitidas (Bronze):
    df["Fecha"] = _to_dt(df["Fecha"])
    bad = int(df["Fecha"].isna().sum())
    if bad > 0:
        _warn(f"Hay {bad} filas con Fecha no parseable. Se eliminan en Bronze por imposibilidad técnica.")
        df = df[df["Fecha"].notna()].copy()

    # Casteos técnicos a string (evita mezcla rara de tipos al escribir parquet)
    for c in ["Variedad", "Bloque", "Destino"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # El resto (Grado, Tallos, peso_menos_vegetativo) queda “tal cual venga”
    # (sin reglas de negocio ni caps; Silver decide)

    df["bronze_source"] = "balanza_cosecha_sql"
    df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = bronze_dir / "balanza_cosecha_raw.parquet"
    write_parquet(df, out_path)

    _info(f"OK: bronze balanza_cosecha_raw={len(df)} filas -> {out_path}")
    _info(f"Rango fechas: {df['Fecha'].min()} -> {df['Fecha'].max()}")


if __name__ == "__main__":
    main()
