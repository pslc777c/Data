# src/bronze/build_personal_sources.py
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


def _to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("sources", {})
    server = src.get("sql_server", "")
    driver = src.get("odbc_driver", "ODBC Driver 17 for SQL Server")
    user = src.get("sql_user", "")
    password = src.get("sql_password", "")

    db = src.get("sql_db_gh", src.get("sql_db_ghu", "GestionHumana"))
    schema = src.get("sql_schema_ghu", "dbo")
    table_personal = src.get("sql_table_personal", "Personal")

    if not server:
        raise ValueError("Config: define sources.sql_server en settings.yaml")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    query = f"""
        SELECT
            Codigo_Personal AS codigo_personal,
            Activo_o_Inactivo
        FROM {schema}.{table_personal}
    """

    _info(f"Leyendo Personal desde {db}.{schema}.{table_personal} ...")
    with pyodbc.connect(conn_str) as conn:
        df = pd.read_sql(query, conn)

    df.columns = [str(c).strip() for c in df.columns]

    if "codigo_personal" not in df.columns:
        raise ValueError("Personal: no se obtuvo columna 'codigo_personal'. Revisa query/tabla.")
    if "Activo_o_Inactivo" not in df.columns:
        raise ValueError("Personal: no se obtuvo columna 'Activo_o_Inactivo'. Revisa query/tabla.")

    df["codigo_personal"] = _to_int64(df["codigo_personal"])
    df["Activo_o_Inactivo"] = df["Activo_o_Inactivo"].astype(str).str.strip()

    # Quitar nulos imposibles técnicamente (sin código)
    df = df[df["codigo_personal"].notna()].copy()

    # Metadatos bronze
    df["bronze_source"] = "personal_sql"
    df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = bronze_dir / "personal_raw.parquet"
    write_parquet(df, out_path)

    _info(f"OK: bronze personal_raw={len(df)} filas -> {out_path}")
    _info(f"Activo_o_Inactivo (top): {df['Activo_o_Inactivo'].value_counts(dropna=False).head(10).to_dict()}")


if __name__ == "__main__":
    main()
