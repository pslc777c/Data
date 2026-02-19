# src/bronze/build_balanza_mermas_sources.py
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

    if not (server and db):
        raise ValueError("Config: define sources.sql_server y sources.sql_db en settings.yaml")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    # Recorte temporal permitido en BRONZE (solo por volumen, no por negocio)
    dt_min = pd.to_datetime(cfg.get("bronze", {}).get("balanza_fecha_min", "2025-01-01"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    q_2a = f"""
        SELECT
            Fecha,
            Origen,
            Seccion,
            Variedad,
            Lote,
            codigo_actividad,
            Grado,
            peso_balanza,
            tallos,
            num_bunches
        FROM {schema}.BAL_View_Balanza_2A
        WHERE Fecha >= ?
    """

    q_2 = f"""
        SELECT
            fecha_entrega,
            Lote,
            Grado,
            Destino,
            Tallos,
            peso_neto,
            variedad,
            tipo_pelado,
            Origen,
            producto
        FROM dbo.BAL_View_Balanza_2
        WHERE fecha_entrega >= ?

    """

    with pyodbc.connect(conn_str) as conn:
        _info(f"Leyendo BAL_View_Balanza_2A >= {dt_min.date()} ...")
        df2a = pd.read_sql(q_2a, conn, params=[dt_min])

        _info(f"Leyendo BAL_View_Balanza_2 >= {dt_min.date()} ...")
        df2 = pd.read_sql(q_2, conn, params=[dt_min])

    df2a.columns = [str(c).strip() for c in df2a.columns]
    df2.columns = [str(c).strip() for c in df2.columns]

    # Tipos mínimos (permitidos en BRONZE)
    df2a["Fecha"] = _to_dt(df2a["Fecha"])
    df2 = df2[df2["fecha_entrega"].notna()].copy()
    df2["fecha_entrega"] = _to_dt(df2["fecha_entrega"])

    df2a["bronze_source"] = "BAL_View_Balanza_2A"
    df2a["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    df2["bronze_source"] = "BAL_View_Balanza_2"
    df2["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out2a = bronze_dir / "balanza_2a_raw.parquet"
    out2 = bronze_dir / "balanza_2_raw.parquet"

    write_parquet(df2a, out2a)
    write_parquet(df2, out2)

    _info(f"OK: balanza_2a_raw={len(df2a)} -> {out2a}")
    _info(f"OK: balanza_2_raw={len(df2)} -> {out2}")


if __name__ == "__main__":
    main()
