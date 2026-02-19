# src/bronze/build_ghu_maestro_horas.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pyodbc
import yaml

from common.io import write_parquet


# -------------------------
# Config / helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _read_available_columns(conn: pyodbc.Connection, db: str, schema: str, table_or_view: str) -> set[str]:
    q = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_CATALOG = ? AND TABLE_SCHEMA = ? AND TABLE_NAME = ?
    """
    cols = pd.read_sql(q, conn, params=[db, schema, table_or_view])
    return set(cols["COLUMN_NAME"].astype(str).tolist())


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    src = cfg.get("sources", {})
    server = src.get("sql_server", "")
    driver = src.get("odbc_driver", "ODBC Driver 17 for SQL Server")
    user = src.get("sql_user", "")
    password = src.get("sql_password", "")

    db_gh = src.get("sql_db_gh", "GestionHumana")
    schema_gh = src.get("sql_schema_ghu", "dbo")
    view_ghu = src.get("sql_view_ghu_maestro_horas", "GHU_View_Maestro_Horas")

    if not server:
        raise ValueError("Config: define sources.sql_server en settings.yaml")
    if not (user and password):
        raise ValueError("Config: define sources.sql_user y sources.sql_password en settings.yaml")

    dt_min = pd.to_datetime(cfg.get("bronze", {}).get("ghu_fecha_min", "2024-01-01"))

    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db_gh};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )

    _info(f"Leyendo {db_gh}.{schema_gh}.{view_ghu} desde {dt_min.date()} ...")

    with pyodbc.connect(conn_str) as conn:
        existing = _read_available_columns(conn, db_gh, schema_gh, view_ghu)

        # Base columns (las que ya usabas)
        wanted = [
            "fecha",
            "codigo_personal",
            "nombres",
            "actividad",
            "codigo_actividad",
            "tipo_actividad",
            "area_trabajada",
            "area_original",
            "horas_acumula",
            "bloque",
            "hora_ingreso",
            "hora_salida",
            "horas_presenciales",
            "unidades_producidas",
        ]

        # Nueva columna requerida por Silver (si existe en la vista)
        # Si tu vista la nombra distinto, agrega alias en settings o aquí.
        extra_candidates = ["unidad_medida"]
        for c in extra_candidates:
            if c not in wanted:
                wanted.append(c)

        selected = [c for c in wanted if c in existing]
        missing = [c for c in wanted if c not in existing]
        if missing:
            _warn(f"{db_gh}.{schema_gh}.{view_ghu}: columnas ausentes (se crearán NaN): {missing}")

        select_sql = ", ".join(selected) if selected else "*"

        query = f"""
            SELECT {select_sql}
            FROM {schema_gh}.{view_ghu}
            WHERE fecha >= ?
        """

        df = pd.read_sql(query, conn, params=[dt_min])

    df.columns = [str(c).strip() for c in df.columns]

    # Asegurar presencia de columnas ausentes como NaN (robusto)
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    # Normalizaciones mínimas BRONZE
    df["fecha"] = _to_dt(df["fecha"])
    n_bad = int(df["fecha"].isna().sum())
    if n_bad > 0:
        _warn(f"Hay {n_bad} filas con fecha no parseable. Se eliminan en BRONZE por imposibilidad técnica.")
        df = df[df["fecha"].notna()].copy()

    # Metadatos
    df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = bronze_dir / "ghu_maestro_horas.parquet"
    write_parquet(df, out_path)

    _info(f"OK: bronze ghu_maestro_horas={len(df)} filas -> {out_path}")
    _info(f"Rango fechas: {df['fecha'].min()} -> {df['fecha'].max()}")
    if "tipo_actividad" in df.columns:
        _info(f"Tipo_actividad (top): {df['tipo_actividad'].astype(str).value_counts(dropna=False).head(10).to_dict()}")


if __name__ == "__main__":
    main()
