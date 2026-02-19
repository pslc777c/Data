# src/bronze/build_fenograma_sources.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pyodbc
import yaml

from common.io import write_parquet


# -------------------------
# Helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_excel_any_header(path: str, sheet_name: str, skiprows: int = 0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows, engine="openpyxl")
    return _norm_cols(df)


def _read_indices_sheet_raw(path: str, sheet_name: str) -> pd.DataFrame:
    """
    BRONZE: leer sheet COMPLETO como raw (header=None).
    Requisito: NO inferir tipos. Todo se persiste como string para evitar fallos de parquet.
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl", header=None)
    raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
    return raw


def _sql_connect(server: str, db: str, driver: str, user: str, password: str) -> pyodbc.Connection:
    conn_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={db};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def _read_balanza_filtrada(
    server: str,
    db: str,
    schema: str,
    table: str,
    driver: str,
    user: str,
    password: str,
    bloques: list[str],
    fecha_min: pd.Timestamp,
) -> pd.DataFrame:
    if not bloques:
        return pd.DataFrame(columns=["Bloque", "Fecha"])

    bloques = [str(b).strip() for b in bloques if str(b).strip() != ""]
    seen = set()
    bloques_u = []
    for b in bloques:
        if b not in seen:
            seen.add(b)
            bloques_u.append(b)
    bloques = bloques_u

    out = []
    fecha_min = pd.to_datetime(fecha_min)

    with _sql_connect(server, db, driver, user, password) as conn:
        chunk_size = 800
        for i in range(0, len(bloques), chunk_size):
            chunk = bloques[i : i + chunk_size]
            chunk_esc = [c.replace("'", "''") for c in chunk]
            chunk_sql = ", ".join([f"'{c}'" for c in chunk_esc])

            q = f"""
                SELECT Bloque, Fecha
                FROM {schema}.{table}
                WHERE Bloque IN ({chunk_sql})
                  AND Fecha >= ?
            """
            tmp = pd.read_sql(q, conn, params=[fecha_min])
            out.append(tmp)

    bal = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["Bloque", "Fecha"])
    bal = _norm_cols(bal)
    if "Bloque" in bal.columns:
        bal["Bloque"] = bal["Bloque"].astype(str)
    if "Fecha" in bal.columns:
        bal["Fecha"] = _to_dt(bal["Fecha"])
        bal = bal[bal["Fecha"].notna()].copy()
    return bal


def _force_all_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    BRONZE RAW (índices): forzar 100% a string.
    Esto evita inferencias tipo int64/float en columnas con encabezados/etiquetas mezcladas.
    """
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].astype("string")
    return df


def _force_safe_types_for_parquet(df: pd.DataFrame, prefer_str_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Fenograma XLSM: tipado seguro para parquet.
    """
    df = df.copy()
    prefer_str_cols = prefer_str_cols or []

    for c in prefer_str_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_cols:
        s = df[c]
        sample = s.dropna()
        if sample.empty:
            df[c] = s.astype("string")
            continue

        types = sample.map(lambda x: type(x)).value_counts()
        if len(types) > 1:
            df[c] = s.astype("string")
            continue

        if types.index[0] is str:
            df[c] = s.astype("string")
            continue

        try:
            df[c] = pd.to_numeric(s, errors="raise")
        except Exception:
            df[c] = s.astype("string")

    return df


def _profile_object_cols(df: pd.DataFrame, top_n: int = 5) -> None:
    obj_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "string")]
    if not obj_cols:
        return
    _info(f"Perfil columnas tipo object/string (n={len(obj_cols)}):")
    for c in obj_cols[: min(len(obj_cols), 25)]:
        vals = df[c].dropna().astype(str).head(top_n).tolist()
        _info(f" - {c}: dtype={df[c].dtype} ejemplos={vals}")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Fenograma XLSM (activo) - BRONZE raw con tipado seguro
    # -------------------------
    ruta_xlsm = cfg["sources"].get("fenograma_path", "")
    hoja_fenograma = cfg["sources"].get("fenograma_sheet", "")
    skip_rows = int(cfg.get("sources", {}).get("fenograma_skiprows", 7))

    if not (ruta_xlsm and hoja_fenograma):
        raise ValueError("Config: define sources.fenograma_path y sources.fenograma_sheet")

    df_xlsm = _read_excel_any_header(ruta_xlsm, hoja_fenograma, skiprows=skip_rows)

    prefer_str = ["Fiesta", "Bloques", "Bloque", "Area", "Area ", "Variedad", "S/P", "Pruebas", "Pruebas "]
    df_xlsm = _force_safe_types_for_parquet(df_xlsm, prefer_str_cols=prefer_str)

    df_xlsm["bronze_source"] = "fenograma_xlsm"
    df_xlsm["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

    _profile_object_cols(df_xlsm)

    out_xlsm = bronze_dir / "fenograma_xlsm_raw.parquet"
    write_parquet(df_xlsm, out_xlsm)
    _info(f"OK: fenograma_xlsm_raw={len(df_xlsm)} -> {out_xlsm}")

    bloques = (
        df_xlsm.get("Bloques", pd.Series([], dtype="string"))
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )

    # -------------------------
    # Índices históricos XL/CLO (raw 100% string)
    # -------------------------
    ruta_indices = cfg.get("sources", {}).get("indices_path", "")
    sheet_xl = cfg.get("sources", {}).get("indices_sheet_xl", "")
    sheet_clo = cfg.get("sources", {}).get("indices_sheet_clo", "")

    if ruta_indices and sheet_xl:
        raw_xl = _read_indices_sheet_raw(ruta_indices, sheet_xl)
        raw_xl = _force_all_to_string(raw_xl)
        raw_xl["bronze_source"] = "indices_xl"
        raw_xl["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")
        out_xl = bronze_dir / "indices_xl_raw.parquet"
        write_parquet(raw_xl, out_xl)
        _info(f"OK: indices_xl_raw={len(raw_xl)} -> {out_xl}")
    else:
        _warn("No se generó indices_xl_raw (faltan sources.indices_path o sources.indices_sheet_xl).")

    if ruta_indices and sheet_clo:
        raw_clo = _read_indices_sheet_raw(ruta_indices, sheet_clo)
        raw_clo = _force_all_to_string(raw_clo)
        raw_clo["bronze_source"] = "indices_clo"
        raw_clo["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")
        out_clo = bronze_dir / "indices_clo_raw.parquet"
        write_parquet(raw_clo, out_clo)
        _info(f"OK: indices_clo_raw={len(raw_clo)} -> {out_clo}")
    else:
        _warn("No se generó indices_clo_raw (faltan sources.indices_path o sources.indices_sheet_clo).")

    # -------------------------
    # Balanza SQL (raw filtrado)
    # -------------------------
    fecha_min_hist = pd.to_datetime(cfg.get("sources", {}).get("fecha_min_hist", "2024-01-01"))

    sql_server = cfg["sources"].get("sql_server", "")
    sql_db = cfg["sources"].get("sql_db", "")
    sql_schema = cfg["sources"].get("sql_schema", "dbo")
    sql_table = cfg["sources"].get("sql_table", "")
    odbc_driver = cfg["sources"].get("odbc_driver", "ODBC Driver 17 for SQL Server")

    sql_user = cfg["sources"].get("sql_user", "") or os.environ.get("SQL_USER", "")
    sql_password = cfg["sources"].get("sql_password", "") or os.environ.get("SQL_PASSWORD", "")

    if sql_server and sql_db and sql_table and sql_user and sql_password:
        bal = _read_balanza_filtrada(
            server=sql_server,
            db=sql_db,
            schema=sql_schema,
            table=sql_table,
            driver=odbc_driver,
            user=sql_user,
            password=sql_password,
            bloques=bloques,
            fecha_min=fecha_min_hist,
        )
        bal["bronze_source"] = "balanza_sql"
        bal["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

        out_bal = bronze_dir / "balanza_bloque_fecha_raw.parquet"
        write_parquet(bal, out_bal)
        _info(f"OK: balanza_bloque_fecha_raw={len(bal)} -> {out_bal}")
    else:
        _warn(
            "No se generó balanza_bloque_fecha_raw. "
            "Revisa sources.sql_server/sql_db/sql_table y variables de entorno SQL_USER/SQL_PASSWORD."
        )


if __name__ == "__main__":
    main()
