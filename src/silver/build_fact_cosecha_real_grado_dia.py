from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def bloque_padre_from_bloque(b: pd.Series) -> pd.Series:
    """
    303B -> 303 ; 303 -> 303 ; limpia todo lo no numérico.
    """
    return (b.astype(str)
              .str.upper()
              .str.strip()
              .str.replace(r"[^0-9]", "", regex=True))


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    dist_cfg = cfg.get("dist_grado", {})
    fecha_min = pd.to_datetime(dist_cfg.get("fecha_min", "2024-01-01"))
    destino_excluir = dist_cfg.get("destino_excluir", []) or []
    grados_validos = dist_cfg.get("grados_validos", []) or []
    min_tallos_dia = int(dist_cfg.get("min_tallos_dia", 1))

    variedades_validas = dist_cfg.get("variedades_validas", []) or []

    # Input bronze
    balanza_bronze_name = dist_cfg.get("balanza_cosecha_bronze_file", "balanza_cosecha_raw.parquet")
    in_path = bronze_dir / balanza_bronze_name
    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe Bronze: {in_path}. Ejecuta src/bronze/build_balanza_cosecha_raw.py primero."
        )

    df = pd.read_parquet(in_path)
    df.columns = [str(c).strip() for c in df.columns]

    needed = {"Fecha", "Variedad", "Bloque", "Grado", "Destino", "Tallos"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"balanza_cosecha_raw no tiene columnas requeridas: {missing}")

    # Tipos / limpieza (Silver)
    df["Fecha"] = _norm_date(df["Fecha"])
    df["Variedad"] = df["Variedad"].astype(str).str.strip().str.upper()
    df["Bloque"] = df["Bloque"].astype(str).str.strip()
    df["Destino"] = df["Destino"].astype(str).str.strip()
    df["Grado"] = pd.to_numeric(df["Grado"], errors="coerce").astype("Int64")
    df["Tallos"] = pd.to_numeric(df["Tallos"], errors="coerce").fillna(0.0)

    # Filtros
    df = df[df["Fecha"].notna()].copy()
    df = df[df["Fecha"] >= fecha_min].copy()
    df = df[df["Tallos"] > 0].copy()

    if variedades_validas:
        vv = set(str(x).strip().upper() for x in variedades_validas)
        df = df[df["Variedad"].isin(vv)].copy()

    if destino_excluir:
        df = df[~df["Destino"].isin(destino_excluir)].copy()

    if grados_validos:
        gv = set(int(x) for x in grados_validos)
        df = df[df["Grado"].isin(gv)].copy()

    df["bloque_padre"] = bloque_padre_from_bloque(df["Bloque"])
    df = df[df["bloque_padre"].astype(str).str.len() > 0].copy()

    # Agregación diaria por bloque_padre, variedad, grado
    fact = (df.groupby(["Fecha", "bloque_padre", "Variedad", "Grado"], dropna=False)
              .agg(tallos_real=("Tallos", "sum"))
              .reset_index())

    fact = fact.rename(columns={
        "Fecha": "fecha",
        "Variedad": "variedad",
        "Grado": "grado",
    })

    fact["tallos_real"] = fact["tallos_real"].astype(float)

    # Filtro mínimo por día
    fact = fact[fact["tallos_real"] >= float(min_tallos_dia)].copy()

    fact["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "fact_cosecha_real_grado_dia.parquet"
    write_parquet(fact, out_path)

    print(f"OK: fact_cosecha_real_grado_dia={len(fact)} filas -> {out_path}")
    print("Rango fechas:", fact["fecha"].min(), "->", fact["fecha"].max())
    print("Grados:", sorted(fact["grado"].dropna().unique().tolist())[:20])


if __name__ == "__main__":
    main()
