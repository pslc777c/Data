# src/silver/build_fact_peso_tallo_real_grado_dia.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def bloque_padre_from_bloque(b: pd.Series) -> pd.Series:
    return (
        b.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^0-9]", "", regex=True)
    )


def _require_cols(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name}: faltan columnas requeridas: {missing}. "
            f"Columnas disponibles={list(df.columns)}"
        )


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Input BRONZE: balanza raw
    # -------------------------
    balanza_cfg = cfg.get("balanza", {})
    balanza_file = balanza_cfg.get("balanza_file", "balanza_raw.parquet")
    in_path = bronze_dir / balanza_file

    if not in_path.exists():
        raise FileNotFoundError(
            f"No existe input Bronze de balanza: {in_path}\n"
            "Crea primero la capa Bronze de balanza (la que contiene peso_menos_vegetativo), "
            "o ajusta balanza.balanza_file en settings.yaml."
        )

    df = read_parquet(in_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Requeridas para este fact
    needed = ["Fecha", "Variedad", "Bloque", "Grado", "Tallos", "peso_menos_vegetativo"]
    _require_cols(df, needed, f"balanza bronze ({balanza_file})")

    # -------------------------
    # Filtros config (igual que antes)
    # -------------------------
    dist_cfg = cfg.get("dist_grado", {})
    fecha_min = pd.to_datetime(dist_cfg.get("fecha_min", "2024-01-01"))

    variedades_validas = dist_cfg.get("variedades_validas", []) or []
    vv = set(str(x).strip().upper() for x in variedades_validas) if variedades_validas else None

    # -------------------------
    # Limpieza + tipos
    # -------------------------
    df["Fecha"] = _norm_date(df["Fecha"])
    df["Variedad"] = df["Variedad"].astype(str).str.strip().str.upper()
    df["Bloque"] = df["Bloque"].astype(str).str.strip()
    df["Grado"] = pd.to_numeric(df["Grado"], errors="coerce").astype("Int64")
    df["Tallos"] = pd.to_numeric(df["Tallos"], errors="coerce").fillna(0.0)
    df["peso_menos_vegetativo"] = pd.to_numeric(df["peso_menos_vegetativo"], errors="coerce").fillna(0.0)

    # Filtros operativos (idénticos)
    df = df[df["Fecha"].notna()].copy()
    df = df[df["Fecha"] >= fecha_min].copy()
    df = df[(df["Tallos"] > 0) & (df["peso_menos_vegetativo"] > 0)].copy()
    df = df[df["Grado"].notna()].copy()

    if vv is not None:
        df = df[df["Variedad"].isin(vv)].copy()

    df["bloque_padre"] = bloque_padre_from_bloque(df["Bloque"])
    df = df[df["bloque_padre"].astype(str).str.len() > 0].copy()

    # -------------------------
    # Agregación diaria por bloque_padre, variedad, grado
    # -------------------------
    fact = (
        df.groupby(["Fecha", "bloque_padre", "Variedad", "Grado"], dropna=False)
        .agg(
            tallos_real=("Tallos", "sum"),
            peso_real_kg=("peso_menos_vegetativo", "sum"),
        )
        .reset_index()
    )

    fact = fact.rename(
        columns={
            "Fecha": "fecha",
            "Variedad": "variedad",
            "Grado": "grado",
        }
    )

    fact["tallos_real"] = fact["tallos_real"].astype(float)
    fact["peso_real_kg"] = fact["peso_real_kg"].astype(float)

    # convertir a gramos
    fact["peso_real_g"] = fact["peso_real_kg"] * 1000.0
    fact["peso_tallo_real_g"] = fact["peso_real_g"] / fact["tallos_real"]

    # eliminar kg para no mezclar unidades
    fact = fact.drop(columns=["peso_real_kg"])

    # sanity caps (evitar basura extrema)
    fact = fact[(fact["peso_tallo_real_g"] > 1) & (fact["peso_tallo_real_g"] < 500)].copy()

    fact["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "fact_peso_tallo_real_grado_dia.parquet"
    write_parquet(fact, out_path)

    print(f"OK: fact_peso_tallo_real_grado_dia={len(fact)} filas -> {out_path}")
    print("Rango fechas:", fact["fecha"].min(), "->", fact["fecha"].max())
    print("peso_tallo_real_g describe:\n", fact["peso_tallo_real_g"].describe().to_string())


if __name__ == "__main__":
    main()
