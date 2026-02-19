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


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])

    in_path = silver_dir / "fact_peso_tallo_real_grado_dia.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}. Primero construye fact_peso_tallo_real_grado_dia.")

    df = read_parquet(in_path).copy()

    needed = {"fecha", "tallos_real", "peso_tallo_real_g"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"fact_peso_tallo_real_grado_dia no tiene columnas requeridas: {missing}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["tallos_real"] = pd.to_numeric(df["tallos_real"], errors="coerce").fillna(0.0)
    df["peso_tallo_real_g"] = pd.to_numeric(df["peso_tallo_real_g"], errors="coerce")

    df = df[df["fecha"].notna()].copy()
    df = df[(df["tallos_real"] > 0) & df["peso_tallo_real_g"].notna()].copy()

    # Guardrail
    if df.empty:
        raise ValueError("No hay datos válidos para calcular peso_tallo_promedio_dia (df vacío tras filtros).")

    # Promedio ponderado diario: sum(w*x)/sum(w)
    df["wx"] = df["tallos_real"] * df["peso_tallo_real_g"]

    daily = (df.groupby("fecha", dropna=False)
               .agg(
                   tallos_dia=("tallos_real", "sum"),
                   wx_dia=("wx", "sum"),
               )
               .reset_index())

    daily["peso_tallo_prom_g"] = np.where(
        daily["tallos_dia"] > 0,
        daily["wx_dia"] / daily["tallos_dia"],
        np.nan
    )

    daily = daily[["fecha", "peso_tallo_prom_g", "tallos_dia"]].copy()
    daily["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_peso_tallo_promedio_dia.parquet"
    write_parquet(daily, out_path)

    print(f"OK: dim_peso_tallo_promedio_dia={len(daily)} -> {out_path}")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
