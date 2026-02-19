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
    fact_path = silver_dir / "fact_cosecha_real_grado_dia.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"No existe: {fact_path}. Ejecuta build_fact_cosecha_real_grado_dia primero.")

    fact = read_parquet(fact_path).copy()

    # Validaciones mínimas
    needed = {"fecha", "variedad", "grado", "tallos_real"}
    missing = needed - set(fact.columns)
    if missing:
        raise ValueError(f"fact_cosecha_real_grado_dia no tiene columnas requeridas: {missing}")

    fact["variedad"] = fact["variedad"].astype(str).str.strip()
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce").astype("Int64")
    fact["tallos_real"] = pd.to_numeric(fact["tallos_real"], errors="coerce").fillna(0.0)
    fact = fact[(fact["grado"].notna()) & (fact["tallos_real"] > 0)].copy()

    # 1) Totales por variedad y día (sumando grados)
    day_tot = (fact.groupby(["variedad", "fecha"], dropna=False)
                  .agg(tallos_dia=("tallos_real", "sum"))
                  .reset_index())

    # 2) Join para obtener % por grado por día
    tmp = fact.merge(day_tot, on=["variedad", "fecha"], how="left")
    tmp = tmp[tmp["tallos_dia"] > 0].copy()
    tmp["pct_grado_dia"] = tmp["tallos_real"] / tmp["tallos_dia"]

    # 3) Baseline por variedad+grado: usar mediana (robusto)
    dim = (tmp.groupby(["variedad", "grado"], dropna=False)
              .agg(
                  n_dias=("pct_grado_dia", "count"),
                  pct_grado=("pct_grado_dia", "median"),
              )
              .reset_index())

    # 4) Normalización para que por variedad sume 1.0 (importante)
    s = dim.groupby("variedad")["pct_grado"].sum().rename("sum_pct").reset_index()
    dim = dim.merge(s, on="variedad", how="left")
    dim["pct_grado"] = np.where(dim["sum_pct"] > 0, dim["pct_grado"] / dim["sum_pct"], dim["pct_grado"])
    dim = dim.drop(columns=["sum_pct"])

    dim["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_dist_grado_baseline.parquet"
    write_parquet(dim, out_path)

    # Auditoría
    chk = dim.groupby("variedad")["pct_grado"].sum().describe()
    print(f"OK: dim_dist_grado_baseline={len(dim)} filas -> {out_path}")
    print("Suma pct por variedad (describe):\n", chk.to_string())


if __name__ == "__main__":
    main()
