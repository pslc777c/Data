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
        raise FileNotFoundError(f"No existe: {in_path}. Ejecuta build_fact_peso_tallo_real_grado_dia primero.")

    fact = read_parquet(in_path).copy()

    needed = {"fecha", "variedad", "grado", "peso_tallo_real_g"}
    missing = needed - set(fact.columns)
    if missing:
        raise ValueError(f"Faltan columnas en fact_peso_tallo_real_grado_dia: {missing}")

    fact["variedad"] = fact["variedad"].astype(str).str.strip().str.upper()
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce").astype("Int64")
    fact["peso_tallo_real_g"] = pd.to_numeric(fact["peso_tallo_real_g"], errors="coerce")

    fact = fact[fact["grado"].notna() & fact["peso_tallo_real_g"].notna()].copy()
    fact = fact[(fact["peso_tallo_real_g"] > 1) & (fact["peso_tallo_real_g"] < 500)].copy()

    if fact.empty:
        raise ValueError(
            "fact_peso_tallo_real_grado_dia quedó vacío tras filtros (grado/peso_tallo_real_g). "
            "Revisa el fact upstream y/o los rangos de peso."
        )

    # Baseline robusto por variedad+grado
    dim = (
        fact.groupby(["variedad", "grado"], dropna=False)
            .agg(
                n_dias=("fecha", "nunique"),
                peso_tallo_mediana_g=("peso_tallo_real_g", "median"),
                peso_tallo_p25_g=("peso_tallo_real_g", lambda s: float(np.nanpercentile(s, 25))),
                peso_tallo_p75_g=("peso_tallo_real_g", lambda s: float(np.nanpercentile(s, 75))),
            )
            .reset_index()
    )

    # Guardrail: opcional mínimo de días (default 7)
    min_days = int(cfg.get("pipeline", {}).get("peso_tallo_min_days", 7))
    if int(dim["n_dias"].max()) < min_days:
        raise ValueError(
            f"Baseline peso_tallo: cobertura insuficiente. "
            f"max(n_dias)={int(dim['n_dias'].max())} < min_days={min_days}. "
            "Ejecuta/valida upstream para tener más historia."
        )

    dim["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_peso_tallo_baseline.parquet"
    write_parquet(dim, out_path)

    print(f"OK: dim_peso_tallo_baseline={len(dim)} filas -> {out_path}")
    print("peso_tallo_mediana_g describe:\n", dim["peso_tallo_mediana_g"].describe().to_string())


if __name__ == "__main__":
    main()
