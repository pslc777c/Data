# src/silver/build_dim_dh_baseline_grado_destino.py
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    in_path = silver_dir / "fact_hidratacion_real_post_grado_destino.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}. Ejecuta build_hidratacion_real_from_balanza2.py")

    fact = read_parquet(in_path).copy()
    fact.columns = [str(c).strip() for c in fact.columns]

    need = {"dh_dias", "grado", "destino"}
    miss = sorted(list(need - set(fact.columns)))
    if miss:
        raise ValueError(f"fact_hidratacion_real_post_grado_destino sin columnas: {miss}")

    fact["destino"] = fact["destino"].astype(str).str.strip().str.upper()
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce").astype("Int64")
    fact["dh_dias"] = _to_num(fact["dh_dias"]).astype("Int64")

    fact = fact[
        fact["destino"].notna()
        & fact["grado"].notna()
        & fact["dh_dias"].notna()
        & fact["dh_dias"].between(0, 30)
    ].copy()

    out = (
        fact.groupby(["grado", "destino"], dropna=False)
            .agg(
                n=("dh_dias", "size"),
                dh_dias_med=("dh_dias", "median"),
                dh_dias_p25=("dh_dias", lambda s: int(np.nanpercentile(s.astype(float), 25))),
                dh_dias_p75=("dh_dias", lambda s: int(np.nanpercentile(s.astype(float), 75))),
            )
            .reset_index()
    )

    out["dh_dias_med"] = pd.to_numeric(out["dh_dias_med"], errors="coerce").round().astype("Int64")
    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_dh_baseline_grado_destino.parquet"
    write_parquet(out, out_path)

    print(f"OK -> {out_path} | rows={len(out):,}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
