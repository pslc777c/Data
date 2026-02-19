# src/silver/build_dim_hidratacion_fecha_cosecha_grado_destino.py
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

    need = {"fecha_cosecha", "grado", "destino", "hidr_pct", "peso_base_g"}
    miss = sorted(list(need - set(fact.columns)))
    if miss:
        raise ValueError(f"fact_hidratacion_real_post_grado_destino sin columnas: {miss}")

    fact["fecha_cosecha"] = _norm_date(fact["fecha_cosecha"])
    fact["destino"] = fact["destino"].astype(str).str.strip().str.upper()
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce").astype("Int64")
    fact["hidr_pct"] = _to_num(fact["hidr_pct"])
    fact["peso_base_g"] = _to_num(fact["peso_base_g"]).fillna(0.0)

    fact = fact[
        fact["fecha_cosecha"].notna()
        & fact["grado"].notna()
        & fact["destino"].notna()
        & fact["hidr_pct"].notna()
        & (fact["peso_base_g"] > 0)
    ].copy()

    fact["_w"] = fact["peso_base_g"]
    fact["_hydr_w"] = fact["hidr_pct"] * fact["_w"]

    out = (
        fact.groupby(["fecha_cosecha", "grado", "destino"], dropna=False)
            .agg(
                n=("hidr_pct", "size"),
                tallos=("tallos", "sum") if "tallos" in fact.columns else ("hidr_pct", "size"),
                peso_base_g=("peso_base_g", "sum"),
                sum_hydr_w=("_hydr_w", "sum"),
                sum_w=("_w", "sum"),
            )
            .reset_index()
    )

    out["hidr_pct"] = np.where(out["sum_w"] > 0, out["sum_hydr_w"] / out["sum_w"], np.nan)
    out = out.drop(columns=["sum_hydr_w", "sum_w"], errors="ignore")
    out["factor_hidr"] = (1.0 + out["hidr_pct"]).clip(0.8, 3.0)  # cap amplio
    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_hidratacion_fecha_cosecha_grado_destino.parquet"
    write_parquet(out, out_path)

    print(f"OK -> {out_path} | rows={len(out):,}")
    print("factor_hidr describe:\n", out["factor_hidr"].describe().to_string())


if __name__ == "__main__":
    main()
