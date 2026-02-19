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


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])

    milestones_path = silver_dir / "milestones_ciclo_final.parquet"
    maestro_path = silver_dir / "fact_ciclo_maestro.parquet"

    if not milestones_path.exists():
        raise FileNotFoundError(f"No existe: {milestones_path}")
    if not maestro_path.exists():
        raise FileNotFoundError(f"No existe: {maestro_path}")

    m = read_parquet(milestones_path).copy()
    c = read_parquet(maestro_path).copy()

    # Normalizar
    m["ciclo_id"] = m["ciclo_id"].astype(str)
    m["milestone_code"] = _canon_str(m["milestone_code"])
    m["fecha"] = _to_date(m["fecha"])

    c["ciclo_id"] = c["ciclo_id"].astype(str)
    for col in ["tipo_sp", "variedad", "area"]:
        if col not in c.columns:
            c[col] = "UNKNOWN"
        c[col] = _canon_str(c[col])

    # Mapa variedad (si existe)
    var_map = (cfg.get("mappings", {}).get("variedad_map", {}) or {})
    var_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in var_map.items()}
    c["variedad_std"] = c["variedad"].map(lambda x: var_map.get(x, x))

    # Pivot milestones
    piv = (
        m.pivot_table(index="ciclo_id", columns="milestone_code", values="fecha", aggfunc="min")
         .reset_index()
    )
    for col in ["VEG_START", "HARVEST_START", "HARVEST_END", "POST_START", "POST_END"]:
        if col in piv.columns:
            piv[col] = _to_date(piv[col])

    # Traer segmentación
    seg = c[["ciclo_id", "tipo_sp", "variedad_std", "area"]].drop_duplicates("ciclo_id")
    piv = piv.merge(seg, on="ciclo_id", how="left")

    # Duraciones
    piv["dias_veg"] = (piv["HARVEST_START"] - piv["VEG_START"]).dt.days
    piv["dias_harvest"] = (piv["HARVEST_END"] - piv["HARVEST_START"]).dt.days + 1
    piv["dias_post"] = (piv["POST_END"] - piv["POST_START"]).dt.days + 1

    # Limpieza básica
    piv.loc[piv["dias_veg"] < 0, "dias_veg"] = np.nan
    piv.loc[piv["dias_harvest"] <= 0, "dias_harvest"] = np.nan
    piv.loc[piv["dias_post"] <= 0, "dias_post"] = np.nan

    base = piv[["tipo_sp", "variedad_std", "area", "dias_veg", "dias_harvest", "dias_post"]].copy()
    base = base.dropna(subset=["dias_veg", "dias_harvest"])  # post puede ser opcional

    if base.empty:
        raise ValueError("No hay ciclos con duraciones válidas para calcular medianas.")

    out = (
        base.groupby(["tipo_sp", "variedad_std", "area"], dropna=False)
            .agg(
                mediana_dias_veg=("dias_veg", "median"),
                mediana_dias_harvest=("dias_harvest", "median"),
                mediana_dias_post=("dias_post", "median"),
                n=("dias_veg", "count"),
            )
            .reset_index()
    )

    # Caps razonables (evita basura)
    out["mediana_dias_veg"] = pd.to_numeric(out["mediana_dias_veg"], errors="coerce").clip(0, 180)
    out["mediana_dias_harvest"] = pd.to_numeric(out["mediana_dias_harvest"], errors="coerce").clip(1, 180)
    out["mediana_dias_post"] = pd.to_numeric(out["mediana_dias_post"], errors="coerce").clip(1, 365)

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_mediana_etapas_tipo_sp_variedad_area.parquet"
    write_parquet(out, out_path)

    print(f"OK: {out_path} | rows={len(out):,}")
    print(out[["n"]].describe().to_string())


if __name__ == "__main__":
    main()
