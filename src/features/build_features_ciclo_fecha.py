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


def add_semana_ecuador(df: pd.DataFrame, col_fecha: str = "fecha") -> pd.Series:
    """
    Semana_ como tú la has manejado (ajustando +2 días y semana empezando Sunday).
    Devuelve YYWW en texto.
    """
    f = pd.to_datetime(df[col_fecha], errors="coerce")
    f2 = f + pd.to_timedelta(2, unit="D")
    yy = (f2.dt.year % 100).astype("Int64").astype(str).str.zfill(2)
    ww = f2.dt.isocalendar().week.astype("Int64").astype(str).str.zfill(2)
    return yy + ww


def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    features_dir = Path(cfg["paths"]["features"])
    features_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    fact_path = silver_dir / "fact_ciclo_maestro.parquet"
    grid_path = silver_dir / "grid_ciclo_fecha.parquet"
    milestones_path = silver_dir / "fact_milestones_ciclo.parquet"
    windows_path = silver_dir / "milestone_window_ciclo_final.parquet"

    for p in [fact_path, grid_path, milestones_path, windows_path]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    fact = read_parquet(fact_path)
    grid = read_parquet(grid_path)
    milestones = read_parquet(milestones_path)
    windows = read_parquet(windows_path)

    # Normalizar fechas
    grid["fecha"] = _norm_date(grid["fecha"])
    milestones["fecha"] = _norm_date(milestones["fecha"])
    windows["start_date"] = _norm_date(windows["start_date"])
    windows["end_date"] = _norm_date(windows["end_date"])

    # 1) Base: grid + atributos del ciclo
    cols_fact = [
        "ciclo_id",
        "bloque", "bloque_padre",
        "variedad", "tipo_sp",
        "area", "estado",
        "fecha_sp",
        "fecha_inicio_cosecha",
        "fecha_fin_cosecha",
        "tallos_proy",
    ]
    cols_fact = [c for c in cols_fact if c in fact.columns]

    base = grid.merge(fact[cols_fact], on="ciclo_id", how="left")

    # 2) Pivot de milestones para tener columnas (harvest_start, post_start, etc.)
    piv = (milestones.pivot_table(
            index="ciclo_id",
            columns="milestone_code",
            values="fecha",
            aggfunc="min"
        ).reset_index())

    # Renombres canónicos
    rename_map = {
        "VEG_START": "veg_start",
        "HARVEST_START": "harvest_start",
        "HARVEST_END": "harvest_end",
        "POST_START": "post_start",
        "POST_END": "post_end",
    }
    for k, v in rename_map.items():
        if k in piv.columns:
            piv = piv.rename(columns={k: v})

    base = base.merge(piv, on="ciclo_id", how="left")

    # 3) Etiquetar stage por ventanas (VEG/HARVEST/POST)
    #    Regla: si fecha entre [start_date, end_date] (end inclusive). Si end_date es NaT => ventana abierta.
    #    Precedencia: POST > HARVEST > VEG (para evitar solapes).
    windows2 = windows.copy()
    windows2["end_date_filled"] = windows2["end_date"].fillna(pd.Timestamp("2100-01-01"))

    # Join expandido (puede ser pesado si ventanas están mal; con 180 días suele estar OK)
    # Creamos stage por ciclo/fecha aplicando condiciones por stage.
    base["stage"] = "OUT"

    def apply_stage(stage_name: str):
        w = windows2[windows2["stage"] == stage_name][["ciclo_id", "start_date", "end_date_filled"]]
        tmp = base.merge(w, on="ciclo_id", how="left", suffixes=("", "_w"))
        cond = (tmp["fecha"] >= tmp["start_date"]) & (tmp["fecha"] <= tmp["end_date_filled"])
        return cond

    # Precedencia: VEG primero, luego HARVEST sobreescribe, luego POST sobreescribe
    veg_cond = apply_stage("VEG")
    base.loc[veg_cond, "stage"] = "VEG"

    harv_cond = apply_stage("HARVEST")
    base.loc[harv_cond, "stage"] = "HARVEST"

    post_cond = apply_stage("POST")
    base.loc[post_cond, "stage"] = "POST"

    # 4) Features de tiempo (días)
    # d_desde_sp
    if "fecha_sp" in base.columns:
        base["fecha_sp"] = _norm_date(base["fecha_sp"])
        base["d_desde_sp"] = (base["fecha"] - base["fecha_sp"]).dt.days.astype("Int64")
    else:
        base["d_desde_sp"] = pd.Series([pd.NA] * len(base), dtype="Int64")

    # distancias a hitos (si existen)
    for col in ["harvest_start", "post_start"]:
        if col in base.columns:
            base[f"d_hasta_{col}"] = (base[col] - base["fecha"]).dt.days.astype("Int64")
        else:
            base[f"d_hasta_{col}"] = pd.Series([pd.NA] * len(base), dtype="Int64")

    # 5) Calendario
    base["anio"] = base["fecha"].dt.year.astype("Int64")
    base["mes"] = base["fecha"].dt.month.astype("Int64")
    base["semana_"] = add_semana_ecuador(base, "fecha")

    # 6) Auditoría mínima
    base["created_at"] = datetime.now().isoformat(timespec="seconds")

    # 7) Escritura
    out_path = features_dir / "features_ciclo_fecha.parquet"
    write_parquet(base, out_path)

    print(f"OK: features_ciclo_fecha={len(base)} filas -> {out_path}")
    print("Stage counts:\n", base["stage"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
