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


def _ensure_required_cols(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {sorted(missing)}")


def build_fact_milestones(fact_ciclo: pd.DataFrame, dh_days: int, post_tail_days: int) -> pd.DataFrame:
    """
    Devuelve tabla larga:
      ciclo_id, milestone_code, fecha, source, created_at
    """
    df = fact_ciclo.copy()

    required = {"ciclo_id", "fecha_sp", "fecha_inicio_cosecha", "fecha_fin_cosecha"}
    _ensure_required_cols(df, required, "fact_ciclo_maestro")

    df["ciclo_id"] = df["ciclo_id"].astype(str).str.strip()
    df["fecha_sp"] = _norm_date(df["fecha_sp"])
    df["fecha_inicio_cosecha"] = _norm_date(df["fecha_inicio_cosecha"])
    df["fecha_fin_cosecha"] = _norm_date(df["fecha_fin_cosecha"])

    rows: list[pd.DataFrame] = []

    def add_milestone(code: str, fecha: pd.Series, source: str) -> None:
        tmp = df[["ciclo_id"]].copy()
        tmp["milestone_code"] = code
        tmp["fecha"] = _norm_date(fecha)
        tmp["source"] = source
        rows.append(tmp)

    add_milestone("VEG_START", df["fecha_sp"], "fenograma")
    add_milestone("HARVEST_START", df["fecha_inicio_cosecha"], "fenograma/balanza")
    add_milestone("HARVEST_END", df["fecha_fin_cosecha"], "fenograma/balanza")

    post_start = df["fecha_inicio_cosecha"] + pd.to_timedelta(int(dh_days), unit="D")
    add_milestone("POST_START", post_start, f"rule:harvest_start+{int(dh_days)}")

    post_end = df["fecha_fin_cosecha"] + pd.to_timedelta(int(dh_days) + int(post_tail_days), unit="D")
    add_milestone("POST_END", post_end, f"rule:harvest_end+{int(dh_days)}+{int(post_tail_days)}")

    milestones = pd.concat(rows, ignore_index=True)

    milestones = milestones[milestones["fecha"].notna()].copy()
    milestones["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Unicidad por ciclo_id + milestone_code
    milestones = (
        milestones.sort_values(["ciclo_id", "milestone_code", "fecha"], ascending=[True, True, True])
        .drop_duplicates(subset=["ciclo_id", "milestone_code"], keep="first")
        .reset_index(drop=True)
    )

    return milestones


def build_windows_from_milestones(fact_ciclo: pd.DataFrame, dh_days: int, post_tail_days: int) -> pd.DataFrame:
    """
    Ventanas por etapa en formato largo:
      ciclo_id, stage, start_date, end_date, rule
    Reglas:
      VEG: fecha_sp -> (harvest_start - 1)
      HARVEST: harvest_start -> harvest_end
      POST: (harvest_start+dh) -> (harvest_end+dh+tail)
    """
    df = fact_ciclo.copy()

    required = {"ciclo_id", "fecha_sp", "fecha_inicio_cosecha", "fecha_fin_cosecha"}
    _ensure_required_cols(df, required, "fact_ciclo_maestro")

    df["ciclo_id"] = df["ciclo_id"].astype(str).str.strip()
    df["fecha_sp"] = _norm_date(df["fecha_sp"])
    df["fecha_inicio_cosecha"] = _norm_date(df["fecha_inicio_cosecha"])
    df["fecha_fin_cosecha"] = _norm_date(df["fecha_fin_cosecha"])

    # VEG
    veg_start = df["fecha_sp"]
    veg_end = df["fecha_inicio_cosecha"] - pd.to_timedelta(1, unit="D")
    veg_end = veg_end.where(df["fecha_inicio_cosecha"].notna(), pd.NaT)

    # HARVEST
    harv_start = df["fecha_inicio_cosecha"]
    harv_end = df["fecha_fin_cosecha"]

    # POST
    post_start = df["fecha_inicio_cosecha"] + pd.to_timedelta(int(dh_days), unit="D")
    post_end = df["fecha_fin_cosecha"] + pd.to_timedelta(int(dh_days) + int(post_tail_days), unit="D")
    post_start = post_start.where(df["fecha_inicio_cosecha"].notna(), pd.NaT)
    post_end = post_end.where(df["fecha_fin_cosecha"].notna(), pd.NaT)

    windows: list[pd.DataFrame] = []

    def add_window(stage: str, s: pd.Series, e: pd.Series, rule: str) -> None:
        tmp = df[["ciclo_id"]].copy()
        tmp["stage"] = stage
        tmp["start_date"] = _norm_date(s)
        tmp["end_date"] = _norm_date(e)
        tmp["rule"] = rule
        windows.append(tmp)

    add_window("VEG", veg_start, veg_end, "fecha_sp -> harvest_start-1")
    add_window("HARVEST", harv_start, harv_end, "harvest_start -> harvest_end")
    add_window("POST", post_start, post_end, f"harvest_start+{int(dh_days)} -> harvest_end+{int(dh_days)}+{int(post_tail_days)}")

    out = pd.concat(windows, ignore_index=True)

    # Quitar ventanas sin start_date
    out = out[out["start_date"].notna()].copy()

    # Regla de integridad: end_date no puede ser < start_date
    bad = out["end_date"].notna() & (out["end_date"] < out["start_date"])
    out.loc[bad, "end_date"] = pd.NaT

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Dedup defensivo
    out = (
        out.sort_values(["ciclo_id", "stage", "start_date"], ascending=[True, True, True])
        .drop_duplicates(subset=["ciclo_id", "stage"], keep="first")
        .reset_index(drop=True)
    )

    return out


def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    fact_path = silver_dir / "fact_ciclo_maestro.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"No existe: {fact_path}")

    fact = read_parquet(fact_path)

    dh_days = int(cfg.get("milestones", {}).get("dh_days", 7))
    post_tail_days = int(cfg.get("milestones", {}).get("post_tail_days", 2))

    milestones = build_fact_milestones(fact, dh_days=dh_days, post_tail_days=post_tail_days)
    windows = build_windows_from_milestones(fact, dh_days=dh_days, post_tail_days=post_tail_days)

    write_parquet(milestones, silver_dir / "fact_milestones_ciclo.parquet")
    write_parquet(windows, silver_dir / "milestone_window_ciclo.parquet")

    print(f"OK milestones: {len(milestones)} filas -> fact_milestones_ciclo.parquet")
    print(f"OK windows:    {len(windows)} filas -> milestone_window_ciclo.parquet")
    print(f"Reglas: DH={dh_days}, post_tail={post_tail_days}")


if __name__ == "__main__":
    main()
