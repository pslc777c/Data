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


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])

    milestones_path = silver_dir / "milestones_ciclo_final.parquet"
    maestro_path = silver_dir / "fact_ciclo_maestro.parquet"
    med_path = silver_dir / "dim_mediana_etapas_tipo_sp_variedad_area.parquet"

    if not milestones_path.exists():
        raise FileNotFoundError(f"No existe: {milestones_path}")
    if not maestro_path.exists():
        raise FileNotFoundError(f"No existe: {maestro_path}")
    if not med_path.exists():
        raise FileNotFoundError(f"No existe: {med_path}")

    m = read_parquet(milestones_path).copy()
    c = read_parquet(maestro_path).copy()
    med = read_parquet(med_path).copy()

    # ---- Maestro
    c["ciclo_id"] = c["ciclo_id"].astype(str)
    for col in ["tipo_sp", "variedad", "area", "estado"]:
        if col not in c.columns:
            c[col] = "UNKNOWN"
        c[col] = _canon_str(c[col])

    var_map = (cfg.get("mappings", {}).get("variedad_map", {}) or {})
    var_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in var_map.items()}
    c["variedad_std"] = c["variedad"].map(lambda x: var_map.get(x, x))

    sp_col = _pick_first_existing(c, ["fecha_sp", "sp_date", "s_p", "sp"])
    if sp_col is None:
        raise ValueError("fact_ciclo_maestro: no encontré columna fecha_sp (o equivalente).")
    c["fecha_sp"] = _to_date(c[sp_col])

    # ---- Milestones reales (pivot)
    m["ciclo_id"] = m["ciclo_id"].astype(str)
    m["milestone_code"] = _canon_str(m["milestone_code"])
    m["fecha"] = _to_date(m["fecha"])

    piv = (
        m.pivot_table(index="ciclo_id", columns="milestone_code", values="fecha", aggfunc="min")
         .reset_index()
    )
    for col in ["VEG_START", "HARVEST_START", "HARVEST_END", "POST_START", "POST_END"]:
        if col in piv.columns:
            piv[col] = _to_date(piv[col])

    # ---- Base por ciclo
    seg = c[["ciclo_id", "tipo_sp", "variedad_std", "area", "fecha_sp", "estado"]].drop_duplicates("ciclo_id")
    base = seg.merge(piv, on="ciclo_id", how="left")

    # ---- Medianas por segmento + fallback global
    for col in ["tipo_sp", "variedad_std", "area"]:
        med[col] = _canon_str(med[col])

    med["mediana_dias_veg"] = pd.to_numeric(med["mediana_dias_veg"], errors="coerce")
    med["mediana_dias_harvest"] = pd.to_numeric(med["mediana_dias_harvest"], errors="coerce")
    if "mediana_dias_post" in med.columns:
        med["mediana_dias_post"] = pd.to_numeric(med["mediana_dias_post"], errors="coerce")
    else:
        med["mediana_dias_post"] = np.nan

    base = base.merge(
        med[["tipo_sp", "variedad_std", "area", "mediana_dias_veg", "mediana_dias_harvest", "mediana_dias_post"]],
        on=["tipo_sp", "variedad_std", "area"],
        how="left",
    )

    g_veg = float(np.nanmedian(med["mediana_dias_veg"].values)) if len(med) else 60.0
    g_harv = float(np.nanmedian(med["mediana_dias_harvest"].values)) if len(med) else 30.0
    g_post = float(np.nanmedian(med["mediana_dias_post"].values)) if len(med) else 180.0

    if not np.isfinite(g_veg):
        g_veg = 60.0
    if not np.isfinite(g_harv):
        g_harv = 30.0
    if not np.isfinite(g_post):
        g_post = 180.0

    base["mediana_dias_veg"] = base["mediana_dias_veg"].fillna(g_veg).clip(0, 180)
    base["mediana_dias_harvest"] = base["mediana_dias_harvest"].fillna(g_harv).clip(1, 180)
    base["mediana_dias_post"] = base["mediana_dias_post"].fillna(g_post).clip(1, 365)

    # ---- Resolver fechas finales (real si existe; si no inferir)
    base["veg_start_final"] = base.get("VEG_START")
    base.loc[base["veg_start_final"].isna(), "veg_start_final"] = base.loc[base["veg_start_final"].isna(), "fecha_sp"]

    base["harvest_start_final"] = base.get("HARVEST_START")
    miss_hs = base["harvest_start_final"].isna() & base["veg_start_final"].notna()
    base.loc[miss_hs, "harvest_start_final"] = base.loc[miss_hs, "veg_start_final"] + pd.to_timedelta(
        base.loc[miss_hs, "mediana_dias_veg"], unit="D"
    )

    base["harvest_end_final"] = base.get("HARVEST_END")
    miss_he = base["harvest_end_final"].isna() & base["harvest_start_final"].notna()
    base.loc[miss_he, "harvest_end_final"] = base.loc[miss_he, "harvest_start_final"] + pd.to_timedelta(
        base.loc[miss_he, "mediana_dias_harvest"] - 1, unit="D"
    )

    base["post_start_final"] = base.get("POST_START")
    miss_ps = base["post_start_final"].isna() & base["harvest_end_final"].notna()
    base.loc[miss_ps, "post_start_final"] = base.loc[miss_ps, "harvest_end_final"] + pd.to_timedelta(1, unit="D")

    base["post_end_final"] = base.get("POST_END")
    miss_pe = base["post_end_final"].isna() & base["post_start_final"].notna()
    base.loc[miss_pe, "post_end_final"] = base.loc[miss_pe, "post_start_final"] + pd.to_timedelta(
        base.loc[miss_pe, "mediana_dias_post"] - 1, unit="D"
    )

    # ---- Ventanas
    rows: list[pd.DataFrame] = []

    def add(stage: str, s_col: str, e_col: str, rule: str) -> None:
        tmp = base[["ciclo_id", s_col, e_col]].copy()
        tmp = tmp.rename(columns={s_col: "start_date", e_col: "end_date"})
        tmp["stage"] = stage
        tmp["rule"] = rule
        rows.append(tmp)

    veg_end = base["harvest_start_final"] - pd.to_timedelta(1, unit="D")
    tmp_veg = base[["ciclo_id", "veg_start_final"]].copy()
    tmp_veg["start_date"] = tmp_veg["veg_start_final"]
    tmp_veg["end_date"] = veg_end
    tmp_veg["stage"] = "VEG"
    tmp_veg["rule"] = "VEG_START (real o SP) -> HARVEST_START-1 (real o inferido)"
    rows.append(tmp_veg[["ciclo_id", "stage", "start_date", "end_date", "rule"]])

    add("HARVEST", "harvest_start_final", "harvest_end_final", "HARVEST (real o inferido)")
    add("POST", "post_start_final", "post_end_final", "POST (real o inferido/regla)")

    win = pd.concat(rows, ignore_index=True)

    win["start_date"] = _to_date(win["start_date"])
    win["end_date"] = _to_date(win["end_date"])

    win = win[win["start_date"].notna() & win["end_date"].notna()].copy()
    win = win[win["start_date"] <= win["end_date"]].copy()

    win["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "milestone_window_ciclo_final.parquet"
    write_parquet(win, out_path)

    print(f"OK: {out_path} | rows={len(win):,}")
    print("Stage counts:\n", win["stage"].value_counts(dropna=False).to_string())

    # auditoría cobertura harvest
    cyc_total = win["ciclo_id"].nunique()
    cyc_h = win[win["stage"].eq("HARVEST")]["ciclo_id"].nunique()
    print("% ciclos con HARVEST (windows):", round(cyc_h / max(cyc_total, 1) * 100, 2))


if __name__ == "__main__":
    main()
