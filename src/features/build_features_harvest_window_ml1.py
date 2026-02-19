from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import yaml

from common.io import read_parquet, write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# Helpers
# -------------------------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _pick_first(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg.get("paths", {}).get("silver", "data/silver"))
    features_dir = Path(cfg.get("paths", {}).get("features", "data/features"))
    features_dir.mkdir(parents=True, exist_ok=True)

    maestro_path = silver_dir / "fact_ciclo_maestro.parquet"
    if not maestro_path.exists():
        raise FileNotFoundError(f"No existe: {maestro_path}")

    df = read_parquet(maestro_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # ---- columnas base
    _require(df, ["ciclo_id"], "fact_ciclo_maestro")
    df["ciclo_id"] = df["ciclo_id"].astype(str)

    # llaves / categóricas
    if "bloque_base" not in df.columns:
        # fallback típicos
        bcol = _pick_first(df, ["bloque_base", "bloque_padre", "bloque"])
        if bcol is None:
            raise ValueError("fact_ciclo_maestro: no encontré bloque_base/bloque_padre/bloque")
        df["bloque_base"] = df[bcol]
    df["bloque_base"] = _canon_int(df["bloque_base"])

    if "tipo_sp" in df.columns:
        df["tipo_sp"] = _canon_str(df["tipo_sp"])
    else:
        df["tipo_sp"] = "UNKNOWN"

    if "area" in df.columns:
        df["area"] = _canon_str(df["area"])
    else:
        df["area"] = "UNKNOWN"

    # variedad_canon: la usaremos sí o sí
    # (asumo que ya existe en maestro; si no existe, intenta desde variedad)
    if "variedad_canon" in df.columns:
        df["variedad_canon"] = _canon_str(df["variedad_canon"])
    elif "variedad" in df.columns:
        # si aquí necesitas map con dim_variedad_canon, lo integramos luego;
        # por ahora canonizo directo para no romper.
        df["variedad_canon"] = _canon_str(df["variedad"])
    else:
        df["variedad_canon"] = "UNKNOWN"

    # tallos_proy opcional (puede ayudar)
    if "tallos_proy" in df.columns:
        df["tallos_proy"] = pd.to_numeric(df["tallos_proy"], errors="coerce")
    else:
        df["tallos_proy"] = np.nan

    # fechas: según tu schema real
    sp_col = _pick_first(df, ["fecha_sp", "sp_date", "fecha_siembra", "fecha_sp_real"])
    hs_col = _pick_first(df, ["fecha_inicio_cosecha", "harvest_start", "inicio_cosecha", "fecha_inicio_real"])
    he_col = _pick_first(df, ["fecha_fin_cosecha", "harvest_end_eff", "harvest_end", "fin_cosecha", "fecha_fin_real"])

    if sp_col is None:
        raise ValueError("fact_ciclo_maestro: no encontré columna de fecha SP (ej: fecha_sp).")

    df["fecha_sp"] = _to_date(df[sp_col])
    df["harvest_start_real"] = _to_date(df[hs_col]) if hs_col is not None else pd.NaT
    df["harvest_end_real"] = _to_date(df[he_col]) if he_col is not None else pd.NaT

    # calendario de SP (para estacionalidad)
    df["sp_month"] = df["fecha_sp"].dt.month
    df["sp_weekofyear"] = df["fecha_sp"].dt.isocalendar().week.astype("Int64")
    df["sp_doy"] = df["fecha_sp"].dt.dayofyear
    df["sp_dow"] = df["fecha_sp"].dt.dayofweek

    # targets reales (solo donde exista)
    df["d_start_real"] = (df["harvest_start_real"] - df["fecha_sp"]).dt.days
    df["n_harvest_days_real"] = (df["harvest_end_real"] - df["harvest_start_real"]).dt.days + 1

    # limpieza targets
    df.loc[df["d_start_real"] < 0, "d_start_real"] = np.nan
    df.loc[df["n_harvest_days_real"] <= 0, "n_harvest_days_real"] = np.nan

    # caps razonables (evitar basura)
    df["d_start_real"] = pd.to_numeric(df["d_start_real"], errors="coerce").clip(0, 180)
    df["n_harvest_days_real"] = pd.to_numeric(df["n_harvest_days_real"], errors="coerce").clip(1, 180)

    # salida: 1 fila por ciclo
    out = df[
        [
            "ciclo_id",
            "bloque_base",
            "variedad_canon",
            "area",
            "tipo_sp",
            "tallos_proy",
            "fecha_sp",
            "sp_month",
            "sp_weekofyear",
            "sp_doy",
            "sp_dow",
            "harvest_start_real",
            "harvest_end_real",
            "d_start_real",
            "n_harvest_days_real",
        ]
    ].drop_duplicates(subset=["ciclo_id"]).copy()

    out["created_at"] = pd.Timestamp.utcnow()

    out_path = features_dir / "features_harvest_window_ml1.parquet"
    write_parquet(out, out_path)

    # prints útiles
    n_total = len(out)
    n_start = int(out["d_start_real"].notna().sum())
    n_days = int(out["n_harvest_days_real"].notna().sum())
    print(f"OK -> {out_path} | rows={n_total:,}")
    print(f"[COVERAGE] d_start_real notna: {n_start:,} ({n_start/max(n_total,1):.2%})")
    print(f"[COVERAGE] n_harvest_days_real notna: {n_days:,} ({n_days/max(n_total,1):.2%})")

    # segmentos chicos (para tu warning)
    seg = (
        out[out["d_start_real"].notna() & out["n_harvest_days_real"].notna()]
        .groupby(["area", "variedad_canon", "tipo_sp"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("n")
    )
    small = seg[seg["n"] < 20].head(50)
    if len(small):
        print("[WARN] segmentos con n < 20 (ML1 puede generalizar peor; pooling/regularización):")
        print(small.to_string(index=False))


if __name__ == "__main__":
    main()
