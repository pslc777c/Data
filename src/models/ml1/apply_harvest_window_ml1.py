from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


FEATURES_PATH = Path("data/features/features_harvest_window_ml1.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/harvest_window")

MEDIANS_PATH = Path("data/silver/dim_mediana_etapas_tipo_sp_variedad_area.parquet")

OUT_PATH = Path("data/gold/pred_harvest_window_ml1.parquet")


# -------------------------
# Helpers
# -------------------------
def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


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


# -------------------------
# Main
# -------------------------
def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir() if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {ver_dir}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_start_path = ver_dir / "model_start_offset.joblib"
    model_days_path = ver_dir / "model_harvest_days.joblib"
    if not model_start_path.exists():
        raise FileNotFoundError(f"No encontré: {model_start_path}")
    if not model_days_path.exists():
        raise FileNotFoundError(f"No encontré: {model_days_path}")

    model_start = load(model_start_path)
    model_days = load(model_days_path)

    # -------------------------
    # Features base (ciclos a predecir)
    # -------------------------
    df = read_parquet(FEATURES_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"ciclo_id", "bloque_base", "fecha_sp", "area", "tipo_sp"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features_harvest_window_ml1: faltan columnas {sorted(miss)}")

    # variedad_canon: si no existe, fallback
    if "variedad_canon" not in df.columns:
        vcol = _pick_first(df, ["variedad_std", "variedad", "variedad_raw", "variedad_id"])
        if vcol is None:
            df["variedad_canon"] = "UNKNOWN"
        else:
            df["variedad_canon"] = df[vcol]

    df["ciclo_id"] = df["ciclo_id"].astype(str)
    df["fecha_sp"] = _to_date(df["fecha_sp"])
    df["bloque_base"] = _canon_int(df["bloque_base"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["area"] = _canon_str(df["area"])
    df["tipo_sp"] = _canon_str(df["tipo_sp"])

    # -------------------------
    # Medianas (fallback): tu esquema real
    # -------------------------
    med = read_parquet(MEDIANS_PATH).copy()
    med.columns = [str(c).strip() for c in med.columns]

    # columnas obligatorias del archivo real
    need_med = {"tipo_sp", "variedad_std", "area", "mediana_dias_veg", "mediana_dias_harvest"}
    miss_med = need_med - set(med.columns)
    if miss_med:
        raise ValueError(
            "dim_mediana_etapas_tipo_sp_variedad_area: faltan columnas esperadas "
            f"{sorted(miss_med)}. Cols={list(med.columns)}"
        )

    med["area"] = _canon_str(med["area"])
    med["tipo_sp"] = _canon_str(med["tipo_sp"])
    med["variedad_canon"] = _canon_str(med["variedad_std"])

    # map a nombres estándar que usa el apply
    med["d_start_med"] = pd.to_numeric(med["mediana_dias_veg"], errors="coerce")
    med["n_days_med"] = pd.to_numeric(med["mediana_dias_harvest"], errors="coerce")

    med = med[["area", "variedad_canon", "tipo_sp", "d_start_med", "n_days_med"]].drop_duplicates()

    # -------------------------
    # Predicción ML1
    # -------------------------
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])

    # asegurar columnas faltantes
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[num_cols + cat_cols]
    d_start_pred = pd.to_numeric(model_start.predict(X), errors="coerce")
    n_days_pred = pd.to_numeric(model_days.predict(X), errors="coerce")

    # clips duros
    d_start_pred = np.clip(d_start_pred, 0, 180)
    n_days_pred = np.clip(n_days_pred, 1, 180)

    out = df[["ciclo_id", "bloque_base", "variedad_canon", "area", "tipo_sp", "fecha_sp"]].copy()
    out["ml1_version"] = ver_dir.name
    out["d_start_pred_ml1"] = d_start_pred
    out["n_harvest_days_pred_ml1"] = n_days_pred

    # -------------------------
    # Fallback a mediana por segmento
    # -------------------------
    out = out.merge(med, on=["area", "variedad_canon", "tipo_sp"], how="left")

    out["d_start_pred_final"] = out["d_start_pred_ml1"]
    out["n_harvest_days_pred_final"] = out["n_harvest_days_pred_ml1"]

    miss_start = out["d_start_pred_final"].isna()
    miss_days = out["n_harvest_days_pred_final"].isna()

    out.loc[miss_start, "d_start_pred_final"] = out.loc[miss_start, "d_start_med"]
    out.loc[miss_days, "n_harvest_days_pred_final"] = out.loc[miss_days, "n_days_med"]

    # fallback global final
    out["d_start_pred_final"] = out["d_start_pred_final"].fillna(90.0).clip(0, 180)
    out["n_harvest_days_pred_final"] = out["n_harvest_days_pred_final"].fillna(40.0).clip(1, 180)

    out["harvest_start_pred"] = out["fecha_sp"] + pd.to_timedelta(out["d_start_pred_final"], unit="D")
    out["harvest_end_pred"] = out["harvest_start_pred"] + pd.to_timedelta(out["n_harvest_days_pred_final"] - 1, unit="D")

    out["start_source"] = np.where(miss_start, "median_segment", "ml1_model")
    out["days_source"] = np.where(miss_days, "median_segment", "ml1_model")

    out["created_at"] = pd.Timestamp.utcnow()

    out = out[
        [
            "ciclo_id",
            "bloque_base",
            "variedad_canon",
            "area",
            "tipo_sp",
            "fecha_sp",
            "harvest_start_pred",
            "harvest_end_pred",
            "d_start_pred_final",
            "n_harvest_days_pred_final",
            "start_source",
            "days_source",
            "ml1_version",
            "created_at",
        ]
    ].sort_values(["bloque_base", "variedad_canon", "fecha_sp"]).reset_index(drop=True)

    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={ver_dir.name}")
    print(f"[COVERAGE] start_source=ml1_model: {float((out['start_source']=='ml1_model').mean()):.2%}")
    print(f"[COVERAGE] days_source=ml1_model: {float((out['days_source']=='ml1_model').mean()):.2%}")

    fmin = pd.to_datetime(out["harvest_start_pred"].min()).date() if len(out) else None
    fmax = pd.to_datetime(out["harvest_end_pred"].max()).date() if len(out) else None
    print(f"[RANGE] harvest_start_pred..harvest_end_pred: {fmin} .. {fmax}")


if __name__ == "__main__":
    main(version=None)
