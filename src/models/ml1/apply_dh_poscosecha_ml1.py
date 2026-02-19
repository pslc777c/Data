from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


IN_UNIVERSE = Path("data/gold/pred_poscosecha_seed_grado_dia_bloque_destino.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/dh_poscosecha")
OUT_PATH = Path("data/gold/pred_poscosecha_ml1_dh_grado_dia_bloque_destino.parquet")

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]


def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def main(version: str | None = None) -> None:
    if version is None:
        ver_dir = _latest_version_dir()
    else:
        ver_dir = REGISTRY_ROOT / version
        if not ver_dir.exists():
            raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_path = ver_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe: {model_path}")

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Universe seed sin columnas: {sorted(miss)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")
    df["destino"] = df["destino"].astype(str).str.upper().str.strip()

    # Features calendario sobre fecha (cosecha proyectada)
    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    # asegurar cols
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    model = load(model_path)

    X = df[NUM_COLS + CAT_COLS]
    pred = model.predict(X)

    # clip + redondeo a int días
    dh = pd.to_numeric(pd.Series(pred), errors="coerce").clip(lower=0, upper=30)
    df["dh_dias_ml1_raw"] = dh
    df["dh_dias_ml1"] = np.rint(df["dh_dias_ml1_raw"]).astype("Int64")

    # fecha_post_pred_ml1
    df["fecha_post_pred_ml1"] = df["fecha"] + pd.to_timedelta(df["dh_dias_ml1"].fillna(0).astype(int), unit="D")

    df["ml1_dh_version"] = ver_dir.name
    df["created_at"] = pd.Timestamp.utcnow()

    # Mantener todo y agregar columnas ML1
    keep_extra = ["dh_dias_ml1_raw", "dh_dias_ml1", "fecha_post_pred_ml1", "ml1_dh_version", "created_at"]
    for c in keep_extra:
        if c not in df.columns:
            df[c] = pd.NA

    write_parquet(df, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(df):,} | version={ver_dir.name} | clip={meta.get('clip_range_apply')}")


if __name__ == "__main__":
    main(version=None)
