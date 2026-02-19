from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


IN_UNIVERSE = Path("data/gold/pred_poscosecha_ml1_dh_grado_dia_bloque_destino.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/hidr_poscosecha")
OUT_PATH = Path("data/gold/pred_poscosecha_ml1_hidr_grado_dia_bloque_destino.parquet")

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
        raise ValueError(f"Universe sin columnas: {sorted(miss)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")
    df["destino"] = df["destino"].astype(str).str.upper().str.strip()

    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    model = load(model_path)
    X = df[NUM_COLS + CAT_COLS]
    pred = model.predict(X)

    lo, hi = meta.get("clip_range_apply", [0.80, 3.00])
    df["factor_hidr_ml1_raw"] = pd.to_numeric(pd.Series(pred), errors="coerce")
    df["factor_hidr_ml1"] = df["factor_hidr_ml1_raw"].clip(lower=float(lo), upper=float(hi))

    df["ml1_hidr_version"] = ver_dir.name
    df["created_at"] = pd.Timestamp.utcnow()

    write_parquet(df, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(df):,} | version={ver_dir.name} | clip={lo}->{hi}")


if __name__ == "__main__":
    main(version=None)
