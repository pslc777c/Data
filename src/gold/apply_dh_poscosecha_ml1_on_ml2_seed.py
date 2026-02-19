from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_seed_grado_dia_bloque_destino.parquet"
REGISTRY_ROOT = ROOT / "models_registry" / "ml1" / "dh_poscosecha"
OUT_PATH = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]


def _latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir(REGISTRY_ROOT) if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe versión: {ver_dir}")

    with open(ver_dir / "metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load(ver_dir / "model.joblib")

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino", "cajas_split_grado_dia"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Seed ML2 sin columnas: {sorted(miss)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")
    df["destino"] = df["destino"].astype(str).str.upper().str.strip()

    # features calendario sobre fecha (cosecha)
    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[NUM_COLS + CAT_COLS]
    pred = model.predict(X)

    # clip + round
    clip_lo, clip_hi = meta.get("clip_range_apply", [0, 30])
    dh = pd.to_numeric(pd.Series(pred), errors="coerce").clip(lower=float(clip_lo), upper=float(clip_hi))
    df["dh_dias_ml1_raw"] = dh
    df["dh_dias_ml1"] = np.rint(df["dh_dias_ml1_raw"]).astype("Int64")

    df["fecha_post_pred_ml1"] = df["fecha"] + pd.to_timedelta(df["dh_dias_ml1"].fillna(0).astype(int), unit="D")

    df["ml1_dh_version"] = ver_dir.name
    df["created_at"] = pd.Timestamp.utcnow()

    write_parquet(df, OUT_PATH)
    print(f"[OK] Wrote: {OUT_PATH} rows={len(df):,} version={ver_dir.name}")


if __name__ == "__main__":
    main(version=None)
