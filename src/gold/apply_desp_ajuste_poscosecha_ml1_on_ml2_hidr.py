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

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino.parquet"

REG_DESP = ROOT / "models_registry" / "ml1" / "desp_poscosecha"
REG_AJ = ROOT / "models_registry" / "ml1" / "ajuste_poscosecha"

OUT_PATH = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino"]


def _latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _load_meta_and_model(root: Path, version: str | None):
    ver_dir = _latest_version_dir(root) if version is None else (root / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe versión: {ver_dir}")
    with open(ver_dir / "metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load(ver_dir / "model.joblib")
    return ver_dir.name, meta, model


def main(version_desp: str | None = None, version_aj: str | None = None) -> None:
    v_desp, meta_desp, model_desp = _load_meta_and_model(REG_DESP, version_desp)
    v_aj, meta_aj, model_aj = _load_meta_and_model(REG_AJ, version_aj)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino", "fecha_post_pred_ml1", "factor_hidr_ml1", "cajas_split_grado_dia"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Universe Hidr ML2 sin columnas: {sorted(miss)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["fecha_post_pred_ml1"] = pd.to_datetime(df["fecha_post_pred_ml1"], errors="coerce").dt.normalize()
    df["destino"] = df["destino"].astype(str).str.upper().str.strip()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")

    # features calendario por fecha_post_pred_ml1 (día real operativo de proceso)
    df["dow"] = df["fecha_post_pred_ml1"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post_pred_ml1"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post_pred_ml1"].dt.isocalendar().week.astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[NUM_COLS + CAT_COLS]

    # DESP
    lo_d, hi_d = meta_desp.get("clip_range_apply", [0.05, 1.00])
    df["factor_desp_ml1_raw"] = pd.to_numeric(pd.Series(model_desp.predict(X)), errors="coerce")
    df["factor_desp_ml1"] = df["factor_desp_ml1_raw"].clip(lower=float(lo_d), upper=float(hi_d))

    # AJUSTE (ojo: en tu fórmula se usa DIVISIÓN / ajuste_ml1)
    lo_a, hi_a = meta_aj.get("clip_range_apply", [0.80, 1.50])
    df["ajuste_ml1_raw"] = pd.to_numeric(pd.Series(model_aj.predict(X)), errors="coerce")
    df["ajuste_ml1"] = df["ajuste_ml1_raw"].clip(lower=float(lo_a), upper=float(hi_a))

    df["ml1_desp_version"] = v_desp
    df["ml1_ajuste_version"] = v_aj
    df["created_at"] = pd.Timestamp.utcnow()

    # cálculo final poscosecha “ML1” pero alimentado por ML2-campo
    df["cajas_postcosecha_ml1"] = (
        pd.to_numeric(df["cajas_split_grado_dia"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["factor_hidr_ml1"], errors="coerce").fillna(1.0)
        * pd.to_numeric(df["factor_desp_ml1"], errors="coerce").fillna(1.0)
        / pd.to_numeric(df["ajuste_ml1"], errors="coerce").replace(0, np.nan).fillna(1.0)
    )

    write_parquet(df, OUT_PATH)
    print(f"[OK] Wrote: {OUT_PATH} rows={len(df):,} desp={v_desp} ajuste={v_aj}")


if __name__ == "__main__":
    main(version_desp=None, version_aj=None)
