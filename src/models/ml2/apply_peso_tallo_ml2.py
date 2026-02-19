from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"
MODELS = DATA / "models" / "ml2"
EVAL = DATA / "eval" / "ml2"

# Dataset (ya consolidado con ML1 + real + clima)
IN_DS = GOLD / "ml2_datasets" / "ds_peso_tallo_ml2_v1.parquet"

# Outputs PROD
OUT_FACTOR_PROD = GOLD / "factors" / "factor_ml2_peso_tallo_grado_dia.parquet"
OUT_FINAL_PROD = GOLD / "pred_peso_tallo_grado_dia_ml2_final.parquet"

# Outputs BACKTEST
OUT_FACTOR_BT = EVAL / "backtest_factor_ml2_peso_tallo_grado_dia.parquet"
OUT_FINAL_BT = EVAL / "backtest_pred_peso_tallo_grado_dia_ml2_final.parquet"


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _load_latest_model() -> tuple[object, dict]:
    metas = sorted(MODELS.glob("peso_tallo_ml2_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No ML2 meta found in {MODELS}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = MODELS / f"peso_tallo_ml2_{meta['run_id']}.pkl"
    import joblib
    model = joblib.load(model_path)
    return model, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="backtest")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model, meta = _load_latest_model()

    df = read_parquet(IN_DS).copy()

    # Canon
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    for c in ["bloque_base", "variedad_canon", "grado", "tipo_sp", "estado", "area"]:
        if c in df.columns:
            df[c] = _canon_str(df[c])

    # Feature set from meta
    for c in meta.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in meta.get("cat_cols", []):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    feature_cols = meta["num_cols"] + meta["cat_cols"]
    X = df[feature_cols].copy()

    pred_log = model.predict(X).astype(float)
    lo, hi = meta["guardrails"]["clip_log_error"]
    pred_log = np.clip(pred_log, lo, hi)

    df["pred_log_error_peso"] = pred_log
    df["pred_ratio_peso"] = np.exp(df["pred_log_error_peso"])

    # Final
    df["peso_tallo_ml1_g"] = pd.to_numeric(df["peso_tallo_ml1_g"], errors="coerce")
    df["peso_tallo_final_g"] = df["peso_tallo_ml1_g"] * df["pred_ratio_peso"]

    # Versioning
    df["ml2_run_id"] = meta["run_id"]
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    # Outputs: factor y final
    cols_factor = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "peso_tallo_ml1_g", "peso_tallo_real_g",
        "pred_log_error_peso", "pred_ratio_peso", "peso_tallo_final_g",
        "ml2_run_id", "created_at",
    ]
    cols_factor = [c for c in cols_factor if c in df.columns]
    out_factor = df[cols_factor].copy()

    cols_final = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "peso_tallo_ml1_g", "peso_tallo_final_g",
        "ml2_run_id", "created_at",
    ]
    cols_final = [c for c in cols_final if c in df.columns]
    out_final = df[cols_final].copy()

    if args.mode == "prod":
        OUT_FACTOR_PROD.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_PROD)
        write_parquet(out_final, OUT_FINAL_PROD)
        print(f"[OK] PROD factor: {OUT_FACTOR_PROD} rows={len(out_factor):,}")
        print(f"[OK] PROD final : {OUT_FINAL_PROD} rows={len(out_final):,}")
    else:
        EVAL.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_BT)
        write_parquet(out_final, OUT_FINAL_BT)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR_BT} rows={len(out_factor):,}")
        print(f"[OK] BACKTEST final : {OUT_FINAL_BT} rows={len(out_final):,}")


if __name__ == "__main__":
    main()
