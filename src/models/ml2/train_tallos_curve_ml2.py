from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import uuid

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models" / "ml2"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_DS = GOLD_DIR / "ml2_datasets" / "ds_tallos_curve_ml2_v2.parquet"

# Guardrails (mismo rango del dataset builder)
CLIP_LOG_ERR_LO = -1.5
CLIP_LOG_ERR_HI = 1.5

VAL_QUANTILE = 0.80  # último 20% fechas para validación


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    df = read_parquet(IN_DS).copy()

    # Canon + fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
    if "variedad_canon" in df.columns:
        df["variedad_canon"] = _canon_str(df["variedad_canon"])
    if "tipo_sp" in df.columns:
        df["tipo_sp"] = _canon_str(df["tipo_sp"])
    if "estado" in df.columns:
        df["estado"] = _canon_str(df["estado"])

    # Target
    y = pd.to_numeric(df["log_error"], errors="coerce")
    m = y.notna() & df["fecha"].notna()
    df = df.loc[m, :].copy()
    y = y.loc[m].astype(float).clip(CLIP_LOG_ERR_LO, CLIP_LOG_ERR_HI)

    # Features (num + cat)
    num_cols = [
        "rel_pos_final",
        "day_in_harvest_final",
        "n_harvest_days_final",
        "tallos_pred_ml1_dia",
        "tallos_real_dia",
        # clima día
        "gdc_dia",
        "rainfall_mm_dia",
        "en_lluvia_dia",
        "temp_avg_dia",
        "solar_energy_j_m2_dia",
        "wind_speed_avg_dia",
        "wind_run_dia",
        # calendario
        "dow", "month", "weekofyear",
    ]
    cat_cols = [
        "bloque_base",
        "variedad_canon",
        "area",
        "tipo_sp",
        "estado",
    ]

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Limpieza
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("")

    X = df[num_cols + cat_cols].copy()

    # Split temporal por fecha
    cutoff = df["fecha"].quantile(VAL_QUANTILE)
    is_val = df["fecha"] >= cutoff

    Xtr, ytr = X.loc[~is_val], y.loc[~is_val]
    Xva, yva = X.loc[is_val], y.loc[is_val]

    # OHE denso
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=6,
        learning_rate=0.05,
        max_iter=350,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(Xtr, ytr)

    # Métricas sobre log_error
    pred_tr = pipe.predict(Xtr)
    pred_va = pipe.predict(Xva)

    mae_tr = float(mean_absolute_error(ytr, pred_tr))
    mae_va = float(mean_absolute_error(yva, pred_va))

    # Diagnóstico: % clipping si se aplica (aún no lo aplicamos aquí)
    pred_va_clip = np.clip(pred_va, CLIP_LOG_ERR_LO, CLIP_LOG_ERR_HI)
    pct_clip_va = float(np.mean((pred_va <= CLIP_LOG_ERR_LO) | (pred_va >= CLIP_LOG_ERR_HI))) if len(pred_va) else np.nan
    mae_va_clip = float(mean_absolute_error(yva, pred_va_clip)) if len(pred_va) else np.nan

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    import joblib
    model_path = MODELS_DIR / f"tallos_curve_ml2_{run_id}.pkl"
    joblib.dump(pipe, model_path)

    # Save meta
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(IN_DS),
        "target": "log_error = log((tallos_real_dia+eps)/(tallos_pred_ml1_dia+eps))",
        "guardrails": {"clip_log_error": [CLIP_LOG_ERR_LO, CLIP_LOG_ERR_HI]},
        "split": {"type": "temporal_quantile", "val_quantile": VAL_QUANTILE, "cutoff_fecha": str(pd.to_datetime(cutoff).date())},
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "metrics": {
            "mae_train_log": mae_tr,
            "mae_val_log": mae_va,
            "mae_val_log_clipped": mae_va_clip,
            "pct_clip_val": pct_clip_va,
            "n_train": int(len(ytr)),
            "n_val": int(len(yva)),
        },
    }
    meta_path = MODELS_DIR / f"tallos_curve_ml2_{run_id}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save eval row
    eval_row = pd.DataFrame([{
        "run_id": run_id,
        "cutoff_fecha": pd.to_datetime(cutoff).normalize(),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "mae_train_log": mae_tr,
        "mae_val_log": mae_va,
        "mae_val_log_clipped": mae_va_clip,
        "pct_clip_val": pct_clip_va,
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])
    eval_path = EVAL_DIR / "ml2_tallos_curve_train_cv.parquet"
    if eval_path.exists():
        old = read_parquet(eval_path)
        eval_row = pd.concat([old, eval_row], ignore_index=True)
    write_parquet(eval_row, eval_path)

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Meta  saved: {meta_path}")
    print(f"[OK] Eval  saved: {eval_path}")
    print(f"     MAE log train={mae_tr:.4f}  val={mae_va:.4f}  val_clipped={mae_va_clip:.4f}  pct_clip_val={pct_clip_va:.3%}")


if __name__ == "__main__":
    main()
