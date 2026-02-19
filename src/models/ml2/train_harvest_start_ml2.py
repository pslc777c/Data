# src/models/ml2/train_harvest_start_ml2.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import uuid
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models" / "ml2"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

DS_PATH = GOLD_DIR / "ml2_datasets" / "ds_harvest_start_ml2_v2.parquet"
OUT_KPIS = EVAL_DIR / "ml2_harvest_start_train_cv.parquet"

TARGET = "error_start_days"

NUM_COLS = [
    "days_sp_to_start_pred",
    "n_harvest_days_pred",
    "tallos_proy",
    "dow", "month", "weekofyear",
    "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
    "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
    "solar_cum_sp", "solar_7d",
    "temp_avg_7d",
]
CAT_COLS = [
    "bloque_base",
    "variedad_canon",
    "tipo_sp",
    "area",
    "estado",
]
CLIP_LO, CLIP_HI = -21, 21


def main() -> None:
    df = read_parquet(DS_PATH).copy()

    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]

    df = df.sort_values("as_of_date").reset_index(drop=True)

    X = df[num_cols + cat_cols]
    y = df[TARGET].astype(float)

    # Preprocess
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn older
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
        max_iter=400,
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        mae = float(mean_absolute_error(yte, pred))
        maes.append({"fold": fold, "mae_days": mae, "n_test": int(len(te))})

    pipe.fit(X, y)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"harvest_start_ml2_{run_id}.pkl"
    meta_path = MODELS_DIR / f"harvest_start_ml2_{run_id}_meta.json"

    import joblib
    joblib.dump(pipe, model_path)

    meta = {
        "run_id": run_id,
        "trained_at": datetime.now().isoformat(),
        "dataset": str(DS_PATH),
        "target": TARGET,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cv": maes,
        "cv_mae_mean": float(np.mean([m["mae_days"] for m in maes])) if maes else None,
        "guardrails": {"clip_error_days": [CLIP_LO, CLIP_HI]},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Guardar CV
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    kpi_df = pd.DataFrame(maes)
    kpi_df["run_id"] = run_id
    kpi_df["created_at"] = pd.Timestamp(datetime.now()).normalize()
    write_parquet(kpi_df, OUT_KPIS)

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Meta saved : {meta_path}")
    print(f"[OK] CV KPIs saved: {OUT_KPIS}")


if __name__ == "__main__":
    main()
