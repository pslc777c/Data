from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet

TRAINSET_PATH = Path("data/features/trainset_curva_beta_multiplier_dia.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/curva_beta_multiplier_dia")

NUM_COLS = [
    "day_in_harvest","rel_pos","n_harvest_days",
    "pct_avance_real","dia_rel_cosecha_real","gdc_acum_real",
    "rainfall_mm_dia","horas_lluvia","temp_avg_dia","solar_energy_j_m2_dia",
    "wind_speed_avg_dia","wind_run_dia","gdc_dia",
    "dias_desde_sp","gdc_acum_desde_sp",
    "dow","month","weekofyear",
]
CAT_COLS = ["variedad_canon", "area", "tipo_sp"]

def _make_version() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def main() -> None:
    if not TRAINSET_PATH.exists():
        raise FileNotFoundError(f"No existe: {TRAINSET_PATH}")

    created_at = pd.Timestamp.utcnow()
    version = _make_version()
    out_dir = REGISTRY_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(TRAINSET_PATH).copy()

    # Ensure cols exist
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    for c in CAT_COLS:
        df[c] = _canon_str(df[c].fillna("UNKNOWN"))

    y = pd.to_numeric(df["y_log_mult"], errors="coerce")
    ok = y.notna()
    df = df[ok].copy()
    y = y[ok].astype(float)

    # weights: dar más peso a días con mayor real (para que aprenda supresión por clima también se puede)
    if "tallos_real_dia" in df.columns:
        w = pd.to_numeric(df["tallos_real_dia"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w = np.sqrt(np.clip(w, 0.0, None))
        w = np.where(w > 0, w / np.median(w[w > 0]), 1.0)
        w = np.clip(w, 0.2, 5.0)
    else:
        w = np.ones(len(df), dtype=float)

    # X
    X = df[NUM_COLS + CAT_COLS].copy()
    for c in NUM_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = pd.get_dummies(X, columns=CAT_COLS, dummy_na=True)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=120,     # más alto para suavidad
        l2_regularization=1e-4,
        random_state=42,
    )
    model.fit(X, y.to_numpy(dtype=float), sample_weight=w)

    dump(model, out_dir / "model_log_mult.joblib")

    meta = {
        "created_at": str(created_at),
        "version": version,
        "target": "y_log_mult = log(share_real/(share_beta+eps)) (daily multiplier on beta prior)",
        "feature_cols_numeric": NUM_COLS,
        "feature_cols_categorical": CAT_COLS,
        "feature_names": list(X.columns),
        "n_train_rows": int(len(X)),
        "clip_note": "apply will clip multiplier exp(y) to stable bounds",
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK -> {out_dir} | n_train_rows={len(X):,}")

if __name__ == "__main__":
    main()
