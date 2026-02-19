from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import hashlib

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
MODELS = DATA / "models" / "ml2"
EVAL = DATA / "eval" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_ajuste_poscosecha_ml2_v1.parquet"


def _hash8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _safe_ohe() -> OneHotEncoder:
    # sklearn compatibility: sparse_output vs sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _wmae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    w = np.asarray(w, dtype="float64")
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w >= 0)
    if not np.any(m):
        return float("nan")
    denom = float(np.sum(w[m]))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true[m] - y_pred[m]) * w[m]) / denom)


def main() -> None:
    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # solo filas con real para entrenamiento
    m = df["factor_ajuste_real"].notna() & df["log_ratio_ajuste"].notna()
    d = df.loc[m].copy()

    if d.empty:
        raise ValueError("Dataset no tiene filas con real (factor_ajuste_real). No se puede entrenar.")

    # features
    num_cols = ["dow", "month", "weekofyear"]
    cat_cols = ["destino", "grado"]

    for c in num_cols:
        if c not in d.columns:
            d[c] = pd.NA
    for c in cat_cols:
        if c not in d.columns:
            d[c] = "UNKNOWN"

    X = d[num_cols + cat_cols].copy()
    y = pd.to_numeric(d["log_ratio_ajuste"], errors="coerce").astype("float64").values
    w = pd.to_numeric(d.get("w", 1.0), errors="coerce").fillna(0.0).astype("float64").values

    # split temporal simple: 80/20 por fecha_post_pred_used
    d["fecha_post_pred_used"] = pd.to_datetime(d["fecha_post_pred_used"], errors="coerce").dt.normalize()
    order = d["fecha_post_pred_used"].fillna(pd.Timestamp("1970-01-01")).values
    idx = np.argsort(order)

    cut = int(0.80 * len(d))
    tr_idx = idx[:cut]
    va_idx = idx[cut:]

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    w_tr, w_va = w[tr_idx], w[va_idx]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", _safe_ohe(), cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    pred_tr = pipe.predict(X_tr)
    pred_va = pipe.predict(X_va)

    mae_tr = _wmae(y_tr, pred_tr, w_tr)
    mae_va = _wmae(y_va, pred_va, w_va)

    # save
    MODELS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = _hash8(f"{ts}|ajuste_poscosecha_ml2|{mae_va:.6f}")
    model_name = f"ajuste_poscosecha_ml2_{ts}_{tag}.pkl"
    meta_name = f"ajuste_poscosecha_ml2_{ts}_{tag}_meta.json"

    model_path = MODELS / model_name
    meta_path = MODELS / meta_name

    dump(pipe, model_path)

    meta = {
        "model_file": model_name,
        "created_at_utc": datetime.utcnow().isoformat(),
        "target": "log_ratio_ajuste = log(real/ml1) clipped",
        "clip_range_target": [-1.2, 1.2],
        "clip_range_apply": [0.50, 2.00],
        "features_num": num_cols,
        "features_cat": cat_cols,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "mae_log_train_w": float(mae_tr),
        "mae_log_val_w": float(mae_va),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out_eval = pd.DataFrame([{
        "model_file": model_name,
        "mae_log_train_w": mae_tr,
        "mae_log_val_w": mae_va,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "created_at": pd.Timestamp.utcnow(),
    }])
    eval_path = EVAL / "ml2_ajuste_poscosecha_train_val.parquet"
    write_parquet(out_eval, eval_path)

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Meta  saved: {meta_path}")
    print(f"[OK] Eval  saved: {eval_path}")
    print(f"     MAE log train={mae_tr:.4f}  val={mae_va:.4f}")


if __name__ == "__main__":
    main()
