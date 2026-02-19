from __future__ import annotations

from pathlib import Path
from datetime import datetime
import hashlib
import json

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
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_desp_poscosecha_ml2_v1.parquet"
OUT_EVAL = EVAL / "ml2_desp_poscosecha_train_val.parquet"


NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]
TARGET_COL = "log_ratio_desp_clipped"
WEIGHT_COL = "tallos_w"


def _mae(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return float("nan")
    err = np.abs(a[m] - b[m])
    if w is None:
        return float(np.mean(err))
    ww = np.asarray(w, dtype=float)[m]
    denom = float(np.sum(ww))
    if denom <= 0:
        return float(np.mean(err))
    return float(np.sum(err * ww) / denom)


def _fingerprint(df: pd.DataFrame) -> str:
    # hash liviano para versionado
    h = hashlib.sha256()
    h.update(str(df.shape).encode("utf-8"))
    h.update(str(df[TARGET_COL].dropna().describe().to_dict()).encode("utf-8"))
    return h.hexdigest()[:8]


def main() -> None:
    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = set(NUM_COLS + CAT_COLS + [TARGET_COL, WEIGHT_COL, "has_real", "fecha_post_pred_used"])
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Dataset sin columnas: {sorted(miss)}")

    d = df.loc[df["has_real"].astype(bool)].copy()
    d["fecha_post_pred_used"] = pd.to_datetime(d["fecha_post_pred_used"], errors="coerce").dt.normalize()

    # Split temporal: últimos 30 días (por fecha_post) a validación
    max_date = d["fecha_post_pred_used"].max()
    if pd.isna(max_date):
        raise ValueError("No hay fecha_post_pred_used válida en dataset con real.")

    cut = max_date - pd.Timedelta(days=30)

    tr = d.loc[d["fecha_post_pred_used"] < cut].copy()
    va = d.loc[d["fecha_post_pred_used"] >= cut].copy()

    X_tr = tr[NUM_COLS + CAT_COLS].copy()
    y_tr = pd.to_numeric(tr[TARGET_COL], errors="coerce").values
    w_tr = pd.to_numeric(tr[WEIGHT_COL], errors="coerce").fillna(0.0).values

    X_va = va[NUM_COLS + CAT_COLS].copy()
    y_va = pd.to_numeric(va[TARGET_COL], errors="coerce").values
    w_va = pd.to_numeric(va[WEIGHT_COL], errors="coerce").fillna(0.0).values

    # Preprocess: OHE denso (evita error sparse->dense)
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        random_state=42,
    )

    pipe = Pipeline([("prep", pre), ("model", model)])

    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    pred_tr = pipe.predict(X_tr)
    pred_va = pipe.predict(X_va)

    mae_tr = _mae(y_tr, pred_tr, w_tr)
    mae_va = _mae(y_va, pred_va, w_va)

    # Persist
    MODELS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = _fingerprint(d)
    model_name = f"desp_poscosecha_ml2_{ts}_{fp}.pkl"
    meta_name = f"desp_poscosecha_ml2_{ts}_{fp}_meta.json"

    dump(pipe, MODELS / model_name)

    meta = {
        "model_file": model_name,
        "dataset": str(IN_DS),
        "target": TARGET_COL,
        "clip_target": 1.2,
        "clip_factor_final": [0.05, 1.00],
        "split_cut_date": str(cut.date()),
        "mae_train": mae_tr,
        "mae_val": mae_va,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "created_at": ts,
    }

    with open(MODELS / meta_name, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out_eval = pd.DataFrame(
        [
            {
                "split": "train",
                "n": int(len(tr)),
                "mae": mae_tr,
                "cut_date": str(cut.date()),
                "created_at": pd.Timestamp.now().normalize(),
            },
            {
                "split": "val",
                "n": int(len(va)),
                "mae": mae_va,
                "cut_date": str(cut.date()),
                "created_at": pd.Timestamp.now().normalize(),
            },
        ]
    )
    write_parquet(out_eval, OUT_EVAL)

    print(f"[OK] Model saved: {MODELS / model_name}")
    print(f"[OK] Meta  saved: {MODELS / meta_name}")
    print(f"[OK] Eval  saved: {OUT_EVAL}")
    print(f"     MAE log train={mae_tr:.4f}  val={mae_va:.4f}")


if __name__ == "__main__":
    main()
