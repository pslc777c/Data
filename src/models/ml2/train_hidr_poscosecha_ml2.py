from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_hidr_poscosecha_ml2_v1.parquet"


def _ohe_dense() -> OneHotEncoder:
    # compat sklearn
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # solo filas con real
    m = df.get("is_in_real", pd.Series(False, index=df.index)).astype(bool)
    d = df.loc[m].copy()
    if d.empty:
        raise ValueError("DS no tiene filas con real (is_in_real=1). No se puede entrenar.")

    # features
    num_cols = ["dow", "month", "weekofyear", "dh_dias_final"]
    # dh_dias_final podría no existir en dataset; intenta alias
    if "dh_dias_final" not in d.columns:
        for c in ["dh_dias_ml2", "dh_dias_ml1", "dh_dias"]:
            if c in d.columns:
                d = d.rename(columns={c: "dh_dias_final"})
                break
        if "dh_dias_final" not in d.columns:
            d["dh_dias_final"] = np.nan

    cat_cols = ["destino", "grado", "variedad_canon"]

    for c in num_cols:
        if c not in d.columns:
            d[c] = np.nan
    for c in cat_cols:
        if c not in d.columns:
            d[c] = "UNKNOWN"

    X = d[num_cols + cat_cols].copy()
    y = pd.to_numeric(d["log_error_hidr_clipped"], errors="coerce")

    # pesos por tallos (si no hay, 1)
    w = pd.to_numeric(d.get("tallos_w"), errors="coerce").fillna(1.0).clip(lower=0.0)

    # split simple por tiempo (más estable que random)
    # tomamos las últimas ~20% fechas_post_pred como "val" si existe, si no random
    split_col = None
    for c in ["fecha_post_pred_final", "fecha_post_pred_ml2", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in d.columns:
            split_col = c
            break

    if split_col is not None:
        t = pd.to_datetime(d[split_col], errors="coerce").dt.normalize()
        q = t.quantile(0.80)
        train_idx = t <= q
    else:
        rng = np.random.default_rng(42)
        train_idx = rng.random(len(d)) <= 0.80

    X_train, y_train, w_train = X.loc[train_idx], y.loc[train_idx], w.loc[train_idx]
    X_val, y_val, w_val = X.loc[~train_idx], y.loc[~train_idx], w.loc[~train_idx]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", _ohe_dense(), cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.06,
        max_iter=350,
        min_samples_leaf=40,
        l2_regularization=0.0,
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train, model__sample_weight=w_train)

    # eval en log space
    pred_tr = pipe.predict(X_train)
    pred_va = pipe.predict(X_val)

    def _mae(y_true, y_pred, w=None) -> float:
        a = pd.to_numeric(pd.Series(y_true), errors="coerce")
        b = pd.to_numeric(pd.Series(y_pred), errors="coerce")
        m = a.notna() & b.notna()
        if not m.any():
            return float("nan")

        err = (a[m] - b[m]).abs()

        if w is None:
            return float(err.mean())

        ww = pd.to_numeric(pd.Series(w), errors="coerce").fillna(0.0)
        ww = ww[m].clip(lower=0.0)
        denom = float(ww.sum())
        if denom <= 0:
            return float(err.mean())
        return float((err * ww).sum() / denom)

    mae_tr = _mae(y_train, pred_tr, w_train)
    mae_va = _mae(y_val, pred_va, w_val)

    # artefactos
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = f"{np.random.default_rng().integers(0, 16**8):08x}"
    model_name = f"hidr_poscosecha_ml2_{ts}_{uid}.pkl"
    meta_name = f"hidr_poscosecha_ml2_{ts}_{uid}_meta.json"

    MODELS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    dump(pipe, MODELS / model_name)

    meta = {
        "model_file": model_name,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "target": "log_error_hidr_clipped",
        "apply_clip_log": [-1.2, 1.2],
        "final_factor_clip": [0.60, 3.00],
        "features_num": num_cols,
        "features_cat": cat_cols,
        "mae_log_train_w": mae_tr,
        "mae_log_val_w": mae_va,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    with open(MODELS / meta_name, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # guardar tabla simple train/val
    out = pd.DataFrame([{
        "mae_log_train_w": mae_tr,
        "mae_log_val_w": mae_va,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "model_file": model_name,
        "meta_file": meta_name,
        "created_at": pd.Timestamp.now().normalize(),
    }])
    write_parquet(out, EVAL / "ml2_hidr_poscosecha_train_val.parquet")

    print(f"[OK] Model saved: {MODELS / model_name}")
    print(f"[OK] Meta  saved: {MODELS / meta_name}")
    print(f"[OK] Eval  saved: {EVAL / 'ml2_hidr_poscosecha_train_val.parquet'}")
    print(f"     MAE log train(w)={mae_tr:.4f}  val(w)={mae_va:.4f}")


if __name__ == "__main__":
    main()
