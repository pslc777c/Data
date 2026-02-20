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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_dh_poscosecha_ml2_v1.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]

TARGET = "err_dh_days_clipped"
WEIGHT_COL = "tallos_real"


def _run_id(prefix: str = "dh_poscosecha_ml2") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(ts.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{ts}_{h}"


def _make_ohe_dense() -> OneHotEncoder:
    """
    sklearn >=1.2 usa sparse_output; sklearn <1.2 usa sparse.
    Forzamos salida densa para que HistGradientBoostingRegressor no falle.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill_value) -> pd.DataFrame:
    """
    Si falta una columna, la crea con fill_value.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX CLAVE:
    - NUM_COLS: fuerza numeric (coerce -> NaN)
    - CAT_COLS: fuerza string limpio + fillna('UNKNOWN')
    Esto evita que SimpleImputer reciba arrays con dtype incorrecto
    (p.ej. float con strings como 'BLANCO') y reviente.
    """
    df = df.copy()

    # asegurar columnas existen
    df = _ensure_cols(df, NUM_COLS, pd.NA)
    df = _ensure_cols(df, CAT_COLS, "UNKNOWN")

    # num -> numeric
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # cat -> string limpio
    for c in CAT_COLS:
        # string dtype (pandas) + normalize
        s = df[c].astype("string")
        s = s.str.strip()
        s = s.fillna("UNKNOWN")
        # opcional: evitar vacíos
        s = s.replace("", "UNKNOWN")
        df[c] = s

    return df[NUM_COLS + CAT_COLS]


def main() -> None:
    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Train solo donde hay real
    m = df.get("dh_real").notna() & df.get("dh_ml1").notna() & df.get(TARGET).notna()
    tr = df.loc[m].copy()
    if tr.empty:
        raise ValueError("No hay filas con real para entrenar DH ML2.")

    tr["fecha"] = pd.to_datetime(tr["fecha"], errors="coerce").dt.normalize()
    tr = tr.sort_values("fecha")

    # Split temporal: últimas 8 semanas val
    cut = tr["fecha"].max() - pd.Timedelta(days=56)
    is_val = tr["fecha"] >= cut

    train_df = tr.loc[~is_val].copy()
    val_df = tr.loc[is_val].copy()

    # fallback si val es muy pequeño
    if len(val_df) < 1000:
        n = len(tr)
        n_val = max(int(0.2 * n), 1000 if n >= 5000 else int(0.2 * n))
        val_df = tr.tail(n_val).copy()
        train_df = tr.iloc[: max(n - n_val, 1)].copy()

    # === X/y/w (con casteo robusto) ===
    X_train = _prepare_X(train_df)
    y_train = pd.to_numeric(train_df[TARGET], errors="coerce")

    w_train = pd.to_numeric(train_df.get(WEIGHT_COL), errors="coerce").fillna(0.0).values.astype(float)
    w_train = np.where(w_train <= 0, 1.0, w_train)

    X_val = _prepare_X(val_df)
    y_val = pd.to_numeric(val_df[TARGET], errors="coerce")

    w_val = pd.to_numeric(val_df.get(WEIGHT_COL), errors="coerce").fillna(0.0).values.astype(float)
    w_val = np.where(w_val <= 0, 1.0, w_val)

    # Debug mínimo (útil para el video)
    print("[DBG] X_train dtypes:")
    print(X_train.dtypes)
    for c in CAT_COLS:
        if c in X_train.columns:
            sample_vals = X_train[c].dropna().astype(str).unique()[:5]
            print(f"[DBG] cat sample {c}={sample_vals}")

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), NUM_COLS),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", _make_ohe_dense()),  # denso
            ]), CAT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    # Fit con sample_weight
    pipe.fit(X_train, y_train, model__sample_weight=w_train)

    pred_tr = pipe.predict(X_train)
    pred_va = pipe.predict(X_val)

    mae_tr = mean_absolute_error(y_train, pred_tr, sample_weight=w_train)
    mae_va = mean_absolute_error(y_val, pred_va, sample_weight=w_val)

    run_id = _run_id()

    MODELS.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    out_model = MODELS / f"{run_id}.pkl"
    out_meta = MODELS / f"{run_id}_meta.json"
    out_eval = EVAL / "ml2_dh_poscosecha_train_val.parquet"

    dump(pipe, out_model)

    clip_err = float(pd.to_numeric(df.get("clip_err_days", pd.Series([5.0])).iloc[0], errors="coerce") or 5.0)

    meta = {
        "run_id": run_id,
        "created_at_utc": str(pd.Timestamp.utcnow()),
        "dataset": str(IN_DS),
        "target": TARGET,
        "clip_err_days": clip_err,
        "features_num": NUM_COLS,
        "features_cat": CAT_COLS,
        "split": {"type": "time", "val_window_days": 56},
        "metrics": {"mae_train_days": float(mae_tr), "mae_val_days": float(mae_va)},
        "note": "Forced dtypes: num->numeric, cat->string + fillna('UNKNOWN'); OHE dense for HGBR",
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    ev = pd.DataFrame([{
        "run_id": run_id,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "mae_train_days": float(mae_tr),
        "mae_val_days": float(mae_va),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])
    write_parquet(ev, out_eval)

    print(f"[OK] Model saved: {out_model}")
    print(f"[OK] Meta  saved: {out_meta}")
    print(f"[OK] Eval  saved: {out_eval}")
    print(f"     MAE days train={mae_tr:.4f}  val={mae_va:.4f}")


if __name__ == "__main__":
    main()
