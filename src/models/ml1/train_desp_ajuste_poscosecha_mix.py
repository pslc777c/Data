from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime, timezone

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

from common.io import read_parquet


FEATURES_MIX_REAL = Path("data/gold/features_mix_real_dia_destino.parquet")
TARGETS = Path("data/silver/dim_mermas_ajuste_fecha_post_destino.parquet")

REG_DESP = Path("models_registry/ml1/desp_poscosecha")
REG_AJ = Path("models_registry/ml1/ajuste_poscosecha")


def _mk_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main(
    alpha_desp: float = 10.0,
    alpha_aj: float = 0.25,
    test_frac: float = 0.20,   # <- split temporal
) -> None:
    # 1) Load X
    Xdf = read_parquet(FEATURES_MIX_REAL).copy()
    Xdf.columns = [str(c).strip() for c in Xdf.columns]

    need_x = {"fecha_post", "destino"}
    miss = need_x - set(Xdf.columns)
    if miss:
        raise ValueError(f"Features MIX REAL sin columnas: {sorted(miss)}")

    Xdf["fecha_post"] = pd.to_datetime(Xdf["fecha_post"], errors="coerce").dt.normalize()
    Xdf["destino"] = Xdf["destino"].astype(str).str.upper().str.strip()

    # calendario
    Xdf["dow"] = Xdf["fecha_post"].dt.dayofweek.astype("Int64")
    Xdf["month"] = Xdf["fecha_post"].dt.month.astype("Int64")
    Xdf["weekofyear"] = Xdf["fecha_post"].dt.isocalendar().week.astype("Int64")

    # 2) Load y
    ydf = read_parquet(TARGETS).copy()
    ydf.columns = [str(c).strip() for c in ydf.columns]

    need_y = {"fecha_post", "destino", "desp_pct", "ajuste"}
    miss = need_y - set(ydf.columns)
    if miss:
        raise ValueError(f"Targets sin columnas: {sorted(miss)}")

    ydf["fecha_post"] = pd.to_datetime(ydf["fecha_post"], errors="coerce").dt.normalize()
    ydf["destino"] = ydf["destino"].astype(str).str.upper().str.strip()

    df = Xdf.merge(
        ydf[["fecha_post", "destino", "desp_pct", "ajuste"]],
        on=["fecha_post", "destino"],
        how="inner",
        validate="one_to_one",
    )

    # targets numeric
    df["desp_pct"] = pd.to_numeric(df["desp_pct"], errors="coerce")
    df["ajuste"] = pd.to_numeric(df["ajuste"], errors="coerce")
    df = df.dropna(subset=["desp_pct", "ajuste"]).copy()

    if len(df) < 50:
        raise ValueError(f"Muy pocas filas para train: {len(df)}. Revisa cobertura de merge X vs y.")

    # ordenar para split temporal
    df = df.sort_values(["fecha_post", "destino"]).reset_index(drop=True)

    # 3) Feature sets
    cat_cols = ["destino"]

    drop_cols = {"fecha_post", "desp_pct", "ajuste"} | set(cat_cols)
    cand_cols = [c for c in df.columns if c not in drop_cols]

    cov = df[cand_cols].notna().mean()
    cand_cols = cov[cov >= 0.05].index.tolist()

    bad_num = []
    num_cols = []
    for c in cand_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            bad_num.append(c)
            continue
        num_cols.append(c)

    if bad_num:
        print("[WARN] columnas no numéricas removidas de num_cols:", bad_num)

    if len(num_cols) == 0:
        raise ValueError("No quedaron features numéricas después de filtros (coverage>=5% y numeric dtype).")

    # 4) Preprocess + Models
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model_desp = Pipeline([("pre", pre), ("mdl", Ridge(alpha=float(alpha_desp), random_state=42))])
    model_aj = Pipeline([("pre", pre), ("mdl", Ridge(alpha=float(alpha_aj), random_state=42))])

    # ---- split temporal
    cut = int(len(df) * (1.0 - float(test_frac)))
    cut = max(10, min(cut, len(df) - 10))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    X_train = train[cat_cols + num_cols]
    X_test = test[cat_cols + num_cols]

    y_train_d = train["desp_pct"].astype(float)
    y_test_d = test["desp_pct"].astype(float)

    y_train_a = train["ajuste"].astype(float)
    y_test_a = test["ajuste"].astype(float)

    model_desp.fit(X_train, y_train_d)
    model_aj.fit(X_train, y_train_a)

    pred_train_d = model_desp.predict(X_train)
    pred_test_d = model_desp.predict(X_test)

    pred_train_a = model_aj.predict(X_train)
    pred_test_a = model_aj.predict(X_test)

    metrics_common = {
        "n_rows": int(len(df)),
        "split": "time",
        "test_frac": float(test_frac),
        "n_features_num": int(len(num_cols)),
        "features_num": num_cols,
        "features_cat": cat_cols,
        "alpha_desp": float(alpha_desp),
        "alpha_aj": float(alpha_aj),
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "x_source": str(FEATURES_MIX_REAL),
        "y_source": str(TARGETS),
    }

    metrics_desp = {
        **metrics_common,
        "target": "desp_pct",
        "r2_train": float(r2_score(y_train_d, pred_train_d)),
        "r2_test": float(r2_score(y_test_d, pred_test_d)),
        "rmse_train": _rmse(y_train_d, pred_train_d),
        "rmse_test": _rmse(y_test_d, pred_test_d),
        # en apply tú haces factor = 1 - desp_pct
        "clip_range_apply": [0.0, 0.95],
    }

    metrics_aj = {
        **metrics_common,
        "target": "ajuste",
        "r2_train": float(r2_score(y_train_a, pred_train_a)),
        "r2_test": float(r2_score(y_test_a, pred_test_a)),
        "rmse_train": _rmse(y_train_a, pred_train_a),
        "rmse_test": _rmse(y_test_a, pred_test_a),
        "clip_range_apply": [0.80, 1.50],
    }

    v = _mk_version()

    (REG_DESP / v).mkdir(parents=True, exist_ok=True)
    (REG_AJ / v).mkdir(parents=True, exist_ok=True)

    dump(model_desp, REG_DESP / v / "model.joblib")
    dump(model_aj, REG_AJ / v / "model.joblib")

    with open(REG_DESP / v / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_desp, f, ensure_ascii=False, indent=2)

    with open(REG_AJ / v / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_aj, f, ensure_ascii=False, indent=2)

    print(f"[OK] TRAIN ML1 MIX REAL version={v}")
    print("[INFO] split=time | test_frac=", test_frac)
    print("[INFO] desp r2 train/test:", metrics_desp["r2_train"], metrics_desp["r2_test"], "| rmse:", metrics_desp["rmse_test"])
    print("[INFO] aj  r2 train/test:", metrics_aj["r2_train"], metrics_aj["r2_test"], "| rmse:", metrics_aj["rmse_test"])


if __name__ == "__main__":
    main()
