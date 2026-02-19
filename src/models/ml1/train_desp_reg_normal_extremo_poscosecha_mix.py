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
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet


FEATURES_MIX_REAL = Path("data/gold/features_mix_real_dia_destino.parquet")
TARGETS = Path("data/silver/dim_mermas_ajuste_fecha_post_destino.parquet")

REG_DN = Path("models_registry/ml1/desp_poscosecha_reg_normal")
REG_DE = Path("models_registry/ml1/desp_poscosecha_reg_extremo")


def _mk_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _ohe_dense():
    """
    Compatibilidad sklearn:
    - versiones nuevas: sparse_output=False
    - versiones viejas: sparse=False
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _choose_num_cols(df: pd.DataFrame, cat_cols: list[str], min_cov: float = 0.05) -> list[str]:
    drop = {"fecha_post", "desp_pct", "is_extremo", "thr_extremo_dest"} | set(cat_cols)
    cand = [c for c in df.columns if c not in drop]

    if not cand:
        return []

    cov = df[cand].notna().mean()
    cand = cov[cov >= float(min_cov)].index.tolist()

    out: list[str] = []
    bad: list[str] = []
    for c in cand:
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
        else:
            bad.append(c)

    if bad:
        print("[WARN] columnas no numéricas removidas de num_cols:", bad)

    return out


def _time_split_by_unique_dates(df: pd.DataFrame, date_col: str = "fecha_post", test_frac: float = 0.2):
    """
    Split temporal: toma las últimas ~test_frac de fechas únicas como test.
    """
    dates = (
        pd.to_datetime(df[date_col], errors="coerce")
        .dropna()
        .dt.normalize()
        .sort_values()
        .unique()
    )
    if len(dates) < 10:
        # fallback: split simple por fila si hay muy pocas fechas
        cut = int(np.floor(len(df) * (1.0 - test_frac)))
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()

    cut_idx = int(np.floor(len(dates) * (1.0 - test_frac)))
    cut_date = dates[max(0, min(cut_idx, len(dates) - 1))]

    train = df[df[date_col] < cut_date].copy()
    test = df[df[date_col] >= cut_date].copy()

    # si quedara vacío por edge-case, fallback
    if len(train) == 0 or len(test) == 0:
        cut = int(np.floor(len(df) * (1.0 - test_frac)))
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()

    return train, test


def main(
    alpha_normal: float = 10.0,
    q_extreme: float = 0.80,      # <-- CAMBIO: más data extrema
    min_cov: float = 0.05,
    test_frac: float = 0.20,
) -> None:
    # -----------------------------
    # 1) Load X
    # -----------------------------
    Xdf = read_parquet(FEATURES_MIX_REAL).copy()
    Xdf.columns = [str(c).strip() for c in Xdf.columns]
    if not {"fecha_post", "destino"} <= set(Xdf.columns):
        raise ValueError("FEATURES_MIX_REAL debe tener: fecha_post, destino")

    Xdf["fecha_post"] = pd.to_datetime(Xdf["fecha_post"], errors="coerce").dt.normalize()
    Xdf["destino"] = _canon_destino(Xdf["destino"])
    Xdf = Xdf.dropna(subset=["fecha_post", "destino"]).copy()

    # -----------------------------
    # 2) Load y
    # -----------------------------
    ydf = read_parquet(TARGETS).copy()
    ydf.columns = [str(c).strip() for c in ydf.columns]
    if not {"fecha_post", "destino", "desp_pct"} <= set(ydf.columns):
        raise ValueError("TARGETS debe tener: fecha_post, destino, desp_pct")

    ydf["fecha_post"] = pd.to_datetime(ydf["fecha_post"], errors="coerce").dt.normalize()
    ydf["destino"] = _canon_destino(ydf["destino"])
    ydf["desp_pct"] = pd.to_numeric(ydf["desp_pct"], errors="coerce")
    ydf = ydf.dropna(subset=["fecha_post", "destino", "desp_pct"]).copy()

    df = Xdf.merge(
        ydf[["fecha_post", "destino", "desp_pct"]],
        on=["fecha_post", "destino"],
        how="inner",
        validate="one_to_one",
    )

    if len(df) < 80:
        raise ValueError(f"Muy pocas filas para train desp: {len(df)} (esperaba >=80).")

    # -----------------------------
    # 3) Definir extremos por destino
    # -----------------------------
    thr_by_dest = {
        str(d): float(g["desp_pct"].quantile(float(q_extreme)))
        for d, g in df.groupby("destino", dropna=False)
    }
    df["thr_extremo_dest"] = df["destino"].map(thr_by_dest).astype(float)
    df["is_extremo"] = (df["desp_pct"] >= df["thr_extremo_dest"]).astype(int)

    # calendario
    df["dow"] = df["fecha_post"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post"].dt.isocalendar().week.astype("Int64")

    # -----------------------------
    # 4) Features
    # -----------------------------
    cat_cols = ["destino"]
    num_cols = _choose_num_cols(df, cat_cols=cat_cols, min_cov=float(min_cov))
    for c in ["dow", "month", "weekofyear"]:
        if c not in num_cols:
            num_cols.append(c)

    if len(num_cols) == 0:
        raise ValueError("No quedaron columnas numéricas para entrenar (revisa min_cov o features).")

    # Normal: permite sparse OHE
    pre_sparse_ok = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    # Extremo: HGBR requiere denso
    pre_dense = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _ohe_dense()),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    model_normal = Pipeline([("pre", pre_sparse_ok), ("mdl", Ridge(alpha=float(alpha_normal), random_state=42))])
    model_extremo = Pipeline(
        [
            ("pre", pre_dense),
            ("mdl", HistGradientBoostingRegressor(random_state=42, max_depth=3, learning_rate=0.05)),
        ]
    )

    # -----------------------------
    # 5) Split temporal
    # -----------------------------
    train, test = _time_split_by_unique_dates(df, date_col="fecha_post", test_frac=float(test_frac))

    train_n = train[train["is_extremo"] == 0].copy()
    train_e = train[train["is_extremo"] == 1].copy()

    test_n = test[test["is_extremo"] == 0].copy()
    test_e = test[test["is_extremo"] == 1].copy()

    if len(train_n) < 60:
        print("[WARN] pocos datos NORMAL en train:", len(train_n))
    if len(train_e) < 30:
        print("[WARN] pocos datos EXTREMO en train:", len(train_e), "| considera bajar q_extreme o ampliar historia.")

    X_train_n = train_n[cat_cols + num_cols]
    y_train_n = train_n["desp_pct"].astype(float)

    X_train_e = train_e[cat_cols + num_cols]
    y_train_e = train_e["desp_pct"].astype(float)

    # test global
    X_test_all = test[cat_cols + num_cols]
    y_test_all = test["desp_pct"].astype(float)

    # test por régimen
    X_test_n = test_n[cat_cols + num_cols] if len(test_n) else None
    y_test_n = test_n["desp_pct"].astype(float) if len(test_n) else None

    X_test_e = test_e[cat_cols + num_cols] if len(test_e) else None
    y_test_e = test_e["desp_pct"].astype(float) if len(test_e) else None

    # -----------------------------
    # 6) Fit
    # -----------------------------
    model_normal.fit(X_train_n, y_train_n)
    model_extremo.fit(X_train_e, y_train_e)

    # -----------------------------
    # 7) Métricas
    # -----------------------------
    pred_n_all = model_normal.predict(X_test_all)
    pred_e_all = model_extremo.predict(X_test_all)

    # métricas por régimen (lo correcto)
    metrics_regime = {}
    if X_test_n is not None and len(test_n) >= 20:
        pred_n_on_n = model_normal.predict(X_test_n)
        metrics_regime["normal_on_normal_r2"] = float(r2_score(y_test_n, pred_n_on_n))
        metrics_regime["normal_on_normal_rmse"] = _rmse(y_test_n, pred_n_on_n)
    else:
        metrics_regime["normal_on_normal_r2"] = None
        metrics_regime["normal_on_normal_rmse"] = None

    if X_test_e is not None and len(test_e) >= 10:
        pred_e_on_e = model_extremo.predict(X_test_e)
        metrics_regime["extremo_on_extremo_r2"] = float(r2_score(y_test_e, pred_e_on_e))
        metrics_regime["extremo_on_extremo_rmse"] = _rmse(y_test_e, pred_e_on_e)
    else:
        metrics_regime["extremo_on_extremo_r2"] = None
        metrics_regime["extremo_on_extremo_rmse"] = None

    # “mixto” (sin gate real, pero útil como upper bound simple):
    # usa extremo si is_extremo==1, normal si is_extremo==0 (ground-truth)
    if len(test_e) and len(test_n):
        pred_mix = pd.Series(pred_n_all, index=test.index)
        pred_mix.loc[test["is_extremo"] == 1] = model_extremo.predict(test.loc[test["is_extremo"] == 1, cat_cols + num_cols])
        mix_r2 = float(r2_score(y_test_all, pred_mix))
        mix_rmse = _rmse(y_test_all, pred_mix)
    else:
        mix_r2, mix_rmse = None, None

    metrics_common = {
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "n_rows_train_normal": int(len(train_n)),
        "n_rows_train_extremo": int(len(train_e)),
        "n_rows_test_normal": int(len(test_n)),
        "n_rows_test_extremo": int(len(test_e)),
        "q_extreme": float(q_extreme),
        "thr_by_destino": thr_by_dest,
        "features_cat": cat_cols,
        "features_num": num_cols,
        "x_source": str(FEATURES_MIX_REAL),
        "y_source": str(TARGETS),
        "split": {"type": "time", "test_frac": float(test_frac)},
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "clip_range_apply": [0.0, 0.95],  # desp_pct
        **metrics_regime,
        "mix_using_true_regime_r2": mix_r2,
        "mix_using_true_regime_rmse": mix_rmse,
    }

    metrics_n = {
        **metrics_common,
        "model_type": "ridge_normal",
        "alpha": float(alpha_normal),
        "r2_test_on_all": float(r2_score(y_test_all, pred_n_all)),
        "rmse_test_on_all": _rmse(y_test_all, pred_n_all),
    }

    metrics_e = {
        **metrics_common,
        "model_type": "hgb_extremo",
        "r2_test_on_all": float(r2_score(y_test_all, pred_e_all)),
        "rmse_test_on_all": _rmse(y_test_all, pred_e_all),
    }

    # -----------------------------
    # 8) Save registry
    # -----------------------------
    v = _mk_version()
    (REG_DN / v).mkdir(parents=True, exist_ok=True)
    (REG_DE / v).mkdir(parents=True, exist_ok=True)

    dump(model_normal, REG_DN / v / "model.joblib")
    dump(model_extremo, REG_DE / v / "model.joblib")

    with open(REG_DN / v / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_n, f, ensure_ascii=False, indent=2)
    with open(REG_DE / v / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_e, f, ensure_ascii=False, indent=2)

    print(f"[OK] TRAIN DESP REG NORMAL/EXTREMO version={v}")
    print("[INFO] split=time | test_frac=", float(test_frac), "| q_extreme=", float(q_extreme))
    print("[INFO] train normal/extremo:", len(train_n), len(train_e), "| test normal/extremo:", len(test_n), len(test_e))
    print("[INFO] normal r2/rmse test(all):", metrics_n["r2_test_on_all"], metrics_n["rmse_test_on_all"])
    print("[INFO] extremo r2/rmse test(all):", metrics_e["r2_test_on_all"], metrics_e["rmse_test_on_all"])
    print("[INFO] normal_on_normal r2/rmse:", metrics_regime["normal_on_normal_r2"], metrics_regime["normal_on_normal_rmse"])
    print("[INFO] extremo_on_extremo r2/rmse:", metrics_regime["extremo_on_extremo_r2"], metrics_regime["extremo_on_extremo_rmse"])
    print("[INFO] mix_using_true_regime r2/rmse:", mix_r2, mix_rmse)


if __name__ == "__main__":
    main()
