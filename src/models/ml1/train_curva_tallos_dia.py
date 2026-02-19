from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import warnings

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

from common.io import read_parquet

warnings.filterwarnings("ignore")


FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/curva_tallos_dia")


NUM_COLS = [
    # baseline
    "tallos_pred_baseline_dia",
    # progreso / etapa
    "pct_avance_real",
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    # clima / gdc
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    # SP
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    # calendario
    "dow",
    "month",
    "weekofyear",
]

CAT_COLS = [
    "variedad_canon",
    "area",
    "tipo_sp",
]


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom <= 1e-12:
        return float(np.nan)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _make_pipeline(model) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def _candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0, random_state=0),
        "gbr": GradientBoostingRegressor(random_state=0),
        "hgb": HistGradientBoostingRegressor(random_state=0),
        "rf": RandomForestRegressor(
            n_estimators=400,
            random_state=0,
            n_jobs=-1,
            min_samples_leaf=5,
        ),
    }


def _time_folds(dates: pd.Series, n_folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    d = pd.to_datetime(dates, errors="coerce").dt.normalize()
    d = d.dropna()
    if d.empty:
        return []

    uniq = np.array(sorted(d.unique()))
    if len(uniq) < 10:
        cut = uniq[int(len(uniq) * 0.8)]
        all_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
        tr = (all_dates < cut).to_numpy()
        va = (all_dates >= cut).to_numpy()
        return [(np.where(tr)[0], np.where(va)[0])] if tr.sum() > 0 and va.sum() > 0 else []

    n_folds = int(min(max(2, n_folds), max(2, len(uniq) // 3)))
    valid_blocks = np.array_split(uniq, n_folds + 1)[1:]

    folds = []
    all_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()

    for vb in valid_blocks:
        if len(vb) == 0:
            continue
        valid_start = vb.min()
        valid_end = vb.max()

        tr = (all_dates < valid_start).to_numpy()
        va = ((all_dates >= valid_start) & (all_dates <= valid_end)).to_numpy()
        if tr.sum() > 0 and va.sum() > 0:
            folds.append((np.where(tr)[0], np.where(va)[0]))

    if not folds:
        cut = uniq[int(len(uniq) * 0.8)]
        tr = (all_dates < cut).to_numpy()
        va = (all_dates >= cut).to_numpy()
        if tr.sum() > 0 and va.sum() > 0:
            folds = [(np.where(tr)[0], np.where(va)[0])]

    return folds


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    wape = _wape(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    return {"mae": mae, "wape": wape, "smape": smape}


def main() -> None:
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = REGISTRY_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(FEATURES_PATH).copy()

    need = {"fecha", "bloque_base", "variedad_canon", "tallos_pred_baseline_dia", "factor_tallos_dia_clipped"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features_curva_cosecha_bloque_dia sin columnas: {sorted(miss)}")

    # target
    df["y"] = pd.to_numeric(df["factor_tallos_dia_clipped"], errors="coerce")

    # training rows: donde hay y finito
    train = df[np.isfinite(df["y"].to_numpy())].copy()
    if train.empty:
        raise ValueError("No hay filas con target (factor_tallos_dia_clipped) finito. Revisa real coverage.")

    # asegurar columnas
    for c in NUM_COLS:
        if c not in train.columns:
            train[c] = np.nan
    for c in CAT_COLS:
        if c not in train.columns:
            train[c] = "UNKNOWN"

    # folds temporales
    folds = _time_folds(train["fecha"], n_folds=4)
    if not folds:
        raise ValueError("No pude construir splits temporales. Revisa rango de fechas en FEATURES.")

    X_all = train[NUM_COLS + CAT_COLS]
    y_all = train["y"].to_numpy(dtype=float)

    models = _candidate_models()

    summary: dict = {
        "version": version,
        "created_at_utc": datetime.utcnow().isoformat(),
        "features_path": str(FEATURES_PATH),
        "n_rows_train_total": int(len(train)),
        "n_folds": int(len(folds)),
        "best_model": None,
        "metrics": {},
        "model_path": None,
    }

    best_name = None
    best_score = None
    best_model = None

    for name, model in models.items():
        pipe = _make_pipeline(model)

        fold_stats = []
        for tr_idx, va_idx in folds:
            X_tr = X_all.iloc[tr_idx]
            y_tr = y_all[tr_idx]
            X_va = X_all.iloc[va_idx]
            y_va = y_all[va_idx]

            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)

            fold_stats.append(_score_fold(y_va, pred))

        maes = np.array([m["mae"] for m in fold_stats], dtype=float)
        wapes = np.array([m["wape"] for m in fold_stats], dtype=float)

        mae_mean = float(np.nanmean(maes))
        mae_std = float(np.nanstd(maes))
        wape_mean = float(np.nanmean(wapes))

        # score compuesto (más bajo es mejor)
        score = 0.60 * mae_mean + 0.30 * (wape_mean if np.isfinite(wape_mean) else mae_mean) + 0.10 * mae_std

        summary["metrics"][name] = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "wape_mean": wape_mean,
            "score": float(score),
            "n_rows": int(len(train)),
            "n_folds": int(len(folds)),
        }

        if (best_score is None) or (score < best_score):
            best_score = score
            best_name = name
            best_model = model

    assert best_name is not None and best_model is not None

    # fit final
    best_pipe = _make_pipeline(best_model)
    best_pipe.fit(X_all, y_all)

    model_path = out_dir / "model_curva_tallos_dia.joblib"
    dump(best_pipe, model_path)

    summary["best_model"] = best_name
    summary["model_path"] = str(model_path).replace("\\", "/")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[ML1 curva_tallos_dia] best={best_name} score={best_score:.6f}")
    print(f"OK -> {out_dir}/ (model + metrics.json)")


if __name__ == "__main__":
    main()
