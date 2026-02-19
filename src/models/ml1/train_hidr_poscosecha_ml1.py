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

FACT_PATH = Path("data/silver/fact_hidratacion_real_post_grado_destino.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/hidr_poscosecha")

# numéricas (calendario)
NUM_COLS = ["dow", "month", "weekofyear"]

# categóricas separadas por tipo (EVITA el crash de most_frequent con mezcla dtype)
CAT_STR_COLS = ["destino"]   # string
CAT_NUM_COLS = ["grado"]     # num/cat


def _make_ohe() -> OneHotEncoder:
    # compat sklearn viejo/nuevo
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_pipeline(model) -> Pipeline:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    cat_str_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", _make_ohe()),
        ]
    )

    cat_num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("onehot", _make_ohe()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat_str", cat_str_pipe, CAT_STR_COLS),
            ("cat_num", cat_num_pipe, CAT_NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # fuerza denso (evita temas con HGB)
    )

    return Pipeline(steps=[("pre", pre), ("model", model)])


def _candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
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
    if d.isna().all():
        return []

    uniq = np.array(sorted(d.dropna().unique()))
    if len(uniq) < 6:
        cut = uniq[int(len(uniq) * 0.8)]
        all_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
        tr = (all_dates < cut).to_numpy()
        va = (all_dates >= cut).to_numpy()
        return [(np.where(tr)[0], np.where(va)[0])] if tr.sum() > 0 and va.sum() > 0 else []

    n_folds = int(min(max(2, n_folds), max(2, len(uniq) // 2)))
    valid_blocks = np.array_split(uniq, n_folds + 1)[1:]

    folds: list[tuple[np.ndarray, np.ndarray]] = []
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


def main() -> None:
    if not FACT_PATH.exists():
        raise FileNotFoundError(f"No existe: {FACT_PATH}")

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = REGISTRY_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    fact = read_parquet(FACT_PATH).copy()
    fact.columns = [str(c).strip() for c in fact.columns]

    need = {"fecha_cosecha", "grado", "destino"}
    miss = need - set(fact.columns)
    if miss:
        raise ValueError(f"fact_hidratacion_real_post_grado_destino sin columnas: {sorted(miss)}")

    # target: factor_hidr si existe; si no, 1+hidr_pct
    if "factor_hidr" in fact.columns:
        fact["y"] = pd.to_numeric(fact["factor_hidr"], errors="coerce")
        target_name = "factor_hidr"
    elif "hidr_pct" in fact.columns:
        fact["y"] = 1.0 + pd.to_numeric(fact["hidr_pct"], errors="coerce")
        target_name = "1+hidr_pct"
    else:
        raise ValueError("No encuentro ni factor_hidr ni hidr_pct en fact_hidratacion_real_post_grado_destino.")

    fact["fecha_cosecha"] = pd.to_datetime(fact["fecha_cosecha"], errors="coerce").dt.normalize()

    # grado num/cat (se queda num, se onehotea como categoría)
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce")

    # destino str
    fact["destino"] = fact["destino"].astype(str).str.upper().str.strip()
    fact.loc[fact["destino"].isin(["", "NAN", "NONE", "NULL"]), "destino"] = np.nan

    # peso base para ponderación si existe
    w = None
    if "peso_base_g" in fact.columns:
        w = pd.to_numeric(fact["peso_base_g"], errors="coerce")
    elif "tallos" in fact.columns:
        w = pd.to_numeric(fact["tallos"], errors="coerce")

    df = fact[fact["fecha_cosecha"].notna() & fact["grado"].notna() & fact["y"].notna()].copy()

    # caps razonables (alineado a tu lógica histórica)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").clip(lower=0.80, upper=3.00)

    if len(df) < 200:
        raise ValueError(f"Poca data para entrenar hidr (n={len(df)}).")

    # features calendario
    df["dow"] = df["fecha_cosecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_cosecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_cosecha"].dt.isocalendar().week.astype("Int64")

    X = df[NUM_COLS + CAT_STR_COLS + CAT_NUM_COLS].copy()
    y = df["y"].astype(float).to_numpy()

    # sample_weight (opcional)
    sample_weight = None
    if w is not None:
        ww = w.loc[df.index]
        ww = pd.to_numeric(ww, errors="coerce").fillna(0.0).astype(float)
        # si viene todo 0, no sirve
        if float(ww.sum()) > 0:
            sample_weight = ww.to_numpy()

    folds = _time_folds(df["fecha_cosecha"], n_folds=4)
    if not folds:
        raise ValueError("No pude armar folds temporales para hidr.")

    models = _candidate_models()

    best_name = None
    best_score = None
    best_model = None
    all_metrics: dict[str, dict] = {}

    for name, model in models.items():
        pipe = _make_pipeline(model)
        maes = []
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs["model__sample_weight"] = sample_weight[tr_idx]

            pipe.fit(X_tr, y_tr, **fit_kwargs)
            pred = pipe.predict(X_va)
            maes.append(float(np.mean(np.abs(y_va - pred))))

        mae_mean = float(np.mean(maes))
        mae_std = float(np.std(maes))
        score = 0.85 * mae_mean + 0.15 * mae_std

        all_metrics[name] = {
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "score": float(score),
            "n_rows": int(len(X)),
            "n_folds": int(len(folds)),
            "uses_sample_weight": bool(sample_weight is not None),
        }

        if (best_score is None) or (score < best_score):
            best_score = score
            best_name = name
            best_model = model

    assert best_name is not None and best_model is not None

    best_pipe = _make_pipeline(best_model)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight

    best_pipe.fit(X, y, **fit_kwargs)

    model_path = out_dir / "model.joblib"
    dump(best_pipe, model_path)

    summary = {
        "version": version,
        "created_at_utc": datetime.utcnow().isoformat(),
        "fact_path": str(FACT_PATH).replace("\\", "/"),
        "target": target_name,
        "clip_range_apply": [0.80, 3.00],
        "features": {
            "num": NUM_COLS,
            "cat_str": CAT_STR_COLS,
            "cat_num": CAT_NUM_COLS,
        },
        "best_model": best_name,
        "metrics": all_metrics,
        "uses_sample_weight": bool(sample_weight is not None),
        "model_path": str(model_path).replace("\\", "/"),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"OK -> {out_dir}/ (model.joblib + metrics.json) | best={best_name} score={best_score:.6f}")


if __name__ == "__main__":
    main()
