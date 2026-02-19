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
from sklearn.dummy import DummyRegressor

from common.io import read_parquet

warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/features/features_peso_tallo_grado_bloque_dia.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/peso_tallo_grado")

# =========================
# Config
# =========================
CLIP_LOW, CLIP_HIGH = 0.60, 1.60

NUM_COLS = [
    "pct_avance_real",
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    "dow",
    "month",
    "weekofyear",
    "peso_tallo_baseline_g",
]

CAT_COLS = ["variedad_canon", "tipo_sp", "area"]

# Candidatos para reconstruir target si viene vacío
REAL_W_CANDS = [
    "peso_tallo_real_g",
    "peso_tallo_real",
    "peso_tallo_g_real",
    "peso_tallo_obs_g",
    "peso_tallo_prom_g",
    "peso_tallo_avg_g",
    "peso_tallo_mediana_g",
    "peso_tallo_g",
    "peso_real_g",
    "peso_real",
]
BASE_W_CANDS = [
    "peso_tallo_baseline_g",
    "peso_tallo_baseline",
    "peso_baseline_g",
    "peso_base_g",
]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _make_ohe() -> OneHotEncoder:
    # compat sklearn viejo/nuevo
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_pipeline(model) -> Pipeline:
    # num -> median (robusto)
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    # cat -> constant (evita bugs dtype mixto)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", _make_ohe()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_COLS),
            ("cat", cat_pipe, CAT_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # fuerza dense
    )
    return Pipeline(steps=[("pre", pre), ("model", model)])


def _candidate_models() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0),
        "gbr": GradientBoostingRegressor(random_state=0),
        "hgb": HistGradientBoostingRegressor(random_state=0),
        "rf": RandomForestRegressor(
            n_estimators=300,
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


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    wape = _wape(y_true, y_pred)
    smape = _smape(y_true, y_pred)
    return {"mae": mae, "wape": wape, "smape": smape}


def _pick_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    cols = list(df.columns)
    for c in cands:
        if c in cols:
            return c
    # match case-insensitive / trimmed
    norm = {str(c).strip().upper(): c for c in cols}
    for c in cands:
        k = str(c).strip().upper()
        if k in norm:
            return norm[k]
    return None


def _ensure_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Garantiza factor_peso_tallo_clipped:
    - si ya hay valores -> listo
    - si está vacío -> intenta reconstruir desde peso_real / peso_baseline
    """
    info = {}

    if "factor_peso_tallo_clipped" in df.columns:
        nn = int(df["factor_peso_tallo_clipped"].notna().sum())
        info["target_present_nonnull"] = nn
        if nn > 0:
            return df, info

    real_col = _pick_col(df, REAL_W_CANDS)
    base_col = _pick_col(df, BASE_W_CANDS)

    info["real_col_used"] = real_col
    info["base_col_used"] = base_col

    if real_col is None or base_col is None:
        # no se puede reconstruir
        if "factor_peso_tallo_clipped" not in df.columns:
            df["factor_peso_tallo_clipped"] = np.nan
        info["target_rebuilt"] = False
        info["target_rebuilt_reason"] = "missing real/base columns"
        return df, info

    real = pd.to_numeric(df[real_col], errors="coerce")
    base = pd.to_numeric(df[base_col], errors="coerce")

    factor = np.where((base > 0) & np.isfinite(base), real / base, np.nan)
    factor = pd.Series(factor, index=df.index)
    factor = factor.replace([np.inf, -np.inf], np.nan)

    df["factor_peso_tallo_raw"] = factor
    df["factor_peso_tallo_clipped"] = pd.to_numeric(df["factor_peso_tallo_raw"], errors="coerce").clip(
        lower=CLIP_LOW, upper=CLIP_HIGH
    )

    info["target_rebuilt"] = True
    info["target_rebuilt_nonnull"] = int(df["factor_peso_tallo_clipped"].notna().sum())
    return df, info


def _fit_constant_pipeline(constant_value: float) -> Pipeline:
    """
    Modelo fallback: predice constante (ej. 1.0).
    Se "fitea" sobre 1 fila dummy para dejar el pipeline serializable.
    """
    model = DummyRegressor(strategy="constant", constant=float(constant_value))
    pipe = _make_pipeline(model)

    X_dummy = pd.DataFrame(
        {
            **{c: [0.0] for c in NUM_COLS},
            **{c: ["UNKNOWN"] for c in CAT_COLS},
        }
    )
    y_dummy = np.array([float(constant_value)], dtype=float)
    pipe.fit(X_dummy, y_dummy)
    return pipe


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"No existe: {FEATURES_PATH}")

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = REGISTRY_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_parquet(FEATURES_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "peso_tallo_baseline_g"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"features_peso_tallo_grado_bloque_dia.parquet sin columnas: {sorted(miss)}")

    # Canon llaves/fechas
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_int(df["bloque_base"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])

    # Asegurar cat cols
    for c in ["tipo_sp", "area"]:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    df["tipo_sp"] = _canon_str(df["tipo_sp"].fillna("UNKNOWN"))
    df["area"] = _canon_str(df["area"].fillna("UNKNOWN"))

    # Asegurar num cols (evita strings raros en num)
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Construir/garantizar target
    df, tinfo = _ensure_target(df)

    # Train set = donde target válido + fecha válida + grado válido
    train_df = df[
        df["factor_peso_tallo_clipped"].notna()
        & df["fecha"].notna()
        & df["grado"].notna()
    ].copy()

    grades = sorted(df["grado"].dropna().astype(int).unique().tolist())
    models = _candidate_models()

    summary: dict = {
        "version": version,
        "created_at_utc": datetime.utcnow().isoformat(),
        "features_path": str(FEATURES_PATH).replace("\\", "/"),
        "target": "factor_peso_tallo_clipped",
        "clip_range": [CLIP_LOW, CLIP_HIGH],
        "n_rows_total": int(len(df)),
        "n_rows_train_total": int(len(train_df)),
        "grades_seen": [int(g) for g in grades],
        "target_build_info": tinfo,
        "best_by_grade": {},
    }

    if not grades:
        # no grados => igual guardo métricas y salgo
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[WARN] No hay grados en features. OK -> {out_dir}/metrics.json")
        return

    if train_df.empty:
        # no hay target: guardo modelos constantes por grado para no romper downstream
        print("[WARN] No hay filas con target para entrenar. Se generan modelos constantes (1.0) por grado.")
        for g in grades:
            g = int(g)
            pipe = _fit_constant_pipeline(1.0)
            model_path = out_dir / f"model_grade_{g}.joblib"
            dump(pipe, model_path)
            summary["best_by_grade"][str(g)] = {
                "best_model": "dummy_const_1.0",
                "metrics": {},
                "model_path": str(model_path).replace("\\", "/"),
                "n_rows_grade": 0,
            }

        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"OK -> {out_dir}/ (models const + metrics.json)")
        return

    # Entrenamiento por grado
    for g in grades:
        g = int(g)
        gdf = train_df[train_df["grado"].astype(int) == g].copy()

        if len(gdf) < 30:
            # fallback modelo constante por grado
            pipe = _fit_constant_pipeline(1.0)
            model_path = out_dir / f"model_grade_{g}.joblib"
            dump(pipe, model_path)
            summary["best_by_grade"][str(g)] = {
                "best_model": "dummy_const_1.0",
                "metrics": {},
                "model_path": str(model_path).replace("\\", "/"),
                "n_rows_grade": int(len(gdf)),
                "note": "poca data; fallback const",
            }
            print(f"[ML1 peso_tallo] grado={g} -> fallback const=1.0 (poca data n={len(gdf)})")
            continue

        # Features/target
        X = gdf[NUM_COLS + CAT_COLS].copy()
        for c in CAT_COLS:
            X[c] = _canon_str(X[c].fillna("UNKNOWN"))

        y = pd.to_numeric(gdf["factor_peso_tallo_clipped"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 30:
            pipe = _fit_constant_pipeline(1.0)
            model_path = out_dir / f"model_grade_{g}.joblib"
            dump(pipe, model_path)
            summary["best_by_grade"][str(g)] = {
                "best_model": "dummy_const_1.0",
                "metrics": {},
                "model_path": str(model_path).replace("\\", "/"),
                "n_rows_grade": int(len(gdf)),
                "note": "target inválido; fallback const",
            }
            print(f"[ML1 peso_tallo] grado={g} -> fallback const=1.0 (target inválido)")
            continue

        X = X.loc[gdf.index[mask]].copy()
        y = y[mask]

        folds = _time_folds(gdf.loc[X.index, "fecha"], n_folds=4)
        if not folds:
            # fallback split 80/20 por fecha
            d = pd.to_datetime(gdf.loc[X.index, "fecha"], errors="coerce").dt.normalize()
            d = d.dropna()
            if d.empty:
                folds = []
            else:
                cut = d.quantile(0.8)
                d_arr = pd.to_datetime(gdf.loc[X.index, "fecha"], errors="coerce").dt.normalize().to_numpy()
                tr_idx = np.where(d_arr < np.datetime64(cut))[0]
                va_idx = np.where(d_arr >= np.datetime64(cut))[0]
                if len(tr_idx) > 0 and len(va_idx) > 0:
                    folds = [(tr_idx, va_idx)]

        if not folds:
            pipe = _fit_constant_pipeline(1.0)
            model_path = out_dir / f"model_grade_{g}.joblib"
            dump(pipe, model_path)
            summary["best_by_grade"][str(g)] = {
                "best_model": "dummy_const_1.0",
                "metrics": {},
                "model_path": str(model_path).replace("\\", "/"),
                "n_rows_grade": int(len(X)),
                "note": "no folds; fallback const",
            }
            print(f"[ML1 peso_tallo] grado={g} -> fallback const=1.0 (no folds)")
            continue

        best_name = None
        best_score = None
        best_model = None
        all_metrics: dict[str, dict] = {}

        for name, model in models.items():
            pipe = _make_pipeline(model)
            fold_stats = []
            for tr_idx, va_idx in folds:
                X_tr = X.iloc[tr_idx]
                y_tr = y[tr_idx]
                X_va = X.iloc[va_idx]
                y_va = y[va_idx]

                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_va)

                fold_stats.append(_score_fold(y_va, pred))

            maes = np.array([m["mae"] for m in fold_stats], dtype=float)
            wapes = np.array([m["wape"] for m in fold_stats], dtype=float)
            smapes = np.array([m["smape"] for m in fold_stats], dtype=float)

            mae_mean = float(np.nanmean(maes))
            mae_std = float(np.nanstd(maes))
            wape_mean = float(np.nanmean(wapes))
            smape_mean = float(np.nanmean(smapes))

            # score híbrido (MAE + estabilidad)
            score = 0.60 * mae_mean + 0.25 * (wape_mean if np.isfinite(wape_mean) else mae_mean) + 0.15 * mae_std

            all_metrics[name] = {
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "wape_mean": wape_mean,
                "smape_mean": smape_mean,
                "score": float(score),
                "n_rows": int(len(X)),
                "n_folds": int(len(folds)),
            }

            if (best_score is None) or (score < best_score):
                best_score = score
                best_name = name
                best_model = model

        assert best_name is not None and best_model is not None

        best_pipe = _make_pipeline(best_model)
        best_pipe.fit(X, y)

        model_path = out_dir / f"model_grade_{g}.joblib"
        dump(best_pipe, model_path)

        summary["best_by_grade"][str(g)] = {
            "best_model": best_name,
            "metrics": all_metrics,
            "model_path": str(model_path).replace("\\", "/"),
            "n_rows_grade": int(len(X)),
        }

        print(f"[ML1 peso_tallo] grado={g} best={best_name} score={best_score:.6f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"OK -> {out_dir}/ (models + metrics.json)")


if __name__ == "__main__":
    main()
