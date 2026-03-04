from __future__ import annotations

import argparse
import inspect
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common.io import read_parquet, write_parquet
from models.ml1.train_multitask_nn_ml1 import CAT_COLS, NUM_COLS, TARGET_CLIPS, TARGET_COLS
from models.ml1.zero_inflated import ZeroInflatedRegressor as StableZeroInflatedRegressor


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
IN_DS = DATA_DIR / "gold" / "ml1_nn" / "ds_ml1_nn_v1.parquet"
EVAL_DIR = DATA_DIR / "eval" / "ml1_nn"
MODELS_DIR = DATA_DIR / "models" / "ml1_nn_target_models"

FIELD_TARGETS = {
    "target_d_start",
    "target_n_harvest_days",
    "target_factor_tallos_dia",
    "target_share_grado",
    "target_factor_peso_tallo",
}

POST_TARGETS = {
    "target_dh_dias",
    "target_factor_hidr",
    "target_factor_desp",
    "target_factor_ajuste",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: object
    dense_ohe: bool
    max_train_rows: int | None = None


class ZeroInflatedRegressor(BaseEstimator, RegressorMixin):
    """Two-stage regressor: P(y>0) * E[y|y>0], with optional hard-zero threshold."""

    def __init__(
        self,
        classifier=None,
        regressor=None,
        zero_threshold: float = 0.35,
        min_positive: float = 1e-6,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.zero_threshold = float(zero_threshold)
        self.min_positive = float(min_positive)

    def fit(self, x, y, sample_weight=None):
        y_arr = np.asarray(y, dtype=np.float64)
        pos = y_arr > self.min_positive

        if self.classifier is None:
            self.classifier_ = HistGradientBoostingClassifier(
                random_state=42,
                max_depth=5,
                learning_rate=0.05,
                max_iter=250,
                min_samples_leaf=10,
            )
        else:
            self.classifier_ = clone(self.classifier)

        if self.regressor is None:
            self.regressor_ = HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=42,
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=10,
            )
        else:
            self.regressor_ = clone(self.regressor)

        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if sw is not None and len(sw) != len(y_arr):
            sw = None

        # classifier on binary target
        y_bin = pos.astype(np.float64)
        try:
            if sw is None:
                self.classifier_.fit(x, y_bin)
            else:
                self.classifier_.fit(x, y_bin, sample_weight=sw)
        except TypeError:
            self.classifier_.fit(x, y_bin)

        # regressor on positives
        if bool(pos.any()):
            x_pos = x[pos]
            y_pos = y_arr[pos]
            sw_pos = sw[pos] if sw is not None else None
            try:
                if sw_pos is None:
                    self.regressor_.fit(x_pos, y_pos)
                else:
                    self.regressor_.fit(x_pos, y_pos, sample_weight=sw_pos)
            except TypeError:
                self.regressor_.fit(x_pos, y_pos)
            self.has_positive_ = True
            self.pos_mean_ = float(np.average(y_pos, weights=sw_pos)) if sw_pos is not None and sw_pos.sum() > 0 else float(np.mean(y_pos))
        else:
            self.has_positive_ = False
            self.pos_mean_ = 0.0

        return self

    def _predict_pos_proba(self, x) -> np.ndarray:
        if hasattr(self.classifier_, "predict_proba"):
            p = np.asarray(self.classifier_.predict_proba(x), dtype=np.float64)[:, 1]
        else:
            p = np.asarray(self.classifier_.predict(x), dtype=np.float64)
        return np.clip(p, 0.0, 1.0)

    def predict(self, x):
        p_pos = self._predict_pos_proba(x)
        if self.has_positive_:
            y_pos = np.asarray(self.regressor_.predict(x), dtype=np.float64)
            y_pos = np.where(np.isfinite(y_pos), y_pos, self.pos_mean_)
            y_pos = np.clip(y_pos, 0.0, None)
        else:
            y_pos = np.full(len(p_pos), self.pos_mean_, dtype=np.float64)

        y_hat = p_pos * y_pos
        y_hat = np.where(p_pos < self.zero_threshold, 0.0, y_hat)
        return y_hat.astype(np.float64)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("benchmark_per_target_models_ml1")
    ap.add_argument("--dataset", default=str(IN_DS))
    ap.add_argument("--val-quantile", type=float, default=0.70)
    ap.add_argument("--min-val-n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--targets",
        default=",".join(TARGET_COLS),
        help="Comma-separated target names.",
    )
    ap.add_argument(
        "--save-models",
        action="store_true",
        help="Persist selected model per target.",
    )
    ap.add_argument(
        "--include-rf",
        action="store_true",
        help="Include RandomForest candidate (slower).",
    )
    ap.add_argument(
        "--include-extra",
        action="store_true",
        help="Include heavier candidates (gbr/etr/hgb_deep).",
    )
    ap.add_argument(
        "--disable-zero-inflated",
        action="store_true",
        help="Disable zero-inflated candidate for target_factor_tallos_dia.",
    )
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _prepare_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = read_parquet(path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha_evento"} | set(TARGET_COLS)
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Dataset missing required columns: {sorted(miss)}")

    df["fecha_evento"] = pd.to_datetime(df["fecha_evento"], errors="coerce").dt.normalize()
    if "has_any_target" in df.columns:
        df = df[df["has_any_target"] == 1].copy()
    df = df[df["fecha_evento"].notna()].copy()

    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = _canon_str(df[c])

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for t in TARGET_COLS:
        df[t] = pd.to_numeric(df[t], errors="coerce")
        mcol = f"mask_{t}"
        if mcol not in df.columns:
            df[mcol] = df[t].notna().astype("float32")
        df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0).clip(lower=0.0).astype(np.float32)

    return df.reset_index(drop=True)


def _build_preprocessor(num_cols: list[str], cat_cols: list[str], dense_ohe: bool) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_ohe)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=not dense_ohe)

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    return ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )


def _weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "n": 0.0,
            "w": 0.0,
            "sse": np.nan,
            "sst": np.nan,
        }

    yt = y_true[m]
    yp = y_pred[m]
    ww = w[m]
    wsum = float(ww.sum())
    err = yp - yt
    mae = float(np.sum(np.abs(err) * ww) / max(wsum, 1e-12))
    sse = float(np.sum((err**2) * ww))
    rmse = float(np.sqrt(sse / max(wsum, 1e-12)))
    ybar = float(np.sum(yt * ww) / max(wsum, 1e-12))
    sst = float(np.sum(((yt - ybar) ** 2) * ww))
    r2 = float(1.0 - sse / sst) if sst > 1e-12 else np.nan
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n": float(m.sum()),
        "w": wsum,
        "sse": sse,
        "sst": sst,
    }


def _safe_target_list(raw: str) -> list[str]:
    out: list[str] = []
    for t in [x.strip() for x in raw.split(",") if x.strip()]:
        if t not in TARGET_COLS:
            raise ValueError(f"Unknown target '{t}'. Allowed: {TARGET_COLS}")
        out.append(t)
    if not out:
        raise ValueError("No targets provided.")
    return out


def _build_candidates(
    seed: int,
    include_rf: bool,
    include_extra: bool,
    target: str,
    include_zero_inflated: bool,
) -> list[Candidate]:
    cands = [
        Candidate(
            name="dummy_median",
            estimator=DummyRegressor(strategy="median"),
            dense_ohe=False,
        ),
        Candidate(
            name="ridge",
            estimator=Ridge(alpha=4.0, random_state=seed),
            dense_ohe=False,
        ),
        Candidate(
            name="hgb",
            estimator=HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=seed,
                max_depth=6,
                learning_rate=0.05,
                max_iter=350,
                min_samples_leaf=10,
                l2_regularization=0.8,
                early_stopping=True,
                validation_fraction=0.15,
            ),
            dense_ohe=True,
        ),
    ]
    if include_extra:
        cands.extend(
            [
                Candidate(
                    name="hgb_deep",
                    estimator=HistGradientBoostingRegressor(
                        loss="squared_error",
                        random_state=seed,
                        max_depth=10,
                        learning_rate=0.03,
                        max_iter=450,
                        min_samples_leaf=6,
                        l2_regularization=0.3,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                    dense_ohe=True,
                ),
                Candidate(
                    name="gbr",
                    estimator=GradientBoostingRegressor(
                        random_state=seed,
                        learning_rate=0.05,
                        n_estimators=400,
                        max_depth=4,
                        min_samples_leaf=15,
                        subsample=0.9,
                        loss="squared_error",
                    ),
                    dense_ohe=True,
                    max_train_rows=120_000,
                ),
                Candidate(
                    name="etr",
                    estimator=ExtraTreesRegressor(
                        n_estimators=160,
                        max_depth=None,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                    dense_ohe=False,
                    max_train_rows=120_000,
                ),
            ]
        )
    if include_rf:
        cands.append(
            Candidate(
                name="rf",
                estimator=RandomForestRegressor(
                    n_estimators=80,
                    max_depth=14,
                    min_samples_leaf=3,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
                dense_ohe=False,
                max_train_rows=120_000,
            )
        )
    if include_zero_inflated and target == "target_factor_tallos_dia":
        cands.append(
            Candidate(
                name="hgb_zero_inflated",
                estimator=StableZeroInflatedRegressor(
                    classifier=HistGradientBoostingClassifier(
                        random_state=seed,
                        max_depth=5,
                        learning_rate=0.05,
                        max_iter=260,
                        min_samples_leaf=12,
                    ),
                    regressor=HistGradientBoostingRegressor(
                        loss="squared_error",
                        random_state=seed,
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=320,
                        min_samples_leaf=10,
                        l2_regularization=0.6,
                        early_stopping=True,
                        validation_fraction=0.15,
                    ),
                    zero_threshold=0.38,
                    min_positive=0.03,
                ),
                dense_ohe=True,
            )
        )
    return cands


def _fit_with_optional_weights(pipe: Pipeline, x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> Pipeline:
    est = pipe.named_steps["model"]
    supports_weight = "sample_weight" in inspect.signature(est.fit).parameters
    if supports_weight:
        pipe.fit(x, y, model__sample_weight=w)
    else:
        pipe.fit(x, y)
    return pipe


def _subsample_train(
    x: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if max_rows is None or len(x) <= max_rows:
        return x, y, w
    rng = np.random.default_rng(seed)
    probs = np.asarray(w, dtype=np.float64)
    probs = np.where(np.isfinite(probs) & (probs > 0), probs, 0.0)
    if probs.sum() <= 0:
        idx = rng.choice(len(x), size=max_rows, replace=False)
    else:
        probs = probs / probs.sum()
        idx = rng.choice(len(x), size=max_rows, replace=False, p=probs)
    idx = np.sort(idx.astype(np.int64))
    return x.iloc[idx], y[idx], w[idx]


def _weighted_cycle_day_metrics_for_tallos(
    sub_val: pd.DataFrame,
    target_col: str,
    mask_col: str,
    pred_val: np.ndarray,
) -> dict[str, float]:
    cols = ["ciclo_id", "fecha_evento", target_col, mask_col]
    miss = [c for c in cols if c not in sub_val.columns]
    if miss:
        return {
            "mae_day": np.nan,
            "rmse_day": np.nan,
            "r2_day": np.nan,
            "w_day": 0.0,
            "n_day": 0.0,
            "zero_fp_rate": np.nan,
            "zero_fn_rate": np.nan,
        }

    v = sub_val[cols].copy()
    v["pred"] = pd.to_numeric(pd.Series(pred_val, index=v.index), errors="coerce")
    v[target_col] = pd.to_numeric(v[target_col], errors="coerce")
    v[mask_col] = pd.to_numeric(v[mask_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    v["fecha_evento"] = pd.to_datetime(v["fecha_evento"], errors="coerce").dt.normalize()
    v = v.dropna(subset=["ciclo_id", "fecha_evento"])
    if v.empty:
        return {
            "mae_day": np.nan,
            "rmse_day": np.nan,
            "r2_day": np.nan,
            "w_day": 0.0,
            "n_day": 0.0,
            "zero_fp_rate": np.nan,
            "zero_fn_rate": np.nan,
        }

    g = (
        v.groupby(["ciclo_id", "fecha_evento"], dropna=False, as_index=False)
        .agg(
            y_true=(target_col, "median"),
            y_pred=("pred", "median"),
            w=(mask_col, "max"),
        )
    )
    mt = _weighted_metrics(
        y_true=pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=np.float64),
        y_pred=pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=np.float64),
        w=pd.to_numeric(g["w"], errors="coerce").to_numpy(dtype=np.float64),
    )
    y_true_d = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=np.float64)
    y_pred_d = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
    w_d = pd.to_numeric(g["w"], errors="coerce").to_numpy(dtype=np.float64)
    m = np.isfinite(y_true_d) & np.isfinite(y_pred_d) & np.isfinite(w_d) & (w_d > 0.0)
    zero_true = y_true_d <= 0.03
    zero_pred = y_pred_d <= 0.05
    if bool(m.any()):
        w_zero = w_d[m & zero_true]
        w_nonzero = w_d[m & (~zero_true)]
        fp_w = float(w_d[m & (~zero_true) & zero_pred].sum())
        fn_w = float(w_d[m & zero_true & (~zero_pred)].sum())
        denom_fp = float(w_nonzero.sum())
        denom_fn = float(w_zero.sum())
        zero_fp_rate = (fp_w / denom_fp) if denom_fp > 1e-12 else 0.0
        zero_fn_rate = (fn_w / denom_fn) if denom_fn > 1e-12 else 0.0
    else:
        zero_fp_rate = np.nan
        zero_fn_rate = np.nan

    return {
        "mae_day": float(mt["mae"]) if np.isfinite(mt["mae"]) else np.nan,
        "rmse_day": float(mt["rmse"]) if np.isfinite(mt["rmse"]) else np.nan,
        "r2_day": float(mt["r2"]) if np.isfinite(mt["r2"]) else np.nan,
        "w_day": float(mt["w"]) if np.isfinite(mt["w"]) else 0.0,
        "n_day": float(mt["n"]) if np.isfinite(mt["n"]) else 0.0,
        "zero_fp_rate": float(zero_fp_rate) if np.isfinite(zero_fp_rate) else np.nan,
        "zero_fn_rate": float(zero_fn_rate) if np.isfinite(zero_fn_rate) else np.nan,
    }


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    df = _prepare_df(Path(args.dataset))
    targets = _safe_target_list(args.targets)
    all_candidate_names: set[str] = set()
    cutoff_global = pd.to_datetime(df["fecha_evento"].quantile(args.val_quantile)).normalize()

    rows: list[dict[str, object]] = []
    winners: list[dict[str, object]] = []
    global_sse = 0.0
    global_sst = 0.0
    global_abs = 0.0
    global_w = 0.0
    domain_stats: dict[str, dict[str, float]] = {
        "field": {"sse": 0.0, "sst": 0.0, "abs": 0.0, "w": 0.0},
        "post": {"sse": 0.0, "sst": 0.0, "abs": 0.0, "w": 0.0},
    }

    models_out_dir = MODELS_DIR / run_id
    if args.save_models:
        models_out_dir.mkdir(parents=True, exist_ok=True)

    for t in targets:
        candidates = _build_candidates(
            seed=args.seed,
            include_rf=bool(args.include_rf),
            include_extra=bool(args.include_extra),
            target=t,
            include_zero_inflated=not bool(args.disable_zero_inflated),
        )
        all_candidate_names.update([c.name for c in candidates])

        mcol = f"mask_{t}"
        sub = df[(df[mcol] > 0) & df[t].notna()].copy()
        if sub.empty:
            continue

        tr = sub["fecha_evento"] < cutoff_global
        va = ~tr
        split_mode = "global_quantile"
        if int(va.sum()) < args.min_val_n or int(tr.sum()) < args.min_val_n:
            cutoff_t = pd.to_datetime(sub["fecha_evento"].quantile(args.val_quantile)).normalize()
            tr = sub["fecha_evento"] < cutoff_t
            va = ~tr
            split_mode = "target_quantile_fallback"

        if int(va.sum()) == 0 or int(tr.sum()) == 0:
            order = np.argsort(sub["fecha_evento"].to_numpy())
            cut = int(len(order) * 0.8)
            tr_idx = order[:cut]
            va_idx = order[cut:]
            tr = np.zeros(len(sub), dtype=bool)
            va = np.zeros(len(sub), dtype=bool)
            tr[tr_idx] = True
            va[va_idx] = True
            split_mode = "target_order_fallback"

        active_num = [c for c in NUM_COLS if pd.to_numeric(sub.loc[tr, c], errors="coerce").notna().any()]
        active_cat = [c for c in CAT_COLS if sub.loc[tr, c].notna().any()]
        if not active_num and not active_cat:
            rows.append(
                {
                    "run_id": run_id,
                    "target": t,
                    "model_name": "none",
                    "split_mode": split_mode,
                    "cutoff_fecha_evento": str(cutoff_global.date()),
                    "n_train": int(tr.sum()),
                    "n_val": int(va.sum()),
                    "w_train": float(0.0),
                    "w_val": float(0.0),
                    "status": "error",
                    "error": "No active features for target train split.",
                }
            )
            continue

        x_tr = sub.loc[tr, active_num + active_cat]
        x_va = sub.loc[va, active_num + active_cat]
        y_tr = pd.to_numeric(sub.loc[tr, t], errors="coerce").to_numpy(dtype=np.float64)
        y_va = pd.to_numeric(sub.loc[va, t], errors="coerce").to_numpy(dtype=np.float64)
        w_tr = pd.to_numeric(sub.loc[tr, mcol], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        w_va = pd.to_numeric(sub.loc[va, mcol], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

        best: dict[str, object] | None = None
        best_pipe: Pipeline | None = None
        lo, hi = TARGET_CLIPS[t]

        for ci, cand in enumerate(candidates):
            pre = _build_preprocessor(num_cols=active_num, cat_cols=active_cat, dense_ohe=cand.dense_ohe)
            pipe = Pipeline([("pre", pre), ("model", cand.estimator)])

            x_fit, y_fit, w_fit = _subsample_train(
                x=x_tr,
                y=y_tr,
                w=w_tr,
                max_rows=cand.max_train_rows,
                seed=args.seed + ci,
            )

            try:
                pipe = _fit_with_optional_weights(pipe, x_fit, y_fit, w_fit)
            except Exception as exc:
                rows.append(
                    {
                        "run_id": run_id,
                        "target": t,
                        "model_name": cand.name,
                        "n_features_num": int(len(active_num)),
                        "n_features_cat": int(len(active_cat)),
                        "split_mode": split_mode,
                        "cutoff_fecha_evento": str(cutoff_global.date()),
                        "n_train": int(len(x_tr)),
                        "n_val": int(len(x_va)),
                        "w_train": float(w_tr.sum()),
                        "w_val": float(w_va.sum()),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            pred_tr = np.clip(pipe.predict(x_tr), lo, hi)
            pred_va = np.clip(pipe.predict(x_va), lo, hi)

            mt_tr = _weighted_metrics(y_true=y_tr, y_pred=pred_tr, w=w_tr)
            mt_va = _weighted_metrics(y_true=y_va, y_pred=pred_va, w=w_va)
            mt_day = (
                _weighted_cycle_day_metrics_for_tallos(
                    sub_val=sub.loc[va].copy(),
                    target_col=t,
                    mask_col=mcol,
                    pred_val=pred_va,
                )
                if t == "target_factor_tallos_dia"
                else {
                    "mae_day": np.nan,
                    "rmse_day": np.nan,
                    "r2_day": np.nan,
                    "w_day": 0.0,
                    "n_day": 0.0,
                    "zero_fp_rate": np.nan,
                    "zero_fn_rate": np.nan,
                }
            )

            row = {
                "run_id": run_id,
                "target": t,
                "model_name": cand.name,
                "n_features_num": int(len(active_num)),
                "n_features_cat": int(len(active_cat)),
                "split_mode": split_mode,
                "cutoff_fecha_evento": str(cutoff_global.date()),
                "n_train": int(len(x_tr)),
                "n_val": int(len(x_va)),
                "w_train": float(w_tr.sum()),
                "w_val": float(w_va.sum()),
                "val_mae": float(mt_va["mae"]),
                "val_rmse": float(mt_va["rmse"]),
                "val_r2": float(mt_va["r2"]) if np.isfinite(mt_va["r2"]) else np.nan,
                "val_mae_day": float(mt_day["mae_day"]) if np.isfinite(mt_day["mae_day"]) else np.nan,
                "val_rmse_day": float(mt_day["rmse_day"]) if np.isfinite(mt_day["rmse_day"]) else np.nan,
                "val_r2_day": float(mt_day["r2_day"]) if np.isfinite(mt_day["r2_day"]) else np.nan,
                "n_val_day": float(mt_day["n_day"]),
                "w_val_day": float(mt_day["w_day"]),
                "val_zero_fp_rate": float(mt_day["zero_fp_rate"]) if np.isfinite(mt_day["zero_fp_rate"]) else np.nan,
                "val_zero_fn_rate": float(mt_day["zero_fn_rate"]) if np.isfinite(mt_day["zero_fn_rate"]) else np.nan,
                "train_mae": float(mt_tr["mae"]),
                "train_rmse": float(mt_tr["rmse"]),
                "train_r2": float(mt_tr["r2"]) if np.isfinite(mt_tr["r2"]) else np.nan,
                "val_sse": float(mt_va["sse"]) if np.isfinite(mt_va["sse"]) else np.nan,
                "val_sst": float(mt_va["sst"]) if np.isfinite(mt_va["sst"]) else np.nan,
                "status": "ok",
                "error": None,
            }
            rows.append(row)

            if t == "target_factor_tallos_dia":
                zfp = row["val_zero_fp_rate"] if np.isfinite(row["val_zero_fp_rate"]) else 1.0
                zfn = row["val_zero_fn_rate"] if np.isfinite(row["val_zero_fn_rate"]) else 1.0
                zerr = 0.7 * zfp + 0.3 * zfn
                cur = (
                    zerr,
                    row["val_mae_day"] if np.isfinite(row["val_mae_day"]) else np.inf,
                    row["val_mae"],
                    -row["val_r2"] if np.isfinite(row["val_r2"]) else np.inf,
                )
            else:
                cur = (row["val_mae"], -row["val_r2"] if np.isfinite(row["val_r2"]) else np.inf)
            if best is None:
                best = row
                best_pipe = pipe
            else:
                if t == "target_factor_tallos_dia":
                    pfp = best["val_zero_fp_rate"] if np.isfinite(best["val_zero_fp_rate"]) else 1.0
                    pfn = best["val_zero_fn_rate"] if np.isfinite(best["val_zero_fn_rate"]) else 1.0
                    perr = 0.7 * pfp + 0.3 * pfn
                    prev = (
                        perr,
                        best["val_mae_day"] if np.isfinite(best["val_mae_day"]) else np.inf,
                        best["val_mae"],
                        -best["val_r2"] if np.isfinite(best["val_r2"]) else np.inf,
                    )
                else:
                    prev = (best["val_mae"], -best["val_r2"] if np.isfinite(best["val_r2"]) else np.inf)
                if cur < prev:
                    best = row
                    best_pipe = pipe

        if best is None:
            continue

        winners.append(
            {
                "run_id": run_id,
                "target": t,
                "selected_model": best["model_name"],
                "val_mae": best["val_mae"],
                "val_rmse": best["val_rmse"],
                "val_r2": best["val_r2"],
                "val_mae_day": best.get("val_mae_day", np.nan),
                "val_rmse_day": best.get("val_rmse_day", np.nan),
                "val_r2_day": best.get("val_r2_day", np.nan),
                "val_zero_fp_rate": best.get("val_zero_fp_rate", np.nan),
                "val_zero_fn_rate": best.get("val_zero_fn_rate", np.nan),
                "n_val": best["n_val"],
                "w_val": best["w_val"],
                "split_mode": split_mode,
            }
        )
        if np.isfinite(best.get("val_sse", np.nan)) and np.isfinite(best.get("val_sst", np.nan)):
            global_sse += float(best["val_sse"])
            global_sst += float(best["val_sst"])

        # rebuild val abs weighted from selected model row
        global_abs += float(best["val_mae"]) * float(best["w_val"])
        global_w += float(best["w_val"])

        domain = "field" if t in FIELD_TARGETS else ("post" if t in POST_TARGETS else "post")
        domain_stats[domain]["abs"] += float(best["val_mae"]) * float(best["w_val"])
        domain_stats[domain]["w"] += float(best["w_val"])
        if np.isfinite(best.get("val_sse", np.nan)) and np.isfinite(best.get("val_sst", np.nan)):
            domain_stats[domain]["sse"] += float(best["val_sse"])
            domain_stats[domain]["sst"] += float(best["val_sst"])

        if args.save_models and best_pipe is not None:
            out_m = models_out_dir / f"{t}__{best['model_name']}.joblib"
            dump(best_pipe, out_m)

    bench = pd.DataFrame(rows)
    win = pd.DataFrame(winners)
    if not bench.empty:
        bench = bench.sort_values(["target", "status", "val_mae"], kind="mergesort").reset_index(drop=True)
    if not win.empty:
        win = win.sort_values(["target"], kind="mergesort").reset_index(drop=True)

    weighted_mae_global = float(global_abs / global_w) if global_w > 0 else np.nan
    weighted_r2_global = float(1.0 - global_sse / global_sst) if global_sst > 1e-12 else np.nan
    r2_avg_global = float(np.nanmean(win["val_r2"].to_numpy(dtype=float))) if not win.empty else np.nan

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(Path(args.dataset).resolve()),
        "targets": targets,
        "val_quantile": float(args.val_quantile),
        "min_val_n": int(args.min_val_n),
        "seed": int(args.seed),
        "n_candidates_total": int(len(all_candidate_names)),
        "candidate_names": sorted(all_candidate_names),
        "zero_inflated_enabled": bool(not args.disable_zero_inflated),
        "selected_models": {str(r["target"]): str(r["selected_model"]) for r in winners},
        "global_metrics": {
            "weighted_mae_global": weighted_mae_global,
            "weighted_r2_global": weighted_r2_global,
            "r2_avg_targets": r2_avg_global,
            "w_val_total": float(global_w),
        },
        "domain_metrics": {
            "field": {
                "weighted_mae": (
                    float(domain_stats["field"]["abs"] / domain_stats["field"]["w"])
                    if domain_stats["field"]["w"] > 0
                    else np.nan
                ),
                "weighted_r2": (
                    float(1.0 - domain_stats["field"]["sse"] / domain_stats["field"]["sst"])
                    if domain_stats["field"]["sst"] > 1e-12
                    else np.nan
                ),
                "w_val_total": float(domain_stats["field"]["w"]),
            },
            "post": {
                "weighted_mae": (
                    float(domain_stats["post"]["abs"] / domain_stats["post"]["w"])
                    if domain_stats["post"]["w"] > 0
                    else np.nan
                ),
                "weighted_r2": (
                    float(1.0 - domain_stats["post"]["sse"] / domain_stats["post"]["sst"])
                    if domain_stats["post"]["sst"] > 1e-12
                    else np.nan
                ),
                "w_val_total": float(domain_stats["post"]["w"]),
            },
        },
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_bench = EVAL_DIR / f"ml1_target_model_benchmark_{run_id}.parquet"
    out_win = EVAL_DIR / f"ml1_target_model_winners_{run_id}.parquet"
    out_summary = EVAL_DIR / f"ml1_target_model_benchmark_summary_{run_id}.json"
    out_hybrid = EVAL_DIR / f"ml1_target_model_hybrid_summary_{run_id}.json"
    write_parquet(bench, out_bench)
    write_parquet(win, out_win)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    out_summary.write_text(payload, encoding="utf-8")
    out_hybrid.write_text(payload, encoding="utf-8")

    print(f"[OK] benchmark: {out_bench}")
    print(f"[OK] winners  : {out_win}")
    print(f"[OK] summary  : {out_summary}")
    print(f"[OK] hybrid   : {out_hybrid}")
    print(f"     weighted_mae_global={weighted_mae_global:.6f}")
    print(f"     weighted_r2_global={weighted_r2_global:.6f}")
    print(f"     r2_avg_targets={r2_avg_global:.6f}")
    if args.save_models:
        print(f"     selected models saved under: {models_out_dir}")


if __name__ == "__main__":
    main()
