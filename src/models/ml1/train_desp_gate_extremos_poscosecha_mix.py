from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime, timezone

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from common.io import read_parquet


FEATURES_MIX_REAL = Path("data/gold/features_mix_real_dia_destino.parquet")
TARGETS = Path("data/silver/dim_mermas_ajuste_fecha_post_destino.parquet")

REG_GATE = Path("models_registry/ml1/desp_poscosecha_gate")


def _mk_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _best_threshold(y_true: np.ndarray, p: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        yhat = (p >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_t)


def _choose_num_cols(
    df: pd.DataFrame,
    cat_cols: list[str],
    min_cov: float = 0.05,
) -> list[str]:
    """
    SOLO numéricas, y excluye explícitamente targets/labels/aux.
    """
    banned = {
        "fecha_post",
        "destino",
        "desp_pct",
        "y_extremo",
        "is_extremo",
        "thr_extremo_dest",
        "thr_extremo",
    } | set(cat_cols)

    cand = [c for c in df.columns if c not in banned]
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


def main(
    q_extreme: float = 0.80,      # <- alineado con tu análisis de regímenes
    min_cov: float = 0.05,
    test_frac: float = 0.20,      # <- split temporal
) -> None:
    # X
    Xdf = read_parquet(FEATURES_MIX_REAL).copy()
    Xdf.columns = [str(c).strip() for c in Xdf.columns]
    if not {"fecha_post", "destino"} <= set(Xdf.columns):
        raise ValueError("FEATURES_MIX_REAL debe tener: fecha_post, destino")

    Xdf["fecha_post"] = pd.to_datetime(Xdf["fecha_post"], errors="coerce").dt.normalize()
    Xdf["destino"] = _canon_destino(Xdf["destino"])
    Xdf = Xdf.dropna(subset=["fecha_post", "destino"]).copy()

    # y
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

    if len(df) < 50:
        raise ValueError(f"Muy pocas filas para train gate: {len(df)}")

    # ordenar para split temporal
    df = df.sort_values(["fecha_post", "destino"]).reset_index(drop=True)

    # label extremo por destino (cuantil)
    thr_by_dest = {
        str(d): float(g["desp_pct"].quantile(q_extreme))
        for d, g in df.groupby("destino", dropna=False)
    }
    df["thr_extremo_dest"] = df["destino"].map(thr_by_dest).astype(float)
    df["y_extremo"] = (df["desp_pct"] >= df["thr_extremo_dest"]).astype(int)

    # calendario
    df["dow"] = df["fecha_post"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post"].dt.isocalendar().week.astype("Int64")

    cat_cols = ["destino"]
    num_cols = _choose_num_cols(df, cat_cols=cat_cols, min_cov=float(min_cov))

    # asegurar calendario siempre
    for c in ["dow", "month", "weekofyear"]:
        if c not in num_cols:
            num_cols.append(c)

    pre = ColumnTransformer(
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
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    # ---- split temporal
    cut = int(len(df) * (1.0 - float(test_frac)))
    cut = max(10, min(cut, len(df) - 10))
    train = df.iloc[:cut].copy()
    test = df.iloc[cut:].copy()

    X_train = train[cat_cols + num_cols]
    X_test = test[cat_cols + num_cols]
    y_train = train["y_extremo"].astype(int).to_numpy()
    y_test = test["y_extremo"].astype(int).to_numpy()

    clf.fit(X_train, y_train)

    p_train = clf.predict_proba(X_train)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    auc_train = float(roc_auc_score(y_train, p_train)) if len(np.unique(y_train)) > 1 else float("nan")
    auc_test = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan")

    gate_thr = _best_threshold(y_train, p_train)

    metrics = {
        "model_type": "gate_extremos_logreg",
        "n_rows": int(len(df)),
        "split": "time",
        "test_frac": float(test_frac),
        "q_extreme": float(q_extreme),
        "thr_by_destino": thr_by_dest,
        "gate_threshold": float(gate_thr),
        "auc_train": auc_train,
        "auc_test": auc_test,
        "features_cat": cat_cols,
        "features_num": num_cols,
        "x_source": str(FEATURES_MIX_REAL),
        "y_source": str(TARGETS),
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
    }

    v = _mk_version()
    (REG_GATE / v).mkdir(parents=True, exist_ok=True)
    dump(clf, REG_GATE / v / "model.joblib")
    with open(REG_GATE / v / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] TRAIN GATE EXTREMOS version={v}")
    print("[INFO] split=time | test_frac=", test_frac, "| q_extreme=", q_extreme)
    print("[INFO] AUC train/test:", auc_train, auc_test)
    print("[INFO] gate_threshold:", gate_thr)
    print("[INFO] thr_by_destino:", thr_by_dest)
    print("[INFO] features_num:", num_cols)


if __name__ == "__main__":
    main()
