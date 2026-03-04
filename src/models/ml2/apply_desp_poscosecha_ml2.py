from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino_final.parquet"
IN_REAL_MA = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_FACTOR = EVAL / "backtest_factor_ml2_desp_poscosecha.parquet"
OUT_FINAL = GOLD / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date() -> pd.Timestamp:
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _latest_model(prefix: str) -> Path:
    files = sorted(MODELS.glob(f"{prefix}_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No encuentro modelos en {MODELS} con prefijo {prefix}_*.pkl")
    return files[-1]


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro fecha_post_pred_* en universe.")


def _resolve_factor_desp_ml1(df: pd.DataFrame) -> str:
    for c in ["factor_desp_ml1", "factor_desp_pred_ml1", "factor_desp"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro factor_desp ML1 en universe.")


def main(mode: str = "backtest", model_file: str | None = None) -> None:
    if mode not in {"backtest", "prod"}:
        raise ValueError("--mode debe ser backtest o prod")

    as_of = _as_of_date()

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Universe sin columnas: {sorted(miss)}")

    df["fecha"] = _to_date(df["fecha"])
    df = df.loc[df["fecha"].notna()].copy()

    if mode == "backtest":
        df = df.loc[df["fecha"] <= as_of].copy()

    df["destino"] = _canon_str(df["destino"])
    if pd.api.types.is_numeric_dtype(df["grado"]):
        df["grado"] = _canon_int(df["grado"])
    else:
        df["grado"] = _canon_str(df["grado"])

    fpp = _resolve_fecha_post_pred(df)
    fd_ml1 = _resolve_factor_desp_ml1(df)

    df[fpp] = _to_date(df[fpp])
    df["fecha_post_pred_used"] = df[fpp]

    # Features calendario
    df["dow"] = df["fecha_post_pred_used"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post_pred_used"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post_pred_used"].dt.isocalendar().week.astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[NUM_COLS + CAT_COLS].copy()

    model_path = (MODELS / model_file) if model_file else _latest_model("desp_poscosecha_ml2")
    pipe = load(model_path)

    pred = pd.to_numeric(pd.Series(pipe.predict(X)), errors="coerce")
    df["log_ratio_desp_pred"] = pred

    df["factor_desp_ml1"] = pd.to_numeric(df[fd_ml1], errors="coerce")
    df["factor_desp_final_model"] = df["factor_desp_ml1"] * np.exp(df["log_ratio_desp_pred"].astype(float))
    df["factor_desp_final_model"] = pd.to_numeric(df["factor_desp_final_model"], errors="coerce").clip(lower=0.05, upper=1.00)

    # Real override por fecha_post + destino
    real = read_parquet(IN_REAL_MA).copy()
    real.columns = [str(c).strip() for c in real.columns]
    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["factor_desp_real"] = pd.to_numeric(real["factor_desp"], errors="coerce")

    real_key = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
        .agg(factor_desp_real=("factor_desp_real", "median"))
    )

    df = df.merge(
        real_key,
        left_on=["fecha_post_pred_used", "destino"],
        right_on=["fecha_post", "destino"],
        how="left",
    )

    has_desp_real = df["factor_desp_real"].notna() & df["fecha_post_pred_used"].notna() & (df["fecha_post_pred_used"] <= as_of)
    df["desp_source"] = np.where(has_desp_real, "REAL", "ML2_MODEL")
    df["factor_desp_final"] = np.where(has_desp_real, df["factor_desp_real"], df["factor_desp_final_model"])
    df["factor_desp_final"] = pd.to_numeric(df["factor_desp_final"], errors="coerce").clip(lower=0.05, upper=1.00)

    EVAL.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    fac = df[
        [
            "fecha",
            "fecha_post_pred_used",
            "bloque_base",
            "variedad_canon",
            "grado",
            "destino",
            "factor_desp_ml1",
            "log_ratio_desp_pred",
            "factor_desp_final_model",
            "factor_desp_real",
            "desp_source",
            "factor_desp_final",
        ]
    ].copy()
    fac["model_file"] = model_path.name
    fac["as_of_date"] = as_of
    fac["created_at"] = pd.Timestamp(datetime.now()).normalize()

    if mode == "backtest":
        write_parquet(fac, OUT_FACTOR)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR} rows={len(fac):,}")

    df["as_of_date"] = as_of
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()
    write_parquet(df, OUT_FINAL)

    print(f"[OK] {mode.upper()} final : {OUT_FINAL} rows={len(df):,}")
    print(f"     model={model_path.name} as_of_date={as_of.date()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="backtest", choices=["backtest", "prod"])
    ap.add_argument("--model-file", default=None)
    args = ap.parse_args()

    main(mode=args.mode, model_file=args.model_file)
