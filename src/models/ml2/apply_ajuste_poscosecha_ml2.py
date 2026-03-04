from __future__ import annotations

import argparse
import json
from pathlib import Path

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
MODELS_DIR = DATA / "models" / "ml2"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"
IN_REAL_MA = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_FINAL = GOLD / "pred_poscosecha_ml2_ajuste_grado_dia_bloque_destino_final.parquet"
OUT_BACKTEST = EVAL / "backtest_factor_ml2_ajuste_poscosecha.parquet"


def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _as_of_date_naive() -> pd.Timestamp:
    t = _to_naive_utc(pd.Timestamp.utcnow())
    return t.normalize() - pd.Timedelta(days=1)


def _to_date_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        pass
    return dt.dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro columna fecha_post_pred (final/used/ml1/seed).")


def _resolve_ajuste_ml1(df: pd.DataFrame) -> str:
    for c in ["ajuste_ml1", "factor_ajuste_ml1", "factor_ajuste_seed", "factor_ajuste"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro ajuste ML1 (esperaba ajuste_ml1 o factor_ajuste_*).")


def _weight_series(df: pd.DataFrame) -> pd.Series:
    for c in ["tallos_w", "tallos", "tallos_total_ml2", "tallos_total"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(1.0, index=df.index, dtype="float64")


def _latest_model(prefix: str = "ajuste_poscosecha_ml2_") -> Path:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"No existe {MODELS_DIR}")
    files = sorted(MODELS_DIR.glob(f"{prefix}*.pkl"))
    if not files:
        raise FileNotFoundError(f"No encontre modelos en {MODELS_DIR} con prefijo {prefix}")
    return files[-1]


def _meta_path_from_model(model_path: Path) -> Path:
    p = str(model_path)
    if p.endswith(".pkl"):
        return Path(p.replace(".pkl", "_meta.json"))
    return model_path.with_suffix(".json")


def main(mode: str = "backtest", model_file: str | None = None) -> None:
    as_of = _as_of_date_naive()

    model_path = Path(model_file) if model_file else _latest_model()
    meta_path = _meta_path_from_model(model_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "fecha" not in df.columns or "destino" not in df.columns:
        raise ValueError("Universe debe tener fecha y destino.")

    df["fecha"] = _to_date_naive(df["fecha"])
    df["destino"] = _canon_str(df["destino"])

    fecha_post_col = _resolve_fecha_post_pred(df)
    df[fecha_post_col] = _to_date_naive(df[fecha_post_col])

    ajuste_ml1_col = _resolve_ajuste_ml1(df)
    df[ajuste_ml1_col] = pd.to_numeric(df[ajuste_ml1_col], errors="coerce")

    df = df[df["fecha"].notna()].copy()
    if mode.lower() == "backtest":
        df = df[df["fecha"] <= as_of].copy()

    df["dow"] = df[fecha_post_col].dt.dayofweek.astype("Int64")
    df["month"] = df[fecha_post_col].dt.month.astype("Int64")
    df["weekofyear"] = df[fecha_post_col].dt.isocalendar().week.astype("Int64")
    df["w"] = _weight_series(df)

    num_cols = ["dow", "month", "weekofyear"]
    cat_cols = ["destino"]
    if "grado" in df.columns:
        cat_cols.append("grado")
        df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[num_cols + cat_cols]

    model = load(model_path)
    pred = model.predict(X)

    clip = float(meta.get("clip_delta", 1.2))
    df["delta_log_ajuste_ml2_raw"] = pd.to_numeric(pd.Series(pred), errors="coerce")
    df["delta_log_ajuste_ml2"] = df["delta_log_ajuste_ml2_raw"].clip(lower=-clip, upper=clip)

    base = pd.to_numeric(df[ajuste_ml1_col], errors="coerce").replace(0, np.nan)
    df["factor_ajuste_final_model"] = (base * np.exp(df["delta_log_ajuste_ml2"])).astype(float)

    clip_factor = meta.get("clip_factor_apply", [0.50, 2.00])
    try:
        lo, hi = float(clip_factor[0]), float(clip_factor[1])
    except Exception:
        lo, hi = 0.50, 2.00
    df["factor_ajuste_final_model"] = df["factor_ajuste_final_model"].clip(lower=lo, upper=hi)

    # Real override por fecha_post + destino
    real = read_parquet(IN_REAL_MA).copy()
    real.columns = [str(c).strip() for c in real.columns]
    real["fecha_post"] = _to_date_naive(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])

    if "factor_ajuste" in real.columns:
        real["factor_ajuste_real"] = pd.to_numeric(real["factor_ajuste"], errors="coerce")
    elif "ajuste" in real.columns:
        # ajuste es el inverso de factor_ajuste en este proyecto
        aj = pd.to_numeric(real["ajuste"], errors="coerce")
        real["factor_ajuste_real"] = np.where(aj > 0, 1.0 / aj, np.nan)
    else:
        raise KeyError("No encuentro factor_ajuste ni ajuste en dim_mermas_ajuste_fecha_post_destino.")

    real_key = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
        .agg(factor_ajuste_real=("factor_ajuste_real", "median"))
    )

    df["fecha_post_pred_used"] = df[fecha_post_col]
    df = df.merge(
        real_key,
        left_on=["fecha_post_pred_used", "destino"],
        right_on=["fecha_post", "destino"],
        how="left",
    )

    has_real = df["factor_ajuste_real"].notna() & df["fecha_post_pred_used"].notna() & (df["fecha_post_pred_used"] <= as_of)
    df["ajuste_source"] = np.where(has_real, "REAL", "ML2_MODEL")
    df["factor_ajuste_final"] = np.where(has_real, df["factor_ajuste_real"], df["factor_ajuste_final_model"])
    df["factor_ajuste_final"] = pd.to_numeric(df["factor_ajuste_final"], errors="coerce").clip(lower=lo, upper=hi)

    df["ml2_ajuste_model_file"] = model_path.name
    df["ml2_ajuste_clip_delta"] = clip
    df["as_of_date"] = as_of
    df["created_at"] = pd.Timestamp.utcnow()

    if mode.lower() == "backtest":
        bt_cols = [
            "fecha", "fecha_post_pred_used", "destino", "delta_log_ajuste_ml2", "w",
            "factor_ajuste_final_model", "factor_ajuste_real", "ajuste_source", "factor_ajuste_final",
            "as_of_date", "created_at"
        ]
        if "grado" in df.columns:
            bt_cols.insert(3, "grado")
        bt = df[bt_cols].copy()
        EVAL.mkdir(parents=True, exist_ok=True)
        write_parquet(bt, OUT_BACKTEST)
        print(f"[OK] BACKTEST factor: {OUT_BACKTEST} rows={len(bt):,}")

    write_parquet(df, OUT_FINAL)
    print(f"[OK] {mode.upper()} final : {OUT_FINAL} rows={len(df):,}")
    print(f"     model={model_path.name} clip_delta=+-{clip} as_of_date={as_of.date()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="backtest", choices=["backtest", "prod"])
    p.add_argument("--model_file", default=None)
    args = p.parse_args()
    main(mode=args.mode, model_file=args.model_file)
