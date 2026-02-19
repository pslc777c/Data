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
EVAL = DATA / "eval" / "ml2"
MODELS_DIR = DATA / "models" / "ml2"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"

OUT_FINAL = GOLD / "pred_poscosecha_ml2_ajuste_grado_dia_bloque_destino_final.parquet"
OUT_BACKTEST = EVAL / "backtest_factor_ml2_ajuste_poscosecha.parquet"


# ----------------------------
# TZ-safe helpers
# ----------------------------
def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _as_of_date_naive() -> pd.Timestamp:
    # regla: hoy-1
    t = _to_naive_utc(pd.Timestamp.utcnow())
    return (t.normalize() - pd.Timedelta(days=1))


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
        raise FileNotFoundError(f"No encontré modelos en {MODELS_DIR} con prefijo {prefix}")
    return files[-1]


def _meta_path_from_model(model_path: Path) -> Path:
    # ✅ NO usar with_suffix("_meta.json") (inválido)
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

    # ✅ tz-safe
    df["fecha"] = _to_date_naive(df["fecha"])
    df["destino"] = _canon_str(df["destino"])

    fecha_post_col = _resolve_fecha_post_pred(df)
    df[fecha_post_col] = _to_date_naive(df[fecha_post_col])

    ajuste_ml1_col = _resolve_ajuste_ml1(df)
    df[ajuste_ml1_col] = pd.to_numeric(df[ajuste_ml1_col], errors="coerce")

    # ✅ filtro as_of consistente
    df = df[df["fecha"].notna()].copy()

    # features calendario
    df["dow"] = df[fecha_post_col].dt.dayofweek.astype("Int64")
    df["month"] = df[fecha_post_col].dt.month.astype("Int64")
    df["weekofyear"] = df[fecha_post_col].dt.isocalendar().week.astype("Int64")
    df["w"] = _weight_series(df)

    # columnas para el modelo
    NUM_COLS = ["dow", "month", "weekofyear"]
    CAT_COLS = ["destino"]
    if "grado" in df.columns:
        CAT_COLS.append("grado")
        df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    X = df[NUM_COLS + CAT_COLS]

    model = load(model_path)
    pred = model.predict(X)

    # delta: log(real/ml1)
    clip = float(meta.get("clip_delta", 1.2))
    df["delta_log_ajuste_ml2_raw"] = pd.to_numeric(pd.Series(pred), errors="coerce")
    df["delta_log_ajuste_ml2"] = df["delta_log_ajuste_ml2_raw"].clip(lower=-clip, upper=clip)

    # aplicar: ajuste_final = ajuste_ml1 * exp(delta)
    base = pd.to_numeric(df[ajuste_ml1_col], errors="coerce").replace(0, np.nan)
    df["factor_ajuste_final"] = (base * np.exp(df["delta_log_ajuste_ml2"])).astype(float)

    # guardarraíles
    clip_factor = meta.get("clip_factor_apply", [0.50, 2.00])
    try:
        lo, hi = float(clip_factor[0]), float(clip_factor[1])
    except Exception:
        lo, hi = 0.50, 2.00
    df["factor_ajuste_final"] = df["factor_ajuste_final"].clip(lower=lo, upper=hi)

    df["ml2_ajuste_model_file"] = model_path.name
    df["ml2_ajuste_clip_delta"] = clip
    df["as_of_date"] = as_of
    df["created_at"] = pd.Timestamp.utcnow()

    # backtest factor output + final output
    if mode.lower() == "backtest":
        bt_cols = ["fecha", fecha_post_col, "destino", "delta_log_ajuste_ml2", "w", "as_of_date", "created_at"]
        if "grado" in df.columns:
            bt_cols.insert(3, "grado")
        bt = df[bt_cols].copy()
        EVAL.mkdir(parents=True, exist_ok=True)
        write_parquet(bt, OUT_BACKTEST)
        print(f"[OK] BACKTEST factor: {OUT_BACKTEST} rows={len(bt):,}")

    write_parquet(df, OUT_FINAL)
    print(f"[OK] BACKTEST final : {OUT_FINAL} rows={len(df):,}")
    print(f"     model={model_path.name} clip_delta=±{clip} as_of_date={as_of.date()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="backtest", choices=["backtest", "prod"])
    p.add_argument("--model_file", default=None)
    args = p.parse_args()
    main(mode=args.mode, model_file=args.model_file)
