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
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

# ✅ IMPORTANTE: usar FULL (no el dh “delgado”)
IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"

OUT_FACTOR = EVAL / "backtest_factor_ml2_dh_poscosecha.parquet"
OUT_FINAL = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino_final.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date_today_minus_1() -> pd.Timestamp:
    return (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).normalize()


def _latest_model(prefix: str) -> tuple[Path, dict]:
    files = sorted(MODELS.glob(f"{prefix}_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No hay modelos en {MODELS} con prefijo {prefix}_*.pkl")
    model_path = files[-1]
    meta_path = MODELS / (model_path.stem + "_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"No existe meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model_path, meta


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _resolve_dh_ml1_col(df: pd.DataFrame) -> str:
    c = _pick_first(df, ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias", "dh_dias_med"])
    if c is None:
        raise KeyError("No encuentro DH baseline en universe. Espero: dh_dias_ml1 / dh_dias_pred_ml1 / dh_dias / dh_dias_med")
    return c


def _resolve_fecha_post_pred_ml1(df: pd.DataFrame) -> str | None:
    return _pick_first(df, ["fecha_post_pred_ml1", "fecha_post_pred", "fecha_post_pred_seed"])


def main(mode: str = "backtest") -> None:
    as_of_date = _as_of_date_today_minus_1()
    created_at = pd.Timestamp.utcnow()

    model_path, meta = _latest_model("dh_poscosecha_ml2")
    pipe = load(model_path)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Universe sin columnas: {sorted(miss)}")

    # Canon
    df["fecha"] = _to_date(df["fecha"])
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["bloque_base"] = _canon_str(df["bloque_base"])

    # filtro backtest <= as_of_date
    df = df.loc[df["fecha"].notna()].copy()

    dh_ml1_col = _resolve_dh_ml1_col(df)
    df["dh_ml1_used"] = pd.to_numeric(df[dh_ml1_col], errors="coerce")

    # Features calendario sobre fecha de cosecha (misma lógica que tu ML1)
    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    # X
    # (si quieres meter grado/destino como cat, hazlo igual que train; aquí asumo que tu train ya coincide)
    X = df[["dow", "month", "weekofyear", "destino", "grado"]].copy()
    # Asegurar tipos
    X["destino"] = X["destino"].astype(str)
    X["grado"] = pd.to_numeric(X["grado"], errors="coerce").astype("Int64").astype(str)

    # Predicción delta (en días)
    pred_delta = pd.to_numeric(pd.Series(pipe.predict(X)), errors="coerce")

    clip_delta = float(meta.get("clip_delta_days", 5))
    pred_delta_clip = pred_delta.clip(lower=-clip_delta, upper=clip_delta)

    # final DH
    df["dh_delta_ml2_raw"] = pred_delta
    df["dh_delta_ml2"] = pred_delta_clip

    df["dh_dias_final"] = (
        pd.to_numeric(df["dh_ml1_used"], errors="coerce")
        .fillna(0.0)
        .astype(float)
        + df["dh_delta_ml2"].fillna(0.0).astype(float)
    )
    df["dh_dias_final"] = np.rint(df["dh_dias_final"]).astype("Int64").clip(lower=0, upper=30)

    # fecha_post_pred_final (si ya existe una fecha_post_pred_ml1, la recalculamos; si no, la creamos)
    df["fecha_post_pred_final"] = df["fecha"] + pd.to_timedelta(df["dh_dias_final"].fillna(0).astype(int), unit="D")

    # meta
    df["ml2_dh_model_file"] = model_path.name
    df["as_of_date"] = as_of_date
    df["created_at"] = created_at

    # factor output (ligero)
    out_factor = df[[
        "fecha", "bloque_base", "variedad_canon", "grado", "destino",
        "dh_ml1_used", "dh_delta_ml2_raw", "dh_delta_ml2", "dh_dias_final",
        "fecha_post_pred_final",
        "ml2_dh_model_file", "as_of_date", "created_at"
    ]].copy()

    EVAL.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    write_parquet(out_factor, OUT_FACTOR)
    write_parquet(df, OUT_FINAL)

    print(f"[OK] BACKTEST factor: {OUT_FACTOR} rows={len(out_factor):,}")
    print(f"[OK] BACKTEST final : {OUT_FINAL} rows={len(df):,}")
    print(f"     model={model_path.name} clip_delta=±{clip_delta} as_of_date={as_of_date.date()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="backtest", choices=["backtest"])
    args = ap.parse_args()

    main(mode=args.mode)
