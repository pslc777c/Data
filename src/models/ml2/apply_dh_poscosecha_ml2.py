from __future__ import annotations

from pathlib import Path
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

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"
IN_REAL_HD = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

OUT_FACTOR = EVAL / "backtest_factor_ml2_dh_poscosecha.parquet"
OUT_FINAL = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino_final.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino", "grado"]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN").replace("", "UNKNOWN")


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
        raise KeyError("No encuentro DH baseline en universe.")
    return c


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill_value) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = fill_value
    return out


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_cols(df, NUM_COLS, pd.NA)
    df = _ensure_cols(df, CAT_COLS, "UNKNOWN")

    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["destino"] = _canon_str(df["destino"])
    df["grado"] = _canon_int(df["grado"]).astype("string").fillna("UNKNOWN")

    return df[NUM_COLS + CAT_COLS]


def main(mode: str = "prod") -> None:
    created_at = pd.Timestamp.utcnow()
    as_of_ref = _as_of_date_today_minus_1()

    model_path, meta = _latest_model("dh_poscosecha_ml2")
    pipe = load(model_path)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Universe sin columnas: {sorted(miss)}")

    df["fecha"] = _to_date(df["fecha"])
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["bloque_base"] = _canon_str(df["bloque_base"])
    df = df.loc[df["fecha"].notna()].copy()

    if mode == "backtest":
        df = df.loc[df["fecha"] <= as_of_ref].copy()

    dh_ml1_col = _resolve_dh_ml1_col(df)
    df["dh_ml1_used"] = pd.to_numeric(df[dh_ml1_col], errors="coerce")

    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    X = _prepare_X(df)

    pred_delta = pd.to_numeric(pd.Series(pipe.predict(X)), errors="coerce")
    clip_err = float(meta.get("clip_err_days", 5.0))
    pred_delta_clip = pred_delta.clip(lower=-clip_err, upper=clip_err)

    df["dh_delta_ml2_raw"] = pred_delta
    df["dh_delta_ml2"] = pred_delta_clip

    df["dh_dias_model"] = (
        pd.to_numeric(df["dh_ml1_used"], errors="coerce").fillna(0.0).astype(float)
        + df["dh_delta_ml2"].fillna(0.0).astype(float)
    )
    df["dh_dias_model"] = np.rint(df["dh_dias_model"]).astype("Int64").clip(lower=0, upper=30)

    # Real override
    real_hd = read_parquet(IN_REAL_HD).copy()
    real_hd.columns = [str(c).strip() for c in real_hd.columns]
    real_hd["fecha_cosecha"] = _to_date(real_hd["fecha_cosecha"])
    real_hd["fecha_post"] = _to_date(real_hd["fecha_post"])
    real_hd["destino"] = _canon_str(real_hd["destino"])
    real_hd["grado"] = _canon_int(real_hd["grado"])

    if "dh_dias" in real_hd.columns:
        real_hd["dh_real"] = pd.to_numeric(real_hd["dh_dias"], errors="coerce")
    else:
        real_hd["dh_real"] = (real_hd["fecha_post"] - real_hd["fecha_cosecha"]).dt.days

    if "tallos" in real_hd.columns:
        real_hd["tallos"] = pd.to_numeric(real_hd["tallos"], errors="coerce").fillna(0.0)
    else:
        real_hd["tallos"] = 0.0

    real_agg = (
        real_hd.groupby(["fecha_cosecha", "grado", "destino"], dropna=False, as_index=False)
        .agg(
            dh_real=("dh_real", "median"),
            fecha_post_real=("fecha_post", "min"),
            tallos_real=("tallos", "sum"),
        )
        .rename(columns={"fecha_cosecha": "fecha"})
    )

    df = df.merge(real_agg, on=["fecha", "grado", "destino"], how="left")
    has_dh_real = df["dh_real"].notna() & df["fecha_post_real"].notna() & (df["fecha_post_real"] <= as_of_ref)

    df["dh_source"] = np.where(has_dh_real, "REAL", "ML2_MODEL")
    df["dh_dias_final"] = np.where(has_dh_real, np.rint(df["dh_real"]), df["dh_dias_model"])
    df["dh_dias_final"] = pd.to_numeric(df["dh_dias_final"], errors="coerce").astype("Int64").clip(lower=0, upper=30)

    df["fecha_post_pred_final"] = np.where(
        has_dh_real,
        df["fecha_post_real"],
        df["fecha"] + pd.to_timedelta(df["dh_dias_final"].fillna(0).astype(int), unit="D"),
    )
    df["fecha_post_pred_final"] = _to_date(df["fecha_post_pred_final"])

    df["ml2_dh_model_file"] = model_path.name
    df["created_at"] = created_at
    df["as_of_date"] = as_of_ref

    EVAL.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)

    if mode == "backtest":
        out_factor = df[[
            "fecha", "bloque_base", "variedad_canon", "grado", "destino",
            "dh_ml1_used", "dh_delta_ml2_raw", "dh_delta_ml2", "dh_dias_model", "dh_dias_final",
            "dh_real", "dh_source", "fecha_post_real", "fecha_post_pred_final",
            "ml2_dh_model_file", "as_of_date", "created_at"
        ]].copy()
        write_parquet(out_factor, OUT_FACTOR)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR} rows={len(out_factor):,}")

    write_parquet(df, OUT_FINAL)

    msg = f"[OK] {mode.upper()} final : {OUT_FINAL} rows={len(df):,}\n"
    msg += f"     model={model_path.name} clip_err=+-{clip_err} as_of_date={as_of_ref.date()}"
    print(msg)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prod", choices=["prod", "backtest"])
    args = ap.parse_args()

    main(mode=args.mode)
