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
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino_final.parquet"

OUT_FACTOR = EVAL / "backtest_factor_ml2_hidr_poscosecha.parquet"
OUT_FINAL = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino_final.parquet"


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


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    c = _pick_first(df, ["fecha_post_pred_final", "fecha_post_pred_ml2", "fecha_post_pred_ml1", "fecha_post_pred"])
    if c is None:
        raise KeyError(
            "No encuentro fecha_post_pred. Espero una de: "
            "fecha_post_pred_final / fecha_post_pred_ml2 / fecha_post_pred_ml1 / fecha_post_pred"
        )
    return c


def _resolve_factor_hidr_ml1(df: pd.DataFrame) -> str:
    c = _pick_first(df, ["factor_hidr_ml1", "factor_hidr_pred_ml1", "factor_hidr"])
    if c is None:
        raise KeyError(
            "No encuentro factor_hidr ML1 en universe. Espero: factor_hidr_ml1 / factor_hidr_pred_ml1 / factor_hidr"
        )
    return c


def main(mode: str = "prod") -> None:
    """
    mode:
      - prod: corre sobre todo el universo y escribe OUT_FINAL (y opcionalmente OUT_FACTOR si quieres)
      - backtest: recorta a fecha<=as_of_date y además escribe OUT_FACTOR
    """
    as_of_date = _as_of_date_today_minus_1()
    created_at = pd.Timestamp.utcnow()

    model_path, meta = _latest_model("hidr_poscosecha_ml2")
    pipe = load(model_path)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Universe sin columnas: {sorted(miss)}")

    fecha_post_col = _resolve_fecha_post_pred(df)
    hidr_ml1_col = _resolve_factor_hidr_ml1(df)

    df["fecha"] = _to_date(df["fecha"])
    df[fecha_post_col] = _to_date(df[fecha_post_col])
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])
    df["bloque_base"] = _canon_str(df["bloque_base"])

    # base: solo filas con fecha válida
    df = df.loc[df["fecha"].notna()].copy()

    # ✅ FIX (tu comentario decía “filtro backtest <= as_of_date” pero no lo aplicabas)
    if mode == "backtest":
        df = df.loc[df["fecha"] <= as_of_date].copy()

    # features calendario sobre fecha_post_pred (la que uses)
    df["dow"] = df[fecha_post_col].dt.dayofweek.astype("Int64")
    df["month"] = df[fecha_post_col].dt.month.astype("Int64")
    df["weekofyear"] = df[fecha_post_col].dt.isocalendar().week.astype("Int64")

    # dh final (si existe)
    if "dh_dias_final" not in df.columns:
        cdh = _pick_first(df, ["dh_dias_ml2", "dh_dias_ml1", "dh_dias"])
        if cdh:
            df = df.rename(columns={cdh: "dh_dias_final"})
    if "dh_dias_final" not in df.columns:
        df["dh_dias_final"] = np.nan

    df["factor_hidr_ml1"] = pd.to_numeric(df[hidr_ml1_col], errors="coerce")

    X = df[["dow", "month", "weekofyear", "dh_dias_final", "destino", "grado", "variedad_canon"]].copy()

    # asegurar tipos cat igual que en train (si tu train los trataba como strings)
    X["destino"] = X["destino"].astype(str)
    X["grado"] = pd.to_numeric(X["grado"], errors="coerce").astype("Int64").astype(str)
    X["variedad_canon"] = X["variedad_canon"].astype(str)

    pred_log = pd.to_numeric(pd.Series(pipe.predict(X)), errors="coerce")

    lo_log, hi_log = meta.get("apply_clip_log", [-1.2, 1.2])
    pred_log_clip = pred_log.clip(lower=float(lo_log), upper=float(hi_log))

    # factor multiplicativo ML2
    df["factor_hidr_ml2_raw"] = np.exp(pred_log)
    df["factor_hidr_ml2"] = np.exp(pred_log_clip)

    lo_f, hi_f = meta.get("final_factor_clip", [0.60, 3.00])
    df["factor_hidr_final"] = (
        df["factor_hidr_ml1"].fillna(1.0).astype(float) * df["factor_hidr_ml2"].astype(float)
    ).clip(lower=float(lo_f), upper=float(hi_f))

    # meta tracking
    df["ml2_hidr_model_file"] = model_path.name
    df["created_at"] = created_at
    df["as_of_date"] = as_of_date

    # factor output (ligero)
    out_factor = df[[
        "fecha", "bloque_base", "variedad_canon", "grado", "destino",
        fecha_post_col, "factor_hidr_ml1", "factor_hidr_ml2_raw", "factor_hidr_ml2", "factor_hidr_final",
        "ml2_hidr_model_file", "as_of_date", "created_at"
    ]].copy().rename(columns={fecha_post_col: "fecha_post_pred_used"})

    GOLD.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)

    # ✅ backtest escribe OUT_FACTOR; prod puede no escribirlo (pero no molesta si lo dejas)
    if mode == "backtest":
        write_parquet(out_factor, OUT_FACTOR)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR} rows={len(out_factor):,}")

    write_parquet(df, OUT_FINAL)
    print(f"[OK] {mode.upper()} final : {OUT_FINAL} rows={len(df):,}")
    print(f"     model={model_path.name} as_of_date={as_of_date.date()} final_clip={lo_f}->{hi_f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # ✅ FIX CLAVE: permitir prod para que el runner no reviente
    ap.add_argument("--mode", default="prod", choices=["prod", "backtest"])
    args = ap.parse_args()

    main(mode=args.mode)
