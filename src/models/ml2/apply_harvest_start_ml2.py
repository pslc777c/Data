from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models" / "ml2"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"
IN_GRID_ML1 = GOLD_DIR / "universe_harvest_grid_ml1.parquet"
IN_CLIMA = SILVER_DIR / "dim_clima_bloque_dia.parquet"

OUT_FACTOR_PROD = GOLD_DIR / "factors" / "factor_ml2_harvest_start.parquet"
OUT_FINAL_PROD = GOLD_DIR / "pred_harvest_start_final_ml2.parquet"

OUT_FACTOR_BT = EVAL_DIR / "backtest_factor_ml2_harvest_start.parquet"
OUT_FINAL_BT = EVAL_DIR / "backtest_pred_harvest_start_final_ml2.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return (a / b).fillna(0.0)


def _build_cycle_header(grid: pd.DataFrame) -> pd.DataFrame:
    g = grid.copy()
    g["harvest_start_pred"] = _to_date(g["harvest_start_pred"])
    g["harvest_end_pred"] = _to_date(g["harvest_end_pred"])
    g["fecha_sp"] = _to_date(g["fecha_sp"])

    head = (
        g.groupby("ciclo_id", as_index=False)
        .agg(
            bloque_base=("bloque_base", "first"),
            variedad_canon=("variedad_canon", "first"),
            area=("area", "first"),
            tipo_sp=("tipo_sp", "first"),
            estado=("estado", "first"),
            tallos_proy=("tallos_proy", "first"),
            fecha_sp=("fecha_sp", "first"),  # ✅ importante
            harvest_start_pred=("harvest_start_pred", "min"),
            harvest_end_pred=("harvest_end_pred", "max"),
            n_harvest_days_pred=("n_harvest_days_pred", "max"),
            ml1_version=("ml1_version", "max"),
        )
    )

    head["days_sp_to_start_pred"] = (head["harvest_start_pred"] - head["fecha_sp"]).dt.days
    return head


def _features_asof(clima: pd.DataFrame, ciclo_chunk: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    Devuelve features por ciclo_id usando clima entre [fecha_sp, as_of_date].
    IMPORTANT:
      - usa gdc_dia acumulado
      - si un ciclo no tiene clima en el rango, devuelve ceros (no desaparece)
    """
    base = ciclo_chunk[["ciclo_id", "bloque_base", "fecha_sp"]].drop_duplicates().copy()
    base["bloque_base"] = _canon_str(base["bloque_base"])
    base["fecha_sp"] = _to_date(base["fecha_sp"])

    cl = clima.copy()
    cl["fecha"] = _to_date(cl["fecha"])
    cl["bloque_base"] = _canon_str(cl["bloque_base"])
    cl = cl.loc[cl["fecha"] <= as_of_date, :].copy()

    m = base.merge(cl, on="bloque_base", how="left")
    m = m.loc[(m["fecha"] >= m["fecha_sp"]) & (m["fecha"] <= as_of_date), :].copy()

    # Si no hay filas de clima para ningún ciclo, devolvemos ceros
    if m.empty:
        out = base[["ciclo_id"]].copy()
        for c in [
            "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
            "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
            "solar_cum_sp", "solar_7d",
            "temp_avg_7d",
        ]:
            out[c] = 0.0
        return out

    # IMPORTANT: usar gdc_dia y acumular (ignorar gdc_base)
    m["gdc_dia"] = pd.to_numeric(m["gdc_dia"], errors="coerce").fillna(0.0)
    m["rainfall_mm_dia"] = pd.to_numeric(m["rainfall_mm_dia"], errors="coerce").fillna(0.0)
    m["solar_energy_j_m2_dia"] = pd.to_numeric(m["solar_energy_j_m2_dia"], errors="coerce").fillna(0.0)
    m["temp_avg_dia"] = pd.to_numeric(m["temp_avg_dia"], errors="coerce")
    m["en_lluvia_dia"] = pd.to_numeric(m["en_lluvia_dia"], errors="coerce").fillna(0.0)

    m = m.sort_values(["ciclo_id", "fecha"])

    def _roll_sum(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(window=w, min_periods=1).sum()

    def _roll_mean(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(window=w, min_periods=1).mean()

    m["gdc_7d"] = m.groupby("ciclo_id")["gdc_dia"].transform(lambda s: _roll_sum(s, 7))
    m["gdc_14d"] = m.groupby("ciclo_id")["gdc_dia"].transform(lambda s: _roll_sum(s, 14))
    m["rain_7d"] = m.groupby("ciclo_id")["rainfall_mm_dia"].transform(lambda s: _roll_sum(s, 7))
    m["solar_7d"] = m.groupby("ciclo_id")["solar_energy_j_m2_dia"].transform(lambda s: _roll_sum(s, 7))
    m["enlluvia_days_7d"] = m.groupby("ciclo_id")["en_lluvia_dia"].transform(lambda s: _roll_sum(s, 7))
    m["temp_avg_7d"] = m.groupby("ciclo_id")["temp_avg_dia"].transform(lambda s: _roll_mean(s, 7))

    m["gdc_cum_sp"] = m.groupby("ciclo_id")["gdc_dia"].cumsum()
    m["rain_cum_sp"] = m.groupby("ciclo_id")["rainfall_mm_dia"].cumsum()
    m["solar_cum_sp"] = m.groupby("ciclo_id")["solar_energy_j_m2_dia"].cumsum()

    last = m.groupby("ciclo_id", as_index=False).tail(1).copy()
    last["days_from_sp"] = (as_of_date - last["fecha_sp"]).dt.days
    last["days_from_sp"] = pd.to_numeric(last["days_from_sp"], errors="coerce").fillna(0).clip(lower=0)
    last["gdc_per_day"] = _safe_div(last["gdc_cum_sp"], last["days_from_sp"].replace(0, np.nan))

    feat_cols = [
        "ciclo_id",
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
        "temp_avg_7d",
    ]
    last = last[feat_cols].copy()

    # ✅ garantizar que todos los ciclos queden (si algunos no tuvieron filas)
    out = base[["ciclo_id"]].merge(last, on="ciclo_id", how="left")

    # imputaciones
    for c in [
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["temp_avg_7d"] = pd.to_numeric(out["temp_avg_7d"], errors="coerce")
    # si todo es NaN -> 0
    med = out["temp_avg_7d"].median()
    if pd.isna(med):
        med = 0.0
    out["temp_avg_7d"] = out["temp_avg_7d"].fillna(med)

    return out


def _load_latest_model() -> tuple[object, dict]:
    metas = sorted(MODELS_DIR.glob("harvest_start_ml2_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No ML2 meta found in {MODELS_DIR}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = MODELS_DIR / f"harvest_start_ml2_{meta['run_id']}.pkl"
    import joblib
    model = joblib.load(model_path)
    return model, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="prod")
    p.add_argument(
        "--asof",
        default=None,
        help="Override as_of_date (YYYY-MM-DD) for prod mode. If not set, uses today-1.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model, meta = _load_latest_model()

    ciclo = read_parquet(IN_CICLO).copy()
    grid = read_parquet(IN_GRID_ML1).copy()
    clima = read_parquet(IN_CLIMA).copy()

    # canon
    ciclo["bloque_base"] = _canon_str(ciclo["bloque_base"])
    ciclo["fecha_sp"] = _to_date(ciclo["fecha_sp"])
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])

    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    head = _build_cycle_header(grid)
    df = ciclo.merge(head, on="ciclo_id", how="inner", suffixes=("", "_ml1"))

    # -----------------------
    # Define as_of_date
    # -----------------------
    if args.mode == "prod":
        if args.asof:
            as_of = pd.Timestamp(args.asof).normalize()
        else:
            as_of = pd.Timestamp(datetime.now()).normalize() - pd.Timedelta(days=1)
        df["as_of_date"] = as_of

    else:  # backtest
        # as_of = (fecha_inicio_cosecha - 1) por ciclo (si no hay real, se excluye del backtest)
        df = df.loc[df["fecha_inicio_cosecha"].notna(), :].copy()
        df["as_of_date"] = df["fecha_inicio_cosecha"] - pd.to_timedelta(1, unit="D")

    # features calendario (por fila)
    df["dow"] = df["as_of_date"].dt.dayofweek
    df["month"] = df["as_of_date"].dt.month
    df["weekofyear"] = df["as_of_date"].dt.isocalendar().week.astype(int)

    # -----------------------
    # Build clima features as-of
    # En prod: un solo as_of.
    # En backtest: muchos as_of distintos -> procesar por grupo para eficiencia.
    # -----------------------
    feats = []
    for as_of_date, chunk in df.groupby("as_of_date"):
        feats.append(_features_asof(clima, chunk[["ciclo_id", "bloque_base", "fecha_sp"]], pd.Timestamp(as_of_date)))
    feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(columns=["ciclo_id"])

    df = df.merge(feat, on="ciclo_id", how="left")

    # imputaciones num según meta
    for c in meta.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # temp (ya viene imputada en _features_asof pero mantenemos robustez)
    if "temp_avg_7d" in df.columns:
        df["temp_avg_7d"] = pd.to_numeric(df["temp_avg_7d"], errors="coerce")
        med = df["temp_avg_7d"].median()
        if pd.isna(med):
            med = 0.0
        df["temp_avg_7d"] = df["temp_avg_7d"].fillna(med)

    feature_cols = meta["num_cols"] + meta["cat_cols"]
    X = df[feature_cols].copy()

    pred_err = model.predict(X).astype(float)
    lo, hi = meta["guardrails"]["clip_error_days"]
    pred_err = np.clip(pred_err, lo, hi)

    df["pred_error_start_days"] = pred_err
    df["harvest_start_final"] = df["harvest_start_pred"] + pd.to_timedelta(df["pred_error_start_days"], unit="D")

    out_factor = df[[
        "ciclo_id", "bloque_base", "variedad", "variedad_canon", "estado",
        "fecha_sp", "fecha_inicio_cosecha",
        "as_of_date",
        "harvest_start_pred", "pred_error_start_days", "harvest_start_final",
        "ml1_version",
    ]].copy()

    out_factor["ml2_run_id"] = meta["run_id"]
    out_factor["created_at"] = pd.Timestamp(datetime.now()).normalize()

    out_final = out_factor[[
        "ciclo_id", "bloque_base", "variedad_canon", "fecha_sp",
        "as_of_date",
        "harvest_start_pred", "harvest_start_final",
        "ml1_version", "ml2_run_id", "created_at",
    ]].copy()

    # -----------------------
    # Save
    # -----------------------
    if args.mode == "prod":
        OUT_FACTOR_PROD.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_PROD)
        write_parquet(out_final, OUT_FINAL_PROD)
        print(f"[OK] PROD factor: {OUT_FACTOR_PROD} rows={len(out_factor):,}")
        print(f"[OK] PROD final : {OUT_FINAL_PROD} rows={len(out_final):,}")
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_BT)
        write_parquet(out_final, OUT_FINAL_BT)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR_BT} rows={len(out_factor):,}")
        print(f"[OK] BACKTEST final : {OUT_FINAL_BT} rows={len(out_final):,}")


if __name__ == "__main__":
    main()
