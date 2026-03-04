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
IN_REAL = SILVER_DIR / "fact_cosecha_real_grado_dia.parquet"

# ML2 Inicio (SoH) outputs
IN_SOH_FINAL_PROD = GOLD_DIR / "pred_harvest_start_final_ml2.parquet"
IN_SOH_FACTOR_PROD = GOLD_DIR / "factors" / "factor_ml2_harvest_start.parquet"

IN_SOH_FINAL_BT = EVAL_DIR / "backtest_pred_harvest_start_final_ml2.parquet"
IN_SOH_FACTOR_BT = EVAL_DIR / "backtest_factor_ml2_harvest_start.parquet"

# Outputs
OUT_FACTOR_PROD = GOLD_DIR / "factors" / "factor_ml2_harvest_horizon.parquet"
OUT_FINAL_PROD = GOLD_DIR / "pred_harvest_horizon_final_ml2.parquet"

OUT_FACTOR_BT = EVAL_DIR / "backtest_factor_ml2_harvest_horizon.parquet"
OUT_FINAL_BT = EVAL_DIR / "backtest_pred_harvest_horizon_final_ml2.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    return (a / b).fillna(0.0)


def _resolve_block_base(df: pd.DataFrame) -> pd.DataFrame:
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
        return df
    if "bloque_padre" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_padre"])
        return df
    if "bloque" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque"])
        return df
    raise KeyError("No encuentro bloque_base / bloque_padre / bloque en fact_cosecha_real_grado_dia.")


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
            fecha_sp=("fecha_sp", "first"),
            harvest_start_pred=("harvest_start_pred", "min"),
            harvest_end_pred=("harvest_end_pred", "max"),
            n_harvest_days_pred=("n_harvest_days_pred", "max"),
            ml1_version=("ml1_version", "max"),
        )
    )
    head["days_sp_to_start_pred"] = (head["harvest_start_pred"] - head["fecha_sp"]).dt.days
    return head


def _features_asof(clima: pd.DataFrame, base: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """
    base: columns ciclo_id, bloque_base, fecha_sp
    returns 1 row per ciclo_id
    """
    base = base[["ciclo_id", "bloque_base", "fecha_sp"]].drop_duplicates().copy()
    base["bloque_base"] = _canon_str(base["bloque_base"])
    base["fecha_sp"] = _to_date(base["fecha_sp"])

    cl = clima.copy()
    cl["fecha"] = _to_date(cl["fecha"])
    cl["bloque_base"] = _canon_str(cl["bloque_base"])
    cl = cl.loc[cl["fecha"] <= as_of_date, :].copy()

    m = base.merge(cl, on="bloque_base", how="left")
    m = m.loc[(m["fecha"] >= m["fecha_sp"]) & (m["fecha"] <= as_of_date), :].copy()

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

    out = base[["ciclo_id"]].merge(last, on="ciclo_id", how="left")

    for c in [
        "gdc_cum_sp", "gdc_7d", "gdc_14d", "gdc_per_day",
        "rain_cum_sp", "rain_7d", "enlluvia_days_7d",
        "solar_cum_sp", "solar_7d",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["temp_avg_7d"] = pd.to_numeric(out["temp_avg_7d"], errors="coerce")
    med = out["temp_avg_7d"].median()
    if pd.isna(med):
        med = 0.0
    out["temp_avg_7d"] = out["temp_avg_7d"].fillna(med)

    return out


def _load_latest_model() -> tuple[object, dict]:
    metas = sorted(MODELS_DIR.glob("harvest_horizon_ml2_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No ML2 meta found in {MODELS_DIR}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = MODELS_DIR / f"harvest_horizon_ml2_{meta['run_id']}.pkl"
    import joblib
    model = joblib.load(model_path)
    return model, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="prod")
    p.add_argument("--asof", default=None, help="Override as_of_date (YYYY-MM-DD) for prod mode; default=today-1.")
    return p.parse_args()


def _pick_soh_inputs(mode: str) -> tuple[Path, Path]:
    if mode == "backtest":
        return IN_SOH_FINAL_BT, IN_SOH_FACTOR_BT
    return IN_SOH_FINAL_PROD, IN_SOH_FACTOR_PROD


def main() -> None:
    args = _parse_args()
    model, meta = _load_latest_model()

    ciclo = read_parquet(IN_CICLO).copy()
    grid = read_parquet(IN_GRID_ML1).copy()
    clima = read_parquet(IN_CLIMA).copy()
    real = read_parquet(IN_REAL).copy()

    ciclo["bloque_base"] = _canon_str(ciclo["bloque_base"])
    ciclo["fecha_sp"] = _to_date(ciclo["fecha_sp"])
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])
    ciclo["fecha_fin_cosecha"] = _to_date(ciclo["fecha_fin_cosecha"])
    ciclo["estado"] = _canon_str(ciclo["estado"])

    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    real["fecha"] = _to_date(real["fecha"])
    real = _resolve_block_base(real)
    real["tallos_real"] = pd.to_numeric(real["tallos_real"], errors="coerce").fillna(0.0)

    head = _build_cycle_header(grid)
    df = ciclo.merge(head, on="ciclo_id", how="inner", suffixes=("", "_ml1"))

    # as_of_date
    if args.mode == "prod":
        if args.asof:
            as_of = pd.Timestamp(args.asof).normalize()
        else:
            as_of = pd.Timestamp(datetime.now()).normalize() - pd.Timedelta(days=1)
        df["as_of_date"] = as_of
    else:
        # backtest causal: usar inicio real - 1
        df = df.loc[df["fecha_inicio_cosecha"].notna(), :].copy()
        df["as_of_date"] = df["fecha_inicio_cosecha"] - pd.to_timedelta(1, unit="D")

    # Cerrado/activo: en cerrados no recalcular por ML2.
    df["is_closed_cycle"] = (
        df["estado"].eq("CERRADO")
        | (df["fecha_fin_cosecha"].notna() & (df["fecha_fin_cosecha"] <= df["as_of_date"]))
    )
    df["is_active_cycle"] = ~df["is_closed_cycle"]

    # --- SoH (ML2 Inicio): harvest_start_final + pred_error_start_days ---
    soh_final_path, soh_factor_path = _pick_soh_inputs(args.mode)

    if not soh_final_path.exists() or not soh_factor_path.exists():
        raise FileNotFoundError(
            f"Missing SoH inputs for mode={args.mode}. Need:\n  {soh_final_path}\n  {soh_factor_path}"
        )

    soh_final = read_parquet(soh_final_path).copy()
    soh_factor = read_parquet(soh_factor_path).copy()

    soh_final["harvest_start_pred"] = _to_date(soh_final["harvest_start_pred"])
    soh_final["harvest_start_final"] = _to_date(soh_final["harvest_start_final"])
    soh_factor["pred_error_start_days"] = pd.to_numeric(soh_factor["pred_error_start_days"], errors="coerce")

    soh_map = soh_final[["ciclo_id", "harvest_start_final"]].drop_duplicates("ciclo_id")
    err_map = soh_factor[["ciclo_id", "pred_error_start_days"]].drop_duplicates("ciclo_id")

    df = df.merge(soh_map, on="ciclo_id", how="left").merge(err_map, on="ciclo_id", how="left")
    df["pred_error_start_days"] = pd.to_numeric(df["pred_error_start_days"], errors="coerce").fillna(0.0)

    # fallback si algún ciclo no tiene harvest_start_final (debería ser raro)
    df["harvest_start_final"] = df["harvest_start_final"].fillna(df["harvest_start_pred"])

    # calendario
    df["dow"] = df["as_of_date"].dt.dayofweek
    df["month"] = df["as_of_date"].dt.month
    df["weekofyear"] = df["as_of_date"].dt.isocalendar().week.astype(int)
    df["days_from_sp"] = (df["as_of_date"] - df["fecha_sp"]).dt.days

    # clima features por grupo (eficiente en backtest)
    feats = []
    for as_of_date, chunk in df.groupby("as_of_date"):
        base = chunk[["ciclo_id", "bloque_base", "fecha_sp"]]
        feats.append(_features_asof(clima, base, pd.Timestamp(as_of_date)))
    feat = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame(columns=["ciclo_id"])

    df = df.merge(feat, on="ciclo_id", how="left")

    # imputaciones numéricas como en meta
    for c in meta.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # cat
    for c in meta.get("cat_cols", []):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    feature_cols = meta["num_cols"] + meta["cat_cols"]
    X = df[feature_cols].copy()

    pred_err_raw = pd.to_numeric(pd.Series(model.predict(X)), errors="coerce").fillna(0.0).astype(float)
    lo, hi = meta["guardrails"]["clip_error_days"]
    pred_err_clip = np.clip(pred_err_raw, lo, hi)
    pred_err_days = np.rint(pred_err_clip).astype("Int64")

    df["pred_error_horizon_days_raw"] = pred_err_raw
    df["pred_error_horizon_days_model"] = pred_err_days

    # Horizon ML2 puro (antes de anclaje)
    df["n_harvest_days_pred"] = pd.to_numeric(df["n_harvest_days_pred"], errors="coerce").fillna(0.0)
    df["n_harvest_days_ml2_model"] = (
        df["n_harvest_days_pred"] + df["pred_error_horizon_days_model"].fillna(0).astype(float)
    )
    df["n_harvest_days_ml2_model"] = np.rint(df["n_harvest_days_ml2_model"]).astype("Int64")

    min_h = int(meta["guardrails"]["min_horizon_days"])
    df["n_harvest_days_ml2_model"] = df["n_harvest_days_ml2_model"].clip(lower=min_h)

    # Anclaje a real de horizonte.
    has_end_real = (
        df["fecha_fin_cosecha"].notna()
        & (df["fecha_fin_cosecha"] >= df["harvest_start_final"])
        & (df["fecha_fin_cosecha"] <= df["as_of_date"])
    )
    n_real = (df["fecha_fin_cosecha"] - df["harvest_start_final"]).dt.days + 1
    n_real = pd.to_numeric(n_real, errors="coerce").astype("Int64")

    # Progreso real acumulado al as_of (solo para ciclos activos, para extender horizonte si van atrasados).
    cyc = df[["ciclo_id", "bloque_base", "harvest_start_final", "as_of_date", "tallos_proy"]].drop_duplicates().copy()
    m = cyc.merge(real[["fecha", "bloque_base", "tallos_real"]], on="bloque_base", how="left")
    m = m.loc[(m["fecha"] >= m["harvest_start_final"]) & (m["fecha"] <= m["as_of_date"]), :].copy()
    real_cum = (
        m.groupby("ciclo_id", as_index=False)
        .agg(tallos_real_cum_asof=("tallos_real", "sum"))
    )
    df = df.merge(real_cum, on="ciclo_id", how="left")
    df["tallos_real_cum_asof"] = pd.to_numeric(df["tallos_real_cum_asof"], errors="coerce").fillna(0.0)

    df["tallos_proy"] = pd.to_numeric(df["tallos_proy"], errors="coerce")
    df["real_progress_asof"] = np.where(
        df["tallos_proy"] > 0,
        (df["tallos_real_cum_asof"] / df["tallos_proy"]).clip(lower=0.0, upper=1.0),
        np.nan,
    )

    elapsed_days = ((df["as_of_date"] - df["harvest_start_final"]).dt.days + 1).clip(lower=1)
    elapsed_days = pd.to_numeric(elapsed_days, errors="coerce").fillna(1.0)
    df["expected_progress_model"] = (
        elapsed_days / pd.to_numeric(df["n_harvest_days_ml2_model"], errors="coerce").replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    df["progress_gap"] = df["real_progress_asof"] - df["expected_progress_model"]

    implied_h = np.where(
        pd.to_numeric(df["real_progress_asof"], errors="coerce").fillna(0.0) > 0,
        np.ceil(elapsed_days / pd.to_numeric(df["real_progress_asof"], errors="coerce").clip(lower=1e-6)),
        np.nan,
    )
    df["implied_horizon_from_real_progress"] = pd.to_numeric(implied_h, errors="coerce")

    lag_mask = (
        df["is_active_cycle"]
        & (~has_end_real)
        & df["real_progress_asof"].notna()
        & df["expected_progress_model"].notna()
        & (df["real_progress_asof"] > 0)
        & (df["real_progress_asof"] < (df["expected_progress_model"] * 0.98))
    )
    df["is_lagging_progress"] = lag_mask

    n_active = pd.to_numeric(df["n_harvest_days_ml2_model"], errors="coerce").fillna(min_h)
    n_active = np.where(lag_mask, np.maximum(n_active, df["implied_horizon_from_real_progress"]), n_active)
    n_active = pd.to_numeric(n_active, errors="coerce")

    # Cerrado: no recalcular (usar real si existe; fallback ML1).
    n_closed = np.where(
        has_end_real,
        pd.to_numeric(n_real, errors="coerce"),
        np.rint(pd.to_numeric(df["n_harvest_days_pred"], errors="coerce")),
    )

    n_final = pd.Series(np.where(df["is_closed_cycle"], n_closed, n_active), index=df.index)

    max_h = int(meta["guardrails"].get("max_horizon_days", 120))
    n_final = pd.to_numeric(n_final, errors="coerce").fillna(min_h).astype("Int64").clip(lower=min_h, upper=max_h)

    df["has_end_real"] = has_end_real
    df["horizon_source"] = np.select(
        [
            df["is_closed_cycle"] & has_end_real,
            df["is_closed_cycle"] & (~has_end_real),
            (~df["is_closed_cycle"]) & has_end_real,
            (~df["is_closed_cycle"]) & lag_mask,
        ],
        ["REAL_CLOSED", "ML1_CLOSED", "REAL_ACTIVE", "ML2_ACTIVE_LAG_EXT"],
        default="ML2_ACTIVE",
    )
    df["n_harvest_days_final"] = n_final
    df["pred_error_horizon_days"] = (
        df["n_harvest_days_final"] - np.rint(df["n_harvest_days_pred"]).astype("Int64")
    )

    # End final coherente con inicio final + horizonte final
    df["harvest_end_final"] = (
        df["harvest_start_final"] + pd.to_timedelta(df["n_harvest_days_final"].fillna(min_h).astype(int) - 1, unit="D")
    ).dt.normalize()

    out_factor = df[[
        "ciclo_id", "bloque_base", "variedad", "variedad_canon", "estado",
        "fecha_sp", "fecha_inicio_cosecha",
        "as_of_date",
        "harvest_start_pred", "harvest_start_final",
        "n_harvest_days_pred",
        "pred_error_start_days",
        "pred_error_horizon_days_raw",
        "pred_error_horizon_days_model",
        "pred_error_horizon_days",
        "n_harvest_days_ml2_model",
        "implied_horizon_from_real_progress",
        "tallos_real_cum_asof",
        "real_progress_asof",
        "expected_progress_model",
        "progress_gap",
        "is_lagging_progress",
        "n_harvest_days_final",
        "is_active_cycle",
        "is_closed_cycle",
        "has_end_real",
        "horizon_source",
        "harvest_end_pred",
        "harvest_end_final",
        "ml1_version",
    ]].copy()

    out_factor["ml2_run_id"] = meta["run_id"]
    out_factor["created_at"] = pd.Timestamp(datetime.now()).normalize()

    out_final = out_factor[[
        "ciclo_id", "bloque_base", "variedad_canon", "fecha_sp",
        "as_of_date",
        "harvest_start_pred", "harvest_start_final",
        "n_harvest_days_pred", "n_harvest_days_final",
        "pred_error_horizon_days",
        "is_lagging_progress", "real_progress_asof", "expected_progress_model",
        "is_active_cycle", "is_closed_cycle", "has_end_real", "horizon_source",
        "harvest_end_pred", "harvest_end_final",
        "ml1_version", "ml2_run_id", "created_at",
    ]].copy()

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
