from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
SILVER_DIR = DATA_DIR / "silver"
OUT_PATH = DATA_DIR / "gold" / "ml1_nn" / "ds_ml1_nn_v1.parquet"
IN_FACT_PESO_REAL = SILVER_DIR / "fact_peso_tallo_real_grado_dia.parquet"
IN_DIM_VARIEDAD = SILVER_DIR / "dim_variedad_canon.parquet"
IN_POST_SEED = DATA_DIR / "gold" / "pred_poscosecha_seed_grado_dia_bloque_destino.parquet"
IN_PRED_KG_ML1 = DATA_DIR / "gold" / "pred_kg_grado_dia_ml1_full.parquet"
IN_CLIMA = SILVER_DIR / "dim_clima_bloque_dia.parquet"

TARGET_COLS = [
    "target_d_start",
    "target_n_harvest_days",
    "target_factor_tallos_dia",
    "target_share_grado",
    "target_factor_peso_tallo",
    "target_dh_dias",
    "target_factor_hidr",
    "target_factor_desp",
    "target_factor_ajuste",
]

TARGET_CLIPS: dict[str, tuple[float, float]] = {
    "target_d_start": (0.0, 180.0),
    "target_n_harvest_days": (1.0, 180.0),
    "target_factor_tallos_dia": (0.20, 5.00),
    "target_share_grado": (0.00, 1.00),
    "target_factor_peso_tallo": (0.60, 1.60),
    "target_dh_dias": (0.0, 30.0),
    "target_factor_hidr": (0.80, 3.00),
    "target_factor_desp": (0.05, 1.00),
    "target_factor_ajuste": (0.50, 2.00),
}

CATEGORY_COLS = [
    "stage",
    "bloque_base",
    "variedad_canon",
    "area",
    "tipo_sp",
    "grado",
    "destino",
]

NUMERIC_COLS = [
    "tallos_proy",
    "tallos_pred_baseline_grado_dia",
    "tallos_pred_ml1_grado_dia",
    "tallos_pred_ml1_dia",
    "sp_month",
    "sp_weekofyear",
    "sp_doy",
    "sp_dow",
    "day_in_harvest",
    "rel_pos",
    "n_harvest_days",
    "pct_avance_real",
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    "share_grado_baseline",
    "peso_tallo_baseline_g",
    "tallos_pred_baseline_dia",
    "tallos_post",
    "tallos_post_proy",
    "peso_base_g",
    "peso_post_g",
    "gramos_verde_ref",
    "gramos_post_real_ref",
    "kg_verde_ref",
    "kg_post_real_ref",
    "w2_kg",
    "w2a_kg",
    "wideal_kg",
    "desp_pct",
    "share_block_post",
    "cajas_ml1_grado_dia",
    "cajas_split_grado_dia",
    "cajas_post_seed",
    "target_weight",
    "dow",
    "month",
    "weekofyear",
    "dow_post",
    "month_post",
    "weekofyear_post",
    # Autoregressive field dynamics (lagged and moving averages)
    "ar_tallos_real_dia_lag1",
    "ar_tallos_real_dia_roll3",
    "ar_tallos_real_dia_roll7",
    "ar_ratio_real_vs_base_lag1",
    "ar_ratio_real_vs_base_roll3",
    "ar_pct_avance_real_lag1",
    "ar_gdc_dia_roll3",
    "ar_temp_avg_dia_roll3",
    "ar_rainfall_mm_roll3",
]

BASE_COLS = [
    "stage",
    "row_source",
    "fecha_evento",
    "fecha_post",
    "ciclo_id",
    "bloque_base",
    "variedad_canon",
    "area",
    "tipo_sp",
    "grado",
    "destino",
]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _canon_intlike_str(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").round().astype("Int64")
    return v.astype("string").fillna("UNKNOWN")


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str], default=np.nan) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = default
    return out


def _coalesce(df: pd.DataFrame, out_col: str, cands: list[str], default=np.nan) -> None:
    val = pd.Series([default] * len(df), index=df.index)
    for c in cands:
        if c in df.columns:
            val = val.where(val.notna(), df[c])
    df[out_col] = val


def _clip_target(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = np.nan
        return
    lo, hi = TARGET_CLIPS[col]
    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)


def _add_clear_name_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        # Tallos
        "tallos_proy_baseline": "tallos_proy",
        "tallos_grado_dia_baseline": "tallos_pred_baseline_grado_dia",
        "tallos_grado_dia_ML1": "tallos_pred_ml1_grado_dia",
        "tallos_dia_baseline": "tallos_pred_baseline_dia",
        "tallos_dia_ML1": "tallos_pred_ml1_dia",
        "tallos_post_real": "tallos_post",
        "tallos_post_ML1": "tallos_post_proy",
        # Peso/Kg/Gramos
        "gramos_verde_ML1": "gramos_verde_ref",
        "gramos_post_real": "gramos_post_real_ref",
        "kg_verde_ML1": "kg_verde_ref",
        "kg_post_real": "kg_post_real_ref",
        # Cajas
        "cajas_verde_ML1": "cajas_split_grado_dia",
        "cajas_post_seed_baseline": "cajas_post_seed",
    }
    for new_col, src_col in alias_map.items():
        if src_col in out.columns and new_col not in out.columns:
            out[new_col] = out[src_col]
    return out


def _add_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fecha_evento"] = _to_date(out["fecha_evento"])
    out["ciclo_id"] = out["ciclo_id"].astype("string").fillna("UNKNOWN")

    grp = ["ciclo_id", "fecha_evento"]
    day = (
        out.groupby(grp, dropna=False, as_index=False)
        .agg(
            tallos_pred_baseline_dia=("tallos_pred_baseline_dia", "median"),
            target_factor_tallos_dia=("target_factor_tallos_dia", "median"),
            pct_avance_real=("pct_avance_real", "median"),
            gdc_dia=("gdc_dia", "median"),
            temp_avg_dia=("temp_avg_dia", "median"),
            rainfall_mm_dia=("rainfall_mm_dia", "median"),
        )
        .sort_values(grp, kind="mergesort")
        .reset_index(drop=True)
    )

    day["tallos_real_proxy_dia"] = (
        pd.to_numeric(day["tallos_pred_baseline_dia"], errors="coerce")
        * pd.to_numeric(day["target_factor_tallos_dia"], errors="coerce")
    )
    day["ratio_real_vs_base"] = np.where(
        pd.to_numeric(day["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0) > 0.0,
        pd.to_numeric(day["tallos_real_proxy_dia"], errors="coerce")
        / pd.to_numeric(day["tallos_pred_baseline_dia"], errors="coerce"),
        np.nan,
    )

    day = day.sort_values(["ciclo_id", "fecha_evento"], kind="mergesort").reset_index(drop=True)

    gby = day.groupby("ciclo_id", dropna=False)

    day["ar_tallos_real_dia_lag1"] = gby["tallos_real_proxy_dia"].shift(1)
    day["ar_tallos_real_dia_roll3"] = gby["tallos_real_proxy_dia"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean()
    )
    day["ar_tallos_real_dia_roll7"] = gby["tallos_real_proxy_dia"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(7, min_periods=1).mean()
    )
    day["ar_ratio_real_vs_base_lag1"] = gby["ratio_real_vs_base"].shift(1)
    day["ar_ratio_real_vs_base_roll3"] = gby["ratio_real_vs_base"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean()
    )
    day["ar_pct_avance_real_lag1"] = gby["pct_avance_real"].shift(1)
    day["ar_gdc_dia_roll3"] = gby["gdc_dia"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean()
    )
    day["ar_temp_avg_dia_roll3"] = gby["temp_avg_dia"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean()
    )
    day["ar_rainfall_mm_roll3"] = gby["rainfall_mm_dia"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean()
    )

    ar_cols = [
        "ar_tallos_real_dia_lag1",
        "ar_tallos_real_dia_roll3",
        "ar_tallos_real_dia_roll7",
        "ar_ratio_real_vs_base_lag1",
        "ar_ratio_real_vs_base_roll3",
        "ar_pct_avance_real_lag1",
        "ar_gdc_dia_roll3",
        "ar_temp_avg_dia_roll3",
        "ar_rainfall_mm_roll3",
    ]
    # Remove placeholder AR columns before merge to avoid *_x/*_y duplicates.
    drop_cols = [c for c in ar_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    day_take = day[grp + ar_cols].copy()
    out = out.merge(day_take, on=grp, how="left")
    return out


def _with_common_schema(df: pd.DataFrame, stage: str, row_source: str) -> pd.DataFrame:
    out = df.copy()
    out["stage"] = stage
    out["row_source"] = row_source

    out = _ensure_cols(out, BASE_COLS)
    out = _ensure_cols(out, CATEGORY_COLS)
    out = _ensure_cols(out, NUMERIC_COLS)
    out = _ensure_cols(out, TARGET_COLS)

    out["fecha_evento"] = _to_date(out["fecha_evento"])
    out["fecha_post"] = _to_date(out["fecha_post"])
    out["ciclo_id"] = out["ciclo_id"].astype("string").fillna("UNKNOWN")

    for c in ["stage", "row_source", "variedad_canon", "area", "tipo_sp", "destino"]:
        out[c] = _canon_str(out[c])
    out["bloque_base"] = _canon_intlike_str(out["bloque_base"])
    out["grado"] = _canon_intlike_str(out["grado"])

    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for t in TARGET_COLS:
        _clip_target(out, t)

    target_weight = pd.to_numeric(out.get("target_weight", 1.0), errors="coerce").fillna(1.0).clip(lower=0.0)
    for t in TARGET_COLS:
        out[f"mask_{t}"] = np.where(out[t].notna(), target_weight, 0.0).astype(np.float32)

    out["has_any_target"] = (out[[f"mask_{t}" for t in TARGET_COLS]].sum(axis=1) > 0).astype("int8")

    ordered = list(
        dict.fromkeys(
            BASE_COLS
            + CATEGORY_COLS
            + NUMERIC_COLS
            + TARGET_COLS
            + [f"mask_{t}" for t in TARGET_COLS]
            + ["has_any_target"]
        )
    )
    return out[ordered]


def _build_stage_veg() -> pd.DataFrame:
    df = read_parquet(FEATURES_DIR / "features_harvest_window_ml1.parquet").copy()
    df.columns = [str(c).strip() for c in df.columns]

    df["fecha_evento"] = _to_date(df["fecha_sp"])
    df["ciclo_id"] = df["ciclo_id"].astype("string")
    df["bloque_base"] = pd.to_numeric(df["bloque_base"], errors="coerce").astype("Int64")

    df["target_d_start"] = pd.to_numeric(df.get("d_start_real"), errors="coerce")
    df["target_n_harvest_days"] = pd.to_numeric(df.get("n_harvest_days_real"), errors="coerce")

    if IN_CLIMA.exists():
        clima = read_parquet(IN_CLIMA).copy()
        clima.columns = [str(c).strip() for c in clima.columns]
        if {"fecha", "bloque_base"} <= set(clima.columns):
            clima["fecha"] = _to_date(clima["fecha"])
            clima["bloque_base"] = pd.to_numeric(clima["bloque_base"], errors="coerce").astype("Int64")
            c_take = [
                "fecha",
                "bloque_base",
                "rainfall_mm_dia",
                "horas_lluvia",
                "temp_avg_dia",
                "solar_energy_j_m2_dia",
                "wind_speed_avg_dia",
                "wind_run_dia",
                "gdc_dia",
            ]
            c_take = [c for c in c_take if c in clima.columns]
            clima2 = clima[c_take].drop_duplicates(subset=["fecha", "bloque_base"])
            df = df.merge(
                clima2,
                left_on=["fecha_evento", "bloque_base"],
                right_on=["fecha", "bloque_base"],
                how="left",
            )
            if "fecha" in df.columns:
                df = df.drop(columns=["fecha"])

    for c in ["sp_month", "sp_weekofyear", "sp_doy", "sp_dow", "tallos_proy"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = _with_common_schema(df, stage="VEG", row_source="features_harvest_window_ml1")
    return out


def _build_stage_harvest_day() -> pd.DataFrame:
    df = read_parquet(FEATURES_DIR / "features_curva_cosecha_bloque_dia.parquet").copy()
    df.columns = [str(c).strip() for c in df.columns]

    df["fecha_evento"] = _to_date(df["fecha"])
    df["ciclo_id"] = df["ciclo_id"].astype("string")

    _coalesce(df, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce(df, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce(df, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])
    if "dia_rel_cosecha_real" in df.columns:
        drr = pd.to_numeric(df.get("dia_rel_cosecha_real"), errors="coerce")
        m = drr.notna()
        if bool(m.any()):
            # dia_rel_cosecha_real is 0-based in source; convert to 1-based harvest day.
            df.loc[m, "day_in_harvest"] = drr.loc[m] + 1.0

    if "fecha_sp" in df.columns:
        fsp = _to_date(df["fecha_sp"])
        df["sp_month"] = fsp.dt.month
        df["sp_weekofyear"] = fsp.dt.isocalendar().week.astype("Int64")
        df["sp_doy"] = fsp.dt.dayofyear
        df["sp_dow"] = fsp.dt.dayofweek

    df["target_factor_tallos_dia"] = pd.to_numeric(df.get("factor_tallos_dia_clipped"), errors="coerce")

    out = _with_common_schema(df, stage="HARVEST_DAY", row_source="features_curva_cosecha_bloque_dia")
    return out


def _build_stage_harvest_grade() -> pd.DataFrame:
    feat = read_parquet(FEATURES_DIR / "features_cosecha_bloque_fecha.parquet").copy()
    feat.columns = [str(c).strip() for c in feat.columns]

    curva = read_parquet(FEATURES_DIR / "features_curva_cosecha_bloque_dia.parquet").copy()
    curva.columns = [str(c).strip() for c in curva.columns]

    peso = read_parquet(FEATURES_DIR / "features_peso_tallo_grado_bloque_dia.parquet").copy()
    peso.columns = [str(c).strip() for c in peso.columns]

    peso_real = read_parquet(IN_FACT_PESO_REAL).copy()
    peso_real.columns = [str(c).strip() for c in peso_real.columns]

    kg = read_parquet(IN_PRED_KG_ML1).copy()
    kg.columns = [str(c).strip() for c in kg.columns]

    dim_var = read_parquet(IN_DIM_VARIEDAD).copy()
    dim_var.columns = [str(c).strip() for c in dim_var.columns]

    feat["fecha"] = _to_date(feat["fecha"])
    feat["fecha_evento"] = feat["fecha"]
    feat["ciclo_id"] = feat["ciclo_id"].astype("string")

    _coalesce(feat, "area", ["area", "area_x", "area_y"])
    _coalesce(feat, "tipo_sp", ["tipo_sp", "tipo_sp_x", "tipo_sp_y"])

    _coalesce(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce(feat, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])
    if "dia_rel_cosecha_real" in feat.columns:
        drr = pd.to_numeric(feat.get("dia_rel_cosecha_real"), errors="coerce")
        m = drr.notna()
        if bool(m.any()):
            # dia_rel_cosecha_real is 0-based in source; convert to 1-based harvest day.
            feat.loc[m, "day_in_harvest"] = drr.loc[m] + 1.0

    key = ["fecha", "bloque_base", "variedad_canon", "grado"]
    for c in key:
        if c == "fecha":
            peso[c] = _to_date(peso[c])
        elif c in {"bloque_base", "grado"}:
            peso[c] = pd.to_numeric(peso[c], errors="coerce").astype("Int64")
        else:
            peso[c] = _canon_str(peso[c])

    feat["bloque_base"] = pd.to_numeric(feat["bloque_base"], errors="coerce").astype("Int64")
    feat["grado"] = pd.to_numeric(feat["grado"], errors="coerce").astype("Int64")
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])

    curva["fecha"] = _to_date(curva["fecha"])
    curva["bloque_base"] = pd.to_numeric(curva["bloque_base"], errors="coerce").astype("Int64")
    curva["variedad_canon"] = _canon_str(curva["variedad_canon"])
    curva["ciclo_id"] = curva["ciclo_id"].astype("string")
    curva_take = [
        "ciclo_id",
        "fecha",
        "bloque_base",
        "variedad_canon",
        "factor_tallos_dia_clipped",
        "tallos_pred_baseline_dia",
    ]
    curva_take = [c for c in curva_take if c in curva.columns]
    curva2 = curva[curva_take].drop_duplicates(subset=["ciclo_id", "fecha", "bloque_base", "variedad_canon"])
    feat = feat.merge(curva2, on=["ciclo_id", "fecha", "bloque_base", "variedad_canon"], how="left", suffixes=("", "_curva"))

    kg["fecha"] = _to_date(kg["fecha"])
    kg["bloque_base"] = pd.to_numeric(kg["bloque_base"], errors="coerce").astype("Int64")
    kg["grado"] = pd.to_numeric(kg["grado"], errors="coerce").astype("Int64")
    kg["variedad_canon"] = _canon_str(kg["variedad_canon"])
    if "ciclo_id" in kg.columns:
        kg["ciclo_id"] = kg["ciclo_id"].astype("string")
    ktake = [
        "ciclo_id",
        "fecha",
        "bloque_base",
        "variedad_canon",
        "grado",
        "tallos_pred_baseline_grado_dia",
        "tallos_pred_ml1_grado_dia",
    ]
    ktake = [c for c in ktake if c in kg.columns]
    kkey = ["fecha", "bloque_base", "variedad_canon", "grado"]
    if "ciclo_id" in ktake:
        kkey = ["ciclo_id"] + kkey
    kg2 = kg[ktake].drop_duplicates(subset=kkey)
    on_key = ["fecha", "bloque_base", "variedad_canon", "grado"]
    if "ciclo_id" in kg2.columns and "ciclo_id" in feat.columns:
        on_key = ["ciclo_id"] + on_key
    feat = feat.merge(kg2, on=on_key, how="left", suffixes=("", "_kg"))

    peso_take = [
        "fecha",
        "bloque_base",
        "variedad_canon",
        "grado",
        "factor_peso_tallo_clipped",
        "peso_tallo_baseline_g",
    ]
    peso_take = [c for c in peso_take if c in peso.columns]
    peso2 = peso[peso_take].drop_duplicates(subset=key)

    df = feat.merge(peso2, on=key, how="left", suffixes=("", "_peso"))

    _coalesce(df, "peso_tallo_baseline_g", ["peso_tallo_baseline_g", "peso_tallo_baseline_g_peso"])
    _coalesce(df, "tallos_pred_baseline_dia", ["tallos_pred_baseline_dia", "tallos_pred_baseline_dia_curva"])
    _coalesce(df, "factor_tallos_dia_clipped", ["factor_tallos_dia_clipped", "factor_tallos_dia_clipped_curva"])
    _coalesce(df, "tallos_pred_baseline_grado_dia", ["tallos_pred_baseline_grado_dia", "tallos_pred_baseline_grado_dia_kg"])
    _coalesce(df, "tallos_pred_ml1_grado_dia", ["tallos_pred_ml1_grado_dia", "tallos_pred_ml1_grado_dia_kg"])

    grp_day = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    df["tallos_pred_ml1_grado_dia"] = pd.to_numeric(df["tallos_pred_ml1_grado_dia"], errors="coerce")
    df["tallos_pred_ml1_dia"] = df.groupby(grp_day, dropna=False)["tallos_pred_ml1_grado_dia"].transform("sum")

    if "bloque_base" not in peso_real.columns:
        if "bloque_padre" in peso_real.columns:
            peso_real["bloque_base"] = peso_real["bloque_padre"]
        elif "bloque" in peso_real.columns:
            peso_real["bloque_base"] = peso_real["bloque"]
        else:
            peso_real["bloque_base"] = pd.NA

    if "variedad_canon" not in peso_real.columns:
        if "variedad" in peso_real.columns:
            m = dim_var.copy()
            m["variedad_raw_norm"] = _canon_str(m["variedad_raw"])
            m["variedad_canon"] = _canon_str(m["variedad_canon"])
            peso_real["variedad_raw_norm"] = _canon_str(peso_real["variedad"])
            peso_real = peso_real.merge(
                m[["variedad_raw_norm", "variedad_canon"]].drop_duplicates(),
                on="variedad_raw_norm",
                how="left",
            )
            peso_real["variedad_canon"] = peso_real["variedad_canon"].fillna(peso_real["variedad_raw_norm"])
        else:
            peso_real["variedad_canon"] = "UNKNOWN"

    peso_real["fecha"] = _to_date(peso_real["fecha"])
    peso_real["bloque_base"] = pd.to_numeric(peso_real["bloque_base"], errors="coerce").astype("Int64")
    peso_real["grado"] = pd.to_numeric(peso_real["grado"], errors="coerce").astype("Int64")
    peso_real["variedad_canon"] = _canon_str(peso_real["variedad_canon"])
    peso_real["peso_tallo_real_g"] = pd.to_numeric(peso_real.get("peso_tallo_real_g"), errors="coerce")

    pr = (
        peso_real[["fecha", "bloque_base", "variedad_canon", "grado", "peso_tallo_real_g"]]
        .dropna(subset=["fecha", "bloque_base", "variedad_canon", "grado"])
        .groupby(key, dropna=False, as_index=False)
        .agg(peso_tallo_real_g=("peso_tallo_real_g", "mean"))
    )
    df = df.merge(pr, on=key, how="left")

    df["target_share_grado"] = pd.to_numeric(df.get("share_grado_real"), errors="coerce")
    df["target_factor_tallos_dia"] = pd.to_numeric(df.get("factor_tallos_dia_clipped"), errors="coerce")
    fac_from_feat = pd.to_numeric(df.get("factor_peso_tallo_clipped"), errors="coerce")
    fac_from_real = np.where(
        pd.to_numeric(df["peso_tallo_baseline_g"], errors="coerce").fillna(0.0) > 0.0,
        pd.to_numeric(df["peso_tallo_real_g"], errors="coerce")
        / pd.to_numeric(df["peso_tallo_baseline_g"], errors="coerce"),
        np.nan,
    )
    df["target_factor_peso_tallo"] = pd.Series(fac_from_feat, index=df.index).where(
        pd.Series(fac_from_feat, index=df.index).notna(),
        pd.Series(fac_from_real, index=df.index),
    )

    out = _with_common_schema(df, stage="HARVEST_GRADE", row_source="features_cosecha_plus_peso")
    return out


def _build_harvest_detail_for_post() -> pd.DataFrame:
    feat = read_parquet(FEATURES_DIR / "features_cosecha_bloque_fecha.parquet").copy()
    feat.columns = [str(c).strip() for c in feat.columns]

    feat["fecha"] = _to_date(feat["fecha"])
    feat["bloque_base"] = pd.to_numeric(feat["bloque_base"], errors="coerce").astype("Int64")
    feat["grado"] = pd.to_numeric(feat.get("grado"), errors="coerce").astype("Int64")
    if "variedad_canon" not in feat.columns:
        feat["variedad_canon"] = "UNKNOWN"
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    if "ciclo_id" not in feat.columns:
        feat["ciclo_id"] = "UNKNOWN"
    feat["ciclo_id"] = feat["ciclo_id"].astype("string").fillna("UNKNOWN")

    _coalesce(feat, "area", ["area", "area_x", "area_y"])
    _coalesce(feat, "tipo_sp", ["tipo_sp", "tipo_sp_x", "tipo_sp_y"])
    _coalesce(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce(feat, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])
    if "dia_rel_cosecha_real" in feat.columns:
        drr = pd.to_numeric(feat.get("dia_rel_cosecha_real"), errors="coerce")
        m = drr.notna()
        if bool(m.any()):
            # dia_rel_cosecha_real is 0-based in source; convert to 1-based harvest day.
            feat.loc[m, "day_in_harvest"] = drr.loc[m] + 1.0

    key = ["fecha", "bloque_base", "variedad_canon", "grado"]
    num_cols = [
        "tallos_proy",
        "day_in_harvest",
        "rel_pos",
        "n_harvest_days",
        "pct_avance_real",
        "dia_rel_cosecha_real",
        "gdc_acum_real",
        "rainfall_mm_dia",
        "horas_lluvia",
        "temp_avg_dia",
        "solar_energy_j_m2_dia",
        "wind_speed_avg_dia",
        "wind_run_dia",
        "gdc_dia",
        "dias_desde_sp",
        "gdc_acum_desde_sp",
        "share_grado_baseline",
        "peso_tallo_baseline_g",
        "tallos_pred_baseline_dia",
    ]
    for c in num_cols:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")

    agg = {c: "median" for c in num_cols if c in feat.columns}
    agg["ciclo_id"] = "first"
    agg["area"] = "first"
    agg["tipo_sp"] = "first"

    out = (
        feat[key + [c for c in ["ciclo_id", "area", "tipo_sp"] if c in feat.columns] + [c for c in num_cols if c in feat.columns]]
        .dropna(subset=["fecha", "bloque_base", "grado"])
        .groupby(key, dropna=False, as_index=False)
        .agg(agg)
    )

    for c in ["day_in_harvest", "n_harvest_days", "dia_rel_cosecha_real"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round()
    return out


def _build_seed_distribution_for_post() -> pd.DataFrame:
    seed = read_parquet(IN_POST_SEED).copy()
    seed.columns = [str(c).strip() for c in seed.columns]

    need = {"fecha", "grado", "destino", "bloque_base", "variedad_canon", "cajas_split_grado_dia"}
    miss = need - set(seed.columns)
    if miss:
        raise ValueError(f"pred_poscosecha_seed missing columns: {sorted(miss)}")

    seed["fecha"] = _to_date(seed["fecha"])
    seed["grado"] = pd.to_numeric(seed["grado"], errors="coerce").astype("Int64")
    seed["destino"] = _canon_str(seed["destino"])
    seed["bloque_base"] = pd.to_numeric(seed["bloque_base"], errors="coerce").astype("Int64")
    seed["variedad_canon"] = _canon_str(seed["variedad_canon"])

    for c in ["cajas_ml1_grado_dia", "cajas_split_grado_dia", "cajas_post_seed"]:
        if c in seed.columns:
            seed[c] = pd.to_numeric(seed[c], errors="coerce")
        else:
            seed[c] = np.nan

    gkey = ["fecha", "grado", "destino", "bloque_base", "variedad_canon"]
    seed = (
        seed.groupby(gkey, dropna=False, as_index=False)
        .agg(
            fecha_post_pred=("fecha_post_pred", "first"),
            cajas_ml1_grado_dia=("cajas_ml1_grado_dia", "sum"),
            cajas_split_grado_dia=("cajas_split_grado_dia", "sum"),
            cajas_post_seed=("cajas_post_seed", "sum"),
        )
    )

    grp = ["fecha", "grado", "destino"]
    den = seed.groupby(grp, dropna=False)["cajas_split_grado_dia"].transform("sum")
    ngrp = seed.groupby(grp, dropna=False)["cajas_split_grado_dia"].transform("size").clip(lower=1)
    seed["share_block_post"] = np.where(
        pd.to_numeric(den, errors="coerce").fillna(0.0) > 0.0,
        seed["cajas_split_grado_dia"] / den,
        1.0 / ngrp.astype(float),
    )
    return seed


def _build_stage_post() -> pd.DataFrame:
    post_real = read_parquet(SILVER_DIR / "fact_hidratacion_real_post_grado_destino.parquet").copy()
    post_real.columns = [str(c).strip() for c in post_real.columns]

    mer = read_parquet(SILVER_DIR / "dim_mermas_ajuste_fecha_post_destino.parquet").copy()
    mer.columns = [str(c).strip() for c in mer.columns]

    hdet = _build_harvest_detail_for_post()
    seed = _build_seed_distribution_for_post()
    kg = read_parquet(IN_PRED_KG_ML1).copy()
    kg.columns = [str(c).strip() for c in kg.columns]

    # Base post projection comes from seed (future + history), not only real post rows.
    seed_h = seed.merge(
        hdet,
        on=["fecha", "bloque_base", "variedad_canon", "grado"],
        how="left",
        suffixes=("", "_h"),
    )
    for c in ["area", "tipo_sp", "ciclo_id"]:
        if c not in seed_h.columns:
            seed_h[c] = "UNKNOWN"

    # Calendar and event dates
    seed_h["fecha"] = _to_date(seed_h["fecha"])
    seed_h["fecha_post_pred"] = _to_date(seed_h["fecha_post_pred"])
    seed_h["fecha_evento"] = seed_h["fecha"]
    seed_h["fecha_post"] = seed_h["fecha_post_pred"]
    seed_h["dow"] = seed_h["fecha_evento"].dt.dayofweek
    seed_h["month"] = seed_h["fecha_evento"].dt.month
    seed_h["weekofyear"] = seed_h["fecha_evento"].dt.isocalendar().week.astype("Int64")
    seed_h["dow_post"] = seed_h["fecha_post"].dt.dayofweek
    seed_h["month_post"] = seed_h["fecha_post"].dt.month
    seed_h["weekofyear_post"] = seed_h["fecha_post"].dt.isocalendar().week.astype("Int64")

    # Bring kg in green per block/grade/day and split it by destination share.
    kg["fecha"] = _to_date(kg["fecha"])
    kg["bloque_base"] = pd.to_numeric(kg["bloque_base"], errors="coerce").astype("Int64")
    kg["grado"] = pd.to_numeric(kg["grado"], errors="coerce").astype("Int64")
    kg["variedad_canon"] = _canon_str(kg["variedad_canon"])
    kg["kg_ml1_grado_dia"] = pd.to_numeric(kg.get("kg_ml1_grado_dia"), errors="coerce")
    kg["tallos_pred_ml1_grado_dia"] = pd.to_numeric(kg.get("tallos_pred_ml1_grado_dia"), errors="coerce")
    kg["tallos_pred_baseline_grado_dia"] = pd.to_numeric(kg.get("tallos_pred_baseline_grado_dia"), errors="coerce")

    kkey = ["fecha", "bloque_base", "variedad_canon", "grado"]
    kg2 = (
        kg[kkey + ["kg_ml1_grado_dia", "tallos_pred_ml1_grado_dia", "tallos_pred_baseline_grado_dia"]]
        .dropna(subset=["fecha", "bloque_base", "variedad_canon", "grado"])
        .groupby(kkey, dropna=False, as_index=False)
        .agg(
            kg_ml1_grado_dia=("kg_ml1_grado_dia", "sum"),
            tallos_pred_ml1_grado_dia=("tallos_pred_ml1_grado_dia", "sum"),
            tallos_pred_baseline_grado_dia=("tallos_pred_baseline_grado_dia", "sum"),
        )
    )
    df = seed_h.merge(kg2, on=kkey, how="left")

    for c in ["cajas_ml1_grado_dia", "cajas_split_grado_dia", "cajas_post_seed"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["kg_ml1_grado_dia"] = pd.to_numeric(df.get("kg_ml1_grado_dia"), errors="coerce")
    df["tallos_pred_ml1_grado_dia"] = pd.to_numeric(df.get("tallos_pred_ml1_grado_dia"), errors="coerce")
    df["tallos_pred_baseline_grado_dia"] = pd.to_numeric(df.get("tallos_pred_baseline_grado_dia"), errors="coerce")

    share_dest = np.where(
        df["cajas_ml1_grado_dia"].fillna(0.0) > 0.0,
        df["cajas_split_grado_dia"] / df["cajas_ml1_grado_dia"],
        np.nan,
    )
    share_dest = pd.Series(share_dest, index=df.index).fillna(1.0 / 3.0).clip(lower=0.0, upper=1.0)
    df["kg_verde_ref"] = df["kg_ml1_grado_dia"] * share_dest
    df["gramos_verde_ref"] = df["kg_verde_ref"] * 1000.0
    df["tallos_post_proy"] = df["tallos_pred_ml1_grado_dia"] * share_dest
    df["tallos_pred_ml1_dia"] = df.groupby(["fecha_evento", "ciclo_id", "bloque_base", "variedad_canon"], dropna=False)["tallos_pred_ml1_grado_dia"].transform("sum")

    # Post real labels (only where history exists).
    post_real["fecha_cosecha"] = _to_date(post_real["fecha_cosecha"])
    post_real["fecha_post"] = _to_date(post_real["fecha_post"])
    post_real["grado"] = pd.to_numeric(post_real["grado"], errors="coerce").astype("Int64")
    post_real["destino"] = _canon_str(post_real["destino"])
    post_real["dh_dias"] = pd.to_numeric(post_real.get("dh_dias"), errors="coerce")
    post_real["hidr_pct"] = pd.to_numeric(post_real.get("hidr_pct"), errors="coerce")
    post_real["tallos"] = pd.to_numeric(post_real.get("tallos"), errors="coerce")
    post_real["peso_base_g"] = pd.to_numeric(post_real.get("peso_base_g"), errors="coerce")
    post_real["peso_post_g"] = pd.to_numeric(post_real.get("peso_post_g"), errors="coerce")

    rkey = ["fecha_cosecha", "fecha_post", "grado", "destino"]
    real2 = (
        post_real[rkey + ["dh_dias", "hidr_pct", "tallos", "peso_base_g", "peso_post_g"]]
        .dropna(subset=["fecha_cosecha", "fecha_post", "grado", "destino"])
        .groupby(rkey, dropna=False, as_index=False)
        .agg(
            dh_dias=("dh_dias", "median"),
            hidr_pct=("hidr_pct", "median"),
            tallos_real_post=("tallos", "sum"),
            peso_base_g_real_post=("peso_base_g", "sum"),
            peso_post_g_real_post=("peso_post_g", "sum"),
        )
    )
    df = df.merge(
        real2,
        left_on=["fecha_evento", "fecha_post", "grado", "destino"],
        right_on=["fecha_cosecha", "fecha_post", "grado", "destino"],
        how="left",
    )
    if "fecha_cosecha" in df.columns:
        df = df.drop(columns=["fecha_cosecha"])

    # Merma/Ajuste by post date + destination (baseline reference and optional labels).
    mer["fecha_post"] = _to_date(mer["fecha_post"])
    mer["destino"] = _canon_str(mer["destino"])
    for c in ["factor_desp", "factor_ajuste", "w2_kg", "w2a_kg", "wideal_kg", "desp_pct"]:
        if c in mer.columns:
            mer[c] = pd.to_numeric(mer[c], errors="coerce")

    mer_take = [
        "fecha_post",
        "destino",
        "factor_desp",
        "factor_ajuste",
        "w2_kg",
        "w2a_kg",
        "wideal_kg",
        "desp_pct",
    ]
    mer_take = [c for c in mer_take if c in mer.columns]
    mer2 = mer[mer_take].drop_duplicates(subset=["fecha_post", "destino"])
    df = df.merge(mer2, on=["fecha_post", "destino"], how="left")

    # Keep real grams only when available (for diagnostics).
    df["gramos_post_real_ref"] = pd.to_numeric(df.get("peso_post_g_real_post"), errors="coerce")
    df["kg_post_real_ref"] = df["gramos_post_real_ref"] / 1000.0
    df["tallos_post"] = pd.to_numeric(df.get("tallos_real_post"), errors="coerce")
    df["peso_base_g"] = pd.to_numeric(df.get("peso_base_g_real_post"), errors="coerce")
    df["peso_post_g"] = pd.to_numeric(df.get("peso_post_g_real_post"), errors="coerce")

    for c in ["day_in_harvest", "n_harvest_days", "dia_rel_cosecha_real"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round()

    # Targets are available only where real post is present.
    df["target_dh_dias"] = pd.to_numeric(df.get("dh_dias"), errors="coerce")
    df["target_factor_hidr"] = 1.0 + pd.to_numeric(df.get("hidr_pct"), errors="coerce")
    df["target_factor_desp"] = pd.to_numeric(df.get("factor_desp"), errors="coerce")
    df["target_factor_ajuste"] = pd.to_numeric(df.get("factor_ajuste"), errors="coerce")
    df["target_weight"] = pd.to_numeric(df.get("share_block_post"), errors="coerce").fillna(1.0).clip(lower=0.0)

    out = _with_common_schema(df, stage="POST", row_source="seed_poscosecha_with_trace")
    return out


def main() -> None:
    created_at = pd.Timestamp(datetime.now(timezone.utc))

    parts = [
        _build_stage_veg(),
        _build_stage_harvest_grade(),
        _build_stage_post(),
    ]
    ds = pd.concat(parts, ignore_index=True)
    ds = _add_autoregressive_features(ds)
    ds["row_id"] = np.arange(len(ds), dtype=np.int64)
    ds["created_at"] = created_at
    ds = _add_clear_name_aliases(ds)

    ds = ds.sort_values(["fecha_evento", "stage", "row_source"], kind="mergesort").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(ds, OUT_PATH)

    print(f"[OK] Wrote dataset: {OUT_PATH}")
    print(f"     rows={len(ds):,} | rows_with_any_target={int(ds['has_any_target'].sum()):,}")
    print(f"     fecha_evento=[{ds['fecha_evento'].min()} .. {ds['fecha_evento'].max()}]")
    print("     stage rows:")
    print(ds["stage"].value_counts(dropna=False).to_string())
    print("     target coverage:")
    for t in TARGET_COLS:
        cov = float(ds[f"mask_{t}"].mean()) if len(ds) else 0.0
        print(f"       - {t}: {cov:.2%}")


if __name__ == "__main__":
    main()
