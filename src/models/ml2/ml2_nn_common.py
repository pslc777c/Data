from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ml2TargetSpec:
    corr_target: str
    original_target: str
    ml1_pred_col: str
    mode: str  # "add" | "log_ratio"
    corr_clip: tuple[float, float]
    final_clip: tuple[float, float]


ML2_TARGET_SPECS: list[Ml2TargetSpec] = [
    Ml2TargetSpec(
        corr_target="ml2_target_delta_d_start",
        original_target="target_d_start",
        ml1_pred_col="pred_d_start",
        mode="add",
        corr_clip=(-21.0, 21.0),
        final_clip=(0.0, 180.0),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_delta_n_harvest_days",
        original_target="target_n_harvest_days",
        ml1_pred_col="pred_n_harvest_days",
        mode="add",
        corr_clip=(-14.0, 21.0),
        final_clip=(1.0, 180.0),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_factor_tallos_dia",
        original_target="target_factor_tallos_dia",
        ml1_pred_col="pred_factor_tallos_dia",
        mode="log_ratio",
        corr_clip=(-1.5, 1.5),
        final_clip=(0.03, 5.00),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_share_grado",
        original_target="target_share_grado",
        ml1_pred_col="pred_share_grado",
        mode="log_ratio",
        corr_clip=(-1.2, 1.2),
        final_clip=(0.00, 1.00),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_factor_peso_tallo",
        original_target="target_factor_peso_tallo",
        ml1_pred_col="pred_factor_peso_tallo",
        mode="log_ratio",
        corr_clip=(-0.8, 0.8),
        final_clip=(0.60, 1.60),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_delta_dh_dias",
        original_target="target_dh_dias",
        ml1_pred_col="pred_dh_dias",
        mode="add",
        corr_clip=(-5.0, 5.0),
        final_clip=(0.0, 30.0),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_factor_hidr",
        original_target="target_factor_hidr",
        ml1_pred_col="pred_factor_hidr",
        mode="log_ratio",
        corr_clip=(-1.2, 1.2),
        final_clip=(0.60, 3.00),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_factor_desp",
        original_target="target_factor_desp",
        ml1_pred_col="pred_factor_desp",
        mode="log_ratio",
        corr_clip=(-1.2, 1.2),
        final_clip=(0.05, 1.00),
    ),
    Ml2TargetSpec(
        corr_target="ml2_target_logratio_factor_ajuste",
        original_target="target_factor_ajuste",
        ml1_pred_col="pred_factor_ajuste",
        mode="log_ratio",
        corr_clip=(-1.2, 1.2),
        final_clip=(0.50, 2.00),
    ),
]

ML2_CAT_COLS = [
    "stage",
    "row_source",
    "variedad_canon",
    "tipo_sp",
    "grado",
    "destino",
    "estado_ciclo",
]

ML2_NUM_COLS = [
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
    "tallos_post_proy",
    "gramos_verde_ref",
    "kg_verde_ref",
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
    "ar_tallos_real_dia_lag1",
    "ar_tallos_real_dia_roll3",
    "ar_tallos_real_dia_roll7",
    "ar_ratio_real_vs_base_lag1",
    "ar_ratio_real_vs_base_roll3",
    "ar_pct_avance_real_lag1",
    "ar_gdc_dia_roll3",
    "ar_temp_avg_dia_roll3",
    "ar_rainfall_mm_roll3",
    "ar_peso_tallo_real_lag1",
    "ar_peso_tallo_real_roll7",
    "ar_peso_tallo_real_roll14",
    "ar_ratio_peso_real_vs_base_lag1",
    "ar_ratio_peso_real_vs_base_roll7",
    "ar_ratio_peso_real_vs_base_roll14",
    "pred_d_start",
    "pred_n_harvest_days",
    "pred_factor_tallos_dia",
    "pred_share_grado",
    "pred_factor_peso_tallo",
    "pred_dh_dias",
    "pred_factor_hidr",
    "pred_factor_desp",
    "pred_factor_ajuste",
    "is_active_cycle",
    "is_closed_cycle",
]


def canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def corr_from_real_and_ml1(
    y_real: np.ndarray,
    y_ml1: np.ndarray,
    mode: str,
    eps: float = 1e-6,
) -> np.ndarray:
    if mode == "add":
        return y_real - y_ml1
    if mode == "log_ratio":
        return np.log((y_real + eps) / (y_ml1 + eps))
    raise ValueError(f"Unknown mode: {mode}")


def final_from_ml1_and_corr(
    y_ml1: np.ndarray,
    corr: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "add":
        return y_ml1 + corr
    if mode == "log_ratio":
        return y_ml1 * np.exp(corr)
    raise ValueError(f"Unknown mode: {mode}")


def valid_corr_mask(y_real: np.ndarray, y_ml1: np.ndarray, mode: str) -> np.ndarray:
    m = np.isfinite(y_real) & np.isfinite(y_ml1)
    if mode == "log_ratio":
        m &= (y_real > 0.0) & (y_ml1 > 0.0)
    return m


def specs_as_dicts() -> list[dict[str, Any]]:
    return [asdict(s) for s in ML2_TARGET_SPECS]
