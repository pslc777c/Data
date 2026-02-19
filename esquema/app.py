from __future__ import annotations

import streamlit as st
from pyvis.network import Network

from ops.runner import Step


# ==========================================================
# REGISTRY (pegado de tu "original actual")
# ==========================================================
def build_registry() -> list[Step]:
    """
    Registry lineal (orden de ejecución) que asegura que los .py se corran sin fallar
    por dependencias faltantes.

    Nota:
    - outputs_rel=[] => nunca se hace skip por existencia (corre siempre)
    - outputs_rel con glob (ej: models/ml2/harvest_start_ml2_*_meta.json) => skip si hay >=1 match
    """
    steps: list[Step] = []

    # =========================
    # BRONZE
    # =========================
    steps += [
        Step("balanza_cosecha_raw", "bronze", "src/bronze/build_balanza_cosecha_raw.py",
             ["bronze/balanza_cosecha_raw.parquet"]),
        Step("balanza_1c_raw", "bronze", "src/bronze/build_balanza_1c_raw.py",
             ["bronze/balanza_1c_raw.parquet"]),
        Step("balanza_mermas_sources", "bronze", "src/bronze/build_balanza_mermas_sources.py",
             ["bronze/balanza_2_raw.parquet", "bronze/balanza_2a_raw.parquet"]),
        Step("fenograma_sources", "bronze", "src/bronze/build_fenograma_sources.py",
             ["bronze/balanza_bloque_fecha_raw.parquet", "bronze/fenograma_xlsm_raw.parquet",
              "bronze/indices_clo_raw.parquet", "bronze/indices_xl_raw.parquet"]),
        Step("ghu_maestro_horas", "bronze", "src/bronze/build_ghu_maestro_horas.py",
             ["bronze/ghu_maestro_horas.parquet"]),
        Step("personal_sources", "bronze", "src/bronze/build_personal_sources.py",
             ["bronze/personal_raw.parquet"]),
        Step("ventas_sources", "bronze", "src/bronze/build_ventas_sources.py",
             ["bronze/ventas_2025_raw.parquet", "bronze/ventas_2026_raw.parquet"]),
        Step("weather_hour_main", "bronze", "src/bronze/build_weather_hour_main.py",
             ["bronze/weather_hour_main.parquet"]),
        Step("weather_hour_a4", "bronze", "src/bronze/build_weather_hour_a4.py",
             ["bronze/weather_hour_a4.parquet"]),
    ]

    # =========================
    # SILVER (base)
    # =========================
    steps += [
        Step("weather_hour_estado", "silver", "src/silver/build_weather_hour_estado.py",
             ["silver/weather_hour_estado.parquet"]),
        Step("weather_hour_wide", "silver", "src/silver/build_weather_hour_wide.py",
             ["silver/weather_hour_wide.parquet"]),
        Step("ciclo_maestro_from_sources", "silver", "src/silver/build_ciclo_maestro_from_sources.py",
             ["silver/fact_ciclo_maestro.parquet", "silver/grid_ciclo_fecha.parquet"]),
        Step("hidratacion_real_from_balanza2", "silver", "src/silver/build_hidratacion_real_from_balanza2.py",
             ["silver/fact_hidratacion_real_post_grado_destino.parquet",
              "silver/dim_hidratacion_dh_grado_destino.parquet",
              "silver/dim_hidratacion_fecha_post_grado_destino.parquet"]),
        Step("fact_cosecha_real_grado_dia", "silver", "src/silver/build_fact_cosecha_real_grado_dia.py",
             ["silver/fact_cosecha_real_grado_dia.parquet"]),
        Step("fact_peso_tallo_real_grado_dia", "silver", "src/silver/build_fact_peso_tallo_real_grado_dia.py",
             ["silver/fact_peso_tallo_real_grado_dia.parquet"]),
        Step("fact_cosecha_uph_hora_clima", "silver", "src/silver/build_fact_cosecha_uph_hora_clima.py",
             ["silver/fact_cosecha_uph_hora_clima.parquet"]),
        Step("dim_dist_grado_baseline", "silver", "src/silver/build_dim_dist_grado_baseline.py",
             ["silver/dim_dist_grado_baseline.parquet"]),
        Step("dim_peso_tallo_baseline", "silver", "src/silver/build_dim_peso_tallo_baseline.py",
             ["silver/dim_peso_tallo_baseline.parquet"]),
        Step("dim_peso_tallo_promedio_dia", "silver", "src/silver/build_dim_peso_tallo_promedio_dia.py",
             ["silver/dim_peso_tallo_promedio_dia.parquet"]),
        Step("dim_factor_uph_cosecha_clima", "silver", "src/silver/build_dim_factor_uph_cosecha_clima.py",
             ["silver/dim_factor_uph_cosecha_clima.parquet"]),
        Step("dim_mermas_ajuste_fecha_post", "silver", "src/silver/build_dim_mermas_ajuste_fecha_post.py",
             ["silver/dim_mermas_ajuste_fecha_post_destino.parquet"]),
        Step("dim_dh_baseline_grado_destino", "silver", "src/silver/build_dim_dh_baseline_grado_destino.py",
             ["silver/dim_dh_baseline_grado_destino.parquet"]),
        Step("dim_hidratacion_fecha_cosecha_grado_destino", "silver",
             "src/silver/build_dim_hidratacion_fecha_cosecha_grado_destino.py",
             ["silver/dim_hidratacion_fecha_cosecha_grado_destino.parquet"]),
        Step("dim_hidratacion_baseline_grado_destino", "silver",
             "src/silver/build_dim_hidratacion_baseline_grado_destino.py",
             ["silver/dim_hidratacion_baseline_grado_destino.parquet"]),
        Step("dim_variedad_canon", "silver", "src/silver/build_dim_variedad_canon.py",
             ["silver/dim_variedad_canon.parquet"]),
        Step("dim_clima_bloque_dia", "silver", "src/silver/build_dim_clima_bloque_dia.py",
             ["silver/dim_clima_bloque_dia.parquet"]),
        Step("dim_estado_termico_cultivo_bloque_fecha", "silver",
             "src/silver/build_dim_estado_termico_cultivo_bloque_fecha.py",
             ["silver/dim_estado_termico_cultivo_bloque_fecha.parquet"]),
        Step("dim_mix_proceso_semana_from_ventas", "silver",
             "src/silver/build_dim_mix_proceso_semana_from_ventas.py",
             ["silver/dim_mix_proceso_semana.parquet"]),
        Step("fact_capacidad_proceso_hist", "silver", "src/silver/build_fact_capacidad_proceso_hist.py",
             ["silver/fact_capacidad_proceso_hist.parquet",
              "silver/dim_capacidad_baseline_proceso.parquet"]),
        Step("dim_baseline_capacidad_tallos_h_persona", "silver",
             "src/silver/build_dim_baseline_capacidad_tallos_h_persona.py",
             ["silver/dim_baseline_capacidad_tallos_h_persona.parquet"]),
        Step("dim_capacidad_baseline_tallos_proceso", "silver",
             "src/silver/build_dim_capacidad_baseline_tallos_proceso.py",
             ["silver/dim_capacidad_baseline_tallos_proceso.parquet",
              "silver/fact_capacidad_tallos_proceso_hist.parquet"]),
        Step("milestones_windows", "silver", "src/silver/build_milestones_windows.py",
             ["silver/fact_milestones_ciclo.parquet",
              "silver/milestone_window_ciclo.parquet"]),
    ]

    # =========================
    # PREDS
    # =========================
    steps += [
        Step("pred_milestones_baseline", "preds", "src/preds/build_pred_milestones_baseline.py",
             ["preds/pred_milestones_ciclo.parquet"]),
        Step("milestones_final", "preds", "src/silver/build_milestones_final.py",
             ["silver/milestones_ciclo_final.parquet"]),
        Step("dim_mediana_etapas_tipo_sp_variedad_area", "preds",
             "src/silver/build_dim_mediana_etapas_tipo_sp_variedad_area.py",
             ["silver/dim_mediana_etapas_tipo_sp_variedad_area.parquet"]),
        Step("milestone_window_ciclo_final_with_inference", "preds",
             "src/silver/build_milestone_window_ciclo_final_with_inference.py",
             ["silver/milestone_window_ciclo_final.parquet"]),
        Step("dim_cosecha_progress_bloque_fecha", "preds",
             "src/silver/build_dim_cosecha_progress_bloque_fecha.py",
             ["silver/dim_cosecha_progress_bloque_fecha.parquet"]),
        Step("pred_oferta_dia", "preds", "src/preds/build_pred_oferta_dia.py",
             ["preds/pred_oferta_dia.parquet"]),
        Step("pred_oferta_grado", "preds", "src/preds/build_pred_oferta_grado.py",
             ["preds/pred_oferta_grado.parquet"]),
        Step("pred_peso_grado", "preds", "src/preds/build_pred_peso_grado.py",
             ["preds/pred_peso_grado.parquet"]),
        Step("pred_peso_hidratado_grado", "preds", "src/preds/build_pred_peso_hidratado_grado.py",
             ["preds/pred_peso_hidratado_grado.parquet"]),
        Step("pred_peso_final_ajustado_grado", "preds", "src/preds/build_pred_peso_final_ajustado_grado.py",
             ["preds/pred_peso_final_ajustado_grado.parquet"]),
        Step("pred_cajas_from_peso_final", "preds", "src/preds/build_pred_cajas_from_peso_final.py",
             ["preds/pred_cajas_grado.parquet", "preds/pred_cajas_dia.parquet"]),
        Step("pred_tallos_cosecha_dia", "preds", "src/preds/build_pred_tallos_cosecha_dia.py",
             ["preds/pred_tallos_cosecha_dia.parquet"]),
        Step("capacidad_cosecha_dia", "preds", "src/preds/build_capacidad_cosecha_dia.py",
             ["preds/capacidad_cosecha_dia.parquet"]),
        Step("pred_horas_poscosecha", "preds", "src/preds/build_pred_horas_poscosecha.py",
             ["preds/pred_horas_poscosecha_dia.parquet"]),
        Step("pred_plan_horas_dia", "preds", "src/preds/build_pred_plan_horas_dia.py",
             ["preds/pred_plan_horas_dia.parquet"]),
        Step("patch_ciclo_maestro_from_pred_tallos", "preds",
             "src/silver/build_patch_ciclo_maestro_from_pred_tallos.py",
             ["silver/fact_ciclo_maestro_patch_pred_tallos_report.parquet"]),
    ]

    # =========================
    # FEATURES (macro)
    # =========================
    steps += [
        Step("features_ciclo_fecha", "features", "src/features/build_features_ciclo_fecha.py",
             ["features/features_ciclo_fecha.parquet"]),
    ]

    # =========================
    # MODELS (ML1)
    # =========================
    steps += [
        Step("features_harvest_window_ml1", "models", "src/features/build_features_harvest_window_ml1.py",
             ["features/features_harvest_window_ml1.parquet"]),
        Step("apply_harvest_window_ml1", "models", "src/models/ml1/apply_harvest_window_ml1.py",
             ["gold/pred_harvest_window_ml1.parquet"]),
        Step("universe_harvest_grid_ml1", "models", "src/gold/build_universe_harvest_grid_ml1.py",
             ["gold/universe_harvest_grid_ml1.parquet"]),

        Step("features_curva_cosecha_bloque_dia", "models", "src/features/build_features_curva_cosecha_bloque_dia.py",
             ["features/features_curva_cosecha_bloque_dia.parquet"]),
        Step("cap_tallos_real_dia", "models", "src/features/build_cap_tallos_real_dia.py",
             ["gold/dim_cap_tallos_real_dia.parquet"]),
        Step("targets_curva_beta_params", "models", "src/features/build_targets_curva_beta_params.py",
             ["features/trainset_curva_beta_params.parquet"]),
        Step("targets_curva_beta_multiplier_dia", "models", "src/features/build_targets_curva_beta_multiplier_dia.py",
             ["features/trainset_curva_beta_multiplier_dia.parquet"]),
        Step("features_cosecha_bloque_fecha", "models", "src/features/build_features_cosecha_bloque_fecha.py",
             ["features/features_cosecha_bloque_fecha.parquet"]),
        Step("apply_dist_grado_ml1", "models", "src/models/ml1/apply_dist_grado.py",
             ["gold/pred_dist_grado_ml1.parquet"]),
        Step("features_peso_tallo_grado_bloque_dia", "models", "src/features/build_features_peso_tallo_grado_bloque_dia.py",
             ["features/features_peso_tallo_grado_bloque_dia.parquet"]),
        Step("apply_peso_tallo_grado_ml1", "models", "src/models/ml1/apply_peso_tallo_grado.py",
             ["gold/pred_peso_tallo_grado_ml1.parquet"]),

        # curva (escriben shared parquet) => no skip
        Step("apply_curva_beta_params", "models", "src/models/ml1/apply_curva_beta_params.py", []),

        # si los quieres activos:
        # Step("apply_curva_cdf_dia", "models", "src/models/ml1/apply_curva_cdf_dia.py", []),
        # Step("apply_curva_share_dia", "models", "src/models/ml1/apply_curva_share_dia.py", []),
        # Step("apply_curva_tallos_dia", "models", "src/models/ml1/apply_curva_tallos_dia.py", []),
    ]

    # =========================
    # GOLD (ML1 finales)
    # =========================
    steps += [
        Step("pred_tallos_grado_dia_ml1", "gold", "src/gold/build_pred_tallos_grado_dia_ml1.py",
             ["gold/pred_tallos_grado_dia_ml1.parquet"]),
        Step("pred_tallos_grado_dia_ml1_full", "gold", "src/gold/build_pred_tallos_grado_dia_ml1_full.py",
             ["gold/pred_tallos_grado_dia_ml1_full.parquet"]),
        Step("pred_kg_cajas_grado_dia_ml1_full", "gold", "src/gold/build_pred_kg_cajas_grado_dia_ml1_full.py",
             ["gold/pred_kg_grado_dia_ml1_full.parquet",
              "gold/pred_kg_grado_dia_ml1_agg.parquet",
              "gold/pred_cajas_grado_dia_ml1_full.parquet",
              "gold/pred_cajas_grado_dia_ml1_agg.parquet",
              "gold/pred_cajas_dia_ml1_full.parquet"]),
        Step("pred_cajas_postcosecha_seed_mix_grado_dia", "gold",
             "src/gold/build_pred_cajas_postcosecha_seed_mix_grado_dia.py",
             ["gold/pred_poscosecha_seed_grado_dia_bloque_destino.parquet",
              "gold/pred_poscosecha_seed_dia_destino.parquet",
              "gold/pred_poscosecha_seed_dia_total.parquet"]),
        Step("apply_dh_poscosecha_ml1", "gold", "src/models/ml1/apply_dh_poscosecha_ml1.py",
             ["gold/pred_poscosecha_ml1_dh_grado_dia_bloque_destino.parquet"]),
        Step("apply_hidr_poscosecha_ml1", "gold", "src/models/ml1/apply_hidr_poscosecha_ml1.py",
             ["gold/pred_poscosecha_ml1_hidr_grado_dia_bloque_destino.parquet"]),
        Step("apply_desp_ajuste_poscosecha_ml1", "gold", "src/models/ml1/apply_desp_ajuste_poscosecha_ml1.py",
             ["gold/pred_poscosecha_ml1_full_grado_dia_bloque_destino.parquet"]),
        Step("pred_poscosecha_ml1_views", "gold", "src/gold/build_pred_poscosecha_ml1_views.py",
             ["gold/pred_poscosecha_ml1_dia_bloque_destino.parquet",
              "gold/pred_poscosecha_ml1_dia_destino.parquet",
              "gold/pred_poscosecha_ml1_dia_total.parquet"]),
        Step("view_planificacion_campo_tallos", "gold", "src/gold/build_view_planificacion_campo_tallos.py",
             ["gold/view_planificacion_campo_tallos_dia.parquet",
              "gold/view_planificacion_campo_tallos_semana.parquet",
              "gold/view_planificacion_campo_tallos_semana_area.parquet",
              "gold/view_planificacion_campo_tallos_semana_bloque.parquet"]),

        Step("postprocess_curva_share_smooth_ml1", "gold", "src/gold/postprocess_curva_share_smooth_ml1.py", []),
    ]

    # =========================
    # ML2 (ORIGINAL ACTUAL)
    # =========================
    steps += [
        Step("build_ds_harvest_start_ml2_v2", "ml2", "src/gold/build_ds_harvest_start_ml2_v2.py", []),
        Step(
            "train_harvest_start_ml2",
            "ml2",
            "src/models/ml2/train_harvest_start_ml2.py",
            outputs_rel=[
                "models/ml2/harvest_start_ml2_*_meta.json",
                "models/ml2/harvest_start_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_harvest_start_ml2",
            "ml2",
            "src/models/ml2/apply_harvest_start_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_harvest_horizon_ml2_v2", "ml2", "src/gold/build_ds_harvest_horizon_ml2_v2.py", []),
        Step(
            "train_harvest_horizon_ml2",
            "ml2",
            "src/models/ml2/train_harvest_horizon_ml2.py",
            outputs_rel=[
                "models/ml2/harvest_horizon_ml2_*_meta.json",
                "models/ml2/harvest_horizon_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_harvest_horizon_ml2",
            "ml2",
            "src/models/ml2/apply_harvest_horizon_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step(
            "universe_harvest_grid_ml2",
            "ml2",
            "src/gold/build_universe_harvest_grid_ml2.py",
            outputs_rel=["gold/universe_harvest_grid_ml2.parquet"],
        ),

        Step("build_ds_tallos_curve_ml2_v2", "ml2", "src/gold/build_ds_tallos_curve_ml2_v2.py", []),
        Step(
            "train_tallos_curve_ml2",
            "ml2",
            "src/models/ml2/train_tallos_curve_ml2.py",
            outputs_rel=[
                "models/ml2/tallos_curve_ml2_*_meta.json",
                "models/ml2/tallos_curve_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_tallos_curve_ml2",
            "ml2",
            "src/models/ml2/apply_tallos_curve_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_share_grado_ml2_v1", "ml2", "src/gold/build_ds_share_grado_ml2_v1.py", []),
        Step(
            "train_share_grado_ml2",
            "ml2",
            "src/models/ml2/train_share_grado_ml2.py",
            outputs_rel=[
                "models/ml2/share_grado_ml2_*_meta.json",
                "models/ml2/share_grado_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_share_grado_ml2",
            "ml2",
            "src/models/ml2/apply_share_grado_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_peso_tallo_ml2_v1", "ml2", "src/gold/build_ds_peso_tallo_ml2_v1.py", []),
        Step(
            "train_peso_tallo_ml2",
            "ml2",
            "src/models/ml2/train_peso_tallo_ml2.py",
            outputs_rel=[
                "models/ml2/peso_tallo_ml2_*_meta.json",
                "models/ml2/peso_tallo_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_peso_tallo_ml2",
            "ml2",
            "src/models/ml2/apply_peso_tallo_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_dh_poscosecha_ml2_v1", "ml2", "src/gold/build_ds_dh_poscosecha_ml2_v1.py", []),
        Step(
            "train_dh_poscosecha_ml2",
            "ml2",
            "src/models/ml2/train_dh_poscosecha_ml2.py",
            outputs_rel=[
                "models/ml2/dh_poscosecha_ml2_*_meta.json",
                "models/ml2/dh_poscosecha_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_dh_poscosecha_ml2",
            "ml2",
            "src/models/ml2/apply_dh_poscosecha_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_hidr_poscosecha_ml2_v1", "ml2", "src/gold/build_ds_hidr_poscosecha_ml2_v1.py", []),
        Step(
            "train_hidr_poscosecha_ml2",
            "ml2",
            "src/models/ml2/train_hidr_poscosecha_ml2.py",
            outputs_rel=[
                "models/ml2/hidr_poscosecha_ml2_*_meta.json",
                "models/ml2/hidr_poscosecha_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_hidr_poscosecha_ml2",
            "ml2",
            "src/models/ml2/apply_hidr_poscosecha_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_desp_poscosecha_ml2_v1", "ml2", "src/gold/build_ds_desp_poscosecha_ml2_v1.py", []),
        Step(
            "train_desp_poscosecha_ml2",
            "ml2",
            "src/models/ml2/train_desp_poscosecha_ml2.py",
            outputs_rel=[
                "models/ml2/desp_poscosecha_ml2_*_meta.json",
                "models/ml2/desp_poscosecha_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_desp_poscosecha_ml2",
            "ml2",
            "src/models/ml2/apply_desp_poscosecha_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_ds_ajuste_poscosecha_ml2_v1", "ml2", "src/gold/build_ds_ajuste_poscosecha_ml2_v1.py", []),
        Step(
            "train_ajuste_poscosecha_ml2",
            "ml2",
            "src/models/ml2/train_ajuste_poscosecha_ml2.py",
            outputs_rel=[
                "models/ml2/ajuste_poscosecha_ml2_*_meta.json",
                "models/ml2/ajuste_poscosecha_ml2_*.pkl",
            ],
        ),
        Step(
            "apply_ajuste_poscosecha_ml2",
            "ml2",
            "src/models/ml2/apply_ajuste_poscosecha_ml2.py",
            outputs_rel=[],
            args=["--mode", "prod"],
        ),

        Step("build_pred_tallos_ml2_full", "ml2", "src/gold/build_pred_tallos_ml2_full.py", []),
        Step("build_pred_kg_cajas_ml2", "ml2", "src/gold/build_pred_kg_cajas_ml2.py", []),

        Step(
            "build_pred_poscosecha_ml2_seed_mix_grado_dia",
            "ml2",
            "src/gold/build_pred_poscosecha_ml2_seed_mix_grado_dia.py",
            outputs_rel=[
                "gold/pred_poscosecha_ml2_seed_grado_dia_bloque_destino.parquet",
                "gold/pred_poscosecha_ml2_seed_dia_destino.parquet",
                "gold/pred_poscosecha_ml2_seed_dia_total.parquet",
            ],
        ),

        Step(
            "apply_dh_poscosecha_ml1_on_ml2_seed",
            "ml2",
            "src/gold/apply_dh_poscosecha_ml1_on_ml2_seed.py",
            outputs_rel=["gold/pred_poscosecha_ml2_dh_grado_dia_bloque_destino.parquet"],
        ),
        Step(
            "apply_hidr_poscosecha_ml1_on_ml2_dh",
            "ml2",
            "src/gold/apply_hidr_poscosecha_ml1_on_ml2_dh.py",
            outputs_rel=["gold/pred_poscosecha_ml2_hidr_grado_dia_bloque_destino.parquet"],
        ),
        Step(
            "apply_desp_ajuste_poscosecha_ml1_on_ml2_hidr",
            "ml2",
            "src/gold/apply_desp_ajuste_poscosecha_ml1_on_ml2_hidr.py",
            outputs_rel=["gold/pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"],
        ),

        Step(
            "build_pred_poscosecha_ml2_views_from_full",
            "ml2",
            "src/gold/build_pred_poscosecha_ml2_views_from_full.py",
            outputs_rel=[
                "gold/pred_poscosecha_ml2_dia_bloque_destino.parquet",
                "gold/pred_poscosecha_ml2_dia_destino.parquet",
                "gold/pred_poscosecha_ml2_dia_total.parquet",
            ],
        ),

        Step(
            "build_pred_poscosecha_ml2_final_views",
            "ml2",
            "src/gold/build_pred_poscosecha_ml2_final_views.py",
            outputs_rel=[],
        ),
    ]

    # =========================
    # EVAL (ML2 KPIs)
    # =========================
    steps += [
        Step("kpis_ajuste_poscosecha_ml2", "eval", "src/eval/build_kpis_ajuste_poscosecha_ml2.py", []),
        Step("kpis_desp_poscosecha_ml2", "eval", "src/eval/build_kpis_desp_poscosecha_ml2.py", []),
        Step("kpis_dh_poscosecha_ml2", "eval", "src/eval/build_kpis_dh_poscosecha_ml2.py", []),
        Step("kpis_harvest_horizon_ml2", "eval", "src/eval/build_kpis_harvest_horizon_ml2.py", []),
        Step("kpis_harvest_start_ml2", "eval", "src/eval/build_kpis_harvest_start_ml2.py", []),
        Step("kpis_hidr_poscosecha_ml2", "eval", "src/eval/build_kpis_hidr_poscosecha_ml2.py", []),
        Step("kpis_peso_tallo_ml2", "eval", "src/eval/build_kpis_peso_tallo_ml2.py", []),
        Step("kpis_poscosecha_ml2_baseline", "eval", "src/eval/build_kpis_poscosecha_ml2_baseline.py", []),
        Step("kpis_share_grado_ml2", "eval", "src/eval/build_kpis_share_grado_ml2.py", []),
        Step("kpis_tallos_curve_ml2", "eval", "src/eval/build_kpis_tallos_curve_ml2.py", []),
    ]

    # =========================
    # AUDIT (ML1 + ML2)
    # =========================
    steps += [
        Step("audit_universe_harvest_grid_ml1", "audit", "src/audit/audit_universe_harvest_grid_ml1.py",
             ["audit/audit_universe_harvest_grid_ml1_checks.parquet"]),
        Step("audit_dist_grado_ml1", "audit", "src/audit/audit_dist_grado_ml1.py",
             ["audit/audit_dist_grado_ml1_checks.parquet"]),
        Step("audit_poscosecha_seed", "audit", "src/audit/audit_poscosecha_seed.py",
             ["audit/audit_poscosecha_seed_checks.parquet"]),
        Step("audit_ml1_curva_clipping_impact", "audit", "src/audit/audit_ml1_curva_clipping_impact.py",
             ["audit/audit_ml1_curva_clipping_by_cycle.parquet",
              "audit/audit_ml1_curva_clipping_by_relpos.parquet",
              "audit/audit_ml1_curva_clipping_report.parquet"]),
        Step("audit_ml1_curva_share_vs_real", "audit", "src/audit/audit_ml1_curva_share_vs_real.py", []),
        Step("audit_ml1_curva_shares", "audit", "src/audit/audit_ml1_curva_shares.py", []),

        Step("audit_harvest_windows_real", "audit", "src/audit/audit_harvest_windows_real.py", []),
        Step("audit_mass_balance_pred_oferta_vs_gold", "audit", "src/audit/audit_mass_balance_pred_oferta_vs_gold.py", []),
        Step("audit_mismatch_tallos_grado_vs_dia", "audit", "src/audit/audit_mismatch_tallos_grado_vs_dia.py", []),

        Step("audit_ajuste_poscosecha_ml2", "audit", "src/audit/audit_ajuste_poscosecha_ml2.py", []),
        Step("audit_desp_poscosecha_ml2", "audit", "src/audit/audit_desp_poscosecha_ml2.py", []),
        Step("audit_dh_poscosecha_ml2", "audit", "src/audit/audit_dh_poscosecha_ml2.py", []),
        Step("audit_hidr_poscosecha_ml2", "audit", "src/audit/audit_hidr_poscosecha_ml2.py", []),
        Step("audit_harvest_start_ml2", "audit", "src/audit/audit_harvest_start_ml2.py", []),
        Step("audit_harvest_horizon_ml2", "audit", "src/audit/audit_harvest_horizon_ml2.py", []),
        Step("audit_peso_tallo_ml2", "audit", "src/audit/audit_peso_tallo_ml2.py", []),
        Step("audit_share_grado_ml2", "audit", "src/audit/audit_share_grado_ml2.py", []),
        Step("audit_tallos_curve_ml2", "audit", "src/audit/audit_tallos_curve_ml2.py", []),
        Step("audit_poscosecha_ml2_baseline", "audit", "src/audit/audit_poscosecha_ml2_baseline.py", []),
    ]

    return steps


# ==========================================================
# Step -> dict (ROBUSTO a diferentes nombres de atributos)
# ==========================================================
def _pick_attr(obj, candidates: list[str], default=None):
    """Devuelve el primer atributo existente y usable."""
    for a in candidates:
        v = getattr(obj, a, None)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        return v
    return default


def build_steps():
    """
    Convierte list[Step] -> list[dict] para el UI.
    En tu ops.runner.Step, el path del script suele ser `py_rel` (no `script`).
    """
    reg = build_registry()

    def _get_name(step):
        return _pick_attr(step, ["name", "step", "step_name", "id"], default=str(step))

    def _get_layer(step):
        return _pick_attr(step, ["layer", "stage", "group"], default="(unknown)")

    def _get_script(step):
        # IMPORTANTE: soporta `py_rel` (tu caso típico), y otros nombres comunes
        return _pick_attr(step, ["script", "py_rel", "py", "path", "filepath", "file"], default="(unknown)")

    def _get_outputs(step):
        outs = _pick_attr(step, ["outputs_rel", "outputs", "out_rel", "out"], default=[])
        return list(outs or [])

    def _get_args(step):
        a = _pick_attr(step, ["args", "argv", "cli_args"], default=[])
        return list(a or [])

    steps = []
    for s in reg:
        steps.append(
            {
                "name": _get_name(s),
                "layer": _get_layer(s),
                "script": _get_script(s),
                "outputs": _get_outputs(s),
                "args": _get_args(s),
            }
        )
    return steps


# ==========================================================
# UI helpers
# ==========================================================
LAYER_COLORS = {
    "bronze": "#b87333",
    "silver": "#9aa0a6",
    "preds": "#7e57c2",
    "features": "#26a69a",
    "models": "#1e88e5",
    "gold": "#f9a825",
    "ml2": "#8e24aa",
    "eval": "#ff7043",
    "audit": "#ef5350",
}


def neighbors_indices(n: int, idx: int, k: int) -> set[int]:
    lo = max(0, idx - k)
    hi = min(n - 1, idx + k)
    return set(range(lo, hi + 1))


def _tooltip_html(s: dict) -> str:
    outs = s.get("outputs") or []
    args = s.get("args") or []

    if outs:
        skip_rule = "skip si existe >=1 output (o match glob)"
        outs_html = "<br>".join(outs)
    else:
        skip_rule = "NO skip (outputs vacíos / overwrite / shared)"
        outs_html = "(none/overwrite/shared)"

    args_html = " ".join(args) if args else "(none)"

    return (
        f"<b>{s['name']}</b>"
        f"<br><br><b>layer</b>: {s['layer']}"
        f"<br><b>script</b>: {s['script']}"
        f"<br><b>args</b>: {args_html}"
        f"<br><b>skip</b>: {skip_rule}"
        f"<br><b>outputs</b>:<br>{outs_html}"
    )


def make_pyvis_html(steps, selected_name: str | None, focus_k: int | None):
    # Edges como orden lineal (registry order)
    n = len(steps)
    name_to_i = {s["name"]: i for i, s in enumerate(steps)}

    keep_idx = set(range(n))
    if selected_name and focus_k is not None and selected_name in name_to_i:
        i = name_to_i[selected_name]
        keep_idx = neighbors_indices(n, i, focus_k)

    keep_steps = [steps[i] for i in range(n) if i in keep_idx]
    keep_names = {s["name"] for s in keep_steps}

    net = Network(height="700px", width="100%", directed=True, bgcolor="#0e1117", font_color="#e6e6e6")
    net.toggle_physics(True)

    net.set_options(
        """
        var options = {
          "nodes": {
            "shape": "dot",
            "scaling": { "min": 10, "max": 35 },
            "font": { "size": 16 }
          },
          "edges": {
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
            "smooth": { "type": "dynamic" },
            "color": { "inherit": true }
          },
          "physics": {
            "barnesHut": { "gravitationalConstant": -15000, "springLength": 160, "springConstant": 0.05 },
            "minVelocity": 0.75
          }
        }
        """
    )

    # Nodes
    for s in keep_steps:
        layer = s["layer"]
        color = LAYER_COLORS.get(layer, "#888888")
        title = _tooltip_html(s)

        is_sel = bool(selected_name and s["name"] == selected_name)
        size = 28 if is_sel else 18
        border = "#ffffff" if is_sel else "#222222"

        net.add_node(
            s["name"],
            label=s["name"],
            title=title,
            color={"background": color, "border": border},
            size=size,
            group=layer,
        )

    # Linear edges
    for i in range(len(steps) - 1):
        a = steps[i]["name"]
        b = steps[i + 1]["name"]
        if a in keep_names and b in keep_names:
            net.add_edge(a, b)

    return net.generate_html()


# ==========================================================
# APP
# ==========================================================
st.set_page_config(page_title="Macro Plan - Pipeline Explorer", layout="wide")
st.title("📌 Macro Plan - Pipeline Explorer")
st.caption("Grafo interactivo del pipeline en orden del registry. Filtra por capa y enfoca pasos.")

steps_all = build_steps()

# Orden de layers según el dict (si sale algo nuevo, al final)
layer_order = list(LAYER_COLORS.keys())
layers_all = sorted(
    {s["layer"] for s in steps_all},
    key=lambda x: layer_order.index(x) if x in layer_order else 999,
)

with st.sidebar:
    st.header("Controles")

    # Debug opcional (útil para confirmar nombres reales en Step)
    # reg0 = build_registry()[0]
    # st.write("Step attrs:", sorted([a for a in dir(reg0) if not a.startswith("_")]))

    layers_sel = st.multiselect(
        "Layers",
        options=layers_all,
        default=layers_all,
    )

    steps = [s for s in steps_all if s["layer"] in set(layers_sel)]

    step_names = [s["name"] for s in steps]
    selected = st.selectbox("Selecciona un step", options=["(none)"] + step_names, index=0)

    mode = st.radio("Modo", ["Full", "Focus"], horizontal=True)
    k = None
    if mode == "Focus" and selected != "(none)":
        k = st.slider("Vecinos +/- k pasos", min_value=3, max_value=60, value=12, step=1)

    st.divider()
    st.write("Tips:")
    st.write("- Zoom con rueda")
    st.write("- Drag nodes")
    st.write("- Hover para ver script/args/outputs + regla de skip")

selected_name = None if selected == "(none)" else selected

col1, col2 = st.columns([2, 1], gap="large")

with col2:
    st.subheader("🔎 Detalle")
    if selected_name:
        s = next(x for x in steps if x["name"] == selected_name)
        st.code(f"{s['layer']}::{s['name']}", language="text")

        st.write("**Script**")
        st.code(s["script"], language="text")

        st.write("**Args**")
        if s.get("args"):
            st.code(" ".join(s["args"]), language="text")
        else:
            st.write("- (none)")

        st.write("**Outputs (outputs_rel/outputs)**")
        if s["outputs"]:
            for o in s["outputs"]:
                st.write(f"- `{o}`")
        else:
            st.write("- (none / overwrite / shared parquet)")
    else:
        st.info("Selecciona un step para ver detalles.")

    st.subheader("📦 Conteo")
    st.write(f"Steps visibles: **{len(steps)}**")
    st.write("Por layer:")
    for ly in layers_sel:
        st.write(f"- {ly}: {sum(1 for x in steps if x['layer'] == ly)}")

with col1:
    st.subheader("🧠 Grafo")
    html = make_pyvis_html(steps, selected_name=selected_name, focus_k=k)
    st.components.v1.html(html, height=740, scrolling=True)

st.divider()
st.caption("Si quieres, también puedo generar un modo 'dataflow' (nodo=parquet) o exportar a SVG/PDF.")
