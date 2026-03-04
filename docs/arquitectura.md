# Arquitectura Del Pipeline (Desde Registry)

Documento generado automaticamente desde `src/ops/registry.py`.

- Total de steps: **139**
- Ultima generacion: **2026-02-25 23:37:55 UTC**

## Flujo End-To-End

![Flujo End-To-End](assets/arquitectura_01_overview.svg)

## Fabrica ML2 (Cadena De Modelado)

![Fabrica ML2](assets/arquitectura_02_ml2_factory.svg)

## Cadenas Silver -> Gold (Proceso De Negocio + Modelos)

![Cadenas Silver Gold](assets/arquitectura_03_silver_gold.svg)

> Nota: estos diagramas estan en SVG estatico para evitar errores del preview con Mermaid.

## Step-By-Step Del Registry

### Bronze

1. `balanza_cosecha_raw` -> `src/bronze/build_balanza_cosecha_raw.py`
2. `balanza_1c_raw` -> `src/bronze/build_balanza_1c_raw.py`
3. `balanza_mermas_sources` -> `src/bronze/build_balanza_mermas_sources.py`
4. `fenograma_sources` -> `src/bronze/build_fenograma_sources.py`
5. `ghu_maestro_horas` -> `src/bronze/build_ghu_maestro_horas.py`
6. `personal_sources` -> `src/bronze/build_personal_sources.py`
7. `ventas_sources` -> `src/bronze/build_ventas_sources.py`
8. `weather_hour_main` -> `src/bronze/build_weather_hour_main.py`
9. `weather_hour_a4` -> `src/bronze/build_weather_hour_a4.py`

### Silver

1. `weather_hour_estado` -> `src/silver/build_weather_hour_estado.py`
2. `weather_hour_wide` -> `src/silver/build_weather_hour_wide.py`
3. `ciclo_maestro_from_sources` -> `src/silver/build_ciclo_maestro_from_sources.py`
4. `hidratacion_real_from_balanza2` -> `src/silver/build_hidratacion_real_from_balanza2.py`
5. `fact_cosecha_real_grado_dia` -> `src/silver/build_fact_cosecha_real_grado_dia.py`
6. `fact_peso_tallo_real_grado_dia` -> `src/silver/build_fact_peso_tallo_real_grado_dia.py`
7. `fact_cosecha_uph_hora_clima` -> `src/silver/build_fact_cosecha_uph_hora_clima.py`
8. `dim_dist_grado_baseline` -> `src/silver/build_dim_dist_grado_baseline.py`
9. `dim_peso_tallo_baseline` -> `src/silver/build_dim_peso_tallo_baseline.py`
10. `dim_peso_tallo_promedio_dia` -> `src/silver/build_dim_peso_tallo_promedio_dia.py`
11. `dim_factor_uph_cosecha_clima` -> `src/silver/build_dim_factor_uph_cosecha_clima.py`
12. `dim_mermas_ajuste_fecha_post` -> `src/silver/build_dim_mermas_ajuste_fecha_post.py`
13. `dim_dh_baseline_grado_destino` -> `src/silver/build_dim_dh_baseline_grado_destino.py`
14. `dim_hidratacion_fecha_cosecha_grado_destino` -> `src/silver/build_dim_hidratacion_fecha_cosecha_grado_destino.py`
15. `dim_hidratacion_baseline_grado_destino` -> `src/silver/build_dim_hidratacion_baseline_grado_destino.py`
16. `dim_variedad_canon` -> `src/silver/build_dim_variedad_canon.py`
17. `dim_clima_bloque_dia` -> `src/silver/build_dim_clima_bloque_dia.py`
18. `dim_estado_termico_cultivo_bloque_fecha` -> `src/silver/build_dim_estado_termico_cultivo_bloque_fecha.py`
19. `dim_mix_proceso_semana_from_ventas` -> `src/silver/build_dim_mix_proceso_semana_from_ventas.py`
20. `fact_capacidad_proceso_hist` -> `src/silver/build_fact_capacidad_proceso_hist.py`
21. `dim_baseline_capacidad_tallos_h_persona` -> `src/silver/build_dim_baseline_capacidad_tallos_h_persona.py`
22. `dim_capacidad_baseline_tallos_proceso` -> `src/silver/build_dim_capacidad_baseline_tallos_proceso.py`
23. `milestones_windows` -> `src/silver/build_milestones_windows.py`

### Preds

1. `pred_milestones_baseline` -> `src/preds/build_pred_milestones_baseline.py`
2. `milestones_final` -> `src/silver/build_milestones_final.py`
3. `dim_mediana_etapas_tipo_sp_variedad_area` -> `src/silver/build_dim_mediana_etapas_tipo_sp_variedad_area.py`
4. `milestone_window_ciclo_final_with_inference` -> `src/silver/build_milestone_window_ciclo_final_with_inference.py`
5. `dim_cosecha_progress_bloque_fecha` -> `src/silver/build_dim_cosecha_progress_bloque_fecha.py`
6. `pred_oferta_dia` -> `src/preds/build_pred_oferta_dia.py`
7. `pred_oferta_grado` -> `src/preds/build_pred_oferta_grado.py`
8. `pred_peso_grado` -> `src/preds/build_pred_peso_grado.py`
9. `pred_peso_hidratado_grado` -> `src/preds/build_pred_peso_hidratado_grado.py`
10. `pred_peso_final_ajustado_grado` -> `src/preds/build_pred_peso_final_ajustado_grado.py`
11. `pred_cajas_from_peso_final` -> `src/preds/build_pred_cajas_from_peso_final.py`
12. `pred_tallos_cosecha_dia` -> `src/preds/build_pred_tallos_cosecha_dia.py`
13. `capacidad_cosecha_dia` -> `src/preds/build_capacidad_cosecha_dia.py`
14. `pred_horas_poscosecha` -> `src/preds/build_pred_horas_poscosecha.py`
15. `pred_plan_horas_dia` -> `src/preds/build_pred_plan_horas_dia.py`
16. `patch_ciclo_maestro_from_pred_tallos` -> `src/silver/build_patch_ciclo_maestro_from_pred_tallos.py`

### Features

1. `features_ciclo_fecha` -> `src/features/build_features_ciclo_fecha.py`

### Models ML1

1. `features_harvest_window_ml1` -> `src/features/build_features_harvest_window_ml1.py`
2. `apply_harvest_window_ml1` -> `src/models/ml1/apply_harvest_window_ml1.py`
3. `universe_harvest_grid_ml1` -> `src/gold/build_universe_harvest_grid_ml1.py`
4. `features_curva_cosecha_bloque_dia` -> `src/features/build_features_curva_cosecha_bloque_dia.py`
5. `cap_tallos_real_dia` -> `src/features/build_cap_tallos_real_dia.py`
6. `targets_curva_beta_params` -> `src/features/build_targets_curva_beta_params.py`
7. `targets_curva_beta_multiplier_dia` -> `src/features/build_targets_curva_beta_multiplier_dia.py`
8. `features_cosecha_bloque_fecha` -> `src/features/build_features_cosecha_bloque_fecha.py`
9. `apply_dist_grado_ml1` -> `src/models/ml1/apply_dist_grado.py`
10. `features_peso_tallo_grado_bloque_dia` -> `src/features/build_features_peso_tallo_grado_bloque_dia.py`
11. `apply_peso_tallo_grado_ml1` -> `src/models/ml1/apply_peso_tallo_grado.py`
12. `apply_curva_beta_params` -> `src/models/ml1/apply_curva_beta_params.py`

### Gold ML1

1. `pred_tallos_grado_dia_ml1` -> `src/gold/build_pred_tallos_grado_dia_ml1.py`
2. `pred_tallos_grado_dia_ml1_full` -> `src/gold/build_pred_tallos_grado_dia_ml1_full.py`
3. `pred_kg_cajas_grado_dia_ml1_full` -> `src/gold/build_pred_kg_cajas_grado_dia_ml1_full.py`
4. `pred_cajas_postcosecha_seed_mix_grado_dia` -> `src/gold/build_pred_cajas_postcosecha_seed_mix_grado_dia.py`
5. `build_gold_features_mix_pred_dia_destino_base` -> `src/gold/build_gold_features_mix_pred_dia_destino_base.py`
6. `build_gold_features_b2a_mix_pred_dia_destino` -> `src/gold/build_gold_features_b2a_mix_pred_dia_destino.py`
7. `patch_gold_features_mix_pred_dia_destino` -> `src/gold/build_gold_features_mix_pred_dia_destino.py`
8. `apply_dh_poscosecha_ml1` -> `src/models/ml1/apply_dh_poscosecha_ml1.py`
9. `apply_hidr_poscosecha_ml1` -> `src/models/ml1/apply_hidr_poscosecha_ml1.py`
10. `apply_desp_ajuste_poscosecha_ml1` -> `src/models/ml1/apply_desp_ajuste_poscosecha_ml1.py`
11. `pred_poscosecha_ml1_views` -> `src/gold/build_pred_poscosecha_ml1_views.py`
12. `view_planificacion_campo_tallos` -> `src/gold/build_view_planificacion_campo_tallos.py`
13. `postprocess_curva_share_smooth_ml1` -> `src/gold/postprocess_curva_share_smooth_ml1.py`

### ML2

1. `build_ds_harvest_start_ml2_v2` -> `src/gold/build_ds_harvest_start_ml2_v2.py`
2. `train_harvest_start_ml2` -> `src/models/ml2/train_harvest_start_ml2.py`
3. `apply_harvest_start_ml2` -> `src/models/ml2/apply_harvest_start_ml2.py`
4. `build_ds_harvest_horizon_ml2_v2` -> `src/gold/build_ds_harvest_horizon_ml2_v2.py`
5. `train_harvest_horizon_ml2` -> `src/models/ml2/train_harvest_horizon_ml2.py`
6. `apply_harvest_horizon_ml2` -> `src/models/ml2/apply_harvest_horizon_ml2.py`
7. `universe_harvest_grid_ml2` -> `src/gold/build_universe_harvest_grid_ml2.py`
8. `build_ds_tallos_curve_ml2_v2` -> `src/gold/build_ds_tallos_curve_ml2_v2.py`
9. `train_tallos_curve_ml2` -> `src/models/ml2/train_tallos_curve_ml2.py`
10. `apply_tallos_curve_ml2` -> `src/models/ml2/apply_tallos_curve_ml2.py`
11. `build_ds_share_grado_ml2_v1` -> `src/gold/build_ds_share_grado_ml2_v1.py`
12. `train_share_grado_ml2` -> `src/models/ml2/train_share_grado_ml2.py`
13. `apply_share_grado_ml2` -> `src/models/ml2/apply_share_grado_ml2.py`
14. `build_ds_peso_tallo_ml2_v1` -> `src/gold/build_ds_peso_tallo_ml2_v1.py`
15. `train_peso_tallo_ml2` -> `src/models/ml2/train_peso_tallo_ml2.py`
16. `apply_peso_tallo_ml2` -> `src/models/ml2/apply_peso_tallo_ml2.py`
17. `build_pred_tallos_ml2_full` -> `src/gold/build_pred_tallos_ml2_full.py`
18. `build_pred_kg_cajas_ml2` -> `src/gold/build_pred_kg_cajas_ml2.py`
19. `build_pred_poscosecha_ml2_seed_mix_grado_dia` -> `src/gold/build_pred_poscosecha_ml2_seed_mix_grado_dia.py`
20. `apply_dh_poscosecha_ml1_on_ml2_seed` -> `src/gold/apply_dh_poscosecha_ml1_on_ml2_seed.py`
21. `apply_hidr_poscosecha_ml1_on_ml2_dh` -> `src/gold/apply_hidr_poscosecha_ml1_on_ml2_dh.py`
22. `apply_desp_ajuste_poscosecha_ml1_on_ml2_hidr` -> `src/gold/apply_desp_ajuste_poscosecha_ml1_on_ml2_hidr.py`
23. `build_ds_dh_poscosecha_ml2_v1` -> `src/gold/build_ds_dh_poscosecha_ml2_v1.py`
24. `train_dh_poscosecha_ml2` -> `src/models/ml2/train_dh_poscosecha_ml2.py`
25. `apply_dh_poscosecha_ml2` -> `src/models/ml2/apply_dh_poscosecha_ml2.py`
26. `build_ds_hidr_poscosecha_ml2_v1` -> `src/gold/build_ds_hidr_poscosecha_ml2_v1.py`
27. `train_hidr_poscosecha_ml2` -> `src/models/ml2/train_hidr_poscosecha_ml2.py`
28. `apply_hidr_poscosecha_ml2` -> `src/models/ml2/apply_hidr_poscosecha_ml2.py`
29. `build_ds_desp_poscosecha_ml2_v1` -> `src/gold/build_ds_desp_poscosecha_ml2_v1.py`
30. `train_desp_poscosecha_ml2` -> `src/models/ml2/train_desp_poscosecha_ml2.py`
31. `apply_desp_poscosecha_ml2` -> `src/models/ml2/apply_desp_poscosecha_ml2.py`
32. `build_ds_ajuste_poscosecha_ml2_v1` -> `src/gold/build_ds_ajuste_poscosecha_ml2_v1.py`
33. `train_ajuste_poscosecha_ml2` -> `src/models/ml2/train_ajuste_poscosecha_ml2.py`
34. `apply_ajuste_poscosecha_ml2` -> `src/models/ml2/apply_ajuste_poscosecha_ml2.py`
35. `build_pred_poscosecha_ml2_views_from_full` -> `src/gold/build_pred_poscosecha_ml2_views_from_full.py`
36. `build_pred_poscosecha_ml2_final_views` -> `src/gold/build_pred_poscosecha_ml2_final_views.py`

### Eval

1. `kpis_ajuste_poscosecha_ml2` -> `src/eval/build_kpis_ajuste_poscosecha_ml2.py`
2. `kpis_desp_poscosecha_ml2` -> `src/eval/build_kpis_desp_poscosecha_ml2.py`
3. `kpis_dh_poscosecha_ml2` -> `src/eval/build_kpis_dh_poscosecha_ml2.py`
4. `kpis_harvest_horizon_ml2` -> `src/eval/build_kpis_harvest_horizon_ml2.py`
5. `kpis_harvest_start_ml2` -> `src/eval/build_kpis_harvest_start_ml2.py`
6. `kpis_hidr_poscosecha_ml2` -> `src/eval/build_kpis_hidr_poscosecha_ml2.py`
7. `kpis_peso_tallo_ml2` -> `src/eval/build_kpis_peso_tallo_ml2.py`
8. `kpis_poscosecha_ml2_baseline` -> `src/eval/build_kpis_poscosecha_ml2_baseline.py`
9. `kpis_share_grado_ml2` -> `src/eval/build_kpis_share_grado_ml2.py`
10. `kpis_tallos_curve_ml2` -> `src/eval/build_kpis_tallos_curve_ml2.py`

### Audit

1. `audit_universe_harvest_grid_ml1` -> `src/audit/audit_universe_harvest_grid_ml1.py`
2. `audit_dist_grado_ml1` -> `src/audit/audit_dist_grado_ml1.py`
3. `audit_poscosecha_seed` -> `src/audit/audit_poscosecha_seed.py`
4. `audit_ml1_curva_clipping_impact` -> `src/audit/audit_ml1_curva_clipping_impact.py`
5. `audit_ml1_curva_share_vs_real` -> `src/audit/audit_ml1_curva_share_vs_real.py`
6. `audit_ml1_curva_shares` -> `src/audit/audit_ml1_curva_shares.py`
7. `audit_harvest_windows_real` -> `src/audit/audit_harvest_windows_real.py`
8. `audit_mass_balance_pred_oferta_vs_gold` -> `src/audit/audit_mass_balance_pred_oferta_vs_gold.py`
9. `audit_mismatch_tallos_grado_vs_dia` -> `src/audit/audit_mismatch_tallos_grado_vs_dia.py`
10. `audit_ajuste_poscosecha_ml2` -> `src/audit/audit_ajuste_poscosecha_ml2.py`
11. `audit_desp_poscosecha_ml2` -> `src/audit/audit_desp_poscosecha_ml2.py`
12. `audit_dh_poscosecha_ml2` -> `src/audit/audit_dh_poscosecha_ml2.py`
13. `audit_hidr_poscosecha_ml2` -> `src/audit/audit_hidr_poscosecha_ml2.py`
14. `audit_harvest_start_ml2` -> `src/audit/audit_harvest_start_ml2.py`
15. `audit_harvest_horizon_ml2` -> `src/audit/audit_harvest_horizon_ml2.py`
16. `audit_peso_tallo_ml2` -> `src/audit/audit_peso_tallo_ml2.py`
17. `audit_share_grado_ml2` -> `src/audit/audit_share_grado_ml2.py`
18. `audit_tallos_curve_ml2` -> `src/audit/audit_tallos_curve_ml2.py`
19. `audit_poscosecha_ml2_baseline` -> `src/audit/audit_poscosecha_ml2_baseline.py`
