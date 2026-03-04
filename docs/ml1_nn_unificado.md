# ML1 NN Unificado (Multitarea)

Este flujo crea una sola base y una sola red neuronal para aprender de forma conjunta:

- `VEG`: inicio de cosecha (`d_start`) y horizonte (`n_harvest_days`)
- `HARVEST_DAY`: forma diaria de cosecha (`factor_tallos_dia`)
- `HARVEST_GRADE`: share por grado y factor de peso de tallo
- `POST`: `dh`, hidratacion, desperdicio y ajuste

## 1) Base Unica

Script:

`src/gold/build_ds_ml1_nn_v1.py`

Salida:

`data/gold/ml1_nn/ds_ml1_nn_v1.parquet`

La base queda en formato multitarea:

- Una fila = un evento de etapa (`stage`)
- Features comunes + features de etapa
- Targets en columnas `target_*`
- Mascaras `mask_target_*` para indicar en que filas existe cada target

Esto permite entrenar una sola red, aunque no todos los targets existan en todas las filas.

## 2) Entrenamiento de Red Neuronal Unica

Script:

`src/models/ml1/train_multitask_nn_ml1.py`

Modelo:

- MLP con trunk compartido (`hidden1`, `hidden2`)
- Salida multi-target (todas las variables objetivo)
- Loss MSE enmascarada por target (`mask_*`)
- Split temporal por `fecha_evento`
- Early stopping por `mae_val_avg`

Artefactos:

- Modelo: `data/models/ml1_nn/ml1_multitask_nn_<run_id>.npz`
- Metadata: `data/models/ml1_nn/ml1_multitask_nn_<run_id>_meta.json`
- Historial: `data/eval/ml1_nn/ml1_multitask_nn_train_history_<run_id>.parquet`
- Relevancia estadistica (p-values): `data/eval/ml1_nn/ml1_nn_feature_pvalues_<run_id>.parquet`
- Reporte PDF del run: `data/eval/ml1_nn/ml1_nn_run_report_<run_id>.pdf`

## 3) Aplicacion del Modelo

Script:

`src/models/ml1/apply_multitask_nn_ml1.py`

Salida:

`data/gold/ml1_nn/pred_ml1_multitask_nn_<run_id>.parquet`

Se agregan columnas `pred_*` para cada target.

Adicionalmente, para poscosecha se generan salidas finales de kg/cajas ajustadas:

- `data/gold/ml1_nn/pred_ml1_multitask_nn_post_final_<run_id>.parquet`
- `data/gold/ml1_nn/pred_ml1_multitask_nn_post_dia_destino_<run_id>.parquet`
- `data/gold/ml1_nn/pred_ml1_multitask_nn_post_dia_total_<run_id>.parquet`

## 4) Comandos

Desde raiz del repo:

```powershell
$env:PYTHONPATH="$PWD;$PWD/src"
python src/gold/build_ds_ml1_nn_v1.py
python src/models/ml1/train_multitask_nn_ml1.py
python src/models/ml1/apply_multitask_nn_ml1.py
python src/analisis/build_pdf_ml1_nn_run_report.py
```

Opciones utiles:

```powershell
python src/models/ml1/train_multitask_nn_ml1.py --epochs 60 --batch-size 4096 --hidden1 128 --hidden2 64
python src/models/ml1/apply_multitask_nn_ml1.py --run-id <run_id>
python src/analisis/build_pdf_ml1_nn_run_report.py --run-id <run_id>
```

## 5) Sobre p-values

La red neuronal no usa p-values para optimizar.  
Aqui se calculan como diagnostico estadistico (f_regression) por target, para validar si variables como `dia de cosecha` estan asociadas a `hidratacion`, `dh`, etc.

## 6) Benchmark Per-Target (Auto-Selección)

Script:

`src/models/ml1/benchmark_per_target_models_ml1.py`

Objetivo:

- Entrenar varios modelos por target en la misma corrida (`dummy_median`, `ridge`, `hgb`)
- Opcionalmente ampliar candidatos con `hgb_deep`, `gbr`, `etr`
- Seleccionar automaticamente el mejor por target (minimizando `val_mae`)
- Reportar metricas globales y por dominio (`field` vs `post`) del conjunto seleccionado
  (`weighted_mae_global`, `weighted_r2_global`, `r2_avg_targets`, `domain_metrics`)

Comando:

```powershell
$env:PYTHONPATH="$PWD;$PWD/src"
python src/models/ml1/benchmark_per_target_models_ml1.py --val-quantile 0.70 --min-val-n 40 --save-models
```

Con busqueda ampliada (mas costosa):

```powershell
python src/models/ml1/benchmark_per_target_models_ml1.py --val-quantile 0.70 --min-val-n 40 --include-extra --save-models
```

Salidas:

- `data/eval/ml1_nn/ml1_target_model_benchmark_<run_id>.parquet`
- `data/eval/ml1_nn/ml1_target_model_winners_<run_id>.parquet`
- `data/eval/ml1_nn/ml1_target_model_benchmark_summary_<run_id>.json`
- Modelos seleccionados: `data/models/ml1_nn_target_models/<run_id>/`

## 7) Apply Hibrido Por Target

Script:

`src/models/ml1/apply_target_model_hybrid_ml1.py`

Objetivo:

- Aplicar un modelo distinto por target (segun summary hibrido de benchmark)
- Mantener salida en formato ML1 con `pred_*`, alias `*_ML1` y vistas postcosecha

Comando:

```powershell
$env:PYTHONPATH="$PWD;$PWD/src"
python src/models/ml1/apply_target_model_hybrid_ml1.py --summary data/eval/ml1_nn/ml1_target_model_hybrid_summary_<field_run>__<post_run>.json
```

Salidas:

- `data/gold/ml1_nn/pred_ml1_target_hybrid_<apply_run_id>.parquet`
- `data/gold/ml1_nn/pred_ml1_target_hybrid_post_final_<apply_run_id>.parquet`
- `data/gold/ml1_nn/pred_ml1_target_hybrid_post_dia_destino_<apply_run_id>.parquet`
- `data/gold/ml1_nn/pred_ml1_target_hybrid_post_dia_total_<apply_run_id>.parquet`
- Metadata de apply: `data/eval/ml1_nn/ml1_target_hybrid_apply_meta_<apply_run_id>.json`
