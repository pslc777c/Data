
select fecha_post_pred, variedad_canon, sum(nullif(cajas_postcosecha_ml1,0)) as suma_cajas 
from gold.pred_poscosecha_ml1_full_grado_dia_bloque_destino
group by fecha_post_pred, variedad_canon
order by fecha_post_pred desc;

select fecha_post_pred, sum(nullif(cajas_postcosecha_ml1,0)) as suma_cajas 
from gold.pred_poscosecha_ml1_full_grado_dia_bloque_destino
group by fecha_post_pred
order by fecha_post_pred desc;

drop table if exists gold.pred_poscosecha_ml1_dia_bloque_destino;
drop table if exists gold.pred_poscosecha_ml1_dia_total;
drop table if exists silver.fact_ciclo_maestro;
drop schema if EXISTS silver;



SELECT
  COALESCE(p.semana, a.semana) AS semana,
  
  (p.suma_cajas_finca - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_finca_cuartofrio,
  (p.suma_cajas_iso   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_iso_cuartofrio,  
  (a.presupuesto   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_presupuesto_cuartofrio,
  p.suma_cajas_iso,
  p.suma_cajas_finca,
  p.diff_iso_menos_finca,
  a.venta,
  a.produccion,
  p.suma_cajas_finca - a.produccion AS diff_finca_menos_produccion,
  p.suma_cajas_iso   - a.produccion AS diff_iso_menos_produccion,
  p.suma_cajas_finca - a.presupuesto AS diff_finca_menos_presupuesto,
  p.suma_cajas_iso   - a.presupuesto AS diff_iso_menos_presupuesto

FROM (
  -- PROYECCION = ISO vs FINCA
  SELECT
    COALESCE(i.semana, f.semana) AS semana,
    i.suma_cajas_iso,
    f.suma_cajas_finca,
    COALESCE(i.suma_cajas_iso, 0) - COALESCE(f.suma_cajas_finca, 0) AS diff_iso_menos_finca
  FROM (
    -- ISO (LUN-DOM)
    SELECT
      right(strftime(CAST(fecha_post_pred AS DATE), '%G'), 2)
        || strftime(CAST(fecha_post_pred AS DATE), '%V') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml1, 0)) AS suma_cajas_iso
    FROM gold.pred_poscosecha_ml1_full_grado_dia_bloque_destino
    GROUP BY 1
  ) i
  FULL OUTER JOIN (
    -- FINCA (DOM-SAB)
    SELECT
      right(strftime(week_start_sun, '%Y'), 2)
        || lpad(strftime(week_start_sun, '%U'), 2, '0') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml1, 0)) AS suma_cajas_finca
    FROM (
      SELECT
        (CAST(fecha_post_pred AS DATE)
          - CAST(EXTRACT(dow FROM CAST(fecha_post_pred AS DATE)) AS INTEGER)
        ) AS week_start_sun,
        cajas_postcosecha_ml1
      FROM gold.pred_poscosecha_ml1_full_grado_dia_bloque_destino
    ) t
    GROUP BY 1
  ) f
    ON i.semana = f.semana
) p
FULL OUTER JOIN (
  -- PROGRAMA
  SELECT
    lpad(CAST(semana AS VARCHAR), 4, '0') AS semana,
    venta,
    produccion,
    presupuesto,
    cajas_cuartofrio
  FROM gold.programa_produccion
) a
  ON p.semana = a.semana
WHERE CAST(COALESCE(p.semana, a.semana) AS INTEGER) BETWEEN 2501 AND 2621
ORDER BY 1 asc
;


select * from gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino limit 20;
describe gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino;

SELECT
  COALESCE(p.semana, a.semana) AS semana,
  
  (p.suma_cajas_finca - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_finca_cuartofrio,
  (p.suma_cajas_iso   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_iso_cuartofrio,  
  (a.presupuesto   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_presupuesto_cuartofrio,
  p.suma_cajas_iso,
  p.suma_cajas_finca,
  p.diff_iso_menos_finca,
  a.venta,
  a.produccion,
  p.suma_cajas_finca - a.produccion AS diff_finca_menos_produccion,
  p.suma_cajas_iso   - a.produccion AS diff_iso_menos_produccion,
  p.suma_cajas_finca - a.presupuesto AS diff_finca_menos_presupuesto,
  p.suma_cajas_iso   - a.presupuesto AS diff_iso_menos_presupuesto

FROM (
  -- PROYECCION = ISO vs FINCA
  SELECT
    COALESCE(i.semana, f.semana) AS semana,
    i.suma_cajas_iso,
    f.suma_cajas_finca,
    COALESCE(i.suma_cajas_iso, 0) - COALESCE(f.suma_cajas_finca, 0) AS diff_iso_menos_finca
  FROM (
    -- ISO (LUN-DOM)
    SELECT
      right(strftime(CAST(fecha_post_pred_final AS DATE), '%G'), 2)
        || strftime(CAST(fecha_post_pred_final AS DATE), '%V') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml2_final, 0)) AS suma_cajas_iso
    FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    GROUP BY 1
  ) i
  FULL OUTER JOIN (
    -- FINCA (DOM-SAB)
    SELECT
      right(strftime(week_start_sun, '%Y'), 2)
        || lpad(strftime(week_start_sun, '%U'), 2, '0') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml2_final, 0)) AS suma_cajas_finca
    FROM (
      SELECT
        (CAST(fecha_post_pred_ml1 AS DATE)
          - CAST(EXTRACT(dow FROM CAST(fecha_post_pred_final AS DATE)) AS INTEGER)
        ) AS week_start_sun,
        cajas_postcosecha_ml2_final
      FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    ) t
    GROUP BY 1
  ) f
    ON i.semana = f.semana
) p
FULL OUTER JOIN (
  -- PROGRAMA
  SELECT
    lpad(CAST(semana AS VARCHAR), 4, '0') AS semana,
    venta,
    produccion,
    presupuesto,
    cajas_cuartofrio
  FROM gold.programa_produccion
) a
  ON p.semana = a.semana
WHERE CAST(COALESCE(p.semana, a.semana) AS INTEGER) BETWEEN 2501 AND 2604

ORDER BY 1 asc
;


SELECT
  COALESCE(p.semana, a.semana) AS semana,

  (p.suma_cajas_finca - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_finca_cuartofrio,
  (p.suma_cajas_iso   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_iso_cuartofrio,
  (a.presupuesto      - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_presupuesto_cuartofrio,

  p.suma_cajas_iso,
  p.suma_cajas_finca,
  p.diff_iso_menos_finca,

  a.produccion,

  p.suma_cajas_finca - a.produccion  AS diff_finca_menos_produccion,
  p.suma_cajas_iso   - a.produccion  AS diff_iso_menos_produccion,
  p.suma_cajas_finca - a.presupuesto AS diff_finca_menos_presupuesto,
  p.suma_cajas_iso   - a.presupuesto AS diff_iso_menos_presupuesto

FROM (
  -- PROYECCION = ISO vs FINCA
  SELECT
    COALESCE(i.semana, f.semana) AS semana,
    i.suma_cajas_iso,
    f.suma_cajas_finca,
    COALESCE(i.suma_cajas_iso, 0) - COALESCE(f.suma_cajas_finca, 0) AS diff_iso_menos_finca
  FROM (
    -- ISO (LUN-DOM)
    SELECT
      right(strftime(CAST(fecha_post_pred_final AS DATE), '%G'), 2)
        || strftime(CAST(fecha_post_pred_final AS DATE), '%V') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml2_final, 0)) AS suma_cajas_iso
    FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    GROUP BY 1
  ) i
  FULL OUTER JOIN (
    -- FINCA (DOM-SAB)
    SELECT
      right(strftime(week_start_sun, '%Y'), 2)
        || lpad(strftime(week_start_sun, '%U'), 2, '0') AS semana,
      SUM(NULLIF(cajas_postcosecha_ml2_final, 0)) AS suma_cajas_finca
    FROM (
      SELECT
        (CAST(fecha_post_pred_ml1 AS DATE)
          - CAST(EXTRACT(dow FROM CAST(fecha_post_pred_final AS DATE)) AS INTEGER)
        ) AS week_start_sun,
        cajas_postcosecha_ml2_final
      FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    ) t
    GROUP BY 1
  ) f
    ON i.semana = f.semana
) p
FULL OUTER JOIN (
  -- PROGRAMA (AHORA SUMADO POR SEMANA)
  SELECT
    lpad(CAST(semana AS VARCHAR), 4, '0') AS semana,
    SUM(produccion)    AS produccion,
    SUM(presupuesto)   AS presupuesto,
    SUM(cajas_cuartofrio) AS cajas_cuartofrio
  FROM gold.programa_produccion
  GROUP BY 1
) a
  ON p.semana = a.semana

WHERE CAST(COALESCE(p.semana, a.semana) AS INTEGER) BETWEEN 2501 AND 2630
ORDER BY 1 ASC
;



SELECT
  COALESCE(p.semana, a.semana) AS semana,
  COALESCE(p.variedad_canon, a.variedad) AS variedad,

  (p.suma_cajas_finca - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_finca_cuartofrio,
  (p.suma_cajas_iso   - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_iso_cuartofrio,
  (a.presupuesto      - a.cajas_cuartofrio)/NULLIF(a.cajas_cuartofrio, 0.001)*100.0 AS kpi_presupuesto_cuartofrio,

  p.suma_cajas_iso,
  p.suma_cajas_finca,
  p.diff_iso_menos_finca,

  a.produccion,

  p.suma_cajas_finca - a.produccion  AS diff_finca_menos_produccion,
  p.suma_cajas_iso   - a.produccion  AS diff_iso_menos_produccion,
  p.suma_cajas_finca - a.presupuesto AS diff_finca_menos_presupuesto,
  p.suma_cajas_iso   - a.presupuesto AS diff_iso_menos_presupuesto

FROM (
  -- ==========================================================
  -- PROYECCION = ISO vs FINCA (por semana + variedad_canon)
  -- ==========================================================
  SELECT
    COALESCE(i.semana, f.semana) AS semana,
    COALESCE(i.variedad_canon, f.variedad_canon) AS variedad_canon,
    COALESCE(i.var_key, f.var_key) AS var_key,

    i.suma_cajas_iso,
    f.suma_cajas_finca,
    COALESCE(i.suma_cajas_iso, 0) - COALESCE(f.suma_cajas_finca, 0) AS diff_iso_menos_finca
  FROM (
    -- ISO (LUN-DOM)
    SELECT
      right(strftime(CAST(fecha_post_pred_final AS DATE), '%G'), 2)
        || strftime(CAST(fecha_post_pred_final AS DATE), '%V') AS semana,
      variedad_canon,
      lower(trim(variedad_canon)) AS var_key,
      SUM(NULLIF(cajas_postcosecha_ml1, 0)) AS suma_cajas_iso
    FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    GROUP BY 1, 2, 3
  ) i
  FULL OUTER JOIN (
    -- FINCA (DOM-SAB)
    SELECT
      right(strftime(week_start_sun, '%Y'), 2)
        || lpad(strftime(week_start_sun, '%U'), 2, '0') AS semana,
      variedad_canon,
      lower(trim(variedad_canon)) AS var_key,
      SUM(NULLIF(cajas_postcosecha_ml1, 0)) AS suma_cajas_finca
    FROM (
      SELECT
        (CAST(fecha_post_pred_ml1 AS DATE)
          - CAST(EXTRACT(dow FROM CAST(fecha_post_pred_final AS DATE)) AS INTEGER)
        ) AS week_start_sun,
        variedad_canon,
        cajas_postcosecha_ml1
      FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
    ) t
    GROUP BY 1, 2, 3
  ) f
    ON i.semana = f.semana
   AND i.var_key = f.var_key
) p
FULL OUTER JOIN (
  -- ==========================================================
  -- PROGRAMA sumado por semana + variedad
  -- ==========================================================
  SELECT
    lpad(CAST(semana AS VARCHAR), 4, '0') AS semana,
    variedad,
    lower(trim(variedad)) AS var_key,
    SUM(produccion)       AS produccion,
    SUM(presupuesto)      AS presupuesto,
    SUM(cajas_cuartofrio) AS cajas_cuartofrio
  FROM gold.programa_produccion
  GROUP BY 1, 2, 3
) a
  ON p.semana = a.semana
 AND p.var_key = a.var_key

WHERE CAST(COALESCE(p.semana, a.semana) AS INTEGER) BETWEEN 2415 AND 2605
ORDER BY 1 ASC, 2 ASC
;


describe FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino;

select
      right(strftime(CAST(fecha_post_pred_final AS DATE), '%G'), 2)
        || strftime(CAST(fecha_post_pred_final AS DATE), '%V') AS semana_posco,
      ROUND(SUM(NULLIF(cajas_postcosecha_ml1, 0)),0) AS suma_cajas_iso_ml1,
			ROUND(SUM(NULLIF(cajas_postcosecha_ml2_final, 0))/0.5,0) AS suma_cajas_iso_ml2
       FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
       group by 1 order by semana_posco desc;

describe gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino;

select
      right(strftime(CAST(fecha AS DATE), '%G'), 2)
        || strftime(CAST(fecha AS DATE), '%V') AS semana_lote,
      ROUND(SUM(NULLIF(cajas_postcosecha_ml1, 0)),0) AS suma_cajas_iso_ml1,
			ROUND(SUM(NULLIF(cajas_postcosecha_ml2_final, 0))/0.5,0) AS suma_cajas_iso_ml2
       FROM gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino
       group by 1 order by semana_lote desc;