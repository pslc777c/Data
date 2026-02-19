--1) Esquema (si no existe)
CREATE SCHEMA IF NOT EXISTS gold;

 --2) Tabla (drop + create para asegurar que quede igual al parquet)
DROP TABLE IF EXISTS gold.pred_poscosecha_ml1_dia_total;

CREATE or replace TABLE gold.pred_poscosecha_ml1_dia_total AS
SELECT *
FROM read_parquet('C:\\Data-LakeHouse\\data\\gold\\pred_poscosecha_ml1_dia_total.parquet');

create or replace table gold.pred_poscosecha_ml1_full_grado_dia_bloque_destino as 
select *
from read_parquet('C:\Data-LakeHouse\data\gold\pred_poscosecha_ml1_full_grado_dia_bloque_destino.parquet');

create or replace table gold.pred_poscosecha_ml2_final_full_grado_dia_bloque_destino as 
select *
from read_parquet('C:\Users\paul.loja\PYPROYECTOS\Data-LakeHouse\Data-LakeHouse\data\gold\pred_poscosecha_ml2_final_full_grado_dia_bloque_destino.parquet');


CREATE OR REPLACE TABLE gold.programa_produccion AS
SELECT *
FROM read_csv_auto(
  'C:\Users\paul.loja\PYPROYECTOS\Data-LakeHouse\Data-LakeHouse\programa_produccion.csv',
 header = true,
 delim = ';'
);