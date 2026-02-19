-- 1) Esquema (si no existe)
--CREATE SCHEMA IF NOT EXISTS silver;

-- 2) Tabla (drop + create para asegurar que quede igual al parquet)
--DROP TABLE IF EXISTS silver.fact_ciclo_maestro;

--CREATE TABLE silver.fact_ciclo_maestro AS
--SELECT *
--FROM read_parquet('C:\Data-LakeHouse\data\silver\fact_ciclo_maestro.parquet');
