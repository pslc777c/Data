import os
import shutil
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIG
# ============================================================
DB_PATH = r"C:\Users\paul.loja\PYPROYECTOS\Data-LakeHouse\Data-LakeHouse\repositorio\data\duckdb\lakehouse.duckdb"

# Si VS Code/DBCode te bloquea el .duckdb, usa una copia de solo-lectura
USE_READ_COPY_IF_LOCKED = True


# ============================================================
# SQL
# - FIX: ahora sí traemos a.cajas_cuartofrio al SELECT final
# ============================================================
SQL = r"""

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

WHERE CAST(COALESCE(p.semana, a.semana) AS INTEGER) BETWEEN 2505 AND 2549
ORDER BY 1 ASC
;

"""


# ============================================================
# HELPERS
# ============================================================
def connect_duckdb(path: str) -> duckdb.DuckDBPyConnection:
    """
    Intenta abrir el .duckdb en read_only.
    Si está bloqueado por otro proceso (VS Code), opcionalmente crea una copia *_read.duckdb.
    """
    try:
        return duckdb.connect(path, read_only=True)
    except duckdb.IOException as e:
        msg = str(e).lower()
        if USE_READ_COPY_IF_LOCKED and ("being utilized" in msg or "utilizado por otro proceso" in msg or "file is already open" in msg):
            base, ext = os.path.splitext(path)
            copy_path = f"{base}_read{ext}"
            print(f"[INFO] DB bloqueada. Creando copia para lectura: {copy_path}")
            shutil.copy2(path, copy_path)
            return duckdb.connect(copy_path, read_only=True)
        raise


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def plot_lines(df: pd.DataFrame, x: str, series: list[tuple[str, str]], title: str, ylab: str, add_zero_line: bool = False):
    plt.figure()
    for col, label in series:
        if col not in df.columns:
            print(f"[WARN] No existe columna '{col}' en df. Saltando esa serie.")
            continue
        plt.plot(df[x], df[col], label=label)

    if add_zero_line:
        plt.axhline(0)

    plt.xticks(rotation=45)
    plt.xlabel("Semana")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# RUN
# ============================================================
con = connect_duckdb(DB_PATH)
df = con.execute(SQL).df()

print("[INFO] Columnas devueltas:", list(df.columns))
print(df.head(3))

# Orden por semana numérica
df["semana"] = df["semana"].astype(str)
df["_semana_int"] = pd.to_numeric(df["semana"], errors="coerce")
df = df.sort_values("_semana_int").drop(columns=["_semana_int"])

# FULL OUTER JOIN => posibles NULLs: convertir numéricos
ensure_numeric(df, [
    "kpi_finca_cuartofrio", "kpi_iso_cuartofrio", "kpi_presupuesto_cuartofrio",
    "cajas_cuartofrio", "produccion", "venta", "presupuesto",
    "suma_cajas_finca", "suma_cajas_iso",
    "diff_iso_menos_finca", "diff_finca_menos_produccion", "diff_iso_menos_produccion",
    "diff_finca_menos_presupuesto", "diff_iso_menos_presupuesto"
])


# ============================================================
# PLOTS
# ============================================================
# 1) KPIs (% vs cuarto frio)
plot_lines(
    df,
    x="semana",
    series=[
        ("kpi_finca_cuartofrio", "FINCA vs Cuarto Frio (%)"),
        ("kpi_iso_cuartofrio", "ISO vs Cuarto Frio (%)"),
        ("kpi_presupuesto_cuartofrio", "Presupuesto vs Cuarto Frio (%)"),
    ],
    title="KPIs semanales vs Cuarto Frio",
    ylab="KPI (%)",
    add_zero_line=True
)

# 2) Niveles (cajas)
plot_lines(
    df,
    x="semana",
    series=[
        ("suma_cajas_finca", "Cajas FINCA (proy)"),
        ("suma_cajas_iso", "Cajas ISO (proy)"),
        ("produccion", "Producción (programa)"),
        ("cajas_cuartofrio", "Cuarto Frio (programa)"),
    ],
    title="Cajas semanales (proyección vs programa)",
    ylab="Cajas",
    add_zero_line=False
)

# 3) Diferencias (cajas)
plot_lines(
    df,
    x="semana",
    series=[
        ("diff_iso_menos_finca", "ISO - FINCA"),
        ("diff_finca_menos_produccion", "FINCA - Producción"),
        ("diff_iso_menos_produccion", "ISO - Producción"),
    ],
    title="Diferencias semanales (cajas)",
    ylab="Diferencia (cajas)",
    add_zero_line=True
)
kpi_cols = [
    "kpi_finca_cuartofrio",
    "kpi_iso_cuartofrio",
    "kpi_presupuesto_cuartofrio",
]

rows = []
for c in kpi_cols:
    if c not in df.columns:
        continue

    x = pd.to_numeric(df[c], errors="coerce")
    x = x.dropna()

    n = int(x.shape[0])
    if n == 0:
        rows.append({"kpi": c, "n_semanas": 0, "MAE_pp": np.nan, "RMSE_pp": np.nan})
        continue

    mae = float(np.mean(np.abs(x)))                 # vs 0
    rmse = float(np.sqrt(np.mean(np.square(x))))    # vs 0

    rows.append({
        "kpi": c,
        "n_semanas": n,
        "MAE_pp": mae,
        "RMSE_pp": rmse,
    })

kpi_err = pd.DataFrame(rows).sort_values("kpi")
print("\n[KPIs] Error (%) vs Cuarto Frío — MAE/RMSE (puntos porcentuales, target=0):")
print(kpi_err.to_string(index=False))