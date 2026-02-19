from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"

# Ajusta a tu ruta real de bronze
IN_PARQUET = BRONZE / "balanza_2a_raw.parquet"

OUT_PARQUET = SILVER / "silver_balanza2a_grado_ideal_dia_variedad_actividad.parquet"

# Filtros según SQL
EXCLUDE_ORIGEN = "GV PELADO"
FILTER_SECCION = "CLASIFICACION"
EXCLUDE_CODIGO_ACT = "PSMC"

KEEP_ACTIVIDADES = {"ARCOIRIS", "BLANCO", "GUIRNALDA", "TINTURADO"}


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _map_codigo_actividad(act: pd.Series) -> pd.Series:
    a = _canon_str(act)
    return a.replace(
        {
            "05CBMB": "BLANCO",
            "CBMC": "BLANCO",
            "CBM": "BLANCO",
            "CBX": "BLANCO",
            "CXLTA1": "TINTURADO",
            "05CTS": "TINTURADO",
            "0504GUFDGU": "GUIRNALDA",
            "CXLTARH": "ARCOIRIS",
        }
    )


def _grado_num(grado: pd.Series) -> pd.Series:
    """
    SQL:
      WHEN Grado LIKE '%GR' AND Grado NOT LIKE 'BQT%' AND Grado NOT LIKE 'PET%'
        THEN TRY_CAST(REPLACE(Grado, 'GR', '') AS float)
    """
    g = _canon_str(grado)
    is_gr = g.str.endswith("GR", na=False)
    is_bqt = g.str.startswith("BQT", na=False)
    is_pet = g.str.startswith("PET", na=False)

    m = is_gr & ~is_bqt & ~is_pet
    out = pd.Series(np.nan, index=g.index, dtype="float64")
    out.loc[m] = pd.to_numeric(g.loc[m].str.replace("GR", "", regex=False), errors="coerce")
    return out


def main() -> None:
    if not IN_PARQUET.exists():
        raise FileNotFoundError(f"No existe IN_PARQUET={IN_PARQUET} (ROOT={ROOT})")

    df = read_parquet(IN_PARQUET).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Esperado del raw (según tu SQL)
    need = {
        "Fecha",
        "Lote",
        "codigo_actividad",
        "Grado",
        "Origen",
        "Seccion",
        "Variedad",
        "peso_balanza",
        "tallos",
        "num_bunches",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Parquet raw sin columnas: {sorted(miss)}")

    # Canon / tipos
    df["Fecha"] = _to_date(df["Fecha"])
    df["Origen"] = _canon_str(df["Origen"])
    df["Seccion"] = _canon_str(df["Seccion"])
    df["Variedad"] = _canon_str(df["Variedad"])
    df["codigo_actividad"] = _map_codigo_actividad(df["codigo_actividad"])
    df["Grado"] = _canon_str(df["Grado"])

    df["PESOKG"] = _to_float(df["peso_balanza"]) / 1000.0
    df["tallos"] = _to_float(df["tallos"])
    df["num_bunches"] = _to_float(df["num_bunches"])

    # Filtros (igual SQL)
    df = df.loc[df["Fecha"].notna()].copy()
    df = df.loc[df["Origen"].ne(EXCLUDE_ORIGEN)].copy()
    df = df.loc[df["Seccion"].eq(FILTER_SECCION)].copy()
    df = df.loc[df["codigo_actividad"].ne(EXCLUDE_CODIGO_ACT)].copy()

    # Keep actividades finales
    df["codigo_actividad"] = _canon_str(df["codigo_actividad"])
    df = df.loc[df["codigo_actividad"].isin(KEEP_ACTIVIDADES)].copy()

    # ============================
    # ROW_LEVEL (como tu CTE)
    # ============================

    # TALLOSTOTALES = tallos / num_bunches (con NULLIF)
    denom = df["num_bunches"].replace({0.0: np.nan})
    df["TALLOSTOTALES"] = df["tallos"] / denom

    # Grado_num y Grado_Ideal
    df["Grado_num"] = _grado_num(df["Grado"])

    # Grado_Ideal = Grado_num / TALLOSTOTALES con guards
    denom2 = df["TALLOSTOTALES"].replace({0.0: np.nan})
    df["Grado_Ideal"] = df["Grado_num"] / denom2

    # ============================
    # AGREGACIÓN (tu segundo query)
    # ============================

    out = (
        df.groupby(["Fecha", "Variedad", "codigo_actividad", "Grado_Ideal"], as_index=False)
        .agg(
            Peso_Balanza_2A=("PESOKG", "sum"),
            Tallos_Totales=("TALLOSTOTALES", "sum"),
        )
        .sort_values(["Fecha", "Variedad", "codigo_actividad", "Grado_Ideal"], ascending=[False, True, True, True])
    )

    # Persist
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_PARQUET)

    print(f"[OK] silver B2A grado_ideal: {OUT_PARQUET} rows={len(out):,}")
    print("     cols:", list(out.columns))
    # quick sanity
    gi = pd.to_numeric(out["Grado_Ideal"], errors="coerce")
    if gi.notna().any():
        print(f"     Grado_Ideal p50={float(gi.quantile(0.50)):.4f}  p01={float(gi.quantile(0.01)):.4f}  p99={float(gi.quantile(0.99)):.4f}")


if __name__ == "__main__":
    main()
