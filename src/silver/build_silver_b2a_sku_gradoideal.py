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
SILVER = DATA / "silver" / "balanzas"

IN_PATH = BRONZE / "balanza_2a_raw.parquet"
OUT_PATH = SILVER / "silver_b2a_sku_gradoideal_dia_destino_variedad.parquet"

DESTINOS_OK = {"ARCOIRIS", "BLANCO", "GUIRNALDA", "TINTURADO"}


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _map_actividad_to_destino(s: pd.Series) -> pd.Series:
    s = _canon_str(s)
    repl = {
        "05CBMB": "BLANCO",
        "CBMC": "BLANCO",
        "CBM": "BLANCO",
        "CBX": "BLANCO",
        "CXLTA1": "TINTURADO",
        "05CTS": "TINTURADO",
        "0504GUFDGU": "GUIRNALDA",
        "CXLTARH": "ARCOIRIS",
    }
    return s.replace(repl)


def main():
    SILVER.mkdir(parents=True, exist_ok=True)

    df = read_parquet(IN_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Base
    df["Fecha"] = _to_date(df.get("Fecha"))
    df["codigo_actividad_raw"] = _canon_str(df.get("codigo_actividad"))

    df["Grado"] = _canon_str(df.get("Grado"))
    df["Origen"] = _canon_str(df.get("Origen"))
    df["Seccion"] = _canon_str(df.get("Seccion"))
    df["Variedad"] = _canon_str(df.get("Variedad"))

    df["PESOKG"] = pd.to_numeric(df.get("peso_balanza"), errors="coerce") / 1000.0
    df["tallos"] = pd.to_numeric(df.get("tallos"), errors="coerce")
    df["num_bunches"] = pd.to_numeric(df.get("num_bunches"), errors="coerce")

    # Filtros (equivalente SQL)
    df = df.loc[
        df["Fecha"].notna()
        & (df["Origen"] != "GV PELADO")
        & (df["Seccion"] == "CLASIFICACION")
        & (df["codigo_actividad_raw"] != "PSMC")
    ].copy()

    # Destino (CASE del SQL)
    df["Destino"] = _map_actividad_to_destino(df["codigo_actividad_raw"])

    # Row-level
    df["TALLOSTOTALES"] = df["tallos"] / df["num_bunches"].replace({0: np.nan})

    grado = df["Grado"].fillna("")
    cond_sku = grado.str.endswith("GR") & (~grado.str.startswith("BQT")) & (~grado.str.startswith("PET"))
    df["SKU"] = np.where(
        cond_sku,
        pd.to_numeric(grado.str.replace("GR", "", regex=False), errors="coerce"),
        np.nan,
    )

    df["Grado_Ideal"] = df["SKU"] / df["TALLOSTOTALES"].where(df["TALLOSTOTALES"] != 0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # WHERE Destino IN (...)
    df = df.loc[df["Destino"].isin(DESTINOS_OK)].copy()

    # AGG final (Destino en vez de codigo_actividad)
    out = (
        df.groupby(["Fecha", "Variedad", "Destino", "SKU", "Grado_Ideal"], as_index=False)
        .agg(
            Peso_Balanza_2A=("PESOKG", "sum"),
            Tallos_Totales=("TALLOSTOTALES", "sum"),
        )
        .sort_values(["Fecha", "Variedad", "Destino", "SKU", "Grado_Ideal"], ascending=[False, True, True, True, True])
        .reset_index(drop=True)
    )

    write_parquet(out, OUT_PATH)
    print(f"[OK] Silver B2A (SQL-equivalente) guardado en {OUT_PATH}")


if __name__ == "__main__":
    main()
