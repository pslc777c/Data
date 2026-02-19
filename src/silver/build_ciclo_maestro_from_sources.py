# src/silver/build_ciclo_maestro_from_sources.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet
from common.ids import make_bloque_id, make_variedad_id, make_ciclo_id
from common.timegrid import build_grid_ciclo_fecha


# -------------------------
# Helpers (tuyos, ajustados)
# -------------------------
def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def to_datetime_safe(x):
    return pd.to_datetime(x, errors="coerce")


def to_numeric_safe(x):
    return pd.to_numeric(x, errors="coerce")


def normalize_date_col(df: pd.DataFrame, col: str):
    df = df.copy()
    df[col] = to_datetime_safe(df[col]).dt.normalize()
    return df


def reemplazos_area(series: pd.Series) -> pd.Series:
    return series.astype(str).replace({"SSJ": "A-4", "MM1": "MH1", "MM2": "MH2"})


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# BRONZE RAW readers (indices: col_0..col_n)
# -------------------------
def _strip_cell(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()


def _ensure_cols_are_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def _rebuild_header_like_xl(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Replica tu lógica XL original (Excel):
      raw = raw[~raw[0].isna() & (raw[0].astype(str).str.strip() != "")].copy()
      raw = raw.iloc[1:].copy()
      raw.columns = raw.iloc[0]
      df = raw.iloc[1:].copy()

    En Bronze: raw viene como col_0..col_n (todo string).
    """
    raw = _ensure_cols_are_strings(raw)
    col0 = "col_0" if "col_0" in raw.columns else raw.columns[0]
    rr = raw.copy()

    mask = rr[col0].notna() & (rr[col0].astype(str).str.strip() != "")
    rr = rr[mask].copy()

    if len(rr) < 2:
        return pd.DataFrame()

    rr = rr.iloc[1:].copy()

    header = rr.iloc[0].tolist()
    header = [re.sub(r"\s+", " ", _strip_cell(h).replace("\n", " ").replace("\r", " ")).strip() for h in header]

    rr = rr.iloc[1:].copy()
    rr.columns = header
    rr = normalizar_columnas(rr)
    return rr


def _rebuild_header_like_clo(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Replica tu lógica CLO original (Excel):
      raw = raw[~raw[0].isna() & (raw[0].astype(str).str.strip() != "")].copy()
      raw.columns = raw.iloc[0]
      df = raw.iloc[1:].copy()

    En Bronze: raw viene como col_0..col_n (todo string).
    """
    raw = _ensure_cols_are_strings(raw)
    col0 = "col_0" if "col_0" in raw.columns else raw.columns[0]
    rr = raw.copy()

    mask = rr[col0].notna() & (rr[col0].astype(str).str.strip() != "")
    rr = rr[mask].copy()

    if len(rr) < 2:
        return pd.DataFrame()

    header = rr.iloc[0].tolist()
    header = [re.sub(r"\s+", " ", _strip_cell(h).replace("\n", " ").replace("\r", " ")).strip() for h in header]

    rr = rr.iloc[1:].copy()
    rr.columns = header
    rr = normalizar_columnas(rr)
    return rr


# -------------------------
# Fenograma activo (tu lógica)
# -------------------------
def fenograma_activo(df_fenograma_xlsm: pd.DataFrame, bal: pd.DataFrame, hoy: pd.Timestamp) -> pd.DataFrame:
    df = normalizar_columnas(df_fenograma_xlsm)

    if "Area " in df.columns and "Area" not in df.columns:
        df = df.rename(columns={"Area ": "Area"})

    df["Bloques"] = df["Bloques"].astype(str)

    valores_excluir = {
        None, 0, 500, 625, 750, 875, 1000, 6025,
        "%1000 GR", "%350 GR", "%500 GR", "%750 GR",
        "00 a 20", "21 a 40", "41 a 60", "61 a 80", "81 a 100",
        "ANDES", "Cajas por cosechar entre:", "CAMPO", "CLOUD", "CV",
        "Distribucion de cosecha en el campo", "DISTRIBUCION POR GRADOS",
        "DOMINGO", "FLOR GRANDE = FG", "FLOR PEQUEÑA = FP", "GENERAL",
        "JUEVES", "LUNES", "MARTES", "MH1", "MH2", "MIERCOLES",
        "MILLION CLOUD", "MM1", "MM2", "OTRO",
        "PROYECCION CAJAS POR GRADOS",
        "SABADO", "SI", "SJP", "SSJ", "Tallos/caja",
        "TOTAL", "Total", "Total cajas", "TOTAL GYPSOS",
        "Total tallos", "VENTAS", "VIERNES", "x",
        "XLENCE", "XLENCE FIVE STARS"
    }

    if "Fiesta" not in df.columns:
        raise ValueError("fenograma_xlsm_raw: no existe columna 'Fiesta' requerida por la lógica de filtrado.")

    df = df[~df["Fiesta"].isin(valores_excluir)].copy()
    df = df[df["Bloques"].notna()].copy()

    cols = ["Bloques", "Fecha S/P", "Area", "S/P", "Variedad", "Tallos / Bloque"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"fenograma_xlsm_raw: faltan columnas requeridas: {missing}")

    df = df[cols].copy()

    df["Fecha S/P"] = to_datetime_safe(df["Fecha S/P"])
    df["Tallos / Bloque"] = to_numeric_safe(df["Tallos / Bloque"]).fillna(0)

    agg = (
        df.groupby(["Bloques", "S/P", "Variedad", "Area"], dropna=False)
          .agg(**{
              "Fecha S/P": ("Fecha S/P", "max"),
              "Tallos_Proy": ("Tallos / Bloque", "sum")
          })
          .reset_index()
    )

    agg = agg[agg["Area"].notna()].copy()
    agg = agg[(agg["Area"] != 0) & (agg["Area"] != "0")].copy()
    agg["Area"] = reemplazos_area(agg["Area"])

    # Join con balanza para calcular días vegetativos
    tmp = agg.merge(bal, left_on="Bloques", right_on="Bloque", how="left").drop(columns=["Bloque"])
    tmp["Personalizado"] = (tmp["Fecha"] - tmp["Fecha S/P"]).dt.days

    tmp_f = tmp[(tmp["Personalizado"] > 30) | (tmp["Personalizado"].isna())].copy()

    g = (
        tmp_f.groupby(["Bloques", "S/P", "Fecha S/P"], dropna=False)
             .agg(
                 Dias_Vegetativo=("Personalizado", "min"),
                 Dias_Vegetativo_1=("Personalizado", "max")
             )
             .reset_index()
    )

    g["Fecha_Inicio_Cosecha (Primer Tallo)"] = g["Fecha S/P"] + pd.to_timedelta(g["Dias_Vegetativo"], unit="D")
    g.loc[g["Dias_Vegetativo"].isna(), "Fecha_Inicio_Cosecha (Primer Tallo)"] = pd.NaT

    fin_calc = g["Fecha S/P"] + pd.to_timedelta(g["Dias_Vegetativo_1"], unit="D")
    g["Fecha_Fin_Cosecha"] = fin_calc

    cond1 = g["Dias_Vegetativo"].isna()
    cond2 = g["Dias_Vegetativo_1"] <= g["Dias_Vegetativo"]
    cond3 = fin_calc >= (hoy - pd.Timedelta(days=4))
    g.loc[cond1 | cond2 | cond3, "Fecha_Fin_Cosecha"] = pd.NaT

    base = agg.merge(
        g[["Bloques", "Fecha S/P", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha"]],
        on=["Bloques", "Fecha S/P"],
        how="left"
    )

    base = base[base["Fecha S/P"] < hoy].copy()
    base["Estado"] = "ACTIVO"

    base = normalize_date_col(base, "Fecha S/P")
    base = normalize_date_col(base, "Fecha_Inicio_Cosecha (Primer Tallo)")
    base = normalize_date_col(base, "Fecha_Fin_Cosecha")

    return base[[
        "Bloques", "Fecha S/P", "S/P", "Variedad", "Area",
        "Tallos_Proy", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha", "Estado"
    ]]


# -------------------------
# Históricos XL/CLO (desde BRONZE raw indices_*.parquet)
# -------------------------
def fenograma_historia_xl(raw_indices_xl: pd.DataFrame, fecha_min_hist: pd.Timestamp) -> pd.DataFrame:
    df = _rebuild_header_like_xl(raw_indices_xl)
    if df.empty:
        return pd.DataFrame(columns=[
            "Bloques", "Fecha S/P", "S/P", "Variedad", "Area",
            "Tallos_Proy", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha", "Estado"
        ])

    if "Pruebas" in df.columns:
        df = df[df["Pruebas"].isna()].copy()
    elif "Pruebas " in df.columns:
        df = df[df["Pruebas "].isna()].copy()

    df["fecha se s/p"] = to_datetime_safe(df.get("fecha se s/p"))
    df["Fecha FIN cos"] = to_datetime_safe(df.get("Fecha FIN cos"))
    df["Fecha Inc cos"] = to_datetime_safe(df.get("Fecha Inc cos"))

    df = df[df["fecha se s/p"].notna()].copy()
    df = df[df["fecha se s/p"] >= fecha_min_hist].copy()

    needed = ["AREA_PRODUCT", "fecha se s/p", "Fecha Inc cos", "Fecha FIN cos", "Bloque", "p/s", "TALLOS TOTALES EN VERDE"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"indices_xl_raw: faltan columnas requeridas: {missing}")

    df = df[needed].copy()
    df = df[df["Fecha FIN cos"].notna()].copy()

    df = df.rename(columns={
        "Fecha Inc cos": "Fecha_Inicio_Cosecha (Primer Tallo)",
        "Fecha FIN cos": "Fecha_Fin_Cosecha",
        "Bloque": "Bloques",
        "fecha se s/p": "Fecha S/P",
        "p/s": "S/P",
        "TALLOS TOTALES EN VERDE": "Tallos_Proy",
        "AREA_PRODUCT": "Area"
    })

    df["Estado"] = "CERRADO"
    df["Variedad"] = "XL"
    df["Area"] = reemplazos_area(df["Area"])
    df["Bloques"] = df["Bloques"].astype(str)

    df = normalize_date_col(df, "Fecha S/P")
    df = normalize_date_col(df, "Fecha_Inicio_Cosecha (Primer Tallo)")
    df = normalize_date_col(df, "Fecha_Fin_Cosecha")
    df["Tallos_Proy"] = to_numeric_safe(df["Tallos_Proy"])

    return df[[
        "Bloques", "Fecha S/P", "S/P", "Variedad", "Area",
        "Tallos_Proy", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha", "Estado"
    ]]


def fenograma_historia_clo(raw_indices_clo: pd.DataFrame, fecha_min_hist: pd.Timestamp) -> pd.DataFrame:
    df = _rebuild_header_like_clo(raw_indices_clo)
    if df.empty:
        return pd.DataFrame(columns=[
            "Bloques", "Fecha S/P", "S/P", "Variedad", "Area",
            "Tallos_Proy", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha", "Estado"
        ])

    df["Fecha de S/P"] = to_datetime_safe(df.get("Fecha de S/P"))
    df["Fecha FIN cos"] = to_datetime_safe(df.get("Fecha FIN cos"))
    df["Fecha Inc cos"] = to_datetime_safe(df.get("Fecha Inc cos"))

    df = df[df["Fecha de S/P"].notna()].copy()
    df = df[df["Fecha de S/P"] >= fecha_min_hist].copy()

    needed = ["AREA_PRODUCT", "Fecha de S/P", "Fecha Inc cos", "Fecha FIN cos", "BLOQUE", "p/s", "TALLOS TOTALES EN VERDE"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"indices_clo_raw: faltan columnas requeridas: {missing}")

    df = df[needed].copy()
    df = df[df["Fecha FIN cos"].notna()].copy()

    df = df.rename(columns={
        "Fecha Inc cos": "Fecha_Inicio_Cosecha (Primer Tallo)",
        "Fecha FIN cos": "Fecha_Fin_Cosecha",
        "BLOQUE": "Bloques",
        "Fecha de S/P": "Fecha S/P",
        "p/s": "S/P",
        "TALLOS TOTALES EN VERDE": "Tallos_Proy",
        "AREA_PRODUCT": "Area"
    })

    df["Estado"] = "CERRADO"
    df["Variedad"] = "CLO"
    df["Area"] = reemplazos_area(df["Area"])
    df["Bloques"] = df["Bloques"].astype(str)

    df = normalize_date_col(df, "Fecha S/P")
    df = normalize_date_col(df, "Fecha_Inicio_Cosecha (Primer Tallo)")
    df = normalize_date_col(df, "Fecha_Fin_Cosecha")
    df["Tallos_Proy"] = to_numeric_safe(df["Tallos_Proy"])

    return df[[
        "Bloques", "Fecha S/P", "S/P", "Variedad", "Area",
        "Tallos_Proy", "Fecha_Inicio_Cosecha (Primer Tallo)", "Fecha_Fin_Cosecha", "Estado"
    ]]


# -------------------------
# 321A/321B preferencia (tu lógica)
# -------------------------
def preferir_con_letra_con_bloque_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Base numérica (321A -> 321)
    out["Bloque_Base"] = out["Bloques"].astype(str).str.replace(r"\D", "", regex=True)

    grp_cols = ["Bloque_Base", "S/P", "Fecha S/P", "Variedad", "Area"]

    # con letra = distinto al base (321A != 321)
    out["__con_letra"] = out["Bloques"].astype(str) != out["Bloque_Base"].astype(str)

    # si en el grupo hay con letra, quedarse solo con esos; si no hay, dejar todos
    has_letra = out.groupby(grp_cols, dropna=False)["__con_letra"].transform("any")
    out = out[(~has_letra) | (out["__con_letra"])].copy()

    return out.drop(columns=["__con_letra"])



# -------------------------
# Construcción total + salida silver + grid (BRONZE -> SILVER)
# -------------------------
def main() -> None:
    cfg = load_settings()
    hoy = pd.Timestamp(datetime.now().date())

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    fecha_min_hist = pd.to_datetime(cfg.get("sources", {}).get("fecha_min_hist", "2024-01-01"))
    horizon_days = int(cfg["pipeline"]["grid_horizon_days"])

    # 1) Fenograma activo desde BRONZE
    df_xlsm = pd.read_parquet(bronze_dir / "fenograma_xlsm_raw.parquet")
    df_xlsm = normalizar_columnas(df_xlsm)

    # 2) Bloques candidatos (solo para filtrar balanza raw)
    bloques_candidatos = (
        df_xlsm.get("Bloques", pd.Series([], dtype=str))
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
    )

    # 3) Balanza desde BRONZE
    bal = pd.read_parquet(bronze_dir / "balanza_bloque_fecha_raw.parquet")
    bal = normalizar_columnas(bal)
    if "Bloque" not in bal.columns or "Fecha" not in bal.columns:
        raise ValueError("balanza_bloque_fecha_raw.parquet debe tener columnas ['Bloque','Fecha'].")

    bal["Bloque"] = bal["Bloque"].astype(str)
    bal["Fecha"] = to_datetime_safe(bal["Fecha"])
    bal = bal[bal["Fecha"].notna()].copy()

    if bloques_candidatos:
        bal = bal[bal["Bloque"].isin([str(b) for b in bloques_candidatos])].copy()
    bal = bal[bal["Fecha"] >= fecha_min_hist].copy()

    # 4) Activo + históricos desde BRONZE
    activo = fenograma_activo(df_xlsm, bal, hoy)

    frames = [activo]

    idx_xl = bronze_dir / "indices_xl_raw.parquet"
    if idx_xl.exists():
        frames.append(fenograma_historia_xl(pd.read_parquet(idx_xl), fecha_min_hist))

    idx_clo = bronze_dir / "indices_clo_raw.parquet"
    if idx_clo.exists():
        frames.append(fenograma_historia_clo(pd.read_parquet(idx_clo), fecha_min_hist))

    total = pd.concat(frames, ignore_index=True)

    total = normalize_date_col(total, "Fecha S/P")
    total = normalize_date_col(total, "Fecha_Inicio_Cosecha (Primer Tallo)")
    total = normalize_date_col(total, "Fecha_Fin_Cosecha")
    total = preferir_con_letra_con_bloque_base(total)

    # 5) Normalizar a fact_ciclo_maestro (silver)
    fact = total.copy()
    fact["Bloques"] = fact["Bloques"].astype(str)
    fact["Variedad"] = fact["Variedad"].astype(str)
    fact["S/P"] = fact["S/P"].astype(str).str.strip().str.upper()
    fact["Area"] = fact["Area"].astype(str)

    fact = fact.rename(columns={
        "Bloques": "bloque",
        "Variedad": "variedad",
        "S/P": "tipo_sp",
        "Fecha S/P": "fecha_sp",
        "Area": "area",
        "Tallos_Proy": "tallos_proy",
        "Fecha_Inicio_Cosecha (Primer Tallo)": "fecha_inicio_cosecha",
        "Fecha_Fin_Cosecha": "fecha_fin_cosecha",
        "Estado": "estado",
        "Bloque_Base": "bloque_base",
    })

    # Jerarquía de bloque (tu lógica)
    fact["bloque"] = fact["bloque"].astype(str).str.strip()
    fact["bloque_padre"] = fact["bloque_base"].astype(str).str.strip()

    # IDs
    if fact["fecha_sp"].isna().any():
        raise ValueError("fact_ciclo_maestro: fecha_sp tiene nulos (revisar fuentes).")

    fact["bloque_id"] = fact["bloque"].map(make_bloque_id)
    fact["bloque_padre_id"] = fact["bloque_padre"].map(make_bloque_id)

    fact["variedad_id"] = fact["variedad"].map(make_variedad_id)
    fact["ciclo_id"] = [
        make_ciclo_id(b, v, t, f)
        for b, v, t, f in fact[["bloque_id", "variedad_id", "tipo_sp", "fecha_sp"]].itertuples(index=False)
    ]

    # Deduplicación: prioriza ACTIVO sobre CERRADO
    fact["__prio"] = np.where(fact["estado"].astype(str).str.upper().eq("ACTIVO"), 1, 0)
    fact = (
        fact.sort_values(["ciclo_id", "__prio"], ascending=[True, False])
            .drop_duplicates(subset=["ciclo_id"], keep="first")
            .drop(columns=["__prio"])
            .reset_index(drop=True)
    )

    if fact.duplicated(["ciclo_id"]).any():
        raise ValueError("fact_ciclo_maestro: ciclo_id no es único luego de dedupe (revisar reglas).")

    write_parquet(fact, silver_dir / "fact_ciclo_maestro.parquet")

    # 6) Grid
    grid = build_grid_ciclo_fecha(fact, horizon_days=horizon_days)
    write_parquet(grid, silver_dir / "grid_ciclo_fecha.parquet")

    print(f"OK: fact_ciclo_maestro={len(fact)} filas; grid={len(grid)} filas; horizonte={horizon_days} días")


if __name__ == "__main__":
    main()
