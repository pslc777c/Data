from __future__ import annotations

from pathlib import Path
import re
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
SILVER = DATA / "silver" / "ventas"

VENTAS_2026 = BRONZE / "ventas_2026_raw.parquet"
VENTAS_2025 = BRONZE / "ventas_2025_raw.parquet"
CAL_PATH = BRONZE / "calendario.parquet"

OUT_PATH = SILVER / "silver_ventas_b2a_share_sku_gi_dia_destino.parquet"


# ---------------------------
# Helpers
# ---------------------------

def _dedup_columns(cols: list[str]) -> list[str]:
    """
    Hace únicas las columnas agregando sufijos __2, __3, ...
    Mantiene el primer nombre tal cual.
    """
    seen = {}
    out = []
    for c in cols:
        c0 = str(c).strip()
        if c0 == "" or c0.lower() == "nan":
            c0 = "COL"
        k = c0
        if k not in seen:
            seen[k] = 1
            out.append(k)
        else:
            seen[k] += 1
            out.append(f"{k}__{seen[k]}")
    return out


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = _dedup_columns(list(df.columns))
    return df

def _canon_str(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.replace("\t", " ", regex=False)
        .str.strip()
        .str.upper()
    )


def _is_date_col(col: str) -> bool:
    c = str(col).strip()

    # formato PowerQuery: 1/2/2026
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", c):
        return True

    # ISO: 2026-02-01
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", c):
        return True

    # cualquier cosa parseable a fecha (pero evitando nombres comunes)
    bad = {"FECHA", "FECHA DISPONIBLE", "SEMANA", "SEMANA_", "DESTINO", "CLIENTE"}
    if c.upper() in bad:
        return False

    try:
        dt = pd.to_datetime(c, errors="raise")
        # si parsea y está entre 2020..2035 lo aceptamos como columna fecha
        if 2020 <= dt.year <= 2035:
            return True
    except Exception:
        return False

    return False



def _clean_proceso(s: pd.Series) -> pd.Series:
    x = _canon_str(s)
    x = x.str.replace("GARLAND NATURAL", "NATURAL", regex=False)
    x = x.str.replace("GLITTER", "NATURAL", regex=False)
    x = x.str.replace("RAINBOW 2", "RAINBOW", regex=False)
    return x


def _destino_from_proceso(proc: pd.Series) -> pd.Series:
    p = _canon_str(proc)
    return np.select(
        [p.eq("NATURAL"), p.eq("RAINBOW")],
        ["BLANCO", "ARCOIRIS"],
        default="TINTURADO",
    ).astype(str)


def _variedad_from_var(var_col: pd.Series) -> pd.Series:
    v = _canon_str(var_col)
    return np.where(v.isin(["FP", "FP "]), "CLOUD", "XLENCE")


def _tipo_tallo_to_gradoideal(s: pd.Series) -> pd.Series:
    return _canon_str(s).str.replace("GR", "", regex=False)


def _compute_semana_yyww_from_fecha_ajustada(fecha_ajustada: pd.Series) -> pd.Series:
    fa = pd.to_datetime(fecha_ajustada, errors="coerce")
    iso = fa.dt.isocalendar()

    yy = (iso.year % 100).astype("Int64")
    ww = iso.week.astype("Int64")

    out = yy.astype(str).str.zfill(2) + ww.astype(str).str.zfill(2)
    out = out.where(fa.notna(), other=pd.NA)
    return out.astype("string")



def _promote_headers_like_powerquery(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Soporta el caso donde el parquet contiene la grilla "cruda" del sheet,
    con headers embebidos en una fila (equivalente a Table.Skip + PromoteHeaders).
    Detecta fila donde aparece 'FINCA' y promueve esa fila como header.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Si ya tiene FINCA como columna, no hacemos nada
    if "FINCA" in df.columns:
        return df

    # Buscar fila header: cualquier fila que contenga "FINCA" en alguna celda
    # (típico del promoteheaders)
    cand = df.astype(str).apply(lambda r: r.str.upper().str.strip().eq("FINCA").any(), axis=1)
    if not cand.any():
        # fallback: devolver tal cual para que el error sea explícito luego
        return df

    header_idx = int(np.where(cand.values)[0][0])

    # Tomar esa fila como nombres
    new_cols = df.iloc[header_idx].astype(str).str.strip().tolist()

    # El equivalente a: Table.Skip(Datos, header_idx+1)  (ya que promoteheaders consume esa fila)
    df2 = df.iloc[header_idx + 1 :].copy()
    df2.columns = new_cols

    # Quitar columnas totalmente vacías o "ColumnX" raras
    df2 = df2.loc[:, ~pd.Series(df2.columns).astype(str).str.match(r"^Column\d+$", na=False).values]

    # Reset index
    df2 = df2.reset_index(drop=True)
    return df2


def _load_ventas(path: Path) -> pd.DataFrame:
    raw = read_parquet(path).copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    df = _promote_headers_like_powerquery(raw)
    df.columns = [str(c).strip() for c in df.columns]

    # 🔥 CRÍTICO: asegurar columnas únicas para evitar InvalidIndexError en concat
    df = _ensure_unique_columns(df)

    return df



def main() -> None:
    SILVER.mkdir(parents=True, exist_ok=True)

    if not VENTAS_2026.exists():
        raise FileNotFoundError(f"No existe {VENTAS_2026}")

    df26 = _load_ventas(VENTAS_2026)
    dfs = [df26]

    if VENTAS_2025.exists():
        df25 = _load_ventas(VENTAS_2025)
        dfs.append(df25)
    else:
        print(f"[WARN] No encontré {VENTAS_2025} -> usaré solo 2026.")

    for i, dfi in enumerate(dfs):
        if not dfi.columns.is_unique:
            print(f"[DBG] df[{i}] columnas duplicadas:", pd.Series(dfi.columns)[pd.Series(dfi.columns).duplicated()].unique().tolist())

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]

    # Normalizar alias posibles de columnas (por si vienen con variantes)
    rename_map = {}
    for c in df.columns:
        cu = str(c).strip().upper()
        if cu == "ST/BX":
            rename_map[c] = "ST/BX"
        if cu in ["BUN/CAJA", "BUN/CAJ", "BUN/CAJA "]:
            rename_map[c] = "BUN/CAJA"
        if cu in ["TIPO TALLO", "TIPO_TALLO"]:
            rename_map[c] = "Tipo Tallo"
        if cu == "PESO FINAL":
            rename_map[c] = "Peso Final"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Validación de columnas base (ya con promoteheaders)
    need_cols = {"FINCA", "VAR", "Proceso", "Cliente", "Tipo Tallo", "Grado", "ST/BX", "BUN/CAJA"}
    miss = need_cols - set(df.columns)
    if miss:
        # dump rápido para diagnóstico
        print("[DBG] Columnas detectadas:", sorted(df.columns)[:60], "...")
        raise ValueError(f"Ventas raw sin columnas (post-limpieza): {sorted(miss)}")

    # Filtros
    df["FINCA"] = _canon_str(df["FINCA"])
    df["VAR"] = _canon_str(df["VAR"])
    df["Proceso"] = _clean_proceso(df["Proceso"])
    df["Cliente"] = _canon_str(df["Cliente"])

    df = df.loc[
        (df["FINCA"] == "MALIMA")
        & (df["VAR"].isin(["XL", "XL 2", "FP", "FP "]))
        & (~df["Proceso"].isin(["NATURAL PRESERVADA"]))
        & (~df["Cliente"].isin(["ZINVENTARIOS", "ZMONJAS"]))
    ].copy()

    # Derivados
    df["Variedad"] = _variedad_from_var(df["VAR"])
    df["Destino"] = _destino_from_proceso(df["Proceso"])
    df["Grado_Ideal"] = _tipo_tallo_to_gradoideal(df["Tipo Tallo"])

    sku_col = None
    for c in ["Peso Final", "Peso", "SKU"]:
        if c in df.columns:
            sku_col = c
            break
    if sku_col is None:
        raise ValueError("No encontré columna para SKU (esperaba 'Peso Final' o 'Peso' o 'SKU').")

    df["SKU"] = pd.to_numeric(df[sku_col], errors="coerce")

    # Unpivot fechas
    # ---------------------------
    # 4) Unpivot / Normalización a formato long
    # Soporta:
    #  A) Wide: muchas columnas de fecha
    #  B) Ya-long: columnas ["Fecha","Valor"] (o similares)
    # ---------------------------
    cols_up = {c.upper().strip(): c for c in df.columns}

    # Caso B) ya viene long
    if ("FECHA" in cols_up) and (("VALOR" in cols_up) or ("VOLUMEN" in cols_up) or ("CANTIDAD" in cols_up)):
        fecha_c = cols_up["FECHA"]
        valor_c = cols_up.get("VALOR") or cols_up.get("VOLUMEN") or cols_up.get("CANTIDAD")

        long = df.copy()
        long = long.rename(columns={fecha_c: "Fecha", valor_c: "Valor"})
        long["Fecha"] = pd.to_datetime(long["Fecha"], errors="coerce").dt.normalize()
        long["Valor"] = pd.to_numeric(long["Valor"], errors="coerce").fillna(0.0)

    else:
        # Caso A) wide: columnas fecha
        date_cols = [c for c in df.columns if _is_date_col(c)]
        if not date_cols:
            # debug útil
            print("[DBG] No detecté columnas fecha. Ejemplos de columnas:", df.columns[:40].tolist())
            raise ValueError(
                "No encontré columnas fecha (ni wide ni long). "
                "Revisa el parquet: cómo vienen los encabezados de fechas."
            )

        fixed_cols = ["Grado", "ST/BX", "BUN/CAJA", "Proceso", "Variedad", "Destino", "Grado_Ideal", "SKU"]
        keep = [c for c in fixed_cols if c in df.columns] + date_cols
        df_w = df[keep].copy()

        long = df_w.melt(
            id_vars=[c for c in fixed_cols if c in df_w.columns],
            value_vars=date_cols,
            var_name="Fecha",
            value_name="Valor",
        )

    # parseo fecha robusto (dayfirst no siempre aplica)
    long["Fecha"] = pd.to_datetime(long["Fecha"], errors="coerce").dt.normalize()
    long["Valor"] = pd.to_numeric(long["Valor"], errors="coerce").fillna(0.0)


    # Fecha_Disponible y Semana_
    long["Fecha_Disponible"] = long["Fecha"] - pd.Timedelta(days=1)
    fecha_ajustada = long["Fecha_Disponible"] - pd.Timedelta(days=1)  # tu ajuste para semana L-D
    long["Semana_"] = _compute_semana_yyww_from_fecha_ajustada(fecha_ajustada)

    # Tallos totales
    long["ST/BX"] = pd.to_numeric(long["ST/BX"], errors="coerce").fillna(0.0)
    long["BUN/CAJA"] = pd.to_numeric(long["BUN/CAJA"], errors="coerce").fillna(0.0)
    long["Tallos_Totales"] = long["ST/BX"] * long["BUN/CAJA"] * long["Valor"]

    # Weekly agg
    g_keys = ["Semana_", "Variedad", "Destino", "SKU", "Grado_Ideal"]
    weekly = (
        long.groupby(g_keys, as_index=False)
        .agg(
            Peso_Balanza_2A_Semana=("Valor", "sum"),
            Tallos_Totales_Semana=("Tallos_Totales", "sum"),
        )
    )

    totals = (
        weekly.groupby(["Semana_", "Variedad", "Destino"], as_index=False)
        .agg(Peso_Balanza_2A_Total=("Peso_Balanza_2A_Semana", "sum"))
    )

    weekly = weekly.merge(totals, on=["Semana_", "Variedad", "Destino"], how="left")
    weekly["Share_SKU&GI"] = np.where(
        weekly["Peso_Balanza_2A_Total"] > 0,
        weekly["Peso_Balanza_2A_Semana"] / weekly["Peso_Balanza_2A_Total"],
        np.nan,
    )

    # Expand to days using calendario
    if not CAL_PATH.exists():
        raise FileNotFoundError(f"No existe calendario en {CAL_PATH}. Ejecuta: python -m src.bronze.build_bronze_calendario")

    cal = read_parquet(CAL_PATH).copy()
    cal.columns = [str(c).strip() for c in cal.columns]
    if not {"Semana_Año", "Fecha"} <= set(cal.columns):
        raise ValueError("Calendario debe tener columnas: Semana_Año y Fecha")

    cal["Semana_Año"] = cal["Semana_Año"].astype(str).str.strip()
    cal["Fecha"] = pd.to_datetime(cal["Fecha"], errors="coerce").dt.normalize()

    daily = weekly.merge(cal[["Semana_Año", "Fecha"]], left_on="Semana_", right_on="Semana_Año", how="left")
    daily = daily.drop(columns=["Semana_Año"])

    # /7 como en tu query
    daily["Peso_Balanza_2A"] = daily["Peso_Balanza_2A_Semana"] / 7.0
    daily["Tallos_Totales"] = daily["Tallos_Totales_Semana"] / 7.0

    out = daily[[
        "Semana_",
        "Fecha",
        "Destino",
        "Variedad",
        "SKU",
        "Grado_Ideal",
        "Share_SKU&GI",
        "Tallos_Totales",
        "Peso_Balanza_2A",
    ]].copy()

    out["Destino"] = out["Destino"].astype(str).str.upper().str.strip()
    out["Variedad"] = out["Variedad"].astype(str).str.upper().str.strip()

    write_parquet(out, OUT_PATH)

    print(f"[OK] Silver ventas B2A guardado en {OUT_PATH} | rows={len(out):,}")
    print(f"[INFO] semanas={out['Semana_'].nunique():,} fechas={out['Fecha'].nunique():,} destinos={sorted(out['Destino'].dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
