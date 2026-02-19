from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import re

from common.io import write_parquet


# =============================================================================
# Settings
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]  # .../src/silver -> repo root
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    p = SETTINGS_PATH if SETTINGS_PATH.exists() else Path("config/settings.yaml")
    if not p.exists():
        raise FileNotFoundError(f"No existe settings.yaml en {p} (ni en config/settings.yaml).")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# =============================================================================
# Utils
# =============================================================================
def _norm_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"\s+", " ", c)
    return c


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_map = {_norm_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def semana_ventas(fecha_clasificacion: pd.Series) -> pd.Series:
    """
    Semana ventas según tu regla: fecha_clasificacion + 2 días, y semana %U.
    FIX:
    - evitar np.where (ndarray) + Series sin index (alineación rara)
    - soportar NaT
    """
    d = pd.to_datetime(fecha_clasificacion, errors="coerce") + pd.Timedelta(days=2)

    yy = (d.dt.year % 100).astype("Int64")
    ww = pd.to_numeric(d.dt.strftime("%U"), errors="coerce").astype("Int64")
    ww = ww.where(ww.notna() & (ww > 0), 1)

    yy_s = yy.astype("string").str.zfill(2)
    ww_s = ww.astype("string").str.zfill(2)

    out = yy_s + ww_s
    out = out.where(d.notna(), pd.NA)
    return out


def _clean_proceso(s: pd.Series) -> pd.Series:
    """
    Normaliza proceso a 3 buckets:
      NATURAL, RAINBOW, TINT
    y todo lo demás -> OTHER (no afecta tu TOTAL porque sumas solo 3).
    FIX: NO usar np.where (convierte a ndarray y rompe .str)
    """
    s = s.astype(str).str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)

    # normalizaciones directas
    s = s.replace({
        "GARLAND NATURAL": "NATURAL",
        "GLITTER": "NATURAL",
        "RAINBOW 2": "RAINBOW",
        "ARCOIRIS": "RAINBOW",
        "BLANCO": "NATURAL",
        "TINTURADO": "TINT",
        "TINT": "TINT",
    })

    # reglas por contains (manteniendo tipo Series)
    s = s.mask(s.str.contains("TINT", na=False), "TINT")
    s = s.mask(s.str.contains("RAINBOW|ARCO", na=False), "RAINBOW")
    s = s.mask(s.str.contains("NATURAL|BLANCO", na=False), "NATURAL")

    # bucket final
    s = s.where(s.isin(["NATURAL", "RAINBOW", "TINT"]), "OTHER")

    return s


def _clean_tipo_tallo(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.upper()
        .str.replace("GR", "", regex=False)
        .str.strip()
    )


def _looks_like_date_colname(x: str) -> bool:
    s = str(x).strip()
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", s):
        return True
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}\s+\d{2}:\d{2}:\d{2}", s):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}:\d{2}", s):
        return True
    return False


def _is_excel_zero_date(col) -> bool:
    s = str(col).strip()
    if s == "1900-01-01 00:00:00":
        return True
    try:
        dt = pd.to_datetime(col, errors="coerce")
        return pd.notna(dt) and dt.normalize() == pd.Timestamp("1900-01-01")
    except Exception:
        return False


# =============================================================================
# IO: read ventas parquet (bronze wide con headers internos)
# =============================================================================
def read_ventas_bronze_parquet(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe Bronze ventas parquet: {parquet_path}")

    raw = pd.read_parquet(parquet_path)

    meta_cols = [c for c in ["bronze_source", "bronze_extracted_at"] if c in raw.columns]
    df = raw.drop(columns=meta_cols, errors="ignore").copy()

    def _row_contains(row: pd.Series, token: str) -> bool:
        vals = row.astype(str).str.strip().str.upper()
        return (vals == token.upper()).any()

    header_idx = None
    scan_n = min(len(df), 50)
    for i in range(scan_n):
        r = df.iloc[i]
        if _row_contains(r, "Cliente") and _row_contains(r, "FINCA"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(
            f"No pude detectar fila de encabezado en {parquet_path.name}. "
            "Esperaba encontrar 'Cliente' y 'FINCA' en las primeras 50 filas."
        )

    headers = df.iloc[header_idx].copy()

    new_cols = []
    for h in headers.tolist():
        if pd.isna(h):
            new_cols.append(None)
            continue
        if isinstance(h, (pd.Timestamp, datetime, np.datetime64)):
            new_cols.append(pd.to_datetime(h))
        else:
            s = str(h).replace("\n", " ").replace("\r", " ")
            s = re.sub(r"\s+", " ", s).strip()
            new_cols.append(s)

    out = df.iloc[header_idx + 1:].copy()
    out.columns = new_cols

    out = out.loc[:, [c for c in out.columns if c is not None and str(c).strip() != ""]]
    out = out.dropna(axis=1, how="all")

    clean_cols = []
    for c in out.columns:
        if isinstance(c, (pd.Timestamp, datetime, np.datetime64)):
            clean_cols.append(pd.to_datetime(c))
        else:
            s = str(c).replace("\n", " ").replace("\r", " ")
            s = re.sub(r"\s+", " ", s).strip()
            clean_cols.append(s)
    out.columns = clean_cols

    return out


# =============================================================================
# Transform: wide -> long + parse fecha robusto (FIX pyarrow string assignment)
# =============================================================================
def unpivot_ventas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    c_finca = _find_col(df, ["FINCA", "Finca"])
    c_var = _find_col(df, ["VAR", "Var", "Variedad"])
    c_cliente = _find_col(df, ["Cliente", "CLIENTE", "Customer"])
    c_proceso = _find_col(df, ["Proceso", "PROCESO"])
    c_tipo_tallo = _find_col(df, ["Tipo Tallo", "TIPO TALLO", "TipoTallo", "TIPOTALLO", "Tipo_Tallo"])
    c_grado = _find_col(df, ["Grado", "GRADO"])
    c_stbx = _find_col(df, ["ST/BX", "STBX", "ST_BX"])
    c_bunca = _find_col(df, ["BUN/CAJA", "BUN CAJA", "BUNCAJA", "BUN_CAJA"])
    c_peso = _find_col(df, ["Peso", "Peso Final", "PESO", "PESO FINAL"])

    if c_finca:
        df = df[df[c_finca].astype(str).str.strip().str.upper().eq("MALIMA")].copy()

    if c_var:
        vv = df[c_var].astype(str).str.upper().str.strip()
        df = df[vv.isin(["XL", "XL 2"])].copy()

    if c_proceso:
        # filtra NATURAL PRESERVADA robusto
        proc = df[c_proceso].astype(str).str.upper().str.strip()
        df = df[~proc.str.contains("NATURAL PRESERVADA", na=False)].copy()

    if c_cliente:
        df = df[~df[c_cliente].astype(str).str.upper().isin(["ZINVENTARIOS", "ZMONJAS"])].copy()

    if c_proceso:
        df[c_proceso] = _clean_proceso(df[c_proceso])
    if c_tipo_tallo:
        df[c_tipo_tallo] = _clean_tipo_tallo(df[c_tipo_tallo])

    extra_fixed = []
    for name in [
        "Cliente", "Vendedor", "Cl2", "Nota", "Orden", "Producto", "Largo",
        "Perdida Peso", "Proceso 2", "Color", "Mercado", "Caja", "TALLOS SKU",
        "presentacion", "sku", "VAR", "FINCA", "ROT"
    ]:
        col = _find_col(df, [name])
        if col:
            extra_fixed.append(col)

    fixed_real = [c for c in [c_tipo_tallo, c_grado, c_proceso, c_stbx, c_bunca, c_peso] if c] + extra_fixed
    seen = set()
    fixed_real = [c for c in fixed_real if not (c in seen or seen.add(c))]

    value_cols = []
    for c in df.columns:
        if c in fixed_real:
            continue
        if _looks_like_date_colname(c):
            value_cols.append(c)

    if len(value_cols) == 0:
        for c in df.columns:
            if c in fixed_real:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                value_cols.append(c)

    if len(value_cols) == 0:
        raise ValueError(
            "No se detectaron columnas fecha para unpivot.\n"
            f"Columnas disponibles: {list(df.columns)[:40]} ...\n"
            "Solución: confirma si Bronze está wide (fechas como columnas) o long (columna 'Fecha')."
        )

    # excluir columna basura 1900-01-01
    value_cols = [c for c in value_cols if not _is_excel_zero_date(c)]

    out = df.melt(id_vars=fixed_real, value_vars=value_cols, var_name="Fecha", value_name="Valor")

    # --- FIX CLAVE: no asignar datetimes sobre columna string[pyarrow] con .loc ---
    fecha_str = out["Fecha"].astype(str).str.strip()

    is_iso = fecha_str.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$")
    is_iso_dt = fecha_str.str.match(r"^\d{4}-\d{1,2}-\d{1,2}\s+\d{2}:\d{2}:\d{2}$")
    is_slash = fecha_str.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    is_slash_dt = fecha_str.str.match(r"^\d{1,2}/\d{1,2}/\d{4}\s+\d{2}:\d{2}:\d{2}$")

    fecha_dt = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    parsed_iso = pd.to_datetime(fecha_str.where(is_iso), format="%Y-%m-%d", errors="coerce")
    parsed_iso_dt = pd.to_datetime(fecha_str.where(is_iso_dt), format="%Y-%m-%d %H:%M:%S", errors="coerce")

    # Mantengo tu contrato: slash = mm/dd/yyyy
    parsed_slash = pd.to_datetime(fecha_str.where(is_slash), format="%m/%d/%Y", errors="coerce")
    parsed_slash_dt = pd.to_datetime(fecha_str.where(is_slash_dt), format="%m/%d/%Y %H:%M:%S", errors="coerce")

    fecha_dt = fecha_dt.fillna(parsed_iso)
    fecha_dt = fecha_dt.fillna(parsed_iso_dt)
    fecha_dt = fecha_dt.fillna(parsed_slash)
    fecha_dt = fecha_dt.fillna(parsed_slash_dt)

    still = fecha_dt.isna()
    if still.any():
        fecha_dt = fecha_dt.fillna(pd.to_datetime(fecha_str.where(still), errors="coerce"))

    out["Fecha"] = fecha_dt.dt.normalize()

    out["Valor"] = pd.to_numeric(out["Valor"], errors="coerce").fillna(0.0)
    out = out[out["Fecha"].notna()].copy()

    rename_map = {}
    if c_tipo_tallo:
        rename_map[c_tipo_tallo] = "Tipo Tallo"
    if c_proceso:
        rename_map[c_proceso] = "Proceso"
    out = out.rename(columns=rename_map)

    if "Tipo Tallo" not in out.columns:
        out["Tipo Tallo"] = np.nan
    if "Proceso" not in out.columns:
        out["Proceso"] = np.nan

    return out


# =============================================================================
# Build mix semana
# =============================================================================
def build_mix_semana(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    df["Fecha_Clasificacion"] = df["Fecha"] - pd.Timedelta(days=2)
    df["Semana_Ventas"] = semana_ventas(df["Fecha_Clasificacion"])

    tipo_series = df["Tipo Tallo"] if "Tipo Tallo" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    tipo = tipo_series.astype(str).str.upper().str.strip()
    es_otro = tipo.isin(["BQT", "PET"])

    proc_series = df["Proceso"] if "Proceso" in df.columns else pd.Series([np.nan] * len(df), index=df.index)
    df["Proceso"] = _clean_proceso(proc_series)

    netas = (
        df[~es_otro]
        .groupby(["Semana_Ventas", "Proceso"], dropna=False)["Valor"]
        .sum()
        .reset_index()
        .rename(columns={"Valor": "cajas_netas"})
    )

    piv = netas.pivot_table(
        index="Semana_Ventas",
        columns="Proceso",
        values="cajas_netas",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    # asegurar columnas estándar
    for c in ["NATURAL", "RAINBOW", "TINT"]:
        if c not in piv.columns:
            piv[c] = 0.0

    piv["TOTAL"] = piv["NATURAL"] + piv["RAINBOW"] + piv["TINT"]

    # weights sin np.where (más limpio)
    piv["W_Blanco"] = (piv["NATURAL"] / piv["TOTAL"]).where(piv["TOTAL"] > 0)
    piv["W_Arcoiris"] = (piv["RAINBOW"] / piv["TOTAL"]).where(piv["TOTAL"] > 0)
    piv["W_Tinturado"] = (piv["TINT"] / piv["TOTAL"]).where(piv["TOTAL"] > 0)

    out = piv[["Semana_Ventas", "W_Blanco", "W_Arcoiris", "W_Tinturado", "TOTAL"]].copy()
    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = load_settings()

    if "paths" not in cfg or "bronze" not in cfg["paths"] or "silver" not in cfg["paths"]:
        raise ValueError("settings.yaml debe incluir paths.bronze y paths.silver")

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    ventas_cfg = cfg.get("ventas", {})
    p25_name = ventas_cfg.get("ventas_2025_raw_parquet", "ventas_2025_raw.parquet")
    p26_name = ventas_cfg.get("ventas_2026_raw_parquet", "ventas_2026_raw.parquet")

    path_2025 = bronze_dir / p25_name
    path_2026 = bronze_dir / p26_name

    frames: list[pd.DataFrame] = []

    if path_2026.exists():
        df26 = read_ventas_bronze_parquet(path_2026)
        frames.append(unpivot_ventas(df26))

    if path_2025.exists():
        df25 = read_ventas_bronze_parquet(path_2025)
        frames.append(unpivot_ventas(df25))

    if not frames:
        raise FileNotFoundError(
            "No se encontraron ventas raw en Bronze. Esperaba:\n"
            f" - {path_2026}\n"
            f" - {path_2025}\n"
            "Ajusta config/settings.yaml (paths.bronze y ventas.*_raw_parquet) o verifica nombres."
        )

    df_long = pd.concat(frames, ignore_index=True)
    mix = build_mix_semana(df_long)

    # seed DEFAULT por mediana (si todo es NaN, fallback 1/3,1/3,1/3)
    wb = float(np.nanmedian(mix["W_Blanco"].values)) if len(mix) else float("nan")
    wa = float(np.nanmedian(mix["W_Arcoiris"].values)) if len(mix) else float("nan")
    wt = float(np.nanmedian(mix["W_Tinturado"].values)) if len(mix) else float("nan")

    seed = {"W_Blanco": wb, "W_Arcoiris": wa, "W_Tinturado": wt}
    if not np.isfinite(seed["W_Blanco"]):
        seed["W_Blanco"] = 0.0
    if not np.isfinite(seed["W_Arcoiris"]):
        seed["W_Arcoiris"] = 0.0
    if not np.isfinite(seed["W_Tinturado"]):
        seed["W_Tinturado"] = 0.0

    s = seed["W_Blanco"] + seed["W_Arcoiris"] + seed["W_Tinturado"]
    if s <= 0:
        seed = {"W_Blanco": 1 / 3, "W_Arcoiris": 1 / 3, "W_Tinturado": 1 / 3}
    else:
        seed = {k: v / s for k, v in seed.items()}

    seed_row = pd.DataFrame([{
        "Semana_Ventas": "DEFAULT",
        "W_Blanco": seed["W_Blanco"],
        "W_Arcoiris": seed["W_Arcoiris"],
        "W_Tinturado": seed["W_Tinturado"],
        "TOTAL": np.nan,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }])

    mix2 = pd.concat([mix, seed_row], ignore_index=True)

    out_path = silver_dir / "dim_mix_proceso_semana.parquet"
    write_parquet(mix2, out_path)

    print(f"OK: dim_mix_proceso_semana={len(mix2):,} filas -> {out_path}")
    print("Seed DEFAULT (mediana histórica normalizada):", seed)


if __name__ == "__main__":
    main()
