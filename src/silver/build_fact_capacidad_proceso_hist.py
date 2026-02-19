from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet


# =============================================================================
# Paths / Settings
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
# Helpers
# =============================================================================
def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _to_float(s: pd.Series) -> pd.Series:
    # robusto: acepta floats/ints, y también strings con coma decimal
    if s is None:
        return pd.Series(dtype="float64")
    ss = s.copy()
    if ss.dtype != object:
        return pd.to_numeric(ss, errors="coerce")
    return pd.to_numeric(ss.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """
    Devuelve el nombre real de la columna existente en df probando candidatos (case-insensitive).
    """
    cols = list(df.columns)
    cols_l = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        ckey = str(cand).strip().lower()
        if ckey in cols_l:
            return cols_l[ckey]
    if required:
        raise ValueError(
            f"No se encontró columna. Candidatos={candidates}. "
            f"Disponibles={cols}"
        )
    return None


def _norm_unidad(u: pd.Series) -> pd.Series:
    """
    Normaliza unidad a etiquetas estándar. Para este fact queremos KILOS.
    """
    x = _canon_str(u).replace({"NAN": "", "NONE": ""})
    return x.replace(
        {
            "KG": "KILOS",
            "KGS": "KILOS",
            "KILO": "KILOS",
            "KILOGRAMO": "KILOS",
            "KILOGRAMOS": "KILOS",
        }
    )


def map_proceso(codigo_actividad: pd.Series) -> pd.Series:
    c = _canon_str(codigo_actividad)
    return np.select(
        [
            c.eq("CBX"),
            c.eq("CXLTA1"),
            c.eq("CXLTARH"),
            c.isin(["PSMC", "CXLTA", "CXLTAR"]),
        ],
        [
            "BLANCO",
            "TINTURADO",
            "ARCOIRIS",
            "OTROS_RESERVADOS",
        ],
        default="OTROS_RESERVADOS",
    )


def _nanpercentile_safe(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(np.nanpercentile(x, q))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now().isoformat(timespec="seconds")

    # -------------------------
    # Inputs Bronze
    # -------------------------
    ghu_name = cfg.get("ghu", {}).get("ghu_maestro_horas_file", "ghu_maestro_horas.parquet")
    personal_name = cfg.get("ghu", {}).get("personal_file", "personal_raw.parquet")

    ghu_path = bronze_dir / ghu_name
    per_path = bronze_dir / personal_name

    if not ghu_path.exists():
        raise FileNotFoundError(f"No existe Bronze: {ghu_path}")
    if not per_path.exists():
        raise FileNotFoundError(f"No existe Bronze: {per_path}")

    mh = pd.read_parquet(ghu_path)
    per = pd.read_parquet(per_path)

    mh.columns = [str(c).strip() for c in mh.columns]
    per.columns = [str(c).strip() for c in per.columns]

    # -------------------------
    # Resolver columnas (robusto a variaciones)
    # -------------------------
    c_fecha = _pick_col(mh, ["fecha", "Fecha"])
    c_cod_per = _pick_col(mh, ["codigo_personal", "Codigo_Personal", "Código_Personal"])
    c_cod_act = _pick_col(mh, ["codigo_actividad", "Codigo_Actividad", "Código_Actividad"])
    c_unid = _pick_col(mh, ["unidad_medida", "Unidad_Medida", "unidad", "unidad_med"])
    c_hpres = _pick_col(mh, ["horas_presenciales", "Horas_Presenciales", "horas_trabajadas", "Horas_Trabajadas"])
    c_uniprod = _pick_col(mh, ["unidades_producidas", "Unidades_Producidas", "unidades", "unid_prod"])
    c_hacum = _pick_col(mh, ["horas_acumula", "Horas_Acumula", "horas_acumuladas"], required=False)

    # Personal
    p_cod_per = _pick_col(per, ["codigo_personal", "Codigo_Personal", "Código_Personal"])
    p_activo = _pick_col(per, ["Activo_o_Inactivo", "activo_inactivo", "estado", "Estado"], required=False)
    if p_activo is None:
        per["activo_inactivo"] = np.nan
        p_activo = "activo_inactivo"

    # -------------------------
    # Normalización mínima (Silver staging)
    # -------------------------
    df = mh.copy()
    df["fecha"] = _norm_date(df[c_fecha])
    df["codigo_personal"] = df[c_cod_per].astype(str).str.strip()
    df["codigo_actividad"] = _canon_str(df[c_cod_act])
    df["unidad_medida"] = _norm_unidad(df[c_unid])

    df["horas_presenciales"] = _to_float(df[c_hpres])
    df["unidades_producidas"] = _to_float(df[c_uniprod])

    if c_hacum is not None and c_hacum in df.columns:
        df["horas_acumula"] = _to_float(df[c_hacum])
    else:
        df["horas_acumula"] = np.nan

    # -------------------------
    # Join con Personal (LEFT) - dedupe para evitar explosión
    # -------------------------
    per2 = per[[p_cod_per, p_activo]].copy()
    per2.columns = ["codigo_personal", "activo_inactivo"]
    per2["codigo_personal"] = per2["codigo_personal"].astype(str).str.strip()
    per2 = per2.drop_duplicates(subset=["codigo_personal"], keep="last")

    df = df.merge(per2, on="codigo_personal", how="left")

    # -------------------------
    # Proceso + filtros del fact
    # -------------------------
    df["proceso"] = map_proceso(df["codigo_actividad"])

    codigos_interes = {"CBX", "CXLTA1", "CXLTARH", "PSMC", "CXLTA", "CXLTAR"}
    df = df[df["codigo_actividad"].isin(codigos_interes)].copy()

    # Esta etapa: SOLO KILOS
    df = df[df["unidad_medida"].eq("KILOS")].copy()

    # Validaciones básicas
    df = df[df["fecha"].notna()].copy()
    df = df[df["codigo_personal"].astype(str).str.len() > 0].copy()
    df = df[df["horas_presenciales"].notna() & (df["horas_presenciales"] > 0)].copy()
    df = df[df["unidades_producidas"].notna() & (df["unidades_producidas"] > 0)].copy()

    if df.empty:
        raise ValueError("No hay datos válidos para construir fact_capacidad_proceso_hist (df vacío tras filtros).")

    # -------------------------
    # Agregación por día x persona x proceso
    # -------------------------
    fact = (
        df.groupby(["fecha", "proceso", "codigo_personal"], dropna=False)
        .agg(
            activo_inactivo=("activo_inactivo", "last"),
            horas_presenciales_dia=("horas_presenciales", "sum"),
            kg_procesados_dia=("unidades_producidas", "sum"),
            horas_acumula=("horas_acumula", "max"),
        )
        .reset_index()
    )

    fact["kg_h_persona_dia"] = fact["kg_procesados_dia"] / fact["horas_presenciales_dia"]
    fact["kg_h_persona_dia"] = pd.to_numeric(fact["kg_h_persona_dia"], errors="coerce")

    # Outliers: cap por proceso usando percentiles (1% y 99%)
    p1 = fact.groupby("proceso", dropna=False)["kg_h_persona_dia"].transform(lambda s: _nanpercentile_safe(s, 1))
    p99 = fact.groupby("proceso", dropna=False)["kg_h_persona_dia"].transform(lambda s: _nanpercentile_safe(s, 99))

    # si algún grupo quedó sin percentiles (nan), no capear
    fact["kg_h_persona_dia"] = np.where(
        np.isfinite(p1) & np.isfinite(p99),
        fact["kg_h_persona_dia"].clip(lower=p1, upper=p99),
        fact["kg_h_persona_dia"],
    )

    fact["created_at"] = created_at

    out_fact = silver_dir / "fact_capacidad_proceso_hist.parquet"
    write_parquet(fact, out_fact)

    # -------------------------
    # Baseline por proceso
    # -------------------------
    base = (
        fact.groupby("proceso", dropna=False)
        .agg(
            kg_h_persona_mediana=("kg_h_persona_dia", "median"),
            kg_h_persona_p25=("kg_h_persona_dia", lambda s: _nanpercentile_safe(s, 25)),
            kg_h_persona_p75=("kg_h_persona_dia", lambda s: _nanpercentile_safe(s, 75)),
            n_registros=("kg_h_persona_dia", "size"),
        )
        .reset_index()
    )
    base["created_at"] = created_at

    out_base = silver_dir / "dim_capacidad_baseline_proceso.parquet"
    write_parquet(base, out_base)

    print(f"OK: fact_capacidad_proceso_hist={len(fact):,} filas -> {out_fact}")
    print(f"OK: dim_capacidad_baseline_proceso={len(base):,} filas -> {out_base}")
    print("Baseline:\n", base.to_string(index=False))


if __name__ == "__main__":
    main()
