# src/silver/build_dim_baseline_capacidad_tallos_h_persona.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def map_proceso(cod: pd.Series) -> pd.Series:
    c = cod.astype(str).str.upper().str.strip()
    # Solo los que hoy aplican (según lo que definiste)
    return np.select(
        [c.eq("CXLTA1"), c.eq("CXLTARH")],
        ["TINTURADO", "ARCOIRIS"],
        default="OTROS",
    )


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Fechas mínimas (recorte para baseline)
    fecha_min = pd.to_datetime(cfg.get("pipeline", {}).get("capacidad_fecha_min", "2024-01-01"))

    # -------------------------
    # BRONZE inputs
    # -------------------------
    ghu_path = bronze_dir / "ghu_maestro_horas.parquet"
    per_path = bronze_dir / "personal_raw.parquet"

    if not ghu_path.exists():
        raise FileNotFoundError(f"No existe Bronze: {ghu_path}. Ejecuta src/bronze/build_ghu_maestro_horas.py")
    if not per_path.exists():
        raise FileNotFoundError(f"No existe Bronze: {per_path}. Ejecuta src/bronze/build_personal_sources.py")

    df = pd.read_parquet(ghu_path)
    per = pd.read_parquet(per_path)

    df.columns = [str(c).strip() for c in df.columns]
    per.columns = [str(c).strip() for c in per.columns]

    # Validar contrato mínimo
    need_df = [
        "fecha",
        "codigo_personal",
        "codigo_actividad",
        "unidad_medida",
        "horas_presenciales",
        "unidades_producidas",
    ]
    missing_df = [c for c in need_df if c not in df.columns]
    if missing_df:
        raise ValueError(f"ghu_maestro_horas: faltan columnas requeridas: {missing_df}")

    need_per = ["codigo_personal", "Activo_o_Inactivo"]
    missing_per = [c for c in need_per if c not in per.columns]
    if missing_per:
        raise ValueError(f"personal_raw: faltan columnas requeridas: {missing_per}")

    # -------------------------
    # Transformaciones Silver
    # -------------------------
    df["fecha"] = _norm_date(df["fecha"])
    df = df[df["fecha"].notna()].copy()
    df = df[df["fecha"] >= fecha_min].copy()

    df["codigo_personal"] = pd.to_numeric(df["codigo_personal"], errors="coerce").astype("Int64")
    df["codigo_actividad"] = df["codigo_actividad"].astype(str).str.upper().str.strip()
    df["unidad_medida"] = df["unidad_medida"].astype(str).str.upper().str.strip()

    df["horas_presenciales"] = pd.to_numeric(df["horas_presenciales"], errors="coerce").fillna(0.0)
    df["unidades_producidas"] = pd.to_numeric(df["unidades_producidas"], errors="coerce").fillna(0.0)

    # horas_acumula es opcional (según Bronze); si no está, se crea 0.0
    if "horas_acumula" in df.columns:
        df["horas_acumula"] = pd.to_numeric(df["horas_acumula"], errors="coerce").fillna(0.0)
    else:
        df["horas_acumula"] = 0.0

    # Filtro negocio: solo actividades relevantes para tallos/h persona
    df = df[df["codigo_actividad"].isin(["CXLTA1", "CXLTARH"])].copy()

    # Solo tallos (si vienen kilos u otros, se ignoran para este baseline)
    df = df[df["unidad_medida"].str.contains("TALLO", na=False)].copy()

    df["proceso"] = map_proceso(df["codigo_actividad"])
    df = df[df["proceso"].isin(["TINTURADO", "ARCOIRIS"])].copy()

    # Personal join (Silver consume Bronze personal_raw)
    per["codigo_personal"] = pd.to_numeric(per["codigo_personal"], errors="coerce").astype("Int64")
    per["Activo_o_Inactivo"] = per["Activo_o_Inactivo"].astype(str).str.strip()

    df = df.merge(per[["codigo_personal", "Activo_o_Inactivo"]], on="codigo_personal", how="left")
    df = df.rename(columns={"Activo_o_Inactivo": "activo_inactivo"})

    # Consolidación por día/persona/proceso
    daily = (
        df.groupby(["fecha", "proceso", "codigo_personal"], dropna=False)
          .agg(
              horas_presenciales_dia=("horas_presenciales", "sum"),
              tallos_procesados_dia=("unidades_producidas", "sum"),
              horas_acumula_max=("horas_acumula", "max"),
              activo_inactivo=("activo_inactivo", "last"),
          )
          .reset_index()
    )

    daily = daily[(daily["horas_presenciales_dia"] > 0) & (daily["tallos_procesados_dia"] > 0)].copy()
    daily["tallos_h_persona_dia"] = daily["tallos_procesados_dia"] / daily["horas_presenciales_dia"]

    # Sanity: evita valores basura extremos
    daily = daily[(daily["tallos_h_persona_dia"] > 10) & (daily["tallos_h_persona_dia"] < 2000)].copy()

    # Baseline por proceso (mediana y percentiles)
    base = (
        daily.groupby("proceso", dropna=False)
             .agg(
                 tallos_h_persona_mediana=("tallos_h_persona_dia", "median"),
                 tallos_h_persona_p25=("tallos_h_persona_dia", lambda s: float(s.quantile(0.25))),
                 tallos_h_persona_p75=("tallos_h_persona_dia", lambda s: float(s.quantile(0.75))),
                 n_registros=("tallos_h_persona_dia", "count"),
             )
             .reset_index()
    )
    base["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "dim_baseline_capacidad_tallos_h_persona.parquet"
    write_parquet(base, out_path)

    print(f"OK: dim_baseline_capacidad_tallos_h_persona={len(base)} filas -> {out_path}")
    print(base.to_string(index=False))


if __name__ == "__main__":
    main()
