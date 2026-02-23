# src/silver/build_dim_capacidad_baseline_tallos_proceso.py
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


def _to_float(s: pd.Series) -> pd.Series:
    # soporta strings con coma decimal
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

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

    # Validar contrato mínimo (lo que tu SQL traía)
    need_df = [
        "fecha",
        "codigo_personal",
        "codigo_actividad",
        "horas_presenciales",
        "unidades_producidas",
        "unidad_medida",
        "horas_acumula",
    ]
    missing_df = [c for c in need_df if c not in df.columns]
    if missing_df:
        raise ValueError(f"ghu_maestro_horas: faltan columnas requeridas: {missing_df}")

    need_per = ["codigo_personal", "Activo_o_Inactivo"]
    missing_per = [c for c in need_per if c not in per.columns]
    if missing_per:
        raise ValueError(f"personal_raw: faltan columnas requeridas: {missing_per}")

    # Normalizaciones (Silver)
    df["fecha"] = _norm_date(df["fecha"])
    df["codigo_personal"] = df["codigo_personal"].astype(str).str.strip()
    df["codigo_actividad"] = df["codigo_actividad"].astype(str).str.strip().str.upper()
    df["unidad_medida"] = df["unidad_medida"].astype(str).str.strip().str.upper()

    df["horas_presenciales"] = _to_float(df["horas_presenciales"])
    df["unidades_producidas"] = _to_float(df["unidades_producidas"])
    df["horas_acumula"] = _to_float(df["horas_acumula"])

    # Join con Personal (Bronze)
    per["codigo_personal"] = pd.to_numeric(per["codigo_personal"], errors="coerce").astype("Int64")
    per["Activo_o_Inactivo"] = per["Activo_o_Inactivo"].astype(str).str.strip()

    # Ojo: df.codigo_personal está string; convertimos a Int64 para matchear como en los otros scripts
    df["codigo_personal_int"] = pd.to_numeric(df["codigo_personal"], errors="coerce").astype("Int64")
    df = df.merge(
        per[["codigo_personal", "Activo_o_Inactivo"]].rename(columns={"codigo_personal": "codigo_personal_int"}),
        on="codigo_personal_int",
        how="left",
    )
    df = df.rename(columns={"Activo_o_Inactivo": "activo_inactivo"})

    # Map proceso
    df["proceso"] = np.select(
        [df["codigo_actividad"].eq("CXLTA1"), df["codigo_actividad"].eq("CXLTARH")],
        ["TINTURADO", "ARCOIRIS"],
        default="OTROS",
    )

    # Filtrar solo TALL0S (unidad nativa)
    df = df[df["unidad_medida"].eq("TALLOS")].copy()

    # Validaciones
    df = df[df["fecha"].notna()].copy()
    df = df[df["codigo_personal"].astype(str).str.len() > 0].copy()
    df = df[df["horas_presenciales"].notna() & (df["horas_presenciales"] > 0)].copy()
    df = df[df["unidades_producidas"].notna() & (df["unidades_producidas"] > 0)].copy()

    # Agregar por día x persona x proceso
    fact = (
        df.groupby(["fecha", "proceso", "codigo_personal"], dropna=False)
          .agg(
              activo_inactivo=("activo_inactivo", "last"),
              horas_presenciales_dia=("horas_presenciales", "sum"),
              tallos_procesados_dia=("unidades_producidas", "sum"),
              horas_acumula=("horas_acumula", "max"),
          )
          .reset_index()
    )

    fact["tallos_h_persona_dia"] = fact["tallos_procesados_dia"] / fact["horas_presenciales_dia"]

    # Cap por proceso (1%–99%)
    # Cap por proceso (1%–99%)
def cap_percentiles(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    x = g["tallos_h_persona_dia"].astype(float).values
    x = x[~np.isnan(x)]
    if x.size < 5:
         return g

    p1 = float(np.nanpercentile(x, 1))
    p99 = float(np.nanpercentile(x, 99))
    g["tallos_h_persona_dia"] = g["tallos_h_persona_dia"].clip(p1, p99)
    return g

    fact = (
        fact.groupby("proceso", dropna=False, group_keys=False)
            .apply(cap_percentiles, include_groups=True)
            .reset_index(drop=True)
    )


    fact["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_fact = silver_dir / "fact_capacidad_tallos_proceso_hist.parquet"
    write_parquet(fact, out_fact)

    base = (
        fact.groupby("proceso", dropna=False)
            .agg(
                tallos_h_persona_mediana=("tallos_h_persona_dia", "median"),
                tallos_h_persona_p25=("tallos_h_persona_dia", lambda s: float(np.nanpercentile(s, 25))),
                tallos_h_persona_p75=("tallos_h_persona_dia", lambda s: float(np.nanpercentile(s, 75))),
                n_registros=("tallos_h_persona_dia", "size"),
            )
            .reset_index()
    )
    base["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_base = silver_dir / "dim_capacidad_baseline_tallos_proceso.parquet"
    write_parquet(base, out_base)

    print(f"OK: fact_capacidad_tallos_proceso_hist={len(fact)} -> {out_fact}")
    print("Baseline tallos/h/persona:\n", base.to_string(index=False))


if __name__ == "__main__":
    main()
