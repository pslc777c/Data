from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
EVAL = DATA / "eval" / "ml2"

IN_GLOBAL = EVAL / "ml2_dh_poscosecha_eval_global.parquet"
IN_BY_DEST = EVAL / "ml2_dh_poscosecha_eval_by_destino.parquet"
IN_BY_GR = EVAL / "ml2_dh_poscosecha_eval_by_grado.parquet"
IN_DIST = EVAL / "ml2_dh_poscosecha_eval_delta_dist.parquet"

# Para ejemplos: usamos el backtest factor + KPIs recomputados localmente
IN_FACTOR = EVAL / "backtest_factor_ml2_dh_poscosecha.parquet"
IN_REAL = ROOT / "data" / "silver" / "fact_hidratacion_real_post_grado_destino.parquet"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_KPI_GLOBAL = EVAL / f"audit_dh_poscosecha_ml2_kpi_global_{ts}.parquet"
OUT_KPI_BY_DEST = EVAL / f"audit_dh_poscosecha_ml2_kpi_by_destino_{ts}.parquet"
OUT_KPI_BY_GR = EVAL / f"audit_dh_poscosecha_ml2_kpi_by_grado_{ts}.parquet"
OUT_DIST = EVAL / f"audit_dh_poscosecha_ml2_delta_dist_{ts}.parquet"
OUT_EX = EVAL / f"audit_dh_poscosecha_ml2_examples_10x2_{ts}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    return pd.Series(s).astype(str).str.upper().str.strip()


def _canon_int(s) -> pd.Series:
    """
    Compat: algunos entornos (pandas/numpy) no entienden dtype 'Int64'.
    Devolvemos float (con NaN) para poder usar notna() y cálculos.
    """
    if s is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(pd.Series(s), errors="coerce").astype(float)


def main() -> None:
    print("\n=== ML2 DH POSCOSECHA AUDIT ===")

    g = read_parquet(IN_GLOBAL).copy()
    d = read_parquet(IN_BY_DEST).copy()
    gr = read_parquet(IN_BY_GR).copy()
    dist = read_parquet(IN_DIST).copy()

    # Examples
    fac = read_parquet(IN_FACTOR).copy()
    real = read_parquet(IN_REAL).copy()
    fac.columns = [str(c).strip() for c in fac.columns]
    real.columns = [str(c).strip() for c in real.columns]

    fac["fecha"] = _to_date(fac["fecha"])
    fac["destino"] = _canon_str(fac["destino"])
    fac["grado"] = _canon_int(fac["grado"])
    fac["dh_ml1"] = _canon_int(fac.get("dh_ml1"))
    fac["dh_dias_final"] = _canon_int(fac.get("dh_dias_final"))

    real["fecha_cosecha"] = _to_date(real["fecha_cosecha"])
    real["destino"] = _canon_str(real["destino"])
    real["grado"] = _canon_int(real["grado"])
    real["tallos"] = pd.to_numeric(real.get("tallos"), errors="coerce").fillna(0.0).astype(float)
    real["dh_dias"] = _canon_int(real.get("dh_dias"))

    real_g = (
        real.groupby(["fecha_cosecha", "grado", "destino"], dropna=False, as_index=False)
        .agg(dh_real=("dh_dias", "median"), tallos=("tallos", "sum"))
    )

    ex = fac.merge(
        real_g,
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    # asegurar columnas numéricas
    ex["dh_real"] = _canon_int(ex.get("dh_real"))
    ex["dh_ml1"] = _canon_int(ex.get("dh_ml1"))
    ex["dh_dias_final"] = _canon_int(ex.get("dh_dias_final"))

    ex = ex[ex["dh_real"].notna() & ex["dh_ml1"].notna() & ex["dh_dias_final"].notna()].copy()

    ex["err_ml1_days"] = (ex["dh_real"] - ex["dh_ml1"]).astype(float)
    ex["err_ml2_days"] = (ex["dh_real"] - ex["dh_dias_final"]).astype(float)
    ex["abs_err_ml1"] = ex["err_ml1_days"].abs()
    ex["abs_err_ml2"] = ex["err_ml2_days"].abs()
    ex["improvement_abs"] = ex["abs_err_ml1"] - ex["abs_err_ml2"]

    # filtrar tallos>0 para evitar “TOP_BEST” falsos
    ex["tallos"] = pd.to_numeric(ex.get("tallos"), errors="coerce").fillna(0.0).astype(float)
    ex = ex[ex["tallos"] > 0].copy()

    # Score ponderado por impacto
    ex["wscore"] = ex["improvement_abs"] * ex["tallos"]

    # si no hay filas, evitar crash y aún así escribir outputs
    if len(ex) > 0:
        top = ex.sort_values("wscore", ascending=False).head(10).copy()
        top["sample_group"] = "TOP_IMPROVE_10"
        worst = ex.sort_values("wscore", ascending=True).head(10).copy()
        worst["sample_group"] = "TOP_WORSE_10"
        examples = pd.concat([top, worst], ignore_index=True)
    else:
        examples = pd.DataFrame(columns=list(ex.columns) + ["sample_group"])

    # Outputs
    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(g, OUT_KPI_GLOBAL)
    write_parquet(d, OUT_KPI_BY_DEST)
    write_parquet(gr, OUT_KPI_BY_GR)
    write_parquet(dist, OUT_DIST)
    write_parquet(examples, OUT_EX)

    print(f"KPI global parquet : {OUT_KPI_GLOBAL}")
    print(f"KPI por destino    : {OUT_KPI_BY_DEST}")
    print(f"KPI por grado      : {OUT_KPI_BY_GR}")
    print(f"Delta dist parquet : {OUT_DIST}")
    print(f"Examples 10x2      : {OUT_EX}")

    print("\n--- KPI GLOBAL ---")
    print(g.to_string(index=False))
    print("\n--- DELTA DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI DESTINO ---")
    print(d.to_string(index=False))
    print("\n--- KPI GRADO (head) ---")
    print(gr.head(10).to_string(index=False))
    print("\n--- EXAMPLES (10x2) ---")
    keep = [
        c
        for c in [
            "fecha",
            "grado",
            "destino",
            "tallos",
            "dh_ml1",
            "dh_dias_final",
            "dh_real",
            "err_ml1_days",
            "err_ml2_days",
            "improvement_abs",
            "wscore",
            "sample_group",
        ]
        if c in examples.columns
    ]
    if len(examples):
        print(examples[keep].to_string(index=False))
    else:
        print("[WARN] No examples rows (no overlap real vs pred).")


if __name__ == "__main__":
    main()