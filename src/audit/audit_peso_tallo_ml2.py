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
SILVER = DATA / "silver"

IN_FACT = EVAL / "backtest_factor_ml2_peso_tallo_grado_dia.parquet"
IN_TALLOS = SILVER / "fact_cosecha_real_grado_dia.parquet"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_KPI_GLOBAL = EVAL / f"audit_peso_tallo_ml2_kpi_global_{TS}.parquet"
OUT_DIST = EVAL / f"audit_peso_tallo_ml2_ratio_dist_{TS}.parquet"
OUT_BY_GRADO = EVAL / f"audit_peso_tallo_ml2_kpi_by_grado_{TS}.parquet"
OUT_EXAMPLES = EVAL / f"audit_peso_tallo_ml2_examples_10x2_{TS}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else np.nan


def wape_weighted(err_abs: pd.Series, w: pd.Series) -> float:
    err_abs = pd.to_numeric(err_abs, errors="coerce").fillna(0.0)
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return np.nan
    return float(np.sum(err_abs * w) / denom)


def main() -> None:
    df = read_parquet(IN_FACT).copy()

    # Tallos reales para ponderar (impacto en kg y wape ponderado)
    tl = read_parquet(IN_TALLOS).copy()
    tl["fecha"] = _to_date(tl["fecha"])
    tl["grado"] = _canon_str(tl["grado"]) if "grado" in tl.columns else tl["grado"]

    if "bloque_base" in tl.columns:
        tl["bloque_base"] = _canon_str(tl["bloque_base"])
    elif "bloque_padre" in tl.columns:
        tl["bloque_base"] = _canon_str(tl["bloque_padre"])
    elif "bloque" in tl.columns:
        tl["bloque_base"] = _canon_str(tl["bloque"])
    else:
        raise KeyError("fact_cosecha_real_grado_dia no tiene bloque_base/bloque_padre/bloque")

    tl["tallos_real"] = pd.to_numeric(tl["tallos_real"], errors="coerce").fillna(0.0)
    tl = tl.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))

    # Canon fact
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_str(df["bloque_base"])
    df["grado"] = _canon_str(df["grado"])

    for c in ["peso_tallo_ml1_g", "peso_tallo_real_g", "peso_tallo_final_g", "pred_ratio_peso"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Merge tallos weights
    df = df.merge(tl, on=["fecha", "bloque_base", "grado"], how="left")
    df["tallos_real"] = pd.to_numeric(df["tallos_real"], errors="coerce").fillna(0.0)

    # Keep only rows where we have real
    df = df[df["peso_tallo_real_g"].notna()].copy()

    # Errors (g/tallo)
    df["err_ml1_g"] = df["peso_tallo_real_g"] - df["peso_tallo_ml1_g"]
    df["err_ml2_g"] = df["peso_tallo_real_g"] - df["peso_tallo_final_g"]

    df["abs_err_ml1_g"] = df["err_ml1_g"].abs()
    df["abs_err_ml2_g"] = df["err_ml2_g"].abs()

    # Equivalent absolute kg error = abs_err_g * tallos / 1000
    df["abs_err_ml1_kg_equiv"] = (df["abs_err_ml1_g"] * df["tallos_real"]) / 1000.0
    df["abs_err_ml2_kg_equiv"] = (df["abs_err_ml2_g"] * df["tallos_real"]) / 1000.0

    # === KPI GLOBAL ===
    kpi_g = pd.DataFrame([{
        "n_rows": int(len(df)),
        "n_cycles": int(df["ciclo_id"].nunique()) if "ciclo_id" in df.columns else np.nan,
        "mae_ml1_g": mae(df["err_ml1_g"]),
        "mae_ml2_g": mae(df["err_ml2_g"]),
        "wape_wt_ml1_g": wape_weighted(df["abs_err_ml1_g"], df["tallos_real"]),
        "wape_wt_ml2_g": wape_weighted(df["abs_err_ml2_g"], df["tallos_real"]),
        "kg_abs_err_ml1": float(df["abs_err_ml1_kg_equiv"].sum()),
        "kg_abs_err_ml2": float(df["abs_err_ml2_kg_equiv"].sum()),
        "improvement_abs_mae_g": mae(df["err_ml1_g"]) - mae(df["err_ml2_g"]),
        "improvement_abs_wape_wt_g": wape_weighted(df["abs_err_ml1_g"], df["tallos_real"]) - wape_weighted(df["abs_err_ml2_g"], df["tallos_real"]),
        "improvement_abs_kg": float(df["abs_err_ml1_kg_equiv"].sum() - df["abs_err_ml2_kg_equiv"].sum()),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])
    write_parquet(kpi_g, OUT_KPI_GLOBAL)

    # === DISTRIBUCIÓN DE RATIOS ===
    r = pd.to_numeric(df["pred_ratio_peso"], errors="coerce")
    dist = pd.DataFrame([{
        "n": int(r.notna().sum()),
        "ratio_min": float(np.nanmin(r)),
        "ratio_p05": float(np.nanpercentile(r, 5)),
        "ratio_p25": float(np.nanpercentile(r, 25)),
        "ratio_median": float(np.nanmedian(r)),
        "ratio_p75": float(np.nanpercentile(r, 75)),
        "ratio_p95": float(np.nanpercentile(r, 95)),
        "ratio_max": float(np.nanmax(r)),
    }])
    write_parquet(dist, OUT_DIST)

    # === KPI POR GRADO (para ver sesgos por calidad) ===
    rows = []
    for grado, g in df.groupby("grado"):
        rows.append({
            "grado": str(grado),
            "n_rows": int(len(g)),
            "mae_ml1_g": mae(g["err_ml1_g"]),
            "mae_ml2_g": mae(g["err_ml2_g"]),
            "wape_wt_ml1_g": wape_weighted(g["abs_err_ml1_g"], g["tallos_real"]),
            "wape_wt_ml2_g": wape_weighted(g["abs_err_ml2_g"], g["tallos_real"]),
            "kg_abs_err_ml1": float(g["abs_err_ml1_kg_equiv"].sum()),
            "kg_abs_err_ml2": float(g["abs_err_ml2_kg_equiv"].sum()),
            "improvement_abs_kg": float(g["abs_err_ml1_kg_equiv"].sum() - g["abs_err_ml2_kg_equiv"].sum()),
        })
    by_grado = pd.DataFrame(rows).sort_values("improvement_abs_kg", ascending=False)
    write_parquet(by_grado, OUT_BY_GRADO)

    # === EJEMPLOS 10x2 (mejora vs empeora) por ciclo usando impacto kg ===
    if "ciclo_id" in df.columns:
        by_cycle = (
            df.groupby("ciclo_id", as_index=False)
              .agg(
                  kg_abs_err_ml1=("abs_err_ml1_kg_equiv", "sum"),
                  kg_abs_err_ml2=("abs_err_ml2_kg_equiv", "sum"),
                  n_rows=("fecha", "count"),
              )
        )
        by_cycle["improvement_kg"] = by_cycle["kg_abs_err_ml1"] - by_cycle["kg_abs_err_ml2"]

        top_good = by_cycle.sort_values("improvement_kg", ascending=False).head(10).assign(sample_group="TOP_IMPROVE_10")
        top_bad = by_cycle.sort_values("improvement_kg", ascending=True).head(10).assign(sample_group="TOP_WORSE_10")
        ex = pd.concat([top_good, top_bad], ignore_index=True)

        # attach some context (bloque_base, variedad_canon) if available
        ctx_cols = [c for c in ["ciclo_id", "bloque_base", "variedad_canon", "area", "tipo_sp", "estado"] if c in df.columns]
        if ctx_cols:
            ctx = df[ctx_cols].drop_duplicates("ciclo_id")
            ex = ex.merge(ctx, on="ciclo_id", how="left")

        write_parquet(ex, OUT_EXAMPLES)
    else:
        ex = pd.DataFrame()
        write_parquet(ex, OUT_EXAMPLES)

    # === PRINT ===
    print("\n=== ML2 PESO POR TALLO AUDIT ===")
    print(f"KPI global parquet : {OUT_KPI_GLOBAL}")
    print(f"Ratio dist parquet : {OUT_DIST}")
    print(f"KPI por grado      : {OUT_BY_GRADO}")
    print(f"Examples 10x2      : {OUT_EXAMPLES}")
    print("\n--- KPI GLOBAL ---")
    print(kpi_g.to_string(index=False))
    print("\n--- RATIO DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI POR GRADO (top 10 por mejora kg) ---")
    print(by_grado.head(10).to_string(index=False))
    if not ex.empty:
        print("\n--- EXAMPLES 10x2 ---")
        print(ex.to_string(index=False))


if __name__ == "__main__":
    main()
