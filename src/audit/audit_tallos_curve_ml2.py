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

IN_FACT = EVAL / "backtest_factor_ml2_tallos_curve_dia.parquet"
IN_CICLO = SILVER / "fact_ciclo_maestro.parquet"

OUT_KPI_GLOBAL = EVAL / f"audit_tallos_curve_ml2_kpi_global_{datetime.now():%Y%m%d_%H%M%S}.parquet"
OUT_KPI_ESTADO = EVAL / f"audit_tallos_curve_ml2_kpi_by_estado_{datetime.now():%Y%m%d_%H%M%S}.parquet"
OUT_DIST = EVAL / f"audit_tallos_curve_ml2_adjust_dist_{datetime.now():%Y%m%d_%H%M%S}.parquet"
OUT_EXAMPLES = EVAL / f"audit_tallos_curve_ml2_examples_10x2_{datetime.now():%Y%m%d_%H%M%S}.parquet"


def mae(s: pd.Series) -> float:
    return float(np.nanmean(np.abs(s))) if len(s) else np.nan


def wape(y: pd.Series, yhat: pd.Series) -> float:
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else np.nan


def main() -> None:
    df = read_parquet(IN_FACT).copy()
    ciclo = read_parquet(IN_CICLO)[["ciclo_id", "estado"]].drop_duplicates("ciclo_id")

    ciclo["estado"] = ciclo["estado"].astype(str).str.upper().str.strip()
    df = df.merge(ciclo, on="ciclo_id", how="left")

    # Core numeric
    for c in ["tallos_real_dia", "tallos_pred_ml1_dia", "tallos_final_ml2_dia", "pred_ratio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df = df[(df["tallos_real_dia"] > 0) | (df["tallos_pred_ml1_dia"] > 0)].copy()

    # Errors
    df["err_ml1"] = df["tallos_real_dia"] - df["tallos_pred_ml1_dia"]
    df["err_ml2"] = df["tallos_real_dia"] - df["tallos_final_ml2_dia"]

    # === KPI GLOBAL ===
    kpi_g = pd.DataFrame([{
        "n_rows": len(df),
        "n_cycles": df["ciclo_id"].nunique(),
        "mae_ml1": mae(df["err_ml1"]),
        "mae_ml2": mae(df["err_ml2"]),
        "wape_ml1": wape(df["tallos_real_dia"], df["tallos_pred_ml1_dia"]),
        "wape_ml2": wape(df["tallos_real_dia"], df["tallos_final_ml2_dia"]),
        "improvement_abs_mae": mae(df["err_ml1"]) - mae(df["err_ml2"]),
        "improvement_abs_wape": wape(df["tallos_real_dia"], df["tallos_pred_ml1_dia"])
                                 - wape(df["tallos_real_dia"], df["tallos_final_ml2_dia"]),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])
    write_parquet(kpi_g, OUT_KPI_GLOBAL)

    # === DISTRIBUCIÓN AJUSTES ===
    r = df["pred_ratio"]
    dist = pd.DataFrame([{
        "n": len(r),
        "min": r.min(),
        "p05": np.percentile(r, 5),
        "p25": np.percentile(r, 25),
        "median": np.median(r),
        "p75": np.percentile(r, 75),
        "p95": np.percentile(r, 95),
        "max": r.max(),
        "pct_clip": float(((r <= np.exp(-1.5)) | (r >= np.exp(1.5))).mean()),
    }])
    write_parquet(dist, OUT_DIST)

    # === KPI POR ESTADO ===
    rows = []
    for estado, g in df.groupby("estado"):
        rows.append({
            "estado": estado,
            "n_rows": len(g),
            "n_cycles": g["ciclo_id"].nunique(),
            "mae_ml1": mae(g["err_ml1"]),
            "mae_ml2": mae(g["err_ml2"]),
            "wape_ml1": wape(g["tallos_real_dia"], g["tallos_pred_ml1_dia"]),
            "wape_ml2": wape(g["tallos_real_dia"], g["tallos_final_ml2_dia"]),
            "improvement_abs_wape": wape(g["tallos_real_dia"], g["tallos_pred_ml1_dia"])
                                     - wape(g["tallos_real_dia"], g["tallos_final_ml2_dia"]),
        })
    write_parquet(pd.DataFrame(rows), OUT_KPI_ESTADO)

    # === EJEMPLOS 10x2 ===
    by_cycle = (
        df.groupby("ciclo_id", as_index=False)
          .agg(
              estado=("estado", "first"),
              wape_ml1=("tallos_pred_ml1_dia", lambda x: wape(df.loc[x.index, "tallos_real_dia"], x)),
              wape_ml2=("tallos_final_ml2_dia", lambda x: wape(df.loc[x.index, "tallos_real_dia"], x)),
          )
    )
    by_cycle["improvement"] = by_cycle["wape_ml1"] - by_cycle["wape_ml2"]

    top = pd.concat([
        by_cycle.sort_values("improvement", ascending=False).head(10),
        by_cycle.sort_values("improvement", ascending=True).head(10),
    ])
    write_parquet(top, OUT_EXAMPLES)

    # === PRINT ===
    print("\n=== AUDIT ML2 – CURVA DIARIA TALL0S ===")
    print(f"KPI global      : {OUT_KPI_GLOBAL}")
    print(f"KPI por estado  : {OUT_KPI_ESTADO}")
    print(f"Dist ajustes    : {OUT_DIST}")
    print(f"Ejemplos 10x2   : {OUT_EXAMPLES}")
    print("\n--- KPI GLOBAL ---")
    print(kpi_g.to_string(index=False))
    print("\n--- DIST AJUSTES ---")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
