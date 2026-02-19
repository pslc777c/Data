from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"
IN_FACTOR = EVAL_DIR / "backtest_factor_ml2_harvest_horizon.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else float("nan")


def _bias(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x)) if len(x) else float("nan")


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")


def main() -> None:
    ciclo = read_parquet(IN_CICLO).copy()
    fact = read_parquet(IN_FACTOR).copy()

    # Normalizar fechas en ciclo (autoridad de lo real)
    # Nota: ciclo puede tener nombres distintos según tu versión, cubrimos ambos.
    c_ini = _pick_first_existing(ciclo, ["fecha_inicio_cosecha", "fecha_ini_cosecha"])
    c_fin = _pick_first_existing(ciclo, ["fecha_fin_cosecha", "fecha_fin"])

    ciclo[c_ini] = _to_date(ciclo[c_ini])
    ciclo[c_fin] = _to_date(ciclo[c_fin])

    # Para evitar sufijos _x/_y: tomamos de ciclo solo lo que necesitamos
    cols = ["ciclo_id", c_ini, c_fin]
    if "estado" in ciclo.columns:
        cols.insert(1, "estado")
    ciclo_small = ciclo[cols].copy()

    ciclo_small = ciclo_small.rename(columns={c_ini: "fecha_inicio_cosecha_real", c_fin: "fecha_fin_cosecha_real"})

    df = ciclo_small.merge(fact, on="ciclo_id", how="inner")
    # Estado: puede venir en ciclo o en factor. Normalizamos y garantizamos que exista.
    if "estado" not in df.columns:
        if "estado_x" in df.columns:
            df["estado"] = df["estado_x"]
        elif "estado_y" in df.columns:
            df["estado"] = df["estado_y"]
        elif "estado_ml1" in df.columns:
            df["estado"] = df["estado_ml1"]
        else:
            df["estado"] = "UNKNOWN"

    df["estado"] = df["estado"].astype(str).str.upper().str.strip()

    # Real duration
    df["n_harvest_days_real"] = (df["fecha_fin_cosecha_real"] - df["fecha_inicio_cosecha_real"]).dt.days + 1

    # Errors (duración)
    df["err_ml1_days"] = pd.to_numeric(df["n_harvest_days_real"], errors="coerce") - pd.to_numeric(df["n_harvest_days_pred"], errors="coerce")
    df["err_ml2_days"] = pd.to_numeric(df["n_harvest_days_real"], errors="coerce") - pd.to_numeric(df["n_harvest_days_final"], errors="coerce")

    df = df.loc[df["err_ml1_days"].notna() & df["err_ml2_days"].notna(), :].copy()

    # Paths outputs con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_kpi_global = EVAL_DIR / f"audit_harvest_horizon_ml2_kpi_global_{ts}.parquet"
    out_dist = EVAL_DIR / f"audit_harvest_horizon_ml2_adjust_dist_{ts}.parquet"
    out_by_estado = EVAL_DIR / f"audit_harvest_horizon_ml2_kpi_by_estado_{ts}.parquet"
    out_examples = EVAL_DIR / f"audit_harvest_horizon_ml2_examples_10x2_{ts}.parquet"

    # KPI GLOBAL
    kpi_global = pd.DataFrame([{
        "n": int(len(df)),
        "mae_ml1_days": _mae(df["err_ml1_days"]),
        "mae_ml2_days": _mae(df["err_ml2_days"]),
        "bias_ml1_days": _bias(df["err_ml1_days"]),
        "bias_ml2_days": _bias(df["err_ml2_days"]),
        "improvement_abs_days": (_mae(df["err_ml1_days"]) - _mae(df["err_ml2_days"])),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])
    write_parquet(kpi_global, out_kpi_global)

    # DIST AJUSTE
    adj = pd.to_numeric(df["pred_error_horizon_days"], errors="coerce")
    dist = pd.DataFrame([{
        "n_factor_rows": int(adj.notna().sum()),
        "adj_min": float(np.nanmin(adj)) if len(adj) else np.nan,
        "adj_p25": float(np.nanpercentile(adj, 25)) if len(adj) else np.nan,
        "adj_median": float(np.nanmedian(adj)) if len(adj) else np.nan,
        "adj_p75": float(np.nanpercentile(adj, 75)) if len(adj) else np.nan,
        "adj_max": float(np.nanmax(adj)) if len(adj) else np.nan,
        "pct_clip": float(np.nanmean((adj <= -14) | (adj >= 21))) if len(adj) else np.nan,
    }])
    write_parquet(dist, out_dist)

    # KPI POR ESTADO
    rows = []
    for estado, g in df.groupby("estado"):
        rows.append({
            "estado": estado,
            "n": int(len(g)),
            "mae_ml1_days": _mae(g["err_ml1_days"]),
            "mae_ml2_days": _mae(g["err_ml2_days"]),
            "bias_ml1_days": _bias(g["err_ml1_days"]),
            "bias_ml2_days": _bias(g["err_ml2_days"]),
            "improvement_abs_days": (_mae(g["err_ml1_days"]) - _mae(g["err_ml2_days"])),
        })
    by_estado = pd.DataFrame(rows).sort_values(["estado"])
    write_parquet(by_estado, out_by_estado)

    # EJEMPLOS 10x2: 10 CERRADO + 10 ACTIVO (si existen), priorizando mayor |err_ml1|
    df["abs_err_ml1"] = pd.to_numeric(df["err_ml1_days"], errors="coerce").abs()
    examples = (
        df.sort_values("abs_err_ml1", ascending=False)
          .groupby("estado", group_keys=False)
          .head(10)
          .copy()
    )
    write_parquet(examples, out_examples)

    print("\n=== ML2 HARVEST HORIZON AUDIT ===")
    print(f"KPI global parquet : {out_kpi_global}")
    print(f"Dist ajustes parquet: {out_dist}")
    print(f"KPI por estado      : {out_by_estado}")
    print(f"Examples 10x2       : {out_examples}")

    print("\n--- KPI GLOBAL ---")
    print(kpi_global.to_string(index=False))
    print("\n--- AJUSTE DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI POR ESTADO ---")
    print(by_estado.to_string(index=False))
    print("\n--- EXAMPLES (head) ---")
    cols_show = [c for c in [
        "ciclo_id", "estado",
        "fecha_inicio_cosecha_real", "fecha_fin_cosecha_real",
        "n_harvest_days_pred", "pred_error_horizon_days", "n_harvest_days_final",
        "err_ml1_days", "err_ml2_days",
        "ml1_version", "ml2_run_id",
    ] if c in examples.columns]
    print(examples[cols_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
