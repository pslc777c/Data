from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_FACTORS = EVAL_DIR / "backtest_factor_ml2_tallos_curve_dia.parquet"
IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"

OUT_GLOBAL = EVAL_DIR / "ml2_tallos_curve_eval_global.parquet"
OUT_GROUP = EVAL_DIR / "ml2_tallos_curve_eval_by_group.parquet"
OUT_RATIO = EVAL_DIR / "ml2_tallos_curve_eval_ratio_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _mae(err: pd.Series) -> float:
    err = pd.to_numeric(err, errors="coerce")
    return float(np.nanmean(np.abs(err))) if len(err) else float("nan")


def _wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0.0)
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    y_pred = pd.to_numeric(y_pred, errors="coerce").fillna(0.0)
    denom = np.abs(y_true) + np.abs(y_pred)
    m = denom > 0
    if not m.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_true[m] - y_pred[m]) / denom[m]))


def main() -> None:
    df = read_parquet(IN_FACTORS).copy()

    # Attach grouping fields if missing (estado / tipo_sp / area)
    ciclo = read_parquet(IN_CICLO).copy()
    if "ciclo_id" not in ciclo.columns:
        raise KeyError("fact_ciclo_maestro debe tener ciclo_id")

    # Normalizar estado si viene
    if "estado" in ciclo.columns:
        ciclo["estado"] = ciclo["estado"].astype(str).str.upper().str.strip()

    ciclo_cols = ["ciclo_id"]
    for c in ["estado", "tipo_sp", "area"]:
        if c in ciclo.columns:
            ciclo_cols.append(c)

    df = df.merge(ciclo[ciclo_cols].drop_duplicates("ciclo_id"), on="ciclo_id", how="left")

    # Core
    df["tallos_real_dia"] = pd.to_numeric(df["tallos_real_dia"], errors="coerce").fillna(0.0)
    df["tallos_pred_ml1_dia"] = pd.to_numeric(df["tallos_pred_ml1_dia"], errors="coerce").fillna(0.0)
    df["tallos_final_ml2_dia"] = pd.to_numeric(df["tallos_final_ml2_dia"], errors="coerce").fillna(0.0)

    # Solo rows con señal (evita universo vacío)
    df = df.loc[(df["tallos_real_dia"] > 0) | (df["tallos_pred_ml1_dia"] > 0), :].copy()

    # Errors
    df["err_ml1"] = df["tallos_real_dia"] - df["tallos_pred_ml1_dia"]
    df["err_ml2"] = df["tallos_real_dia"] - df["tallos_final_ml2_dia"]

    # Global KPIs
    out_g = pd.DataFrame([{
        "n_rows": int(len(df)),
        "n_cycles": int(df["ciclo_id"].nunique()),
        "mae_ml1": _mae(df["err_ml1"]),
        "mae_ml2": _mae(df["err_ml2"]),
        "wape_ml1": _wape(df["tallos_real_dia"], df["tallos_pred_ml1_dia"]),
        "wape_ml2": _wape(df["tallos_real_dia"], df["tallos_final_ml2_dia"]),
        "smape_ml1": _smape(df["tallos_real_dia"], df["tallos_pred_ml1_dia"]),
        "smape_ml2": _smape(df["tallos_real_dia"], df["tallos_final_ml2_dia"]),
        "improvement_abs_mae": _mae(df["err_ml1"]) - _mae(df["err_ml2"]),
        "improvement_abs_wape": _wape(df["tallos_real_dia"], df["tallos_pred_ml1_dia"]) - _wape(df["tallos_real_dia"], df["tallos_final_ml2_dia"]),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # Ratio distribution sanity
    r = pd.to_numeric(df["pred_ratio"], errors="coerce")
    out_r = pd.DataFrame([{
        "n": int(r.notna().sum()),
        "ratio_min": float(np.nanmin(r)) if len(r) else np.nan,
        "ratio_p05": float(np.nanpercentile(r, 5)) if len(r) else np.nan,
        "ratio_p25": float(np.nanpercentile(r, 25)) if len(r) else np.nan,
        "ratio_median": float(np.nanmedian(r)) if len(r) else np.nan,
        "ratio_p75": float(np.nanpercentile(r, 75)) if len(r) else np.nan,
        "ratio_p95": float(np.nanpercentile(r, 95)) if len(r) else np.nan,
        "ratio_max": float(np.nanmax(r)) if len(r) else np.nan,
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # Grouping: prefer estado, else tipo_sp, else area
    if "estado" in df.columns and df["estado"].notna().any():
        group_col = "estado"
    elif "tipo_sp" in df.columns and df["tipo_sp"].notna().any():
        group_col = "tipo_sp"
    else:
        group_col = "area" if "area" in df.columns else None

    rows = []
    if group_col:
        for k, g in df.groupby(group_col):
            rows.append({
                "group_col": group_col,
                "group_val": str(k),
                "n_rows": int(len(g)),
                "n_cycles": int(g["ciclo_id"].nunique()),
                "mae_ml1": _mae(g["err_ml1"]),
                "mae_ml2": _mae(g["err_ml2"]),
                "wape_ml1": _wape(g["tallos_real_dia"], g["tallos_pred_ml1_dia"]),
                "wape_ml2": _wape(g["tallos_real_dia"], g["tallos_final_ml2_dia"]),
                "improvement_abs_mae": _mae(g["err_ml1"]) - _mae(g["err_ml2"]),
                "improvement_abs_wape": _wape(g["tallos_real_dia"], g["tallos_pred_ml1_dia"]) - _wape(g["tallos_real_dia"], g["tallos_final_ml2_dia"]),
            })
    out_grp = pd.DataFrame(rows)

    # Write
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_grp, OUT_GROUP)
    write_parquet(out_r, OUT_RATIO)

    print(f"[OK] Wrote global: {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote group : {OUT_GROUP} (group_col={group_col}) rows={len(out_grp)}")
    if not out_grp.empty:
        print(out_grp.sort_values("improvement_abs_wape", ascending=False).head(15).to_string(index=False))
    print(f"\n[OK] Wrote ratio : {OUT_RATIO}")
    print(out_r.to_string(index=False))


if __name__ == "__main__":
    main()
