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
GOLD = DATA / "gold"

IN_FINAL = GOLD / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"
IN_REAL = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

TS = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_KPI_G = EVAL / f"audit_desp_poscosecha_ml2_kpi_global_{TS}.parquet"
OUT_KPI_D = EVAL / f"audit_desp_poscosecha_ml2_kpi_by_destino_{TS}.parquet"
OUT_DIST = EVAL / f"audit_desp_poscosecha_ml2_ratio_dist_{TS}.parquet"
OUT_EX = EVAL / f"audit_desp_poscosecha_ml2_examples_10x2_{TS}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def wmae_log_ratio(log_ratio: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(log_ratio, errors="coerce")
    ww = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = x.notna() & (ww > 0)
    if not bool(m.any()):
        return float("nan")
    denom = float(ww[m].sum())
    if denom <= 0:
        return float(np.nanmean(np.abs(x[m].values)))
    return float((np.abs(x[m].values) * ww[m].values).sum() / denom)


def main() -> None:
    print("\n=== ML2 DESP POSCOSECHA AUDIT ===")

    df = read_parquet(IN_FINAL).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha_post_pred_used", "destino", "factor_desp_ml1", "factor_desp_final"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Final sin columnas: {sorted(miss)}")

    df["fecha_post_pred_used"] = _to_date(df["fecha_post_pred_used"])
    df["destino"] = _canon_str(df["destino"])
    df["factor_desp_ml1"] = pd.to_numeric(df["factor_desp_ml1"], errors="coerce")
    df["factor_desp_final"] = pd.to_numeric(df["factor_desp_final"], errors="coerce")

    if "tallos_w" in df.columns:
        w = pd.to_numeric(df["tallos_w"], errors="coerce")
    elif "tallos" in df.columns:
        w = pd.to_numeric(df["tallos"], errors="coerce")
    else:
        w = pd.Series(1.0, index=df.index)

    df["w"] = w.fillna(0.0)


    real = read_parquet(IN_REAL).copy()
    real.columns = [str(c).strip() for c in real.columns]

    need_r = {"fecha_post", "destino", "factor_desp"}
    miss_r = need_r - set(real.columns)
    if miss_r:
        raise ValueError(f"Real sin columnas: {sorted(miss_r)}")

    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["factor_desp_real"] = pd.to_numeric(real["factor_desp"], errors="coerce")

    real2 = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
        .agg(factor_desp_real=("factor_desp_real", "median"))
    )

    x = df.merge(
        real2,
        left_on=["fecha_post_pred_used", "destino"],
        right_on=["fecha_post", "destino"],
        how="left",
    )

    eps = 1e-12
    x["ratio_ml1"] = (x["factor_desp_real"] + eps) / (x["factor_desp_ml1"] + eps)
    x["ratio_ml2"] = (x["factor_desp_real"] + eps) / (x["factor_desp_final"] + eps)
    x["log_ratio_ml1"] = np.log(x["ratio_ml1"])
    x["log_ratio_ml2"] = np.log(x["ratio_ml2"])

    m = x["factor_desp_real"].notna() & x["factor_desp_ml1"].notna() & x["factor_desp_final"].notna()
    d = x.loc[m].copy()

    out_g = pd.DataFrame(
        [
            {
                "n_rows": int(len(d)),
                "n_dates": int(d["fecha_post_pred_used"].nunique()),
                "mae_log_ml1_w": wmae_log_ratio(d["log_ratio_ml1"], d["w"]),
                "mae_log_ml2_w": wmae_log_ratio(d["log_ratio_ml2"], d["w"]),
                "median_ratio_ml1": float(np.nanmedian(d["ratio_ml1"].values)),
                "median_ratio_ml2": float(np.nanmedian(d["ratio_ml2"].values)),
                "improvement_abs_mae_log_w": wmae_log_ratio(d["log_ratio_ml1"], d["w"]) - wmae_log_ratio(d["log_ratio_ml2"], d["w"]),
                "created_at": pd.Timestamp(datetime.now()).normalize(),
            }
        ]
    )

    rows = []
    for dest, g in d.groupby("destino"):
        rows.append(
            {
                "destino": str(dest),
                "n_rows": int(len(g)),
                "n_dates": int(g["fecha_post_pred_used"].nunique()),
                "mae_log_ml1_w": wmae_log_ratio(g["log_ratio_ml1"], g["w"]),
                "mae_log_ml2_w": wmae_log_ratio(g["log_ratio_ml2"], g["w"]),
                "median_ratio_ml2": float(np.nanmedian(g["ratio_ml2"].values)),
                "improvement_abs_mae_log_w": wmae_log_ratio(g["log_ratio_ml1"], g["w"]) - wmae_log_ratio(g["log_ratio_ml2"], g["w"]),
            }
        )
    out_d = pd.DataFrame(rows).sort_values("improvement_abs_mae_log_w", ascending=False)

    dist = pd.DataFrame(
        [
            {
                "n": int(len(d)),
                "ratio_min": float(np.nanmin(d["ratio_ml2"].values)),
                "ratio_p05": float(np.nanpercentile(d["ratio_ml2"].values, 5)),
                "ratio_p25": float(np.nanpercentile(d["ratio_ml2"].values, 25)),
                "ratio_median": float(np.nanmedian(d["ratio_ml2"].values)),
                "ratio_p75": float(np.nanpercentile(d["ratio_ml2"].values, 75)),
                "ratio_p95": float(np.nanpercentile(d["ratio_ml2"].values, 95)),
                "ratio_max": float(np.nanmax(d["ratio_ml2"].values)),
                "created_at": pd.Timestamp(datetime.now()).normalize(),
            }
        ]
    )

    # Examples: score = abs(log_ml1) - abs(log_ml2), ponderado por w
    d["score"] = d["log_ratio_ml1"].abs() - d["log_ratio_ml2"].abs()
    d["wscore"] = d["score"] * d["w"].fillna(0.0)

    best = d.sort_values("wscore", ascending=False).head(10).copy()
    worst = d.sort_values("wscore", ascending=True).head(10).copy()

    best["sample_group"] = "TOP_IMPROVE_10"
    worst["sample_group"] = "TOP_WORSE_10"

    ex = pd.concat([best, worst], ignore_index=True)
    ex_out = ex[
        [
            "fecha",
            "fecha_post_pred_used",
            "destino",
            "w",
            "factor_desp_ml1",
            "factor_desp_final",
            "factor_desp_real",
            "ratio_ml1",
            "ratio_ml2",
            "log_ratio_ml1",
            "log_ratio_ml2",
            "score",
            "wscore",
            "sample_group",
        ]
    ].copy()

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_KPI_G)
    write_parquet(out_d, OUT_KPI_D)
    write_parquet(dist, OUT_DIST)
    write_parquet(ex_out, OUT_EX)

    print(f"KPI global parquet : {OUT_KPI_G}")
    print(f"KPI por destino    : {OUT_KPI_D}")
    print(f"Ratio dist parquet : {OUT_DIST}")
    print(f"Examples 10x2      : {OUT_EX}")

    print("\n--- KPI GLOBAL ---")
    print(out_g.to_string(index=False))
    print("\n--- RATIO DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI POR DESTINO ---")
    print(out_d.to_string(index=False))
    print("\n--- EXAMPLES 10x2 ---")
    print(ex_out.to_string(index=False))


if __name__ == "__main__":
    main()
