from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
EVAL = DATA / "eval" / "ml2"
SILVER = DATA / "silver"

IN_FINAL = DATA / "gold" / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"
IN_REAL = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_GLOBAL = EVAL / "ml2_desp_poscosecha_eval_global.parquet"
OUT_BY_DEST = EVAL / "ml2_desp_poscosecha_eval_by_destino.parquet"
OUT_DIST = EVAL / "ml2_desp_poscosecha_eval_ratio_dist.parquet"


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

    # peso
    # peso (tallos). Si no existe tallos/tallos_w en el universo, usar peso=1.0 por fila
    if "tallos_w" in df.columns:
        w = pd.to_numeric(df["tallos_w"], errors="coerce")
    elif "tallos" in df.columns:
        w = pd.to_numeric(df["tallos"], errors="coerce")
    else:
        w = pd.Series(1.0, index=df.index)

    df["w"] = w.fillna(0.0)


    # Real
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
                "p05_ratio_ml2": float(np.nanpercentile(d["ratio_ml2"].values, 5)),
                "p95_ratio_ml2": float(np.nanpercentile(d["ratio_ml2"].values, 95)),
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

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_d, OUT_BY_DEST)
    write_parquet(dist, OUT_DIST)

    print(f"[OK] Wrote global : {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote destino: {OUT_BY_DEST} rows={len(out_d)}")
    print(out_d.to_string(index=False))
    print(f"\n[OK] Wrote dist   : {OUT_DIST}")
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
