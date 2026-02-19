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
GOLD = DATA / "gold"

IN_FINAL = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino_final.parquet"
IN_REAL = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

OUT_GLOBAL = EVAL / "ml2_hidr_poscosecha_eval_global.parquet"
OUT_BY_DESTINO = EVAL / "ml2_hidr_poscosecha_eval_by_destino.parquet"
OUT_BY_GRADO = EVAL / "ml2_hidr_poscosecha_eval_by_grado.parquet"
OUT_RATIO_DIST = EVAL / "ml2_hidr_poscosecha_eval_ratio_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _factor_from_hidr_pct(hidr_pct: pd.Series) -> pd.Series:
    x = pd.to_numeric(hidr_pct, errors="coerce")
    return np.where(x.isna(), np.nan, np.where(x > 3.5, 1.0 + x / 100.0, x)).astype(float)


def _mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else np.nan


def main() -> None:
    df = read_parquet(IN_FINAL).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"grado", "destino", "factor_hidr_ml1", "factor_hidr_final"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Final sin columnas: {sorted(miss)}")

    # fecha_post_pred usada
    fpp = None
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml2", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            fpp = c
            break
    if fpp is None:
        raise KeyError("No encuentro fecha_post_pred en final.")

    df[fpp] = _to_date(df[fpp])
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = _canon_int(df["grado"])
    df["factor_hidr_ml1"] = pd.to_numeric(df["factor_hidr_ml1"], errors="coerce")
    df["factor_hidr_final"] = pd.to_numeric(df["factor_hidr_final"], errors="coerce")

    # real
    real = read_parquet(IN_REAL).copy()
    real.columns = [str(c).strip() for c in real.columns]
    if "hidr_pct" in real.columns:
        real["factor_hidr_real"] = _factor_from_hidr_pct(real["hidr_pct"])
    else:
        pb = pd.to_numeric(real.get("peso_base_g"), errors="coerce")
        pp = pd.to_numeric(real.get("peso_post_g"), errors="coerce")
        real["factor_hidr_real"] = np.where(pb > 0, pp / pb, np.nan)

    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["grado"] = _canon_int(real["grado"])

    # agregación real por llave
    if "tallos" in real.columns:
        real["tallos"] = pd.to_numeric(real["tallos"], errors="coerce").fillna(0.0)
        g = real.groupby(["fecha_post", "grado", "destino"], dropna=False)
        real2 = g.apply(
            lambda x: pd.Series({
                "factor_hidr_real": float(np.nansum(x["factor_hidr_real"] * x["tallos"]) / np.nansum(x["tallos"])) if np.nansum(x["tallos"]) > 0 else float(np.nanmedian(x["factor_hidr_real"])),
                "tallos_real_sum": float(np.nansum(x["tallos"])),
            })
        ).reset_index()
    else:
        real2 = (
            real.groupby(["fecha_post", "grado", "destino"], dropna=False, as_index=False)
                .agg(factor_hidr_real=("factor_hidr_real", "median"))
        )
        real2["tallos_real_sum"] = np.nan

    mrg = df.merge(
        real2,
        left_on=[fpp, "grado", "destino"],
        right_on=["fecha_post", "grado", "destino"],
        how="left",
    ).drop(columns=["fecha_post"], errors="ignore")

    m = mrg["factor_hidr_real"].notna() & mrg["factor_hidr_ml1"].notna() & mrg["factor_hidr_final"].notna()
    d = mrg.loc[m].copy()
    if d.empty:
        raise ValueError("No hay match con real para evaluar hidratación.")

    eps = 1e-9
    d["ratio_ml1"] = d["factor_hidr_real"] / d["factor_hidr_ml1"].clip(lower=eps)
    d["ratio_ml2"] = d["factor_hidr_real"] / d["factor_hidr_final"].clip(lower=eps)

    d["log_ratio_ml1"] = np.log(d["ratio_ml1"].clip(lower=eps))
    d["log_ratio_ml2"] = np.log(d["ratio_ml2"].clip(lower=eps))

    # peso por tallos
    w = pd.to_numeric(d.get("tallos_real_sum"), errors="coerce").fillna(1.0).clip(lower=0.0)
    if (w.sum() <= 0) or (w.isna().all()):
        w = pd.Series(1.0, index=d.index)

    def wmae(x):
        x = pd.to_numeric(x, errors="coerce")
        mm = x.notna()
        if not mm.any():
            return np.nan
        return float(np.sum(np.abs(x[mm]) * w[mm]) / np.sum(w[mm]))

    out_g = pd.DataFrame([{
        "n_rows": int(len(d)),
        "n_dates": int(d[fpp].nunique()),
        "mae_log_ml1_w": wmae(d["log_ratio_ml1"]),
        "mae_log_ml2_w": wmae(d["log_ratio_ml2"]),
        "median_ratio_ml1": float(np.nanmedian(d["ratio_ml1"])),
        "median_ratio_ml2": float(np.nanmedian(d["ratio_ml2"])),
        "p05_ratio_ml2": float(np.nanpercentile(d["ratio_ml2"], 5)),
        "p95_ratio_ml2": float(np.nanpercentile(d["ratio_ml2"], 95)),
        "improvement_abs_mae_log_w": wmae(d["log_ratio_ml1"]) - wmae(d["log_ratio_ml2"]),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # by destino
    rows = []
    for dest, g in d.groupby("destino"):
        ww = pd.to_numeric(g.get("tallos_real_sum"), errors="coerce").fillna(1.0).clip(lower=0.0)
        if ww.sum() <= 0:
            ww = pd.Series(1.0, index=g.index)

        def _wmae(v):
            v = pd.to_numeric(v, errors="coerce")
            mm = v.notna()
            return float(np.sum(np.abs(v[mm]) * ww[mm]) / np.sum(ww[mm])) if mm.any() else np.nan

        rows.append({
            "destino": str(dest),
            "n_rows": int(len(g)),
            "mae_log_ml1_w": _wmae(g["log_ratio_ml1"]),
            "mae_log_ml2_w": _wmae(g["log_ratio_ml2"]),
            "median_ratio_ml2": float(np.nanmedian(g["ratio_ml2"])),
            "improvement_abs_mae_log_w": _wmae(g["log_ratio_ml1"]) - _wmae(g["log_ratio_ml2"]),
        })
    out_bd = pd.DataFrame(rows).sort_values("improvement_abs_mae_log_w", ascending=False)

    # by grado
    rows = []
    for gr, g in d.groupby("grado"):
        ww = pd.to_numeric(g.get("tallos_real_sum"), errors="coerce").fillna(1.0).clip(lower=0.0)
        if ww.sum() <= 0:
            ww = pd.Series(1.0, index=g.index)

        def _wmae(v):
            v = pd.to_numeric(v, errors="coerce")
            mm = v.notna()
            return float(np.sum(np.abs(v[mm]) * ww[mm]) / np.sum(ww[mm])) if mm.any() else np.nan

        rows.append({
            "grado": int(gr) if pd.notna(gr) else None,
            "n_rows": int(len(g)),
            "mae_log_ml1_w": _wmae(g["log_ratio_ml1"]),
            "mae_log_ml2_w": _wmae(g["log_ratio_ml2"]),
            "median_ratio_ml2": float(np.nanmedian(g["ratio_ml2"])),
            "improvement_abs_mae_log_w": _wmae(g["log_ratio_ml1"]) - _wmae(g["log_ratio_ml2"]),
        })
    out_bg = pd.DataFrame(rows).sort_values("improvement_abs_mae_log_w", ascending=False)

    # ratio dist ml2
    out_dist = pd.DataFrame([{
        "n": int(len(d)),
        "ratio_min": float(np.nanmin(d["ratio_ml2"])),
        "ratio_p05": float(np.nanpercentile(d["ratio_ml2"], 5)),
        "ratio_p25": float(np.nanpercentile(d["ratio_ml2"], 25)),
        "ratio_median": float(np.nanmedian(d["ratio_ml2"])),
        "ratio_p75": float(np.nanpercentile(d["ratio_ml2"], 75)),
        "ratio_p95": float(np.nanpercentile(d["ratio_ml2"], 95)),
        "ratio_max": float(np.nanmax(d["ratio_ml2"])),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_bd, OUT_BY_DESTINO)
    write_parquet(out_bg, OUT_BY_GRADO)
    write_parquet(out_dist, OUT_RATIO_DIST)

    print(f"[OK] Wrote global : {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote destino: {OUT_BY_DESTINO} rows={len(out_bd)}")
    print(out_bd.to_string(index=False))
    print(f"\n[OK] Wrote grado  : {OUT_BY_GRADO} rows={len(out_bg)}")
    print(out_bg.head(13).to_string(index=False))
    print(f"\n[OK] Wrote dist   : {OUT_RATIO_DIST}")
    print(out_dist.to_string(index=False))


if __name__ == "__main__":
    main()
