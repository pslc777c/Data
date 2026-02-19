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

IN_FINAL = GOLD / "pred_poscosecha_ml2_ajuste_grado_dia_bloque_destino_final.parquet"
IN_REAL_MA = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_G = EVAL / "ml2_ajuste_poscosecha_eval_global.parquet"
OUT_D = EVAL / "ml2_ajuste_poscosecha_eval_by_destino.parquet"
OUT_DIST = EVAL / "ml2_ajuste_poscosecha_eval_ratio_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro fecha_post_pred_* en final.")


def _resolve_ajuste_ml1(df: pd.DataFrame) -> str:
    for c in ["factor_ajuste_ml1", "ajuste_ml1", "factor_ajuste_seed", "factor_ajuste"]:
        if c in df.columns:
            return c
    raise KeyError("No encuentro ajuste ML1 en final.")


def _weight_series(df: pd.DataFrame) -> pd.Series:
    for c in ["tallos_w", "tallos", "tallos_total_ml2", "tallos_total"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(1.0, index=df.index, dtype="float64")


def _wmae_log_ratio(ratio: pd.Series, w: pd.Series) -> float:
    r = pd.to_numeric(ratio, errors="coerce")
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    lr = np.log(r.replace(0, np.nan))
    m = lr.notna() & np.isfinite(lr) & w.notna() & (w >= 0)
    if not bool(m.any()):
        return np.nan
    denom = float(w[m].sum())
    if denom <= 0:
        return np.nan
    return float((lr[m].abs() * w[m]).sum() / denom)


def main() -> None:
    df = read_parquet(IN_FINAL).copy()
    df.columns = [str(c).strip() for c in df.columns]

    df["fecha"] = _to_date(df["fecha"])
    df["destino"] = _canon_str(df["destino"])

    fecha_post_col = _resolve_fecha_post_pred(df)
    df[fecha_post_col] = _to_date(df[fecha_post_col])

    aj_ml1_col = _resolve_ajuste_ml1(df)
    df[aj_ml1_col] = pd.to_numeric(df[aj_ml1_col], errors="coerce")

    if "factor_ajuste_final" not in df.columns:
        raise KeyError("No encuentro factor_ajuste_final en pred_poscosecha_ml2_ajuste..._final.")

    df["factor_ajuste_final"] = pd.to_numeric(df["factor_ajuste_final"], errors="coerce")

    # real
    real = read_parquet(IN_REAL_MA).copy()
    real.columns = [str(c).strip() for c in real.columns]
    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])

    if "factor_ajuste" in real.columns:
        rc = "factor_ajuste"
    elif "ajuste" in real.columns:
        rc = "ajuste"
    else:
        raise ValueError("Real MA no trae factor_ajuste ni ajuste.")
    real[rc] = pd.to_numeric(real[rc], errors="coerce")

    real2 = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
            .agg(factor_ajuste_real=(rc, "median"))
    )

    df = df.merge(
        real2.rename(columns={"fecha_post": "fecha_post_key"}),
        left_on=[fecha_post_col, "destino"],
        right_on=["fecha_post_key", "destino"],
        how="left",
    )

    w = _weight_series(df)

    # solo con real
    m = df["factor_ajuste_real"].notna() & df[aj_ml1_col].notna() & df["factor_ajuste_final"].notna()
    d = df.loc[m].copy()
    if d.empty:
        raise ValueError("No hay filas con real para KPIs de ajuste.")

    w2 = _weight_series(d)

    # ratios (pred/real)
    d["ratio_ml1"] = d[aj_ml1_col] / d["factor_ajuste_real"].replace(0, np.nan)
    d["ratio_ml2"] = d["factor_ajuste_final"] / d["factor_ajuste_real"].replace(0, np.nan)

    out_g = pd.DataFrame([{
        "n_rows": int(len(d)),
        "n_dates": int(pd.to_datetime(d[fecha_post_col], errors="coerce").dt.normalize().nunique()),
        "mae_log_ml1_w": _wmae_log_ratio(d["ratio_ml1"], w2),
        "mae_log_ml2_w": _wmae_log_ratio(d["ratio_ml2"], w2),
        "median_ratio_ml1": float(np.nanmedian(pd.to_numeric(d["ratio_ml1"], errors="coerce").values)),
        "median_ratio_ml2": float(np.nanmedian(pd.to_numeric(d["ratio_ml2"], errors="coerce").values)),
        "p05_ratio_ml2": float(np.nanpercentile(pd.to_numeric(d["ratio_ml2"], errors="coerce").dropna().values, 5)),
        "p95_ratio_ml2": float(np.nanpercentile(pd.to_numeric(d["ratio_ml2"], errors="coerce").dropna().values, 95)),
        "improvement_abs_mae_log_w": float(_wmae_log_ratio(d["ratio_ml1"], w2) - _wmae_log_ratio(d["ratio_ml2"], w2)),
        "created_at": pd.Timestamp(datetime.utcnow()).normalize(),
    }])

    rows = []
    for dest, g in d.groupby("destino"):
        ww = _weight_series(g)
        rows.append({
            "destino": str(dest),
            "n_rows": int(len(g)),
            "n_dates": int(pd.to_datetime(g[fecha_post_col], errors="coerce").dt.normalize().nunique()),
            "mae_log_ml1_w": _wmae_log_ratio(g["ratio_ml1"], ww),
            "mae_log_ml2_w": _wmae_log_ratio(g["ratio_ml2"], ww),
            "median_ratio_ml2": float(np.nanmedian(pd.to_numeric(g["ratio_ml2"], errors="coerce").values)),
            "improvement_abs_mae_log_w": float(_wmae_log_ratio(g["ratio_ml1"], ww) - _wmae_log_ratio(g["ratio_ml2"], ww)),
        })
    out_d = pd.DataFrame(rows).sort_values("improvement_abs_mae_log_w", ascending=False)

    r = pd.to_numeric(d["ratio_ml2"], errors="coerce")
    out_dist = pd.DataFrame([{
        "n": int(len(d)),
        "ratio_min": float(np.nanmin(r.values)),
        "ratio_p05": float(np.nanpercentile(r.dropna().values, 5)),
        "ratio_p25": float(np.nanpercentile(r.dropna().values, 25)),
        "ratio_median": float(np.nanmedian(r.values)),
        "ratio_p75": float(np.nanpercentile(r.dropna().values, 75)),
        "ratio_p95": float(np.nanpercentile(r.dropna().values, 95)),
        "ratio_max": float(np.nanmax(r.values)),
        "created_at": pd.Timestamp(datetime.utcnow()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_G)
    write_parquet(out_d, OUT_D)
    write_parquet(out_dist, OUT_DIST)

    print(f"[OK] Wrote global : {OUT_G}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote destino: {OUT_D} rows={len(out_d)}")
    print(out_d.to_string(index=False))
    print(f"\n[OK] Wrote dist   : {OUT_DIST}")
    print(out_dist.to_string(index=False))


if __name__ == "__main__":
    main()
