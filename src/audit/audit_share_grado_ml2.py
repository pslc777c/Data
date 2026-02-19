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

IN_FACTOR = EVAL / "backtest_factor_ml2_share_grado.parquet"
IN_FINAL = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"
IN_REAL = SILVER / "fact_cosecha_real_grado_dia.parquet"
IN_PRED_ML1 = GOLD / "pred_tallos_grado_dia_ml1_full.parquet"

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT_KPI_GLOBAL = EVAL / f"audit_share_grado_ml2_kpi_global_{STAMP}.parquet"
OUT_KPI_BY_ESTADO = EVAL / f"audit_share_grado_ml2_kpi_by_estado_{STAMP}.parquet"
OUT_RATIO_DIST = EVAL / f"audit_share_grado_ml2_ratio_dist_{STAMP}.parquet"
OUT_SANITY = EVAL / f"audit_share_grado_ml2_sanity_{STAMP}.parquet"
OUT_EXAMPLES_10x2 = EVAL / f"audit_share_grado_ml2_examples_10x2_{STAMP}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_block_base(df: pd.DataFrame) -> pd.DataFrame:
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
        return df
    if "bloque_padre" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_padre"])
        return df
    if "bloque" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque"])
        return df
    raise KeyError("No encuentro columna de bloque: bloque_base / bloque_padre / bloque")


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _wape(abs_err: pd.Series, w: pd.Series) -> float:
    a = pd.to_numeric(abs_err, errors="coerce").fillna(0.0).values
    ww = pd.to_numeric(w, errors="coerce").fillna(0.0).values
    denom = float(np.sum(ww))
    if denom <= 0:
        return float("nan")
    return float(np.sum(a * ww) / denom)


def main() -> None:
    fac = read_parquet(IN_FACTOR).copy()
    fin = read_parquet(IN_FINAL).copy()
    real = read_parquet(IN_REAL).copy()
    ml1 = read_parquet(IN_PRED_ML1).copy()

    # Canon
    for df in (fac, fin, ml1):
        df["fecha"] = _to_date(df["fecha"])
        df["bloque_base"] = _canon_str(df["bloque_base"])
        df["grado"] = _canon_str(df["grado"])
        if "variedad_canon" in df.columns:
            df["variedad_canon"] = _canon_str(df["variedad_canon"])

    real["fecha"] = _to_date(real["fecha"])
    real = _resolve_block_base(real)
    real["grado"] = _canon_str(real["grado"])
    real["tallos_real"] = pd.to_numeric(real["tallos_real"], errors="coerce").fillna(0.0)

    # ---- Share real por día/bloque ----
    rg = real.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))
    tot = rg.groupby(["fecha", "bloque_base"], as_index=False).agg(tallos_real_dia=("tallos_real", "sum"))
    rg = rg.merge(tot, on=["fecha", "bloque_base"], how="left")
    rg["share_real"] = np.where(rg["tallos_real_dia"] > 0, rg["tallos_real"] / rg["tallos_real_dia"], np.nan)

    # ---- ML1 (share + tallos pred por grado) ----
    ml1_need = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1", "tallos_pred_ml1_grado_dia"]
    ml1 = ml1[ml1_need].copy()
    ml1["share_grado_ml1"] = pd.to_numeric(ml1["share_grado_ml1"], errors="coerce")
    ml1["tallos_pred_ml1_grado_dia"] = pd.to_numeric(ml1["tallos_pred_ml1_grado_dia"], errors="coerce").fillna(0.0)

    # ---- Factor: baseline share_ml1 y share_final ----
    fac_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]
    if "share_ml1" in fac.columns:
        fac_use = fac[fac_cols + ["share_ml1"]].copy()
    elif "share_grado_ml1" in fac.columns:
        fac_use = fac.rename(columns={"share_grado_ml1": "share_ml1"})[fac_cols + ["share_ml1"]].copy()
    else:
        raise KeyError(f"No encuentro share_ml1/share_grado_ml1 en factor. Columnas: {list(fac.columns)}")

    # fin YA trae share_final (del apply)
    need_fin = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_final", "tallos_final_grado_dia"]
    for c in need_fin:
        if c not in fin.columns:
            raise KeyError(f"Falta columna en FINAL backtest: {c}. Columnas: {list(fin.columns)}")
    fin_use = fin[need_fin].copy()

    # ---- Dataset audit a grano ----
    df = fin_use.merge(fac_use, on=fac_cols, how="left")
    df = df.merge(ml1, on=fac_cols, how="left")
    df = df.merge(rg[["fecha", "bloque_base", "grado", "share_real", "tallos_real", "tallos_real_dia"]],
                  on=["fecha", "bloque_base", "grado"], how="left")

    # Bring estado if present in factor (optional)
    if "estado" in fac.columns:
        tmp = fac[fac_cols + ["estado"]].drop_duplicates(fac_cols)
        tmp["estado"] = _canon_str(tmp["estado"])
        df = df.merge(tmp, on=fac_cols, how="left")
    else:
        df["estado"] = "NA"

    # Cast
    df["share_ml1"] = pd.to_numeric(df["share_ml1"], errors="coerce")
    df["share_final"] = pd.to_numeric(df["share_final"], errors="coerce")
    df["share_real"] = pd.to_numeric(df["share_real"], errors="coerce")
    df["tallos_pred_ml1_grado_dia"] = pd.to_numeric(df["tallos_pred_ml1_grado_dia"], errors="coerce")
    df["tallos_final_grado_dia"] = pd.to_numeric(df["tallos_final_grado_dia"], errors="coerce")
    df["tallos_real"] = pd.to_numeric(df["tallos_real"], errors="coerce")
    df["tallos_real_dia"] = pd.to_numeric(df["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Filter where share_real exists
    m = df["share_real"].notna() & df["share_ml1"].notna() & df["share_final"].notna()
    d = df.loc[m].copy()

    # Errors
    d["err_share_ml1"] = d["share_real"] - d["share_ml1"]
    d["err_share_ml2"] = d["share_real"] - d["share_final"]
    d["abs_err_share_ml1"] = d["err_share_ml1"].abs()
    d["abs_err_share_ml2"] = d["err_share_ml2"].abs()

    d["abs_err_tallos_ml1"] = (d["tallos_real"] - d["tallos_pred_ml1_grado_dia"]).abs()
    d["abs_err_tallos_ml2"] = (d["tallos_real"] - d["tallos_final_grado_dia"]).abs()

    # Ratios (diagnóstico)
    eps = 1e-6
    d["ratio_share_final_over_ml1"] = (d["share_final"] + eps) / (d["share_ml1"] + eps)

    # --- KPI GLOBAL ---
    w = d["tallos_real_dia"]
    kpi_global = pd.DataFrame([{
        "n_rows": int(len(d)),
        "n_cycles": int(d["ciclo_id"].nunique()),
        "mae_share_ml1": _safe_float(np.nanmean(d["abs_err_share_ml1"])),
        "mae_share_ml2": _safe_float(np.nanmean(d["abs_err_share_ml2"])),
        "wape_share_ml1_wt_day": _wape(d["abs_err_share_ml1"], w),
        "wape_share_ml2_wt_day": _wape(d["abs_err_share_ml2"], w),
        "mae_abs_tallos_ml1": _safe_float(np.nanmean(d["abs_err_tallos_ml1"])),
        "mae_abs_tallos_ml2": _safe_float(np.nanmean(d["abs_err_tallos_ml2"])),
        "improvement_abs_mae_share": _safe_float(np.nanmean(d["abs_err_share_ml1"]) - np.nanmean(d["abs_err_share_ml2"])),
        "improvement_abs_mae_abs_tallos": _safe_float(np.nanmean(d["abs_err_tallos_ml1"]) - np.nanmean(d["abs_err_tallos_ml2"])),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # --- KPI POR ESTADO ---
    rows = []
    for estado, g in d.groupby("estado"):
        ww = g["tallos_real_dia"]
        rows.append({
            "estado": str(estado),
            "n_rows": int(len(g)),
            "n_cycles": int(g["ciclo_id"].nunique()),
            "mae_share_ml1": _safe_float(np.nanmean(g["abs_err_share_ml1"])),
            "mae_share_ml2": _safe_float(np.nanmean(g["abs_err_share_ml2"])),
            "wape_share_ml1_wt_day": _wape(g["abs_err_share_ml1"], ww),
            "wape_share_ml2_wt_day": _wape(g["abs_err_share_ml2"], ww),
            "mae_abs_tallos_ml1": _safe_float(np.nanmean(g["abs_err_tallos_ml1"])),
            "mae_abs_tallos_ml2": _safe_float(np.nanmean(g["abs_err_tallos_ml2"])),
            "improvement_abs_mae_share": _safe_float(np.nanmean(g["abs_err_share_ml1"]) - np.nanmean(g["abs_err_share_ml2"])),
        })
    kpi_by_estado = pd.DataFrame(rows).sort_values("improvement_abs_mae_share", ascending=False)

    # --- RATIO DIST ---
    r = d["ratio_share_final_over_ml1"].replace([np.inf, -np.inf], np.nan).dropna()
    ratio_dist = pd.DataFrame([{
        "n": int(len(r)),
        "ratio_min": _safe_float(r.min()),
        "ratio_p05": _safe_float(r.quantile(0.05)),
        "ratio_p25": _safe_float(r.quantile(0.25)),
        "ratio_median": _safe_float(r.median()),
        "ratio_p75": _safe_float(r.quantile(0.75)),
        "ratio_p95": _safe_float(r.quantile(0.95)),
        "ratio_max": _safe_float(r.max()),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # --- SANITY: sum shares == 1 ---
    if "share_final" not in fin.columns:
        raise KeyError("FINAL no tiene share_final para sanity.")
    s = fin.groupby(["ciclo_id", "fecha"], as_index=False).agg(sum_share_final=("share_final", "sum"))
    sanity = pd.DataFrame([{
        "n_days": int(len(s)),
        "sum_share_min": _safe_float(s["sum_share_final"].min()),
        "sum_share_p25": _safe_float(s["sum_share_final"].quantile(0.25)),
        "sum_share_median": _safe_float(s["sum_share_final"].median()),
        "sum_share_p75": _safe_float(s["sum_share_final"].quantile(0.75)),
        "sum_share_max": _safe_float(s["sum_share_final"].max()),
        "pct_close_1pm_1e-6": _safe_float(np.mean(np.isclose(s["sum_share_final"].values, 1.0, atol=1e-6))),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # --- EXAMPLES 10x2 (Top improve / Top worse) por reducción de error tallos ---
    # agregamos por ciclo_id el impacto total
    agg = d.groupby(["ciclo_id", "estado"], as_index=False).agg(
        abs_err_tallos_ml1=("abs_err_tallos_ml1", "sum"),
        abs_err_tallos_ml2=("abs_err_tallos_ml2", "sum"),
        n_rows=("ciclo_id", "size"),
    )
    agg["improvement_abs_tallos"] = agg["abs_err_tallos_ml1"] - agg["abs_err_tallos_ml2"]

    top = agg.sort_values("improvement_abs_tallos", ascending=False).head(10).copy()
    top["sample_group"] = "TOP_IMPROVE_10"
    worst = agg.sort_values("improvement_abs_tallos", ascending=True).head(10).copy()
    worst["sample_group"] = "TOP_WORSE_10"

    examples = pd.concat([top, worst], ignore_index=True)

    # attach representative bloque_base/variedad
    rep = fin[["ciclo_id", "bloque_base", "variedad_canon"]].drop_duplicates("ciclo_id")
    examples = examples.merge(rep, on="ciclo_id", how="left")

    # Write
    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(kpi_global, OUT_KPI_GLOBAL)
    write_parquet(kpi_by_estado, OUT_KPI_BY_ESTADO)
    write_parquet(ratio_dist, OUT_RATIO_DIST)
    write_parquet(sanity, OUT_SANITY)
    write_parquet(examples, OUT_EXAMPLES_10x2)

    print("\n=== ML2 SHARE GRADO AUDIT ===")
    print(f"KPI global parquet : {OUT_KPI_GLOBAL}")
    print(f"KPI por estado     : {OUT_KPI_BY_ESTADO}")
    print(f"Ratio dist parquet : {OUT_RATIO_DIST}")
    print(f"Sanity parquet     : {OUT_SANITY}")
    print(f"Examples 10x2      : {OUT_EXAMPLES_10x2}")

    print("\n--- KPI GLOBAL ---")
    print(kpi_global.to_string(index=False))

    print("\n--- KPI POR ESTADO ---")
    print(kpi_by_estado.to_string(index=False))

    print("\n--- RATIO DIST ---")
    print(ratio_dist.to_string(index=False))

    print("\n--- SANITY ---")
    print(sanity.to_string(index=False))

    print("\n--- EXAMPLES 10x2 ---")
    print(examples.sort_values(["sample_group", "improvement_abs_tallos"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
