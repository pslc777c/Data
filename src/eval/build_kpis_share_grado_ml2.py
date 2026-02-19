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

IN_FACTOR = EVAL / "backtest_factor_ml2_share_grado.parquet"
IN_FINAL = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"
IN_REAL = SILVER / "fact_cosecha_real_grado_dia.parquet"
IN_PRED_ML1 = GOLD / "pred_tallos_grado_dia_ml1_full.parquet"

OUT_GLOBAL = EVAL / "ml2_share_grado_eval_global.parquet"
OUT_BY_GRADO = EVAL / "ml2_share_grado_eval_by_grado.parquet"
OUT_SANITY = EVAL / "ml2_share_grado_eval_sanity.parquet"


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


def mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else np.nan


def wape_weighted(abs_err: pd.Series, w: pd.Series) -> float:
    abs_err = pd.to_numeric(abs_err, errors="coerce").fillna(0.0)
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return np.nan
    return float(np.sum(abs_err * w) / denom)


def main() -> None:
    fac = read_parquet(IN_FACTOR).copy()
    fin = read_parquet(IN_FINAL).copy()
    real = read_parquet(IN_REAL).copy()
    ml1 = read_parquet(IN_PRED_ML1).copy()


    # --- Alias handling: share_final ---
    if "share_final" not in fac.columns:
        # intentos comunes
        cands = [c for c in fac.columns if ("share" in c.lower()) and ("final" in c.lower())]
        if cands:
            fac = fac.rename(columns={cands[0]: "share_final"})
        else:
            raise KeyError(f"No encuentro share_final en factor. Columnas disponibles: {list(fac.columns)}")

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

    # Real share por día/bloque
    rg = real.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))
    tot = rg.groupby(["fecha", "bloque_base"], as_index=False).agg(tallos_real_dia=("tallos_real", "sum"))
    rg = rg.merge(tot, on=["fecha", "bloque_base"], how="left")
    rg["share_real"] = np.where(rg["tallos_real_dia"] > 0, rg["tallos_real"] / rg["tallos_real_dia"], np.nan)

    # ML1 share
    ml1_need = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1", "tallos_pred_ml1_grado_dia"]
    ml1 = ml1[ml1_need].copy()
    ml1["share_grado_ml1"] = pd.to_numeric(ml1["share_grado_ml1"], errors="coerce")
    ml1["tallos_pred_ml1_grado_dia"] = pd.to_numeric(ml1["tallos_pred_ml1_grado_dia"], errors="coerce").fillna(0.0)

    # Merge everything at grain (ciclo_id, fecha, bloque_base, variedad_canon, grado)
    fac_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]
    opt = []
    if "share_ml1" in fac.columns:
        opt.append("share_ml1")
    elif "share_grado_ml1" in fac.columns:
        fac = fac.rename(columns={"share_grado_ml1": "share_ml1"})
        opt.append("share_ml1")

    # fin YA trae share_final (porque apply lo guarda ahí).
    # del factor solo necesitamos share_ml1 (baseline) para comparar.

    fac_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]

    # normalizamos share_ml1 desde factor
    if "share_ml1" in fac.columns:
        fac_use = fac[fac_cols + ["share_ml1"]].copy()
    elif "share_grado_ml1" in fac.columns:
        fac_use = fac.rename(columns={"share_grado_ml1": "share_ml1"})[fac_cols + ["share_ml1"]].copy()
    else:
        raise KeyError(f"No encuentro share_ml1 / share_grado_ml1 en factor. Columnas: {list(fac.columns)}")

    df = fin.merge(
        fac_use,
        on=["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"],
        how="left",
    )



    df = df.merge(ml1, on=["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"], how="left")
    df = df.merge(rg[["fecha", "bloque_base", "grado", "share_real", "tallos_real", "tallos_real_dia"]],
                  on=["fecha", "bloque_base", "grado"], how="left")

    # Cast
    df["share_final"] = pd.to_numeric(df["share_final"], errors="coerce")
    df["share_grado_ml1"] = pd.to_numeric(df["share_grado_ml1"], errors="coerce")
    df["share_real"] = pd.to_numeric(df["share_real"], errors="coerce")

    df["tallos_final_grado_dia"] = pd.to_numeric(df["tallos_final_grado_dia"], errors="coerce")
    df["tallos_real"] = pd.to_numeric(df["tallos_real"], errors="coerce")
    df["tallos_total_ml2"] = pd.to_numeric(df["tallos_total_ml2"], errors="coerce")

    # --- KPI share error ---
    m = df["share_real"].notna() & df["share_grado_ml1"].notna() & df["share_final"].notna()
    d = df.loc[m].copy()

    d["err_share_ml1"] = d["share_real"] - d["share_grado_ml1"]
    d["err_share_ml2"] = d["share_real"] - d["share_final"]
    d["abs_err_share_ml1"] = d["err_share_ml1"].abs()
    d["abs_err_share_ml2"] = d["err_share_ml2"].abs()

    # Weighted by tallos_real_dia (impacto por día)
    w = pd.to_numeric(d["tallos_real_dia"], errors="coerce").fillna(0.0)

    # --- KPI tallos por grado error (impacto directo) ---
    # Error absoluto en tallos (si hay real)
    d["abs_err_tallos_ml1"] = (d["tallos_real"] - d["tallos_pred_ml1_grado_dia"]).abs()
    d["abs_err_tallos_ml2"] = (d["tallos_real"] - d["tallos_final_grado_dia"]).abs()

    # Global
    out_g = pd.DataFrame([{
        "n_rows": int(len(d)),
        "n_cycles": int(d["ciclo_id"].nunique()),
        "mae_share_ml1": mae(d["err_share_ml1"]),
        "mae_share_ml2": mae(d["err_share_ml2"]),
        "wape_share_ml1_wt_day": wape_weighted(d["abs_err_share_ml1"], w),
        "wape_share_ml2_wt_day": wape_weighted(d["abs_err_share_ml2"], w),
        "mae_abs_tallos_ml1": float(np.nanmean(d["abs_err_tallos_ml1"])),
        "mae_abs_tallos_ml2": float(np.nanmean(d["abs_err_tallos_ml2"])),
        "improvement_abs_mae_share": mae(d["err_share_ml1"]) - mae(d["err_share_ml2"]),
        "improvement_abs_mae_abs_tallos": float(np.nanmean(d["abs_err_tallos_ml1"]) - np.nanmean(d["abs_err_tallos_ml2"])),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # By grado
    rows = []
    for grado, g in d.groupby("grado"):
        ww = pd.to_numeric(g["tallos_real_dia"], errors="coerce").fillna(0.0)
        rows.append({
            "grado": str(grado),
            "n_rows": int(len(g)),
            "mae_share_ml1": mae(g["err_share_ml1"]),
            "mae_share_ml2": mae(g["err_share_ml2"]),
            "wape_share_ml1_wt_day": wape_weighted(g["abs_err_share_ml1"], ww),
            "wape_share_ml2_wt_day": wape_weighted(g["abs_err_share_ml2"], ww),
            "mae_abs_tallos_ml1": float(np.nanmean(g["abs_err_tallos_ml1"])),
            "mae_abs_tallos_ml2": float(np.nanmean(g["abs_err_tallos_ml2"])),
            "improvement_abs_mae_share": mae(g["err_share_ml1"]) - mae(g["err_share_ml2"]),
        })
    out_bg = pd.DataFrame(rows).sort_values("improvement_abs_mae_share", ascending=False)

    # --- Sanity: sum shares ---
    s = fac.groupby(["ciclo_id", "fecha"], as_index=False).agg(sum_share_final=("share_final", "sum"))
    sanity = pd.DataFrame([{
        "n_days": int(len(s)),
        "sum_share_min": float(s["sum_share_final"].min()),
        "sum_share_p25": float(s["sum_share_final"].quantile(0.25)),
        "sum_share_median": float(s["sum_share_final"].median()),
        "sum_share_p75": float(s["sum_share_final"].quantile(0.75)),
        "sum_share_max": float(s["sum_share_final"].max()),
        "pct_close_1pm_1e-6": float(np.mean(np.isclose(s["sum_share_final"].values, 1.0, atol=1e-6))),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_bg, OUT_BY_GRADO)
    write_parquet(sanity, OUT_SANITY)

    print(f"[OK] Wrote global: {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote by_grado: {OUT_BY_GRADO} rows={len(out_bg)}")
    print(out_bg.head(10).to_string(index=False))
    print(f"\n[OK] Wrote sanity: {OUT_SANITY}")
    print(sanity.to_string(index=False))


if __name__ == "__main__":
    main()
