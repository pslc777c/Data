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

IN_FACT = EVAL / "backtest_factor_ml2_peso_tallo_grado_dia.parquet"
IN_TALLOS = SILVER / "fact_cosecha_real_grado_dia.parquet"

OUT_GLOBAL = EVAL / "ml2_peso_tallo_eval_global.parquet"
OUT_ESTADO = EVAL / "ml2_peso_tallo_eval_by_estado.parquet"
OUT_RATIO = EVAL / "ml2_peso_tallo_eval_ratio_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _mae(x: pd.Series) -> float:
    return float(np.nanmean(np.abs(x))) if len(x) else np.nan


def _wape_weighted(err_abs: pd.Series, w: pd.Series) -> float:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    err_abs = pd.to_numeric(err_abs, errors="coerce").fillna(0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return np.nan
    return float(np.sum(err_abs * w) / denom)


def main() -> None:
    df = read_parquet(IN_FACT).copy()

    # Tallos reales (peso por tallo debe ponderarse por tallos)
    tl = read_parquet(IN_TALLOS).copy()
    tl["fecha"] = _to_date(tl["fecha"])
    tl["grado"] = _canon_str(tl["grado"]) if "grado" in tl.columns else tl["grado"]
    if "bloque_base" not in tl.columns:
        if "bloque_padre" in tl.columns:
            tl["bloque_base"] = _canon_str(tl["bloque_padre"])
        elif "bloque" in tl.columns:
            tl["bloque_base"] = _canon_str(tl["bloque"])
        else:
            raise KeyError("fact_cosecha_real_grado_dia no tiene bloque_base/bloque_padre/bloque")
    else:
        tl["bloque_base"] = _canon_str(tl["bloque_base"])

    tl["tallos_real"] = pd.to_numeric(tl["tallos_real"], errors="coerce").fillna(0.0)
    tl = tl.groupby(["fecha", "bloque_base", "grado"], as_index=False).agg(tallos_real=("tallos_real", "sum"))

    # Canon fact
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_str(df["bloque_base"])
    df["grado"] = _canon_str(df["grado"])
    if "estado" in df.columns:
        df["estado"] = _canon_str(df["estado"])

    for c in ["peso_tallo_ml1_g", "peso_tallo_real_g", "peso_tallo_final_g", "pred_ratio_peso"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.merge(tl, on=["fecha", "bloque_base", "grado"], how="left")
    df["tallos_real"] = pd.to_numeric(df["tallos_real"], errors="coerce").fillna(0.0)

    # Keep rows where we have real peso_tallo and some weight
    df = df[df["peso_tallo_real_g"].notna()].copy()

    # Errors in g/tallo
    df["err_ml1_g"] = df["peso_tallo_real_g"] - df["peso_tallo_ml1_g"]
    df["err_ml2_g"] = df["peso_tallo_real_g"] - df["peso_tallo_final_g"]

    # Weighted absolute error (by tallos)
    df["abs_err_ml1_g"] = df["err_ml1_g"].abs()
    df["abs_err_ml2_g"] = df["err_ml2_g"].abs()

    # Impact kg (approx)
    df["abs_err_ml1_kg_equiv"] = (df["abs_err_ml1_g"] * df["tallos_real"]) / 1000.0
    df["abs_err_ml2_kg_equiv"] = (df["abs_err_ml2_g"] * df["tallos_real"]) / 1000.0

    out_g = pd.DataFrame([{
        "n_rows": int(len(df)),
        "n_cycles": int(df["ciclo_id"].nunique()) if "ciclo_id" in df.columns else np.nan,
        "mae_ml1_g": _mae(df["err_ml1_g"]),
        "mae_ml2_g": _mae(df["err_ml2_g"]),
        "wape_wt_ml1_g": _wape_weighted(df["abs_err_ml1_g"], df["tallos_real"]),
        "wape_wt_ml2_g": _wape_weighted(df["abs_err_ml2_g"], df["tallos_real"]),
        "kg_abs_err_ml1": float(df["abs_err_ml1_kg_equiv"].sum()),
        "kg_abs_err_ml2": float(df["abs_err_ml2_kg_equiv"].sum()),
        "improvement_abs_mae_g": _mae(df["err_ml1_g"]) - _mae(df["err_ml2_g"]),
        "improvement_abs_wape_wt_g": _wape_weighted(df["abs_err_ml1_g"], df["tallos_real"])
                                   - _wape_weighted(df["abs_err_ml2_g"], df["tallos_real"]),
        "improvement_abs_kg": float(df["abs_err_ml1_kg_equiv"].sum() - df["abs_err_ml2_kg_equiv"].sum()),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # Ratio dist sanity
    r = pd.to_numeric(df["pred_ratio_peso"], errors="coerce")
    out_r = pd.DataFrame([{
        "n": int(r.notna().sum()),
        "ratio_min": float(np.nanmin(r)),
        "ratio_p05": float(np.nanpercentile(r, 5)),
        "ratio_p25": float(np.nanpercentile(r, 25)),
        "ratio_median": float(np.nanmedian(r)),
        "ratio_p75": float(np.nanpercentile(r, 75)),
        "ratio_p95": float(np.nanpercentile(r, 95)),
        "ratio_max": float(np.nanmax(r)),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # By estado if exists
    rows = []
    if "estado" in df.columns and df["estado"].notna().any():
        for est, g in df.groupby("estado"):
            rows.append({
                "estado": str(est),
                "n_rows": int(len(g)),
                "mae_ml1_g": _mae(g["err_ml1_g"]),
                "mae_ml2_g": _mae(g["err_ml2_g"]),
                "wape_wt_ml1_g": _wape_weighted(g["abs_err_ml1_g"], g["tallos_real"]),
                "wape_wt_ml2_g": _wape_weighted(g["abs_err_ml2_g"], g["tallos_real"]),
                "kg_abs_err_ml1": float(g["abs_err_ml1_kg_equiv"].sum()),
                "kg_abs_err_ml2": float(g["abs_err_ml2_kg_equiv"].sum()),
                "improvement_abs_kg": float(g["abs_err_ml1_kg_equiv"].sum() - g["abs_err_ml2_kg_equiv"].sum()),
            })
    out_e = pd.DataFrame(rows)

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_r, OUT_RATIO)
    write_parquet(out_e, OUT_ESTADO)

    print(f"[OK] Wrote global: {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote ratio : {OUT_RATIO}")
    print(out_r.to_string(index=False))
    print(f"\n[OK] Wrote estado: {OUT_ESTADO} rows={len(out_e)}")
    if not out_e.empty:
        print(out_e.sort_values('improvement_abs_kg', ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
