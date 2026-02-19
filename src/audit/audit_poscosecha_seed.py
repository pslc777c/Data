from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_GD_BD = Path("data/gold/pred_poscosecha_seed_grado_dia_bloque_destino.parquet")
IN_DD    = Path("data/gold/pred_poscosecha_seed_dia_destino.parquet")
IN_DT    = Path("data/gold/pred_poscosecha_seed_dia_total.parquet")

OUT_CHECKS  = Path("data/audit/audit_poscosecha_seed_checks.parquet")
OUT_SUMMARY = Path("data/audit/audit_poscosecha_seed_summary.json")

def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pct(x: float) -> float:
    return float(np.round(x * 100.0, 4))


def main() -> None:
    gd = read_parquet(IN_GD_BD).copy()
    dd = read_parquet(IN_DD).copy()
    dt = read_parquet(IN_DT).copy()

    gd.columns = [str(c).strip() for c in gd.columns]
    dd.columns = [str(c).strip() for c in dd.columns]
    dt.columns = [str(c).strip() for c in dt.columns]

    need_gd = [
        "fecha","fecha_post_pred","bloque_base","variedad_canon","grado","destino",
        "cajas_ml1_grado_dia","cajas_split_grado_dia","cajas_post_seed",
        "factor_hidr_seed","factor_desp_seed","factor_ajuste_seed","dh_dias"
    ]
    miss = [c for c in need_gd if c not in gd.columns]
    if miss:
        raise ValueError(f"gold seed GD_BD sin columnas: {miss}")

    # numeric
    for c in ["cajas_ml1_grado_dia","cajas_split_grado_dia","cajas_post_seed",
              "factor_hidr_seed","factor_desp_seed","factor_ajuste_seed"]:
        gd[c] = _to_num(gd[c])

    # 1) Checks de integridad y rangos
    gd["chk_nonneg_split"] = gd["cajas_split_grado_dia"].fillna(0) >= -1e-9
    gd["chk_nonneg_post"]  = gd["cajas_post_seed"].fillna(0) >= -1e-9

    gd["chk_hidr_range"]   = gd["factor_hidr_seed"].between(0.60, 3.00, inclusive="both")
    gd["chk_desp_range"]   = gd["factor_desp_seed"].between(0.05, 1.00, inclusive="both")
    gd["chk_ajus_range"]   = gd["factor_ajuste_seed"].between(0.50, 2.00, inclusive="both")

    gd["chk_dh_range"]     = pd.to_numeric(gd["dh_dias"], errors="coerce").between(0, 30, inclusive="both")

    # 2) Mass-balance: sum por key_supply debe igualar cajas_ml1 (antes del split)
    key_supply = ["fecha","bloque_base","variedad_canon","grado"]
    mb = (
        gd.groupby(key_supply, dropna=False, as_index=False)
          .agg(
              cajas_ml1=("cajas_ml1_grado_dia","max"),
              sum_split=("cajas_split_grado_dia","sum"),
          )
    )
    mb["abs_diff"] = (mb["sum_split"] - mb["cajas_ml1"]).abs()
    mb_ok_rate = float((mb["abs_diff"] <= 1e-9).mean()) if len(mb) else float("nan")
    mb_max_abs = float(mb["abs_diff"].max()) if len(mb) else float("nan")

    # 3) Relación post vs split (factor efectivo)
    gd["factor_efectivo_post_vs_split"] = np.where(
        gd["cajas_split_grado_dia"].abs() > 1e-12,
        gd["cajas_post_seed"] / gd["cajas_split_grado_dia"],
        np.nan
    )

    # 4) Agregados por destino
    by_dest = (
        gd.groupby("destino", dropna=False)
          .agg(
              n=("destino","size"),
              cajas_split=("cajas_split_grado_dia","sum"),
              cajas_post=("cajas_post_seed","sum"),
              hidr_med=("factor_hidr_seed","median"),
              desp_med=("factor_desp_seed","median"),
              ajus_med=("factor_ajuste_seed","median"),
              dh_med=("dh_dias","median"),
              eff_med=("factor_efectivo_post_vs_split","median"),
          )
          .reset_index()
    )

    # 5) Top outliers
    outliers = (
        gd.assign(abs_eff_dev=(gd["factor_efectivo_post_vs_split"] - gd["factor_efectivo_post_vs_split"].median()).abs())
          .sort_values("abs_eff_dev", ascending=False)
          .head(200)
          .copy()
    )

    # 6) Construir checks fila a fila (útil para debug)
    checks = gd[
        ["fecha","bloque_base","variedad_canon","grado","destino","fecha_post_pred",
         "cajas_ml1_grado_dia","cajas_split_grado_dia","cajas_post_seed",
         "factor_hidr_seed","factor_desp_seed","factor_ajuste_seed","dh_dias",
         "factor_efectivo_post_vs_split",
         "chk_nonneg_split","chk_nonneg_post","chk_hidr_range","chk_desp_range","chk_ajus_range","chk_dh_range"
        ]
    ].copy()

    write_parquet(checks, OUT_CHECKS)

    summary = {
        "rows_gd_bd": int(len(gd)),
        "rows_dd": int(len(dd)),
        "rows_dt": int(len(dt)),
        "mass_balance_split_ok_rate": _pct(mb_ok_rate) if np.isfinite(mb_ok_rate) else None,
        "mass_balance_split_max_abs_diff": mb_max_abs,
        "range_ok_rates": {
            "hidr_range_ok": _pct(float(checks["chk_hidr_range"].mean())),
            "desp_range_ok": _pct(float(checks["chk_desp_range"].mean())),
            "ajus_range_ok": _pct(float(checks["chk_ajus_range"].mean())),
            "dh_range_ok": _pct(float(checks["chk_dh_range"].mean())),
        },
        "post_vs_split_factor_quantiles": {
            "p01": float(np.nanquantile(gd["factor_efectivo_post_vs_split"].values, 0.01)),
            "p50": float(np.nanquantile(gd["factor_efectivo_post_vs_split"].values, 0.50)),
            "p99": float(np.nanquantile(gd["factor_efectivo_post_vs_split"].values, 0.99)),
        },
        "by_destino": by_dest.to_dict(orient="records"),
        "top_outliers_preview": outliers[
            ["fecha","bloque_base","variedad_canon","grado","destino","fecha_post_pred",
             "cajas_split_grado_dia","cajas_post_seed","factor_hidr_seed","factor_desp_seed","factor_ajuste_seed",
             "factor_efectivo_post_vs_split"
            ]
        ].head(30).to_dict(orient="records"),
    }

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)


    print(f"OK -> {OUT_CHECKS}")
    print(f"OK -> {OUT_SUMMARY}")
    print(f"[AUDIT] mass_balance_ok_rate={mb_ok_rate:.4f} max_abs_diff={mb_max_abs:.10f}")


if __name__ == "__main__":
    main()
