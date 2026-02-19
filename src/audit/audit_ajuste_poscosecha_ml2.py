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

IN_G = EVAL / "ml2_ajuste_poscosecha_eval_global.parquet"
IN_D = EVAL / "ml2_ajuste_poscosecha_eval_by_destino.parquet"
IN_DIST = EVAL / "ml2_ajuste_poscosecha_eval_ratio_dist.parquet"

# Re-lectura de final para ejemplos
GOLD = DATA / "gold"
SILVER = DATA / "silver"
IN_FINAL = GOLD / "pred_poscosecha_ml2_ajuste_grado_dia_bloque_destino_final.parquet"
IN_REAL_MA = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    ts = _ts()

    out_g = EVAL / f"audit_ajuste_poscosecha_ml2_kpi_global_{ts}.parquet"
    out_d = EVAL / f"audit_ajuste_poscosecha_ml2_kpi_by_destino_{ts}.parquet"
    out_dist = EVAL / f"audit_ajuste_poscosecha_ml2_ratio_dist_{ts}.parquet"
    out_ex = EVAL / f"audit_ajuste_poscosecha_ml2_examples_10x2_{ts}.parquet"

    g = read_parquet(IN_G).copy()
    d = read_parquet(IN_D).copy()
    dist = read_parquet(IN_DIST).copy()

    # Examples: top improve / worst by |log_ratio_ml1|-|log_ratio_ml2|
    df = read_parquet(IN_FINAL).copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["destino"] = _canon_str(df["destino"])

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
        rc = None
    if rc is None:
        raise ValueError("Real MA no trae factor_ajuste/ajuste.")
    real[rc] = pd.to_numeric(real[rc], errors="coerce")
    real2 = real.groupby(["fecha_post", "destino"], dropna=False, as_index=False).agg(factor_ajuste_real=(rc, "median"))

    # columnas
    fecha_post_col = None
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            fecha_post_col = c
            break
    if fecha_post_col is None:
        raise KeyError("No encuentro fecha_post_pred_* en final.")
    df[fecha_post_col] = _to_date(df[fecha_post_col])

    aj_ml1_col = None
    for c in ["factor_ajuste_ml1", "ajuste_ml1", "factor_ajuste_seed", "factor_ajuste"]:
        if c in df.columns:
            aj_ml1_col = c
            break
    if aj_ml1_col is None:
        raise KeyError("No encuentro ajuste ML1 en final.")
    df[aj_ml1_col] = pd.to_numeric(df[aj_ml1_col], errors="coerce")

    df["factor_ajuste_final"] = pd.to_numeric(df["factor_ajuste_final"], errors="coerce")

    df = df.merge(
        real2.rename(columns={"fecha_post": "fecha_post_key"}),
        left_on=[fecha_post_col, "destino"],
        right_on=["fecha_post_key", "destino"],
        how="left",
    )

    m = df["factor_ajuste_real"].notna() & df[aj_ml1_col].notna() & df["factor_ajuste_final"].notna()
    x = df.loc[m].copy()
    if x.empty:
        # igual guardamos los KPI y dist
        EVAL.mkdir(parents=True, exist_ok=True)
        write_parquet(g, out_g)
        write_parquet(d, out_d)
        write_parquet(dist, out_dist)
        write_parquet(pd.DataFrame([]), out_ex)
        print("\n=== ML2 AJUSTE POSCOSECHA AUDIT ===")
        print(f"KPI global parquet : {out_g}")
        print(f"KPI por destino    : {out_d}")
        print(f"Ratio dist parquet : {out_dist}")
        print(f"Examples 10x2      : {out_ex}")
        print("\n[WARN] No hay filas con real para examples.")
        return

    # pesos (si existen)
    w = None
    for c in ["tallos_w", "tallos", "tallos_total_ml2", "tallos_total"]:
        if c in x.columns:
            w = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
            break
    if w is None:
        w = pd.Series(1.0, index=x.index, dtype="float64")

    ratio_ml1 = (x[aj_ml1_col] / x["factor_ajuste_real"].replace(0, np.nan)).astype(float)
    ratio_ml2 = (x["factor_ajuste_final"] / x["factor_ajuste_real"].replace(0, np.nan)).astype(float)

    lr1 = np.log(ratio_ml1.replace(0, np.nan))
    lr2 = np.log(ratio_ml2.replace(0, np.nan))
    score = lr1.abs() - lr2.abs()   # >0 mejora
    wscore = score * w

    x["ratio_ml1"] = ratio_ml1
    x["ratio_ml2"] = ratio_ml2
    x["log_ratio_ml1"] = lr1
    x["log_ratio_ml2"] = lr2
    x["score"] = score
    x["wscore"] = wscore

    cols = [
        "fecha",
        fecha_post_col,
        "destino",
        "grado" if "grado" in x.columns else None,
        aj_ml1_col,
        "factor_ajuste_final",
        "factor_ajuste_real",
        "ratio_ml1",
        "ratio_ml2",
        "log_ratio_ml1",
        "log_ratio_ml2",
        "score",
        "wscore",
    ]
    cols = [c for c in cols if c is not None and c in x.columns]
    top = x.sort_values("wscore", ascending=False).head(10).copy()
    top["sample_group"] = "TOP_IMPROVE_10"
    worst = x.sort_values("wscore", ascending=True).head(10).copy()
    worst["sample_group"] = "TOP_WORSE_10"
    ex = pd.concat([top[cols + ["sample_group"]], worst[cols + ["sample_group"]]], ignore_index=True)

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(g, out_g)
    write_parquet(d, out_d)
    write_parquet(dist, out_dist)
    write_parquet(ex, out_ex)

    print("\n=== ML2 AJUSTE POSCOSECHA AUDIT ===")
    print(f"KPI global parquet : {out_g}")
    print(f"KPI por destino    : {out_d}")
    print(f"Ratio dist parquet : {out_dist}")
    print(f"Examples 10x2      : {out_ex}")
    print("\n--- KPI GLOBAL ---")
    print(g.to_string(index=False))
    print("\n--- RATIO DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI POR DESTINO ---")
    print(d.to_string(index=False))
    print("\n--- EXAMPLES 10x2 ---")
    print(ex.to_string(index=False))


if __name__ == "__main__":
    main()
