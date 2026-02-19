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

IN_GLOBAL = EVAL / "ml2_hidr_poscosecha_eval_global.parquet"
IN_BY_DESTINO = EVAL / "ml2_hidr_poscosecha_eval_by_destino.parquet"
IN_BY_GRADO = EVAL / "ml2_hidr_poscosecha_eval_by_grado.parquet"
IN_RATIO_DIST = EVAL / "ml2_hidr_poscosecha_eval_ratio_dist.parquet"

# Para ejemplos necesitamos final + real:
GOLD = DATA / "gold"
SILVER = DATA / "silver"
IN_FINAL = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino_final.parquet"
IN_REAL = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _factor_from_hidr_pct(hidr_pct: pd.Series) -> pd.Series:
    x = pd.to_numeric(hidr_pct, errors="coerce")
    return np.where(x.isna(), np.nan, np.where(x > 3.5, 1.0 + x / 100.0, x)).astype(float)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_global = EVAL / f"audit_hidr_poscosecha_ml2_kpi_global_{ts}.parquet"
    out_dest = EVAL / f"audit_hidr_poscosecha_ml2_kpi_by_destino_{ts}.parquet"
    out_gr = EVAL / f"audit_hidr_poscosecha_ml2_kpi_by_grado_{ts}.parquet"
    out_dist = EVAL / f"audit_hidr_poscosecha_ml2_ratio_dist_{ts}.parquet"
    out_examples = EVAL / f"audit_hidr_poscosecha_ml2_examples_10x2_{ts}.parquet"

    g = read_parquet(IN_GLOBAL).copy()
    bd = read_parquet(IN_BY_DESTINO).copy()
    bg = read_parquet(IN_BY_GRADO).copy()
    dist = read_parquet(IN_RATIO_DIST).copy()

    # ejemplos
    fin = read_parquet(IN_FINAL).copy()
    fin.columns = [str(c).strip() for c in fin.columns]

    # fecha_post_pred usada
    fpp = None
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml2", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in fin.columns:
            fpp = c
            break
    if fpp is None:
        raise KeyError("No encuentro fecha_post_pred en final para ejemplos.")

    fin[fpp] = _to_date(fin[fpp])
    fin["destino"] = _canon_str(fin["destino"])
    fin["grado"] = _canon_int(fin["grado"])
    fin["factor_hidr_ml1"] = pd.to_numeric(fin.get("factor_hidr_ml1"), errors="coerce")
    fin["factor_hidr_final"] = pd.to_numeric(fin.get("factor_hidr_final"), errors="coerce")

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

    if "tallos" in real.columns:
        real["tallos"] = pd.to_numeric(real["tallos"], errors="coerce").fillna(0.0)
        g2 = real.groupby(["fecha_post", "grado", "destino"], dropna=False)
        real2 = g2.apply(
            lambda x: pd.Series({
                "factor_hidr_real": float(np.nansum(x["factor_hidr_real"] * x["tallos"]) / np.nansum(x["tallos"])) if np.nansum(x["tallos"]) > 0 else float(np.nanmedian(x["factor_hidr_real"])),
                "tallos": float(np.nansum(x["tallos"])),
            })
        ).reset_index()
    else:
        real2 = (
            real.groupby(["fecha_post", "grado", "destino"], dropna=False, as_index=False)
                .agg(factor_hidr_real=("factor_hidr_real", "median"))
        )
        real2["tallos"] = 1.0

    df = fin.merge(
        real2,
        left_on=[fpp, "grado", "destino"],
        right_on=["fecha_post", "grado", "destino"],
        how="left",
    ).drop(columns=["fecha_post"], errors="ignore")

    m = df["factor_hidr_real"].notna() & df["factor_hidr_ml1"].notna() & df["factor_hidr_final"].notna()
    d = df.loc[m].copy()
    eps = 1e-9
    d["ratio_ml1"] = d["factor_hidr_real"] / d["factor_hidr_ml1"].clip(lower=eps)
    d["ratio_ml2"] = d["factor_hidr_real"] / d["factor_hidr_final"].clip(lower=eps)
    d["abs_log_ml1"] = np.abs(np.log(d["ratio_ml1"].clip(lower=eps)))
    d["abs_log_ml2"] = np.abs(np.log(d["ratio_ml2"].clip(lower=eps)))

    # score ponderado por tallos (impacto)
    w = pd.to_numeric(d.get("tallos"), errors="coerce").fillna(1.0).clip(lower=0.0)
    d["wscore"] = d["abs_log_ml2"] * w

    # top improve/worst según mejora en abs_log (ponderado)
    d["impr"] = (d["abs_log_ml1"] - d["abs_log_ml2"]) * w

    best = d.sort_values("impr", ascending=False).head(10).assign(sample_group="TOP_IMPROVE_10")
    worst = d.sort_values("impr", ascending=True).head(10).assign(sample_group="TOP_WORSE_10")
    ex = pd.concat([best, worst], ignore_index=True)

    ex_out = ex[[
        "fecha", fpp, "grado", "destino", "factor_hidr_ml1", "factor_hidr_final",
        "factor_hidr_real", "ratio_ml1", "ratio_ml2", "impr", "wscore", "sample_group"
    ]].rename(columns={fpp: "fecha_post_pred_used"})

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(g, out_global)
    write_parquet(bd, out_dest)
    write_parquet(bg, out_gr)
    write_parquet(dist, out_dist)
    write_parquet(ex_out, out_examples)

    print("\n=== ML2 HIDR POSCOSECHA AUDIT ===")
    print(f"KPI global parquet : {out_global}")
    print(f"KPI por destino    : {out_dest}")
    print(f"KPI por grado      : {out_gr}")
    print(f"Ratio dist parquet : {out_dist}")
    print(f"Examples 10x2      : {out_examples}")

    print("\n--- KPI GLOBAL ---")
    print(g.to_string(index=False))
    print("\n--- RATIO DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI DESTINO ---")
    print(bd.to_string(index=False))
    print("\n--- KPI GRADO (head) ---")
    print(bg.head(13).to_string(index=False))
    print("\n--- EXAMPLES (10x2) ---")
    print(ex_out.to_string(index=False))


if __name__ == "__main__":
    main()
