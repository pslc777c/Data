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
GOLD = DATA / "gold"
SILVER = DATA / "silver"

IN_GLOBAL = EVAL / "ml2_poscosecha_baseline_eval_global.parquet"
IN_BY_DEST = EVAL / "ml2_poscosecha_baseline_eval_by_destino.parquet"
IN_BY_GR = EVAL / "ml2_poscosecha_baseline_eval_by_grado.parquet"
IN_COV = EVAL / "ml2_poscosecha_baseline_eval_coverage.parquet"

# Para ejemplos (top improve/worse) calculamos con el join directo
IN_PRED_FULL = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"
IN_REAL_HIDR_DH = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"
IN_REAL_MERMA_AJ = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_KPI_GLOBAL = EVAL / "audit_poscosecha_baseline_kpi_global_{ts}.parquet"
OUT_KPI_DEST = EVAL / "audit_poscosecha_baseline_kpi_by_destino_{ts}.parquet"
OUT_KPI_GR = EVAL / "audit_poscosecha_baseline_kpi_by_grado_{ts}.parquet"
OUT_COV = EVAL / "audit_poscosecha_baseline_coverage_{ts}.parquet"
OUT_EXAMPLES = EVAL / "audit_poscosecha_baseline_examples_10x2_{ts}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date_default() -> pd.Timestamp:
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    aod = _as_of_date_default()

    g = read_parquet(IN_GLOBAL).copy()
    bd = read_parquet(IN_BY_DEST).copy()
    bg = read_parquet(IN_BY_GR).copy()
    cov = read_parquet(IN_COV).copy()

    # Persist snapshot
    write_parquet(g, Path(str(OUT_KPI_GLOBAL).format(ts=ts)))
    write_parquet(bd, Path(str(OUT_KPI_DEST).format(ts=ts)))
    write_parquet(bg, Path(str(OUT_KPI_GR).format(ts=ts)))
    write_parquet(cov, Path(str(OUT_COV).format(ts=ts)))

    # -------------------------
    # Examples (top improve/worse) sobre error de cajas_postcosecha_ml1 vs real "factor-based"
    # Nota: como no tenemos cajas_post real aquí, usamos un proxy robusto:
    #   score = |log_ratio_hidr| + |log_ratio_desp| + |log_ratio_ajuste| + (abs_err_dh_days/7)
    # y mostramos TOP 10 mejores (menor score) y TOP 10 peores (mayor score)
    # -------------------------
    pred = read_parquet(IN_PRED_FULL).copy()
    pred.columns = [str(c).strip() for c in pred.columns]
    pred["fecha"] = _to_date(pred["fecha"])
    pred["grado"] = _canon_int(pred["grado"])
    pred["destino"] = _canon_str(pred["destino"])

    # columnas pred
    dh_col = None
    for c in ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias"]:
        if c in pred.columns:
            dh_col = c
            break
    fecha_post_pred_col = None
    for c in ["fecha_post_pred_ml1", "fecha_post_pred", "fecha_post_ml1"]:
        if c in pred.columns:
            fecha_post_pred_col = c
            break
    if fecha_post_pred_col:
        pred[fecha_post_pred_col] = _to_date(pred[fecha_post_pred_col])

    hidr_col = None
    for c in ["factor_hidr_ml1", "factor_hidr"]:
        if c in pred.columns:
            hidr_col = c
            break
    desp_col = None
    for c in ["factor_desp_ml1", "factor_desp"]:
        if c in pred.columns:
            desp_col = c
            break
    aj_col = None
    for c in ["ajuste_ml1", "factor_ajuste_ml1", "factor_ajuste"]:
        if c in pred.columns:
            aj_col = c
            break

    pred = pred[(pred["fecha"] <= aod)].copy()
    if fecha_post_pred_col:
        pred = pred[(pred[fecha_post_pred_col] <= aod)].copy()

    real_hd = read_parquet(IN_REAL_HIDR_DH).copy()
    real_hd.columns = [str(c).strip() for c in real_hd.columns]
    real_hd["fecha_cosecha"] = _to_date(real_hd["fecha_cosecha"])
    real_hd["fecha_post"] = _to_date(real_hd["fecha_post"])
    real_hd["grado"] = _canon_int(real_hd["grado"])
    real_hd["destino"] = _canon_str(real_hd["destino"])
    real_hd["tallos"] = pd.to_numeric(real_hd["tallos"], errors="coerce").fillna(0.0)
    real_hd["dh_dias"] = pd.to_numeric(real_hd["dh_dias"], errors="coerce")
    real_hd["peso_base_g"] = pd.to_numeric(real_hd["peso_base_g"], errors="coerce")
    real_hd["peso_post_g"] = pd.to_numeric(real_hd["peso_post_g"], errors="coerce")
    real_hd["factor_hidr_real"] = _safe_div(real_hd["peso_post_g"], real_hd["peso_base_g"])
    real_hd = real_hd[(real_hd["fecha_cosecha"] <= aod) & (real_hd["fecha_post"] <= aod)].copy()

    rg = (
        real_hd.groupby(["fecha_cosecha", "grado", "destino"], as_index=False)
        .agg(
            tallos=("tallos", "sum"),
            dh_real=("dh_dias", "median"),
            factor_hidr_real=("factor_hidr_real", "median"),
            fecha_post_real=("fecha_post", "median"),
        )
    )

    real_ma = read_parquet(IN_REAL_MERMA_AJ).copy()
    real_ma.columns = [str(c).strip() for c in real_ma.columns]
    real_ma["fecha_post"] = _to_date(real_ma["fecha_post"])
    real_ma["destino"] = _canon_str(real_ma["destino"])
    real_ma["factor_desp"] = pd.to_numeric(real_ma["factor_desp"], errors="coerce")
    real_ma["factor_ajuste"] = pd.to_numeric(real_ma["factor_ajuste"], errors="coerce")
    real_ma = real_ma[real_ma["fecha_post"] <= aod].copy()

    df = pred.merge(
        rg,
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    df["fecha_post_key"] = df["fecha_post_real"]
    if fecha_post_pred_col:
        df["fecha_post_key"] = df["fecha_post_key"].where(df["fecha_post_key"].notna(), df[fecha_post_pred_col])

    df = df.merge(
        real_ma.rename(columns={"fecha_post": "fecha_post_key"}),
        on=["fecha_post_key", "destino"],
        how="left",
    )

    # métricas por fila
    df["dh_pred"] = pd.to_numeric(df[dh_col], errors="coerce") if dh_col else np.nan
    df["abs_err_dh"] = (pd.to_numeric(df["dh_pred"], errors="coerce") - pd.to_numeric(df["dh_real"], errors="coerce")).abs()

    df["hidr_pred"] = pd.to_numeric(df[hidr_col], errors="coerce") if hidr_col else np.nan
    df["log_ratio_hidr"] = np.log(_safe_div(df["factor_hidr_real"], df["hidr_pred"]).clip(lower=1e-6))

    df["desp_pred"] = pd.to_numeric(df[desp_col], errors="coerce") if desp_col else np.nan
    df["log_ratio_desp"] = np.log(_safe_div(df["factor_desp"], df["desp_pred"]).clip(lower=1e-6))

    df["aj_pred"] = pd.to_numeric(df[aj_col], errors="coerce") if aj_col else np.nan
    df["log_ratio_aj"] = np.log(_safe_div(df["factor_ajuste"], df["aj_pred"]).clip(lower=1e-6))

    # score proxy (menor = mejor)
    df["score"] = (
        df["abs_err_dh"].fillna(0.0) / 7.0
        + df["log_ratio_hidr"].abs().fillna(0.0)
        + df["log_ratio_desp"].abs().fillna(0.0)
        + df["log_ratio_aj"].abs().fillna(0.0)
    )
    # ponderar por tallos para ranking
    df["wscore"] = df["score"] * pd.to_numeric(df["tallos"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # agregamos por (fecha, grado, destino) para evitar que bloque domine por cantidad de filas
    key = ["fecha", "grado", "destino"]
    ex = (
        df.groupby(key, dropna=False, as_index=False)
          .agg(
              tallos=("tallos", "sum"),
              score=("score", "mean"),
              wscore=("wscore", "sum"),
              abs_err_dh=("abs_err_dh", "mean"),
              log_ratio_hidr=("log_ratio_hidr", "mean"),
              log_ratio_desp=("log_ratio_desp", "mean"),
              log_ratio_aj=("log_ratio_aj", "mean"),
              fecha_post_real=("fecha_post_real", "first"),
              fecha_post_key=("fecha_post_key", "first"),
          )
    )

    top_best = ex.sort_values(["wscore", "tallos"], ascending=[True, False]).head(10).copy()
    top_best["sample_group"] = "TOP_BEST_10"
    top_worst = ex.sort_values(["wscore", "tallos"], ascending=[False, False]).head(10).copy()
    top_worst["sample_group"] = "TOP_WORST_10"

    examples = pd.concat([top_best, top_worst], ignore_index=True)
    out_ex_path = Path(str(OUT_EXAMPLES).format(ts=ts))
    write_parquet(examples, out_ex_path)

    # -------------------------
    # Print
    # -------------------------
    print("\n=== POSCOSECHA ML2 BASELINE AUDIT ===")
    print(f"KPI global parquet : {Path(str(OUT_KPI_GLOBAL).format(ts=ts))}")
    print(f"KPI por destino    : {Path(str(OUT_KPI_DEST).format(ts=ts))}")
    print(f"KPI por grado      : {Path(str(OUT_KPI_GR).format(ts=ts))}")
    print(f"Coverage parquet   : {Path(str(OUT_COV).format(ts=ts))}")
    print(f"Examples 10x2      : {out_ex_path}")

    print("\n--- KPI GLOBAL ---")
    print(g.to_string(index=False))

    print("\n--- COVERAGE ---")
    print(cov.to_string(index=False))

    print("\n--- KPI DESTINO (head) ---")
    print(bd.head(10).to_string(index=False))

    print("\n--- KPI GRADO (head) ---")
    print(bg.head(10).to_string(index=False))

    print("\n--- EXAMPLES (10x2) ---")
    print(examples.to_string(index=False))


if __name__ == "__main__":
    main()
