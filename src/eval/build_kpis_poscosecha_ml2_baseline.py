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
GOLD = DATA / "gold"
SILVER = DATA / "silver"
EVAL = DATA / "eval" / "ml2"

# =========================
# INPUTS
# =========================
IN_PRED_FULL = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"

# Real hidr + DH (por fecha_cosecha/fecha_post, grado, destino)
IN_REAL_HIDR_DH = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

# Real merma/ajuste (por fecha_post, destino)
IN_REAL_MERMA_AJ = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

# =========================
# OUTPUTS
# =========================
OUT_GLOBAL = EVAL / "ml2_poscosecha_baseline_eval_global.parquet"
OUT_BY_DESTINO = EVAL / "ml2_poscosecha_baseline_eval_by_destino.parquet"
OUT_BY_GRADO = EVAL / "ml2_poscosecha_baseline_eval_by_grado.parquet"
OUT_COVERAGE = EVAL / "ml2_poscosecha_baseline_eval_coverage.parquet"


# =========================
# HELPERS
# =========================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date_default() -> pd.Timestamp:
    # Regla operativa: "hoy" no se usa porque no está cerrado → usar hoy-1
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return a / b.replace(0, np.nan)


def mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else np.nan


def wape(abs_err: pd.Series, y_true: pd.Series) -> float:
    abs_err = pd.to_numeric(abs_err, errors="coerce").fillna(0.0)
    y_true = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return np.nan
    return float(np.sum(abs_err) / denom)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def main(as_of_date: str | None = None) -> None:
    AOD = _as_of_date_default() if as_of_date is None else pd.to_datetime(as_of_date).normalize()

    pred = read_parquet(IN_PRED_FULL).copy()
    pred.columns = [str(c).strip() for c in pred.columns]

    real_hd = read_parquet(IN_REAL_HIDR_DH).copy()
    real_hd.columns = [str(c).strip() for c in real_hd.columns]

    real_ma = read_parquet(IN_REAL_MERMA_AJ).copy()
    real_ma.columns = [str(c).strip() for c in real_ma.columns]

    # =========================
    # Canon pred
    # =========================
    _require_cols(pred, ["fecha", "grado", "destino"], "pred_poscosecha_ml2_full")

    pred["fecha"] = _to_date(pred["fecha"])
    pred["grado"] = _canon_int(pred["grado"])
    pred["destino"] = _canon_str(pred["destino"])

    # columnas ML1 esperadas (con fallback)
    # DH
    dh_col = None
    for c in ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias"]:
        if c in pred.columns:
            dh_col = c
            break

    # fecha_post predicha (si existe, mejor)
    fecha_post_pred_col = None
    for c in ["fecha_post_pred_ml1", "fecha_post_pred", "fecha_post_ml1"]:
        if c in pred.columns:
            fecha_post_pred_col = c
            break

    if fecha_post_pred_col:
        pred[fecha_post_pred_col] = _to_date(pred[fecha_post_pred_col])
    elif dh_col:
        pred["fecha_post_pred_ml1"] = pred["fecha"] + pd.to_timedelta(
            pd.to_numeric(pred[dh_col], errors="coerce").fillna(0).astype(int), unit="D"
        )
        fecha_post_pred_col = "fecha_post_pred_ml1"
    else:
        raise KeyError("No encuentro dh ni fecha_post_pred en pred. Espero dh_dias_ml1 o fecha_post_pred_ml1.")

    # Hidr
    hidr_col = None
    for c in ["factor_hidr_ml1", "factor_hidr"]:
        if c in pred.columns:
            hidr_col = c
            break

    # Desp
    desp_col = None
    for c in ["factor_desp_ml1", "factor_desp"]:
        if c in pred.columns:
            desp_col = c
            break

    # Ajuste
    aj_col = None
    for c in ["ajuste_ml1", "factor_ajuste_ml1", "factor_ajuste"]:
        if c in pred.columns:
            aj_col = c
            break

    # filtro as_of
    pred = pred[pred["fecha"] <= AOD].copy()
    pred = pred[pred[fecha_post_pred_col] <= AOD].copy()

    # =========================
    # Canon real hidr/dh
    # =========================
    _require_cols(
        real_hd,
        ["fecha_cosecha", "fecha_post", "dh_dias", "grado", "destino", "tallos", "peso_base_g", "peso_post_g"],
        "fact_hidratacion_real_post_grado_destino",
    )
    real_hd["fecha_cosecha"] = _to_date(real_hd["fecha_cosecha"])
    real_hd["fecha_post"] = _to_date(real_hd["fecha_post"])
    real_hd["grado"] = _canon_int(real_hd["grado"])
    real_hd["destino"] = _canon_str(real_hd["destino"])
    real_hd["tallos"] = pd.to_numeric(real_hd["tallos"], errors="coerce").fillna(0.0)
    real_hd["dh_dias"] = pd.to_numeric(real_hd["dh_dias"], errors="coerce")

    real_hd["peso_base_g"] = pd.to_numeric(real_hd["peso_base_g"], errors="coerce")
    real_hd["peso_post_g"] = pd.to_numeric(real_hd["peso_post_g"], errors="coerce")
    real_hd["factor_hidr_real"] = _safe_div(real_hd["peso_post_g"], real_hd["peso_base_g"])

    # filtro as_of (NO usar hoy)
    real_hd = real_hd[(real_hd["fecha_cosecha"] <= AOD) & (real_hd["fecha_post"] <= AOD)].copy()

    # colapsar real a grano único por (fecha_cosecha, grado, destino)
    # (si hay múltiples destinos/grado por día, sum tallos y ponderar hidratación)
    rg = (
        real_hd.groupby(["fecha_cosecha", "grado", "destino"], as_index=False)
        .agg(
            tallos=("tallos", "sum"),
            dh_dias=("dh_dias", "median"),
            peso_base_g=("peso_base_g", "median"),
            peso_post_g=("peso_post_g", "median"),
            factor_hidr_real=("factor_hidr_real", "median"),
            fecha_post_real=("fecha_post", "median"),
        )
    )

    # =========================
    # Canon real merma/ajuste
    # =========================
    _require_cols(
        real_ma,
        ["fecha_post", "destino", "factor_desp", "factor_ajuste"],
        "dim_mermas_ajuste_fecha_post_destino",
    )
    real_ma["fecha_post"] = _to_date(real_ma["fecha_post"])
    real_ma["destino"] = _canon_str(real_ma["destino"])
    real_ma["factor_desp"] = pd.to_numeric(real_ma["factor_desp"], errors="coerce")
    real_ma["factor_ajuste"] = pd.to_numeric(real_ma["factor_ajuste"], errors="coerce")
    real_ma = real_ma[real_ma["fecha_post"] <= AOD].copy()

    # =========================
    # JOIN: pred vs real (DH/HIDR) por fecha_cosecha (pred.fecha)
    # =========================
    df = pred.merge(
        rg,
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    # JOIN merma/ajuste real por fecha_post_real (prefer) y destino.
    # Si no hay fecha_post_real, usar fecha_post_pred (evaluación “contra calendario”)
    df["fecha_post_key"] = df["fecha_post_real"]
    df["fecha_post_key"] = df["fecha_post_key"].where(df["fecha_post_key"].notna(), df[fecha_post_pred_col])
    df = df.merge(
        real_ma.rename(columns={"fecha_post": "fecha_post_key"}),
        on=["fecha_post_key", "destino"],
        how="left",
        suffixes=("", "_realma"),
    )

    # =========================
    # KPIs por componente
    # =========================
    # Pesos: usar tallos reales como peso
    w = pd.to_numeric(df["tallos"], errors="coerce").fillna(0.0)

    # --- DH ---
    if dh_col:
        df["dh_pred"] = pd.to_numeric(df[dh_col], errors="coerce")
    else:
        df["dh_pred"] = pd.to_numeric(_safe_div((df[fecha_post_pred_col] - df["fecha"]).dt.days, 1.0), errors="coerce")

    df["dh_real"] = pd.to_numeric(df["dh_dias"], errors="coerce")
    m_dh = df["dh_pred"].notna() & df["dh_real"].notna()
    df.loc[m_dh, "err_dh_days"] = df.loc[m_dh, "dh_pred"] - df.loc[m_dh, "dh_real"]
    df.loc[m_dh, "abs_err_dh_days"] = df.loc[m_dh, "err_dh_days"].abs()

    # --- HIDR ---
    if hidr_col:
        df["hidr_pred_factor"] = pd.to_numeric(df[hidr_col], errors="coerce")
    else:
        df["hidr_pred_factor"] = np.nan
    df["hidr_real_factor"] = pd.to_numeric(df["factor_hidr_real"], errors="coerce")

    m_h = df["hidr_pred_factor"].notna() & df["hidr_real_factor"].notna()
    df.loc[m_h, "ratio_hidr"] = _safe_div(df.loc[m_h, "hidr_real_factor"], df.loc[m_h, "hidr_pred_factor"])
    df.loc[m_h, "log_ratio_hidr"] = np.log(df.loc[m_h, "ratio_hidr"].clip(lower=1e-6))

    # --- DESP ---
    if desp_col:
        df["desp_pred_factor"] = pd.to_numeric(df[desp_col], errors="coerce")
    else:
        df["desp_pred_factor"] = np.nan
    df["desp_real_factor"] = pd.to_numeric(df["factor_desp"], errors="coerce")

    m_d = df["desp_pred_factor"].notna() & df["desp_real_factor"].notna()
    df.loc[m_d, "ratio_desp"] = _safe_div(df.loc[m_d, "desp_real_factor"], df.loc[m_d, "desp_pred_factor"])
    df.loc[m_d, "log_ratio_desp"] = np.log(df.loc[m_d, "ratio_desp"].clip(lower=1e-6))

    # --- AJUSTE ---
    if aj_col:
        df["aj_pred_factor"] = pd.to_numeric(df[aj_col], errors="coerce")
    else:
        df["aj_pred_factor"] = np.nan
    df["aj_real_factor"] = pd.to_numeric(df["factor_ajuste"], errors="coerce")

    m_a = df["aj_pred_factor"].notna() & df["aj_real_factor"].notna()
    df.loc[m_a, "ratio_ajuste"] = _safe_div(df.loc[m_a, "aj_real_factor"], df.loc[m_a, "aj_pred_factor"])
    df.loc[m_a, "log_ratio_ajuste"] = np.log(df.loc[m_a, "ratio_ajuste"].clip(lower=1e-6))

    # =========================
    # GLOBAL KPIs
    # =========================
    def _kpi_block(sub: pd.DataFrame) -> dict:
        ww = pd.to_numeric(sub["tallos"], errors="coerce").fillna(0.0)

        out = {
            "n_rows_pred": int(len(sub)),
            "n_rows_with_real_hd": int(sub["tallos"].notna().sum()),
            "as_of_date": AOD,
            "created_at": pd.Timestamp(datetime.now()).normalize(),
            "dh_col_used": str(dh_col),
            "fecha_post_pred_col_used": str(fecha_post_pred_col),
            "hidr_col_used": str(hidr_col),
            "desp_col_used": str(desp_col),
            "ajuste_col_used": str(aj_col),
        }

        # DH
        dsub = sub[sub["err_dh_days"].notna()].copy()
        out["n_dh"] = int(len(dsub))
        out["mae_dh_days"] = mae(dsub["err_dh_days"])
        out["bias_dh_days"] = float(np.nanmean(pd.to_numeric(dsub["err_dh_days"], errors="coerce"))) if len(dsub) else np.nan
        out["wape_abs_dh_days_wt_tallos"] = wape(dsub["abs_err_dh_days"], ww.loc[dsub.index]) if len(dsub) else np.nan

        # Hidr
        hsub = sub[sub["log_ratio_hidr"].notna()].copy()
        out["n_hidr"] = int(len(hsub))
        out["mae_log_ratio_hidr"] = mae(hsub["log_ratio_hidr"])
        out["median_ratio_hidr"] = float(np.nanmedian(pd.to_numeric(hsub["ratio_hidr"], errors="coerce"))) if len(hsub) else np.nan
        out["p05_ratio_hidr"] = float(np.nanpercentile(pd.to_numeric(hsub["ratio_hidr"], errors="coerce"), 5)) if len(hsub) else np.nan
        out["p95_ratio_hidr"] = float(np.nanpercentile(pd.to_numeric(hsub["ratio_hidr"], errors="coerce"), 95)) if len(hsub) else np.nan

        # Desp
        dpsub = sub[sub["log_ratio_desp"].notna()].copy()
        out["n_desp"] = int(len(dpsub))
        out["mae_log_ratio_desp"] = mae(dpsub["log_ratio_desp"])
        out["median_ratio_desp"] = float(np.nanmedian(pd.to_numeric(dpsub["ratio_desp"], errors="coerce"))) if len(dpsub) else np.nan

        # Ajuste
        asub = sub[sub["log_ratio_ajuste"].notna()].copy()
        out["n_ajuste"] = int(len(asub))
        out["mae_log_ratio_ajuste"] = mae(asub["log_ratio_ajuste"])
        out["median_ratio_ajuste"] = float(np.nanmedian(pd.to_numeric(asub["ratio_ajuste"], errors="coerce"))) if len(asub) else np.nan

        return out

    global_df = pd.DataFrame([_kpi_block(df)])

    # =========================
    # BY DESTINO / BY GRADO
    # =========================
    by_dest = []
    for dest, g in df.groupby("destino", dropna=False):
        row = _kpi_block(g)
        row["destino"] = str(dest)
        by_dest.append(row)
    by_dest_df = pd.DataFrame(by_dest).sort_values("n_rows_pred", ascending=False)

    by_gr = []
    for gr, g in df.groupby("grado", dropna=False):
        row = _kpi_block(g)
        row["grado"] = int(gr) if pd.notna(gr) else None
        by_gr.append(row)
    by_gr_df = pd.DataFrame(by_gr).sort_values("n_rows_pred", ascending=False)

    # =========================
    # COVERAGE / SANITY
    # =========================
    cov = pd.DataFrame([{
        "as_of_date": AOD,
        "rows_pred": int(len(pred)),
        "rows_join_real_hd": int(df["tallos"].notna().sum()),
        "coverage_real_hd": float(df["tallos"].notna().mean()) if len(df) else np.nan,
        "rows_join_real_ma": int(df["factor_desp"].notna().sum()),
        "coverage_real_ma": float(df["factor_desp"].notna().mean()) if len(df) else np.nan,
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(global_df, OUT_GLOBAL)
    write_parquet(by_dest_df, OUT_BY_DESTINO)
    write_parquet(by_gr_df, OUT_BY_GRADO)
    write_parquet(cov, OUT_COVERAGE)

    print(f"[OK] Wrote global : {OUT_GLOBAL}")
    print(global_df.to_string(index=False))
    print(f"\n[OK] Wrote destino: {OUT_BY_DESTINO} rows={len(by_dest_df)}")
    print(by_dest_df.head(10).to_string(index=False))
    print(f"\n[OK] Wrote grado  : {OUT_BY_GRADO} rows={len(by_gr_df)}")
    print(by_gr_df.head(10).to_string(index=False))
    print(f"\n[OK] Wrote coverage: {OUT_COVERAGE}")
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main(as_of_date=None)
