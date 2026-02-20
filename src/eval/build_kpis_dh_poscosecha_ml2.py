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

IN_FINAL = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino_final.parquet"
IN_REAL = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

OUT_GLOBAL = EVAL / "ml2_dh_poscosecha_eval_global.parquet"
OUT_BY_DESTINO = EVAL / "ml2_dh_poscosecha_eval_by_destino.parquet"
OUT_BY_GRADO = EVAL / "ml2_dh_poscosecha_eval_by_grado.parquet"
OUT_DIST = EVAL / "ml2_dh_poscosecha_eval_delta_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s) -> pd.Series:
    """
    Robust: devuelve Int64 nullable si existe, si no => Series Int64 vacía/NaN.
    Evita TypeError 'Int64' not understood en entornos raros.
    """
    if s is None:
        return pd.Series([pd.NA], dtype="Int64")
    out = pd.to_numeric(s, errors="coerce")
    try:
        return out.astype("Int64")
    except TypeError:
        # fallback si el pandas/engine no soporta Int64 nullable bien
        return out.astype("float").round().astype("Int64")


def _mae(x: pd.Series, w: pd.Series | None = None) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if w is None:
        return float(np.nanmean(np.abs(x))) if len(x) else np.nan
    w = pd.to_numeric(w, errors="coerce").fillna(0.0).astype(float)
    m = x.notna() & np.isfinite(w)
    if not m.any():
        return np.nan
    ww = w[m].values
    ww = np.where(ww <= 0, 1.0, ww)
    return float(np.sum(np.abs(x[m].values) * ww) / np.sum(ww))


def _dist(s) -> dict:
    """
    FIX: aceptar Series | array | scalar | None.
    Siempre convierte a Series antes de dropna().
    """
    if s is None:
        s = pd.Series([], dtype="float")
    elif not isinstance(s, pd.Series):
        s = pd.Series([s])

    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


def _pick_first_col(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def main() -> None:
    fin = read_parquet(IN_FINAL).copy()
    real = read_parquet(IN_REAL).copy()
    fin.columns = [str(c).strip() for c in fin.columns]
    real.columns = [str(c).strip() for c in real.columns]

    fin["fecha"] = _to_date(fin["fecha"])
    fin["destino"] = _canon_str(fin["destino"])
    fin["grado"] = _canon_int(fin["grado"])

    # Real
    real["fecha_cosecha"] = _to_date(real["fecha_cosecha"])
    real["destino"] = _canon_str(real["destino"])
    real["grado"] = _canon_int(real["grado"])
    real["tallos"] = pd.to_numeric(real.get("tallos"), errors="coerce").fillna(0.0)
    real["dh_dias"] = _canon_int(real.get("dh_dias"))

    real_g = (
        real.groupby(["fecha_cosecha", "grado", "destino"], dropna=False, as_index=False)
            .agg(dh_real=("dh_dias", "median"), tallos_real=("tallos", "sum"))
    )

    df = fin.merge(
        real_g,
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    # Errores (robusto si faltan columnas)
    df["dh_ml1"] = _canon_int(df.get("dh_ml1"))
    df["dh_dias_final"] = _canon_int(df.get("dh_dias_final"))
    df["dh_real"] = _canon_int(df.get("dh_real"))
    df["tallos_real"] = pd.to_numeric(df.get("tallos_real"), errors="coerce").fillna(0.0)

    m = df["dh_real"].notna() & df["dh_ml1"].notna() & df["dh_dias_final"].notna()
    d = df.loc[m].copy()

    d["err_ml1_days"] = (pd.to_numeric(d["dh_real"], errors="coerce") - pd.to_numeric(d["dh_ml1"], errors="coerce"))
    d["err_ml2_days"] = (pd.to_numeric(d["dh_real"], errors="coerce") - pd.to_numeric(d["dh_dias_final"], errors="coerce"))
    d["improvement_abs_days"] = d["err_ml1_days"].abs() - d["err_ml2_days"].abs()

    out_g = pd.DataFrame([{
        "n_rows": int(len(d)),
        "n_dates": int(d["fecha"].nunique()) if len(d) else 0,
        "mae_ml1_days": _mae(d["err_ml1_days"], w=d["tallos_real"]) if len(d) else np.nan,
        "mae_ml2_days": _mae(d["err_ml2_days"], w=d["tallos_real"]) if len(d) else np.nan,
        "bias_ml1_days": float(np.nanmean(d["err_ml1_days"])) if len(d) else np.nan,
        "bias_ml2_days": float(np.nanmean(d["err_ml2_days"])) if len(d) else np.nan,
        "improvement_abs_mae_days": (
            (_mae(d["err_ml1_days"], w=d["tallos_real"]) - _mae(d["err_ml2_days"], w=d["tallos_real"]))
            if len(d) else np.nan
        ),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # By destino (si d está vacío, produce DF vacío con columnas esperadas)
    rows_dest = []
    if len(d):
        for dest, g in d.groupby("destino"):
            rows_dest.append({
                "destino": str(dest),
                "n_rows": int(len(g)),
                "mae_ml1_days": _mae(g["err_ml1_days"], w=g["tallos_real"]),
                "mae_ml2_days": _mae(g["err_ml2_days"], w=g["tallos_real"]),
                "bias_ml1_days": float(np.nanmean(g["err_ml1_days"])),
                "bias_ml2_days": float(np.nanmean(g["err_ml2_days"])),
                "improvement_abs_mae_days": _mae(g["err_ml1_days"], w=g["tallos_real"]) - _mae(g["err_ml2_days"], w=g["tallos_real"]),
            })
    out_d = pd.DataFrame(rows_dest)
    if not out_d.empty and "improvement_abs_mae_days" in out_d.columns:
        out_d = out_d.sort_values("improvement_abs_mae_days", ascending=False)

    # By grado
    rows_gr = []
    if len(d):
        for gr, g in d.groupby("grado"):
            rows_gr.append({
                "grado": int(gr) if pd.notna(gr) else None,
                "n_rows": int(len(g)),
                "mae_ml1_days": _mae(g["err_ml1_days"], w=g["tallos_real"]),
                "mae_ml2_days": _mae(g["err_ml2_days"], w=g["tallos_real"]),
                "bias_ml1_days": float(np.nanmean(g["err_ml1_days"])),
                "bias_ml2_days": float(np.nanmean(g["err_ml2_days"])),
                "improvement_abs_mae_days": _mae(g["err_ml1_days"], w=g["tallos_real"]) - _mae(g["err_ml2_days"], w=g["tallos_real"]),
            })
    out_gd = pd.DataFrame(rows_gr)
    if not out_gd.empty and "improvement_abs_mae_days" in out_gd.columns:
        out_gd = out_gd.sort_values("improvement_abs_mae_days", ascending=False)

    # Dist: FIX + fallback de columna
    # Preferimos delta/err que realmente exista en fin
    dist_col = _pick_first_col(fin, [
        "dh_delta_ml2", "dh_delta_ml2_raw",
        "err_dh_days_pred_ml2",
        "err_ml2_days", "err_ml1_days",
    ])
    dist_src = fin[dist_col] if dist_col else None
    dist = _dist(dist_src)

    out_dist = pd.DataFrame([{
        **dist,
        "source_col": dist_col if dist_col else "NONE",
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_d, OUT_BY_DESTINO)
    write_parquet(out_gd, OUT_BY_GRADO)
    write_parquet(out_dist, OUT_DIST)

    print(f"[OK] Wrote global : {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote destino: {OUT_BY_DESTINO} rows={len(out_d)}")
    if len(out_d):
        print(out_d.to_string(index=False))
    print(f"\n[OK] Wrote grado  : {OUT_BY_GRADO} rows={len(out_gd)}")
    if len(out_gd):
        print(out_gd.head(10).to_string(index=False))
    print(f"\n[OK] Wrote dist   : {OUT_DIST}")
    print(out_dist.to_string(index=False))


if __name__ == "__main__":
    main()