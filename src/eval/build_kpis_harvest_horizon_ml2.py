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
GOLD = DATA / "gold"

# (compat) path viejo/hardcodeado
LEGACY_IN_FINAL_BT = EVAL / "backtest_pred_harvest_horizon_final_ml2.parquet"

OUT_GLOBAL = EVAL / "ml2_harvest_horizon_eval_global.parquet"
OUT_DIST = EVAL / "ml2_harvest_horizon_eval_dist.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _safe_series(obj) -> pd.Series:
    # evita el error tipo "numpy.float64 has no dropna"
    if isinstance(obj, pd.Series):
        return obj
    if obj is None:
        return pd.Series(dtype="float64")
    # si viene escalar, lo convertimos a serie
    return pd.Series([obj])


def _mae(err: pd.Series, w: pd.Series | None = None) -> float:
    err = pd.to_numeric(err, errors="coerce")
    if w is None:
        return float(np.nanmean(np.abs(err))) if len(err) else np.nan
    w = pd.to_numeric(w, errors="coerce").fillna(0.0).astype(float)
    m = err.notna() & np.isfinite(w)
    if not m.any():
        return np.nan
    ww = w[m].values
    ww = np.where(ww <= 0, 1.0, ww)
    return float(np.sum(np.abs(err[m].values) * ww) / np.sum(ww))


def _dist(s: pd.Series) -> dict:
    s = pd.to_numeric(_safe_series(s), errors="coerce").dropna()
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


def _resolve_input_parquet() -> Path:
    """
    Busca el parquet correcto del harvest_horizon ML2.
    Prioriza el legacy si existe, si no: glob en eval/ml2 y gold.
    """
    if LEGACY_IN_FINAL_BT.exists():
        return LEGACY_IN_FINAL_BT

    patterns = [
        (EVAL, "*harvest_horizon*ml2*.parquet"),
        (GOLD, "*harvest_horizon*ml2*.parquet"),
        (EVAL, "*horizon*ml2*.parquet"),
        (GOLD, "*horizon*ml2*.parquet"),
    ]

    hits: list[Path] = []
    for base, pat in patterns:
        hits.extend(sorted(base.glob(pat)))

    hits = [p for p in hits if p.is_file()]
    if not hits:
        raise FileNotFoundError(
            "No encontré ningún parquet de harvest_horizon ML2.\n"
            f"Busqué en:\n"
            f" - {EVAL} con *harvest_horizon*ml2*.parquet / *horizon*ml2*.parquet\n"
            f" - {GOLD} con *harvest_horizon*ml2*.parquet / *horizon*ml2*.parquet\n"
            f"Y tampoco existe el legacy:\n - {LEGACY_IN_FINAL_BT}"
        )

    # más reciente por modified time
    hits.sort(key=lambda p: p.stat().st_mtime)
    return hits[-1]


def main() -> None:
    in_path = _resolve_input_parquet()

    df = read_parquet(in_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Canon mínimos si existen
    c_fecha = _pick_first(df, ["fecha", "fecha_cosecha", "date"])
    if c_fecha:
        df[c_fecha] = _to_date(df[c_fecha])

    c_bloque = _pick_first(df, ["bloque_base", "bloque", "block"])
    if c_bloque:
        df[c_bloque] = _canon_str(df[c_bloque])

    c_var = _pick_first(df, ["variedad_canon", "variedad", "cultivar"])
    if c_var:
        df[c_var] = _canon_str(df[c_var])

    # Intento resolver columnas clave (varían según tu pipeline)
    c_real = _pick_first(df, ["horizon_real", "harvest_horizon_real", "horizon_dias_real", "horizon_days_real"])
    c_ml1 = _pick_first(df, ["horizon_ml1", "harvest_horizon_ml1", "horizon_dias_ml1", "horizon_days_ml1"])
    c_final = _pick_first(df, ["horizon_dias_final", "harvest_horizon_final", "horizon_ml2_final", "horizon_days_final"])

    c_w = _pick_first(df, ["tallos_real", "tallos", "peso_real", "area", "w"])

    # Errores si hay real
    d = df.copy()
    if c_real and c_final:
        d["err_ml2_days"] = pd.to_numeric(d[c_real], errors="coerce") - pd.to_numeric(d[c_final], errors="coerce")
    else:
        d["err_ml2_days"] = np.nan

    if c_real and c_ml1:
        d["err_ml1_days"] = pd.to_numeric(d[c_real], errors="coerce") - pd.to_numeric(d[c_ml1], errors="coerce")
    else:
        d["err_ml1_days"] = np.nan

    # Improvement si hay ambos
    if d["err_ml1_days"].notna().any() and d["err_ml2_days"].notna().any():
        d["improvement_abs_days"] = d["err_ml1_days"].abs() - d["err_ml2_days"].abs()
    else:
        d["improvement_abs_days"] = np.nan

    w = pd.to_numeric(d[c_w], errors="coerce").fillna(0.0) if c_w else None

    out_g = pd.DataFrame([{
        "input_file": str(in_path),
        "n_rows": int(len(d)),
        "n_dates": int(d[c_fecha].nunique()) if c_fecha else None,
        "has_real": bool(c_real is not None),
        "col_real": c_real,
        "col_ml1": c_ml1,
        "col_final": c_final,
        "mae_ml1_days": _mae(d["err_ml1_days"], w=w) if c_real and c_ml1 else np.nan,
        "mae_ml2_days": _mae(d["err_ml2_days"], w=w) if c_real and c_final else np.nan,
        "bias_ml1_days": float(np.nanmean(d["err_ml1_days"])) if c_real and c_ml1 else np.nan,
        "bias_ml2_days": float(np.nanmean(d["err_ml2_days"])) if c_real and c_final else np.nan,
        "improvement_abs_mae_days": (
            (_mae(d["err_ml1_days"], w=w) - _mae(d["err_ml2_days"], w=w))
            if (c_real and c_ml1 and c_final) else np.nan
        ),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # Dist: preferimos columnas de error si existen; si no, intentamos con cualquier delta/err conocido
    c_err_pref = _pick_first(df, [
        "err_horizon_days_pred_ml2",
        "err_harvest_horizon_days_pred_ml2",
        "err_horizon_days_ml2",
        "err_ml2_days",
    ])
    if c_err_pref == "err_ml2_days":
        dist_src = d["err_ml2_days"]
    elif c_err_pref and c_err_pref in df.columns:
        dist_src = df[c_err_pref]
    else:
        dist_src = d["err_ml2_days"]

    dist = _dist(dist_src)
    out_dist = pd.DataFrame([{
        **dist,
        "input_file": str(in_path),
        "dist_source": str(c_err_pref or "err_ml2_days"),
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out_g, OUT_GLOBAL)
    write_parquet(out_dist, OUT_DIST)

    print(f"[OK] Wrote global: {OUT_GLOBAL}")
    print(out_g.to_string(index=False))
    print(f"\n[OK] Wrote dist  : {OUT_DIST}")
    print(out_dist.to_string(index=False))


if __name__ == "__main__":
    main()