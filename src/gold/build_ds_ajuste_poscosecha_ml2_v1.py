from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_desp_grado_dia_bloque_destino_final.parquet"
IN_REAL_MA = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_ajuste_poscosecha_ml2_v1.parquet"


# ----------------------------
# TZ helpers (evita tz-aware vs tz-naive)
# ----------------------------
def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    """Convierte a UTC y deja tz-naive."""
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)


def _as_of_date_naive() -> pd.Timestamp:
    """
    Regla: as_of = hoy-1.
    IMPORTANTE: tz-naive para comparar con columnas datetime64[ns] (naive).
    """
    # utcnow() suele ser tz-naive, pero en algunos entornos puede salir tz-aware; blindamos.
    t = pd.Timestamp.utcnow()
    t = _to_naive_utc(t)
    return (t.normalize() - pd.Timedelta(days=1))


def _to_date_naive(s: pd.Series) -> pd.Series:
    """
    Convierte a datetime y deja tz-naive normalizado.
    """
    dt = pd.to_datetime(s, errors="coerce")
    # Si viene tz-aware, pásalo a UTC y quita tz
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        # si no soporta .dt.tz (por tipos raros), lo dejamos como está
        pass
    return dt.dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _resolve_fecha_post_pred(df: pd.DataFrame) -> str:
    for c in ["fecha_post_pred_final", "fecha_post_pred_used", "fecha_post_pred_ml1", "fecha_post_pred"]:
        if c in df.columns:
            return c
    raise KeyError(
        "No encuentro columna de fecha_post_pred (esperaba fecha_post_pred_final/fecha_post_pred_ml1/...)."
    )


def _resolve_ajuste_ml1(df: pd.DataFrame) -> str:
    for c in ["factor_ajuste_ml1", "ajuste_ml1", "factor_ajuste_seed", "factor_ajuste"]:
        if c in df.columns:
            return c
    raise KeyError(
        "No encuentro columna de ajuste ML1 (esperaba ajuste_ml1 / factor_ajuste_ml1 / factor_ajuste_seed)."
    )


def _weight_series(df: pd.DataFrame) -> pd.Series:
    for c in ["tallos_w", "tallos", "tallos_total_ml2", "tallos_total"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(1.0, index=df.index, dtype="float64")


def main() -> None:
    as_of = _as_of_date_naive()

    uni = read_parquet(IN_UNIVERSE).copy()
    uni.columns = [str(c).strip() for c in uni.columns]

    need = {"fecha", "destino"}
    miss = need - set(uni.columns)
    if miss:
        raise ValueError(f"Universe sin columnas: {sorted(miss)}")

    # ✅ tz-naive
    uni["fecha"] = _to_date_naive(uni["fecha"])
    uni["destino"] = _canon_str(uni["destino"])

    if "grado" in uni.columns:
        uni["grado"] = pd.to_numeric(uni["grado"], errors="coerce").astype("Int64")
    else:
        uni["grado"] = pd.Series(pd.NA, index=uni.index, dtype="Int64")

    fecha_post_col = _resolve_fecha_post_pred(uni)
    uni[fecha_post_col] = _to_date_naive(uni[fecha_post_col])

    ajuste_ml1_col = _resolve_ajuste_ml1(uni)
    uni[ajuste_ml1_col] = pd.to_numeric(uni[ajuste_ml1_col], errors="coerce")

    # ✅ comparación safe: ambos tz-naive
    uni = uni[uni["fecha"].notna() & (uni["fecha"] <= as_of)].copy()

    # ---- Real por fecha_post + destino (NO por grado) ----
    real = read_parquet(IN_REAL_MA).copy()
    real.columns = [str(c).strip() for c in real.columns]

    if "fecha_post" not in real.columns or "destino" not in real.columns:
        raise ValueError("dim_mermas_ajuste_fecha_post_destino debe traer fecha_post y destino.")

    if "factor_ajuste" in real.columns:
        real_factor_col = "factor_ajuste"
    elif "ajuste" in real.columns:
        real_factor_col = "ajuste"
    else:
        raise ValueError("dim_mermas_ajuste_fecha_post_destino no trae factor_ajuste ni ajuste.")

    real["fecha_post"] = _to_date_naive(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real[real_factor_col] = pd.to_numeric(real[real_factor_col], errors="coerce")

    real2 = (
        real.groupby(["fecha_post", "destino"], dropna=False, as_index=False)
            .agg(factor_ajuste_real=(real_factor_col, "median"))
    )

    df = uni.merge(
        real2.rename(columns={"fecha_post": "fecha_post_key"}),
        left_on=[fecha_post_col, "destino"],
        right_on=["fecha_post_key", "destino"],
        how="left",
    )

    # target: log(real / ml1)  (si es >0)
    ml1 = pd.to_numeric(df[ajuste_ml1_col], errors="coerce")
    realv = pd.to_numeric(df["factor_ajuste_real"], errors="coerce")

    ratio = realv / ml1.replace(0, np.nan)
    log_ratio = np.log(ratio.replace([np.inf, -np.inf], np.nan))

    clip = 1.2
    df["log_ratio_ajuste"] = log_ratio.clip(lower=-clip, upper=clip)

    df["w"] = _weight_series(df)

    # calendario sobre fecha_post_pred
    df["dow"] = df[fecha_post_col].dt.dayofweek.astype("Int64")
    df["month"] = df[fecha_post_col].dt.month.astype("Int64")
    df["weekofyear"] = df[fecha_post_col].dt.isocalendar().week.astype("Int64")

    keep = [
        "fecha",
        fecha_post_col,
        "bloque_base" if "bloque_base" in df.columns else None,
        "variedad_canon" if "variedad_canon" in df.columns else None,
        "grado",
        "destino",
        ajuste_ml1_col,
        "factor_ajuste_real",
        "log_ratio_ajuste",
        "w",
        "dow",
        "month",
        "weekofyear",
    ]
    keep = [c for c in keep if c is not None and c in df.columns]

    out = df[keep].copy()
    out = out.rename(columns={fecha_post_col: "fecha_post_pred_used", ajuste_ml1_col: "ajuste_ml1_used"})
    out["as_of_date"] = as_of
    out["created_at"] = pd.Timestamp.utcnow()

    (GOLD / "ml2_datasets").mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_DS)

    n_real = int(out["factor_ajuste_real"].notna().sum())
    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={len(out):,} rows_with_real={n_real:,} as_of_date={as_of.date()} clip=±{clip}")


if __name__ == "__main__":
    main()
