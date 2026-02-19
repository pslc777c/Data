from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
SILVER = DATA / "silver"

# Universe: ya contiene el split por destino y llaves bloque/variedad/grado/destino
IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_seed_grado_dia_bloque_destino.parquet"

# DH baseline aplicado (ML1) sobre seed ML2 (esto te da dh_dias_ml1 y fecha_post_pred_ml1)
IN_DH_ML1_ON_ML2 = GOLD / "pred_poscosecha_ml2_dh_grado_dia_bloque_destino.parquet"

# Real DH + hidratación (tiene dh_dias a nivel fecha_cosecha, fecha_post, grado, destino)
IN_REAL_HD = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"

OUT_DS = GOLD / "ml2_datasets" / "ds_dh_poscosecha_ml2_v1.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _as_of_date() -> pd.Timestamp:
    # Regla: usar hoy-1 (día cerrado)
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def main(clip_err_days: float = 5.0) -> None:
    as_of = _as_of_date()

    uni = read_parquet(IN_UNIVERSE).copy()
    dhm = read_parquet(IN_DH_ML1_ON_ML2).copy()
    real = read_parquet(IN_REAL_HD).copy()

    for df in (uni, dhm, real):
        df.columns = [str(c).strip() for c in df.columns]

    # Canon keys
    uni["fecha"] = _to_date(uni["fecha"])
    uni["destino"] = _canon_str(uni["destino"])
    uni["grado"] = _canon_int(uni["grado"])
    uni["bloque_base"] = _canon_int(uni.get("bloque_base", pd.Series([pd.NA] * len(uni))))
    if "variedad_canon" in uni.columns:
        uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    # Filtrar as_of (no usar hoy)
    uni = uni[uni["fecha"].notna() & (uni["fecha"] <= as_of)].copy()

    # DH ML1 sobre ML2 seed
    dhm["fecha"] = _to_date(dhm["fecha"])
    dhm["destino"] = _canon_str(dhm["destino"])
    dhm["grado"] = _canon_int(dhm["grado"])
    dhm["bloque_base"] = _canon_int(dhm.get("bloque_base", pd.Series([pd.NA] * len(dhm))))
    if "variedad_canon" in dhm.columns:
        dhm["variedad_canon"] = _canon_str(dhm["variedad_canon"])

    # Identificar columna DH ML1
    dh_col = None
    for c in ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias_pred", "dh_dias"]:
        if c in dhm.columns:
            dh_col = c
            break
    if dh_col is None:
        raise KeyError("No encuentro columna dh en pred_poscosecha_ml2_dh_*. Espero dh_dias_ml1 o similar.")

    dhm[dh_col] = _canon_int(dhm[dh_col])

    # Real
    real["fecha_cosecha"] = _to_date(real["fecha_cosecha"])
    real["fecha_post"] = _to_date(real["fecha_post"])
    real["destino"] = _canon_str(real["destino"])
    real["grado"] = _canon_int(real["grado"])

    # tallos real (para peso del KPI / entrenamiento)
    if "tallos" in real.columns:
        real["tallos"] = pd.to_numeric(real["tallos"], errors="coerce").fillna(0.0)
    else:
        real["tallos"] = 0.0

    # dh real (col explícita)
    if "dh_dias" in real.columns:
        real["dh_dias"] = _canon_int(real["dh_dias"])
    else:
        # fallback: (fecha_post - fecha_cosecha)
        real["dh_dias"] = (real["fecha_post"] - real["fecha_cosecha"]).dt.days.astype("Int64")

    # Agregar real a grano (fecha_cosecha, grado, destino) con median para dh y sum tallos
    real_g = (
        real.groupby(["fecha_cosecha", "grado", "destino"], dropna=False, as_index=False)
            .agg(
                dh_real=("dh_dias", "median"),
                tallos_real=("tallos", "sum"),
                fecha_post_real=("fecha_post", "min"),
            )
    )

    # Unimos universe + dh_ml1
    keys = ["fecha", "bloque_base", "variedad_canon", "grado", "destino"]
    need_uni = [c for c in keys if c in uni.columns]
    need_dhm = [c for c in keys if c in dhm.columns]

    df = uni.merge(
        dhm[need_dhm + [dh_col]].rename(columns={dh_col: "dh_ml1"}),
        on=[c for c in keys if c in need_uni and c in need_dhm],
        how="left",
    )

    # Join real por fecha_cosecha=fecha + grado + destino
    df = df.merge(
        real_g,
        left_on=["fecha", "grado", "destino"],
        right_on=["fecha_cosecha", "grado", "destino"],
        how="left",
    )

    # Features calendario sobre fecha_cosecha (fecha)
    df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")

    # Target: error en días
    df["dh_ml1"] = _canon_int(df["dh_ml1"])
    df["dh_real"] = _canon_int(df["dh_real"])

    df["err_dh_days"] = (df["dh_real"].astype("float") - df["dh_ml1"].astype("float"))
    df["err_dh_days_clipped"] = df["err_dh_days"].clip(lower=-float(clip_err_days), upper=float(clip_err_days))

    df["clip_err_days"] = float(clip_err_days)
    df["as_of_date"] = as_of
    df["created_at"] = pd.Timestamp.utcnow()

    # Guardar DS completo (incluye filas sin real; el train filtrará)
    OUT_DS.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df, OUT_DS)

    n_all = len(df)
    n_real = int(df["dh_real"].notna().sum())
    print(f"[OK] Wrote dataset: {OUT_DS}")
    print(f"     rows={n_all:,} rows_with_real={n_real:,} as_of_date={as_of.date()} clip=±{clip_err_days:g}")


if __name__ == "__main__":
    main()
