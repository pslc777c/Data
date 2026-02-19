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

IN_FULL = GOLD / "pred_poscosecha_ml2_ajuste_grado_dia_bloque_destino_final.parquet"

OUT_FULL = GOLD / "pred_poscosecha_ml2_final_full_grado_dia_bloque_destino.parquet"
OUT_DBD = GOLD / "pred_poscosecha_ml2_final_dia_bloque_destino.parquet"
OUT_DD = GOLD / "pred_poscosecha_ml2_final_dia_destino.parquet"
OUT_DT = GOLD / "pred_poscosecha_ml2_final_dia_total.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")
    
def _semana_iso_yyww(d: pd.Series) -> pd.Series:
    dt = pd.to_datetime(d, errors="coerce")
    iso = dt.dt.isocalendar()  # columns: year, week, day
    yy = (iso["year"] % 100).astype("Int64")
    ww = iso["week"].astype("Int64")
    # Devuelve texto 'YYWW' (ej. '2601', '2611')
    return (yy.astype(str).str.zfill(2) + ww.astype(str).str.zfill(2))


def main() -> None:
    df = read_parquet(IN_FULL).copy()
    df.columns = [str(c).strip() for c in df.columns]

    _require(
        df,
        ["fecha", "destino", "cajas_split_grado_dia", "factor_hidr_final", "factor_desp_final", "factor_ajuste_final"],
        "pred_poscosecha_ml2_ajuste...final",
    )

    df["fecha_post_pred_final"] = _to_date(df["fecha_post_pred_final"])
    df["Semana_ISO"] = _semana_iso_yyww(df["fecha_post_pred_final"])
    df["cajas_split_grado_dia"] = pd.to_numeric(df["cajas_split_grado_dia"], errors="coerce").fillna(0.0)
    for c in ["factor_hidr_final", "factor_desp_final", "factor_ajuste_final"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0)

    df["cajas_postcosecha_ml2_final"] = (
        df["cajas_split_grado_dia"].astype(float)
        * df["factor_hidr_final"].astype(float)
        * df["factor_desp_final"].astype(float)
        * df["factor_ajuste_final"].astype(float)
    )

    df["created_at"] = pd.Timestamp.utcnow()

    write_parquet(df, OUT_FULL)
    print(f"[OK] Wrote: {OUT_FULL} rows={len(df):,}")

    # día-bloque-destino
    keys_dbd = ["fecha", "bloque_base", "destino"]
    cols_av = [c for c in keys_dbd if c in df.columns]
    if len(cols_av) == 3:
        out_dbd = (
            df.groupby(keys_dbd, dropna=False, as_index=False)
              .agg(
                  cajas_split=("cajas_split_grado_dia", "sum"),
                  cajas_post_ml2=("cajas_postcosecha_ml2_final", "sum"),
              )
        )
        out_dbd["created_at"] = pd.Timestamp.utcnow()
        write_parquet(out_dbd, OUT_DBD)
        print(f"[OK] Wrote: {OUT_DBD} rows={len(out_dbd):,}")
    else:
        print("[WARN] No se pudo escribir OUT_DBD (faltan columnas de bloque_base).")

    # día-destino
    out_dd = (
        df.groupby(["fecha", "destino"], dropna=False, as_index=False)
          .agg(
              cajas_split=("cajas_split_grado_dia", "sum"),
              cajas_post_ml2=("cajas_postcosecha_ml2_final", "sum"),
          )
    )
    out_dd["created_at"] = pd.Timestamp.utcnow()
    write_parquet(out_dd, OUT_DD)
    print(f"[OK] Wrote: {OUT_DD} rows={len(out_dd):,}")

    # día-total
    out_dt = (
        out_dd.groupby(["fecha"], dropna=False, as_index=False)
              .agg(
                  cajas_split=("cajas_split", "sum"),
                  cajas_post_ml2=("cajas_post_ml2", "sum"),
              )
    )
    out_dt["created_at"] = pd.Timestamp.utcnow()
    write_parquet(out_dt, OUT_DT)
    print(f"[OK] Wrote: {OUT_DT} rows={len(out_dt):,}")


if __name__ == "__main__":
    main()
