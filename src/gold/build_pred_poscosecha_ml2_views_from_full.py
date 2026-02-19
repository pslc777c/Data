from __future__ import annotations

from pathlib import Path
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"

IN_FULL = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"

OUT_DBD = GOLD / "pred_poscosecha_ml2_ml1_dia_bloque_destino.parquet"
OUT_DD = GOLD / "pred_poscosecha_ml2_ml1_dia_destino.parquet"
OUT_DT = GOLD / "pred_poscosecha_ml2_ml1_dia_total.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def main() -> None:
    df = read_parquet(IN_FULL).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "fecha_post_pred_ml1", "bloque_base", "variedad_canon", "grado", "destino", "cajas_postcosecha_ml1"}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Full ML2 sin columnas: {sorted(miss)}")

    df["fecha"] = _to_date(df["fecha"])
    df["fecha_post_pred_ml1"] = _to_date(df["fecha_post_pred_ml1"])

    # 1) día-bloque-destino
    out_dbd = (
        df.groupby(["fecha_post_pred_ml1", "bloque_base", "destino"], dropna=False, as_index=False)
          .agg(
              cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
              cajas_split=("cajas_split_grado_dia", "sum"),
          )
    )
    out_dbd["created_at"] = pd.Timestamp.utcnow()
    write_parquet(out_dbd, OUT_DBD)
    print(f"[OK] Wrote: {OUT_DBD} rows={len(out_dbd):,}")

    # 2) día-destino
    out_dd = (
        out_dbd.groupby(["fecha_post_pred_ml1", "destino"], dropna=False, as_index=False)
               .agg(
                   cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
                   cajas_split=("cajas_split", "sum"),
               )
    )
    out_dd["created_at"] = pd.Timestamp.utcnow()
    write_parquet(out_dd, OUT_DD)
    print(f"[OK] Wrote: {OUT_DD} rows={len(out_dd):,}")

    # 3) día-total
    out_dt = (
        out_dd.groupby(["fecha_post_pred_ml1"], dropna=False, as_index=False)
              .agg(
                  cajas_postcosecha_ml1=("cajas_postcosecha_ml1", "sum"),
                  cajas_split=("cajas_split", "sum"),
              )
    )
    out_dt["created_at"] = pd.Timestamp.utcnow()
    write_parquet(out_dt, OUT_DT)
    print(f"[OK] Wrote: {OUT_DT} rows={len(out_dt):,}")


if __name__ == "__main__":
    main()
