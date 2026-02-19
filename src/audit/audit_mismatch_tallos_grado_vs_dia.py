from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from common.io import read_parquet

GOLD = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")

def main() -> None:
    df = read_parquet(GOLD).copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()

    # Detectar columnas tallos por grado
    cand_base = [c for c in df.columns if c in ("tallos_pred_baseline_grado_dia", "tallos_baseline_grado_dia")]
    cand_ml1  = [c for c in df.columns if c in ("tallos_pred_ml1_grado_dia", "tallos_ml1_grado_dia")]

    if not cand_base or not cand_ml1:
        raise ValueError(f"No encuentro tallos por grado. Cols={list(df.columns)}")

    col_base = cand_base[0]
    col_ml1 = cand_ml1[0]

    # Detectar columnas tallos día (si existen)
    col_base_dia = "tallos_pred_baseline_dia" if "tallos_pred_baseline_dia" in df.columns else None
    col_ml1_dia = "tallos_pred_ml1_dia" if "tallos_pred_ml1_dia" in df.columns else None

    grp = ["fecha", "bloque_base", "variedad_canon"]

    g = df.groupby(grp, dropna=False).agg(
        n_rows=("grado", "size"),
        n_grados=("grado", pd.Series.nunique),
        base_sum=(col_base, "sum"),
        ml1_sum=(col_ml1, "sum"),
        base_nan_rate=(col_base, lambda s: float(pd.isna(s).mean())),
        ml1_nan_rate=(col_ml1, lambda s: float(pd.isna(s).mean())),
        share_base_nan=("share_grado_baseline", lambda s: float(pd.isna(s).mean())) if "share_grado_baseline" in df.columns else ("grado", lambda s: np.nan),
        share_ml1_nan=("share_grado_ml1", lambda s: float(pd.isna(s).mean())) if "share_grado_ml1" in df.columns else ("grado", lambda s: np.nan),
        share_base_sum=("share_grado_baseline", "sum") if "share_grado_baseline" in df.columns else ("grado", lambda s: np.nan),
        share_ml1_sum=("share_grado_ml1", "sum") if "share_grado_ml1" in df.columns else ("grado", lambda s: np.nan),
    ).reset_index()

    if col_base_dia:
        base_day = df.groupby(grp, dropna=False)[col_base_dia].first().reset_index().rename(columns={col_base_dia: "base_dia"})
        g = g.merge(base_day, on=grp, how="left")
        g["diff_base"] = (g["base_sum"] - g["base_dia"]).abs()
    else:
        g["diff_base"] = np.nan

    if col_ml1_dia:
        ml1_day = df.groupby(grp, dropna=False)[col_ml1_dia].first().reset_index().rename(columns={col_ml1_dia: "ml1_dia"})
        g = g.merge(ml1_day, on=grp, how="left")
        g["diff_ml1"] = (g["ml1_sum"] - g["ml1_dia"]).abs()
    else:
        g["diff_ml1"] = np.nan

    eps = 1e-6
    bad = g[(g["diff_base"] > eps) | (g["diff_ml1"] > eps)].copy()

    print("================================================================================")
    print("[A] RESUMEN")
    print("================================================================================")
    print(f"groups total: {len(g):,}")
    print(f"groups mismatch: {len(bad):,} ({len(bad)/max(len(g),1):.4%})")

    if len(bad) == 0:
        print("OK: no hay mismatch.")
        return

    print("================================================================================")
    print("[B] HIPÓTESIS RÁPIDA (por qué falla)")
    print("================================================================================")
    print("Promedios en grupos mismatch:")
    cols_show = ["n_grados","n_rows","base_nan_rate","ml1_nan_rate","share_base_nan","share_ml1_nan","share_base_sum","share_ml1_sum"]
    print(bad[cols_show].describe().to_string())

    # Top casos: mayor diff
    top = bad.sort_values(["diff_ml1","diff_base"], ascending=False).head(30)
    print("================================================================================")
    print("[C] TOP 30 GRUPOS PEOR DIFF")
    print("================================================================================")
    show_cols = grp + ["n_grados","n_rows","diff_base","diff_ml1","base_nan_rate","ml1_nan_rate","share_base_nan","share_ml1_nan","share_base_sum","share_ml1_sum"]
    print(top[show_cols].to_string(index=False))

    # Ejemplo detallado del peor
    worst = top.iloc[0][grp].to_dict()
    print("================================================================================")
    print("[D] DETALLE DEL PEOR GRUPO (filas por grado)")
    print("================================================================================")
    sub = df[
        (df["fecha"] == worst["fecha"]) &
        (df["bloque_base"] == worst["bloque_base"]) &
        (df["variedad_canon"] == worst["variedad_canon"])
    ].copy()

    cols_det = ["grado", col_base, col_ml1]
    for c in ["share_grado_baseline","share_grado_ml1", col_base_dia, col_ml1_dia]:
        if c and c in sub.columns:
            cols_det.append(c)

    sub = sub[cols_det].sort_values("grado")
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
