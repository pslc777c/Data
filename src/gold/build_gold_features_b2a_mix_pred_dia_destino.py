from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from common.io import read_parquet, write_parquet

IN_PATH = Path("data/silver/ventas/silver_ventas_b2a_share_sku_gi_dia_destino.parquet")
OUT_PATH = Path("data/gold/features_b2a_mix_pred_dia_destino.parquet")

def _canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _map_variedad(v: pd.Series) -> pd.Series:
    x = _canon(v)
    # unifica nombres
    x = x.replace({"XL": "XLENCE", "CLO": "CLOUD"})
    return x

def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    return float(-(p * np.log(p)).sum()) if p.size else float("nan")

def _agg_one(g: pd.DataFrame, prefix: str) -> pd.Series:
    p = g["share"].to_numpy(float)
    p = p[np.isfinite(p)]
    ps = np.sort(p)[::-1]

    top1 = float(ps[0]) if ps.size else float("nan")
    top3 = float(ps[:3].sum()) if ps.size >= 3 else float("nan")
    ent = _entropy(p)

    gi_mean = float((g["gi"] * g["share"]).sum())

    gi = g["gi"]
    def band(lo, hi):
        return float(g.loc[gi.between(lo, hi), "share"].sum())

    def sku_share(val):
        return float(g.loc[g["sku"].eq(val), "share"].sum())

    return pd.Series({
        f"{prefix}b2a_entropy": ent,
        f"{prefix}b2a_top1_share": top1,
        f"{prefix}b2a_top3_sum": top3,
        f"{prefix}b2a_gi_mean": gi_mean,
        f"{prefix}b2a_share_gi_15_25": band(15, 25),
        f"{prefix}b2a_share_gi_26_30": band(26, 30),
        f"{prefix}b2a_share_gi_31_35": band(31, 35),
        f"{prefix}b2a_share_gi_36_40": band(36, 40),
        f"{prefix}b2a_share_gi_41_50": band(41, 50),
        f"{prefix}b2a_share_sku_625": sku_share(625),
        f"{prefix}b2a_share_sku_750": sku_share(750),
        f"{prefix}b2a_share_sku_1000": sku_share(1000),
    })

def main() -> None:
    df = read_parquet(IN_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Tu silver trae: Semana_, Fecha, Destino, Variedad, SKU, Grado_Ideal, ... Peso_Balanza_2A
    need = {"Fecha", "Destino", "Variedad", "SKU", "Grado_Ideal"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Ventas silver sin columnas: {sorted(miss)}")

    # peso: intenta encontrar la mejor candidata
    peso_col = None
    for c in ["Peso_Balanza_2A", "Peso", "Valor", "peso_kg", "PESOKG"]:
        if c in df.columns:
            peso_col = c
            break
    if peso_col is None:
        raise ValueError("No encontré columna de peso en ventas silver (esperaba Peso_Balanza_2A o similar).")

    df["fecha_post"] = pd.to_datetime(df["Fecha"], errors="coerce").dt.normalize()
    df["destino"] = _canon(df["Destino"])
    df["variedad"] = _map_variedad(df["Variedad"])
    df["sku"] = pd.to_numeric(df["SKU"], errors="coerce")
    df["gi"] = pd.to_numeric(df["Grado_Ideal"], errors="coerce")
    df["peso"] = pd.to_numeric(df[peso_col], errors="coerce")

    df = df.loc[
        df["fecha_post"].notna()
        & df["destino"].notna()
        & df["variedad"].notna()
        & df["sku"].notna()
        & df["gi"].notna()
        & (df["peso"] > 0)
    ].copy()

    # Share por día×destino×variedad (composición)
    denom = df.groupby(["fecha_post", "destino", "variedad"])["peso"].transform("sum")
    df["share"] = np.where(denom > 0, df["peso"] / denom, np.nan)

    feats_var = (
        df.groupby(["fecha_post", "destino", "variedad"], group_keys=False)
          .apply(lambda g: _agg_one(g, prefix=""))
          .reset_index()
    )

    # wide por variedad: <feat>__<var>
    value_cols = [c for c in feats_var.columns if c not in {"fecha_post", "destino", "variedad"}]
    wide = feats_var.pivot_table(
        index=["fecha_post", "destino"],
        columns="variedad",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{feat}__{var}" for feat, var in wide.columns]
    wide = wide.reset_index()

    # Total: share sobre total día×destino
    denom2 = df.groupby(["fecha_post", "destino"])["peso"].transform("sum")
    df2 = df.copy()
    df2["share"] = np.where(denom2 > 0, df2["peso"] / denom2, np.nan)

    feats_tot = (
        df2.groupby(["fecha_post", "destino"], group_keys=False)
           .apply(lambda g: _agg_one(g, prefix="tot__"))
           .reset_index()
    )

    out = wide.merge(feats_tot, on=["fecha_post", "destino"], how="outer")

    write_parquet(out, OUT_PATH)
    print(f"[OK] {OUT_PATH} | rows={len(out):,} | fechas={out['fecha_post'].min()}..{out['fecha_post'].max()}")
    print("[INFO] destinos:", sorted(out["destino"].dropna().unique().tolist()))
    print("[INFO] cols:", len(out.columns))

if __name__ == "__main__":
    main()
