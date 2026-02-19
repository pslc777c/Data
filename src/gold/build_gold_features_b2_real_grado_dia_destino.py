from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from common.io import read_parquet, write_parquet

IN_PATH = Path("data/silver/fact_hidratacion_real_post_grado_destino.parquet")
OUT_PATH = Path("data/gold/features_b2_real_grado_dia_destino.parquet")

def _canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    return float(-(p * np.log(p)).sum()) if p.size else float("nan")

def main() -> None:
    df = read_parquet(IN_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # columnas esperadas (según lo que me diste)
    need = {"fecha_post", "destino", "grado", "peso_post_g"}
    if not need <= set(df.columns):
        # fallback por si viene "fecha" en vez de fecha_post
        if "fecha_post" not in df.columns and "fecha_post" in df.columns:
            pass
        raise ValueError(f"B2 real sin columnas: {sorted(need - set(df.columns))}")

    df["fecha_post"] = pd.to_datetime(df["fecha_post"], errors="coerce").dt.normalize()
    df["destino"] = _canon(df["destino"])
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce")
    df["peso_kg"] = pd.to_numeric(df["peso_post_g"], errors="coerce") / 1000.0

    df = df.loc[df["fecha_post"].notna() & df["destino"].notna() & df["grado"].notna() & (df["peso_kg"] > 0)].copy()

    # dist por día×destino×grado
    dist = (
        df.groupby(["fecha_post", "destino", "grado"], as_index=False)
          .agg(peso_kg=("peso_kg", "sum"))
    )
    denom = dist.groupby(["fecha_post", "destino"])["peso_kg"].transform("sum")
    dist["share"] = np.where(denom > 0, dist["peso_kg"] / denom, np.nan)

    # features compactas por día×destino
    def agg_one(g: pd.DataFrame) -> pd.Series:
        p = g["share"].to_numpy(float)
        p = p[np.isfinite(p)]
        ps = np.sort(p)[::-1]
        top1 = float(ps[0]) if ps.size else float("nan")
        top3 = float(ps[:3].sum()) if ps.size >= 3 else float("nan")
        ent = _entropy(p)
        mean_gr = float((g["grado"] * g["share"]).sum())

        gr = g["grado"]
        def band(lo, hi):
            return float(g.loc[gr.between(lo, hi), "share"].sum())

        return pd.Series({
            "b2_entropy": ent,
            "b2_top1_share": top1,
            "b2_top3_sum": top3,
            "b2_grado_mean": mean_gr,
            "b2_share_15_25": band(15, 25),
            "b2_share_26_35": band(26, 35),
            "b2_share_36_45": band(36, 45),
            "b2_share_46_55": band(46, 55),
            "b2_share_56_75": band(56, 75),
        })

    out = (
        dist.groupby(["fecha_post", "destino"], group_keys=False)
            .apply(agg_one)
            .reset_index()
    )

    write_parquet(out, OUT_PATH)
    print(f"[OK] {OUT_PATH} | rows={len(out):,} | fechas={out['fecha_post'].min()}..{out['fecha_post'].max()}")

if __name__ == "__main__":
    main()
