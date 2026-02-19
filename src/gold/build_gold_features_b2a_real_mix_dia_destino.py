from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from common.io import read_parquet, write_parquet

IN_PATH = Path("data/silver/balanzas/silver_b2a_sku_gradoideal_dia_destino_variedad.parquet")
OUT_PATH = Path("data/gold/features_b2a_real_mix_dia_destino.parquet")


def _canon(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _map_variedad(v: pd.Series) -> pd.Series:
    x = _canon(v)
    x = x.replace({"CLO": "CLOUD", "XL": "XLENCE", "XLENCE ": "XLENCE", "CLOUD ": "CLOUD"})
    return x


def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    return float(-(p * np.log(p)).sum()) if p.size else float("nan")


def _pick_col(cols: list[str], aliases: list[str]) -> str | None:
    # match exact por lower
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    # match por contains (suave)
    for c in cols:
        cl = c.lower()
        for a in aliases:
            if a.lower() in cl:
                return c
    return None


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

    cols = list(df.columns)

    # aliases esperados
    c_fecha = _pick_col(cols, ["fecha", "Fecha", "fecha_post", "fecha_post_b2a", "date"])
    c_dest  = _pick_col(cols, ["destino", "Destino", "codigo_actividad", "actividad"])
    c_var   = _pick_col(cols, ["variedad", "Variedad", "variedad_canon", "var"])
    c_sku   = _pick_col(cols, ["sku", "SKU"])
    c_gi    = _pick_col(cols, ["grado_ideal", "Grado_Ideal", "GI", "grado ideal"])
    c_peso  = _pick_col(cols, ["peso_kg", "peso kg", "peso", "PESOKG", "Peso_Balanza_2A", "peso_balanza_2a"])

    missing = {
        "fecha": c_fecha,
        "destino": c_dest,
        "variedad": c_var,
        "sku": c_sku,
        "grado_ideal": c_gi,
        "peso_kg": c_peso,
    }
    miss_keys = [k for k, v in missing.items() if v is None]
    if miss_keys:
        print("[ERROR] No pude mapear columnas:", miss_keys)
        print("[INFO] Columnas disponibles:", cols)
        raise ValueError(f"B2A real (variedad) sin columnas mapeables: {miss_keys}")

    # renombrar a estándar
    df = df.rename(columns={
        c_fecha: "fecha",
        c_dest: "destino",
        c_var: "variedad",
        c_sku: "sku",
        c_gi: "grado_ideal",
        c_peso: "peso_kg",
    })

    df["fecha_post"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["destino"] = _canon(df["destino"])
    df["variedad"] = _map_variedad(df["variedad"])
    df["sku"] = pd.to_numeric(df["sku"], errors="coerce")
    df["gi"] = pd.to_numeric(df["grado_ideal"], errors="coerce")
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce")

    df = df.loc[
        df["fecha_post"].notna()
        & df["destino"].notna()
        & df["variedad"].notna()
        & df["sku"].notna()
        & df["gi"].notna()
        & (df["peso_kg"] > 0)
    ].copy()

    # share por (día×destino×variedad)
    denom = df.groupby(["fecha_post", "destino", "variedad"])["peso_kg"].transform("sum")
    df["share"] = np.where(denom > 0, df["peso_kg"] / denom, np.nan)

    feats_var = (
        df.groupby(["fecha_post", "destino", "variedad"], group_keys=False)
          .apply(lambda g: _agg_one(g, prefix=""))
          .reset_index()
    )

    # pivot variedad -> wide
    value_cols = [c for c in feats_var.columns if c not in {"fecha_post", "destino", "variedad"}]
    wide = feats_var.pivot_table(
        index=["fecha_post", "destino"],
        columns="variedad",
        values=value_cols,
        aggfunc="first",
    )
    wide.columns = [f"{feat}__{var}" for feat, var in wide.columns]
    wide = wide.reset_index()

    # total (share sobre total día×destino)
    denom2 = df.groupby(["fecha_post", "destino"])["peso_kg"].transform("sum")
    df2 = df.copy()
    df2["share"] = np.where(denom2 > 0, df2["peso_kg"] / denom2, np.nan)

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
