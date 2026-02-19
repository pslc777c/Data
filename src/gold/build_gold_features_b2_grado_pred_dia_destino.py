from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"

IN_PATH = GOLD / "dist_b2_grado_pred_dia_destino.parquet"
OUT_PATH = GOLD / "features_b2_grado_pred_dia_destino.parquet"


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def main() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)

    df = read_parquet(IN_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha_post", "destino", "grado", "share_b2_grado"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Dist B2 sin columnas: {sorted(miss)}")

    df["fecha_post"] = pd.to_datetime(df["fecha_post"], errors="coerce").dt.normalize()
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce")
    df["share"] = pd.to_numeric(df["share_b2_grado"], errors="coerce")

    df = df.loc[df["fecha_post"].notna() & df["destino"].notna() & df["grado"].notna()].copy()

    # normalizar share por seguridad
    denom = df.groupby(["fecha_post", "destino"])["share"].transform("sum")
    df["share_n"] = np.where(denom > 0, df["share"] / denom, np.nan)

    def agg_one(g: pd.DataFrame) -> pd.Series:
        p = g["share_n"].to_numpy(dtype=float)
        p = p[np.isfinite(p)]
        p_sorted = np.sort(p)[::-1]
        top1 = float(p_sorted[0]) if p_sorted.size >= 1 else float("nan")
        top3 = float(p_sorted[:3].sum()) if p_sorted.size >= 3 else float("nan")
        ent = _entropy(p)

        # grado ponderado
        mean_gr = float((g["grado"] * g["share_n"]).sum())

        # bandas operativas (ajusta si quieres)
        gr = g["grado"]
        def band(lo, hi):
            return float(g.loc[gr.between(lo, hi), "share_n"].sum())

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
        df.groupby(["fecha_post", "destino"], group_keys=False)
          .apply(agg_one)
          .reset_index()
    )

    write_parquet(out, OUT_PATH)
    print(f"[OK] features B2 guardado en {OUT_PATH} | rows={len(out):,}")
    print("[INFO] fechas:", out["fecha_post"].min(), out["fecha_post"].max(), "| destinos:", sorted(out["destino"].unique().tolist()))


if __name__ == "__main__":
    main()
