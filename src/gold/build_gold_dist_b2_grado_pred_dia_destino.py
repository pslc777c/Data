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

IN_UNIVERSE = GOLD / "pred_poscosecha_ml1_hidr_grado_dia_bloque_destino.parquet"
OUT_PATH = GOLD / "dist_b2_grado_pred_dia_destino.parquet"


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha_post_pred_ml1", "destino", "grado", "cajas_split_grado_dia"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Universe sin columnas requeridas: {sorted(miss)}")

    df["fecha_post"] = pd.to_datetime(df["fecha_post_pred_ml1"], errors="coerce").dt.normalize()
    df["destino"] = _canon_str(df["destino"])
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce")
    df["peso_b2_proxy"] = pd.to_numeric(df["cajas_split_grado_dia"], errors="coerce").fillna(0.0)

    # Agregar (sum) a nivel día×destino×grado (colapsa bloques/variedad)
    agg = (
        df.loc[df["fecha_post"].notna() & df["destino"].notna() & df["grado"].notna()]
          .groupby(["fecha_post", "destino", "grado"], as_index=False)
          .agg(peso_b2_proxy=("peso_b2_proxy", "sum"))
    )

    # Share dentro de día×destino
    denom = agg.groupby(["fecha_post", "destino"])["peso_b2_proxy"].transform("sum")
    agg["share_b2_grado"] = np.where(denom > 0, agg["peso_b2_proxy"] / denom, np.nan)

    write_parquet(agg, OUT_PATH)
    print(f"[OK] dist B2 (grado) guardado en {OUT_PATH} | rows={len(agg):,}")
    print("[INFO] fechas:", agg["fecha_post"].min(), agg["fecha_post"].max(), "| destinos:", sorted(agg["destino"].unique().tolist()))


if __name__ == "__main__":
    main()
