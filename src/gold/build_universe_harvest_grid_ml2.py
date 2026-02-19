from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/gold/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"

IN_GRID_ML1 = GOLD_DIR / "universe_harvest_grid_ml1.parquet"

# ML2 outputs
IN_SOH_PROD = GOLD_DIR / "pred_harvest_start_final_ml2.parquet"
IN_HH_PROD = GOLD_DIR / "pred_harvest_horizon_final_ml2.parquet"

OUT_GRID_ML2 = GOLD_DIR / "universe_harvest_grid_ml2.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    grid = read_parquet(IN_GRID_ML1).copy()

    # Canon base
    grid["fecha"] = _to_date(grid["fecha"])
    grid["fecha_sp"] = _to_date(grid["fecha_sp"])
    grid["bloque_base"] = _canon_str(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])
    if "estado" in grid.columns:
        grid["estado"] = _canon_str(grid["estado"])

    # Load ML2 start + horizon (PROD)
    if not IN_SOH_PROD.exists():
        raise FileNotFoundError(f"Missing ML2 SoH prod file: {IN_SOH_PROD}")
    if not IN_HH_PROD.exists():
        raise FileNotFoundError(f"Missing ML2 HH prod file: {IN_HH_PROD}")

    soh = read_parquet(IN_SOH_PROD).copy()
    hh = read_parquet(IN_HH_PROD).copy()

    # Normalize
    soh["harvest_start_final"] = _to_date(soh["harvest_start_final"])
    soh = soh[["ciclo_id", "harvest_start_final"]].drop_duplicates("ciclo_id")

    hh["harvest_end_final"] = _to_date(hh["harvest_end_final"])
    hh["n_harvest_days_final"] = pd.to_numeric(hh["n_harvest_days_final"], errors="coerce")
    hh = hh[["ciclo_id", "harvest_end_final", "n_harvest_days_final"]].drop_duplicates("ciclo_id")

    # Reduce ML1 grid to a header (cycle-level ML1 preds) to keep metadata stable
    g = grid.copy()
    g["harvest_start_pred"] = _to_date(g["harvest_start_pred"])
    g["harvest_end_pred"] = _to_date(g["harvest_end_pred"])

    head = (
        g.groupby("ciclo_id", as_index=False)
        .agg(
            bloque_base=("bloque_base", "first"),
            variedad_canon=("variedad_canon", "first"),
            area=("area", "first"),
            tipo_sp=("tipo_sp", "first"),
            estado=("estado", "first") if "estado" in g.columns else ("ciclo_id", "first"),
            tallos_proy=("tallos_proy", "first"),
            fecha_sp=("fecha_sp", "first"),
            harvest_start_pred=("harvest_start_pred", "min"),
            harvest_end_pred=("harvest_end_pred", "max"),
            n_harvest_days_pred=("n_harvest_days_pred", "max"),
            ml1_version=("ml1_version", "max"),
        )
    )

    # Attach ML2 start + horizon
    head = head.merge(soh, on="ciclo_id", how="left").merge(hh, on="ciclo_id", how="left")

    # Fallbacks (should be rare)
    head["harvest_start_final"] = head["harvest_start_final"].fillna(head["harvest_start_pred"])
    head["harvest_end_final"] = head["harvest_end_final"].fillna(head["harvest_end_pred"])

    # If n_harvest_days_final missing, derive from start/end
    miss_n = head["n_harvest_days_final"].isna()
    if miss_n.any():
        head.loc[miss_n, "n_harvest_days_final"] = (
            (head.loc[miss_n, "harvest_end_final"] - head.loc[miss_n, "harvest_start_final"]).dt.days + 1
        )

    head["n_harvest_days_final"] = pd.to_numeric(head["n_harvest_days_final"], errors="coerce").fillna(0.0)
    head["n_harvest_days_final"] = head["n_harvest_days_final"].clip(lower=7)

    # Build ML2 grid per cycle: fecha range [start_final, end_final]
    parts = []
    for r in head.itertuples(index=False):
        if pd.isna(r.harvest_start_final) or pd.isna(r.harvest_end_final):
            continue
        if r.harvest_end_final < r.harvest_start_final:
            continue
        dates = pd.date_range(start=r.harvest_start_final, end=r.harvest_end_final, freq="D")
        tmp = pd.DataFrame({
            "ciclo_id": r.ciclo_id,
            "fecha": dates,
        })
        tmp["bloque_base"] = r.bloque_base
        tmp["variedad_canon"] = r.variedad_canon
        tmp["area"] = r.area
        tmp["tipo_sp"] = r.tipo_sp
        tmp["estado"] = r.estado if "estado" in head.columns else "UNKNOWN"
        tmp["tallos_proy"] = r.tallos_proy
        tmp["fecha_sp"] = r.fecha_sp

        tmp["harvest_start_pred"] = r.harvest_start_pred
        tmp["harvest_end_pred"] = r.harvest_end_pred
        tmp["n_harvest_days_pred"] = r.n_harvest_days_pred

        tmp["harvest_start_final"] = r.harvest_start_final
        tmp["harvest_end_final"] = r.harvest_end_final
        tmp["n_harvest_days_final"] = r.n_harvest_days_final

        parts.append(tmp)

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if out.empty:
        raise RuntimeError("ML2 universe grid is empty. Check ML2 inputs and cycle headers.")

    # Re-index / features
    out["day_in_harvest_final"] = (out["fecha"] - out["harvest_start_final"]).dt.days.astype(int) + 1
    out["rel_pos_final"] = (out["day_in_harvest_final"] - 1) / (out["n_harvest_days_final"] - 1).replace(0, np.nan)
    out["rel_pos_final"] = out["rel_pos_final"].fillna(0.0).clip(0.0, 1.0)

    # Keep ML1 rel_pos_pred if you want, but this grid is ML2-driven.
    out["dow"] = out["fecha"].dt.dayofweek
    out["month"] = out["fecha"].dt.month
    out["weekofyear"] = out["fecha"].dt.isocalendar().week.astype(int)

    # Versioning
    out["ml2_version"] = "ml2_harvest_start+hh"
    out["created_at"] = pd.Timestamp(datetime.now()).normalize()

    OUT_GRID_ML2.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_GRID_ML2)

    print(f"[OK] Wrote: {OUT_GRID_ML2}")
    print(f"     rows={len(out):,} cycles={out['ciclo_id'].nunique():,} "
          f"fecha_range=[{out['fecha'].min().date()}..{out['fecha'].max().date()}]")


if __name__ == "__main__":
    main()
