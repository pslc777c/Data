from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_HW = Path("data/gold/pred_harvest_window_ml1.parquet")
IN_MAESTRO = Path("data/silver/fact_ciclo_maestro.parquet")

OUT_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _build_grid_from_start_n(
    df: pd.DataFrame,
    start_col: str,
    n_col: str,
) -> pd.DataFrame:
    """
    Construye grid diario EXACTO de n días por fila, empezando en start.
    Evita el mismatch count != n_pred por definición.

    Output: explode columna 'fecha', y añade 'day_in_harvest_pred' (1..n).
    """
    out = df.copy()
    out[start_col] = _to_date(out[start_col])
    out[n_col] = pd.to_numeric(out[n_col], errors="coerce").round().astype("Int64")

    ok = out[start_col].notna() & out[n_col].notna() & (out[n_col].astype(float) > 0)
    bad = ~ok
    if bad.any():
        nbad = int(bad.sum())
        print(f"[WARN] Filas HW inválidas (sin start o n<=0): {nbad:,}. Se excluyen del grid.")
    out = out[ok].copy()
    if out.empty:
        return out.assign(fecha=pd.Series([], dtype="datetime64[ns]"))

    # Construir offsets 0..n-1 por fila sin apply:
    # Creamos un array de longitudes y repetimos filas.
    lens = out[n_col].astype(int).to_numpy()
    idx_rep = np.repeat(np.arange(len(out)), lens)

    g = out.iloc[idx_rep].copy().reset_index(drop=True)
    # offset dentro de fila repetida: 0..n-1
    offset = np.concatenate([np.arange(n, dtype=int) for n in lens])
    g["fecha"] = g[start_col] + pd.to_timedelta(offset, unit="D")
    g["fecha"] = _to_date(g["fecha"])

    g["day_in_harvest_pred"] = pd.Series(offset + 1, dtype="Int64")

    return g


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    hw = read_parquet(IN_HW).copy()
    maestro = read_parquet(IN_MAESTRO).copy()

    hw.columns = [str(c).strip() for c in hw.columns]
    maestro.columns = [str(c).strip() for c in maestro.columns]

    need_hw = {
        "ciclo_id",
        "bloque_base",
        "variedad_canon",
        "area",
        "tipo_sp",
        "fecha_sp",
        "harvest_start_pred",
        "n_harvest_days_pred_final",
    }
    miss = need_hw - set(hw.columns)
    if miss:
        raise ValueError(f"pred_harvest_window_ml1: faltan columnas {sorted(miss)}")

    # Canon HW
    hw["ciclo_id"] = hw["ciclo_id"].astype(str)
    hw["bloque_base"] = _canon_int(hw["bloque_base"])
    hw["variedad_canon"] = _canon_str(hw["variedad_canon"])
    hw["area"] = _canon_str(hw["area"])
    hw["tipo_sp"] = _canon_str(hw["tipo_sp"])
    hw["fecha_sp"] = _to_date(hw["fecha_sp"])
    hw["harvest_start_pred"] = _to_date(hw["harvest_start_pred"])
    hw["n_harvest_days_pred_final"] = pd.to_numeric(hw["n_harvest_days_pred_final"], errors="coerce").round().astype("Int64")

    # Maestro: mínimo para enriquecer
    if "ciclo_id" not in maestro.columns:
        raise ValueError("fact_ciclo_maestro: falta ciclo_id")
    maestro["ciclo_id"] = maestro["ciclo_id"].astype(str)

    if "bloque_base" in maestro.columns:
        maestro["bloque_base"] = _canon_int(maestro["bloque_base"])

    if "variedad_canon" in maestro.columns:
        maestro["variedad_canon"] = _canon_str(maestro["variedad_canon"])
    elif "variedad" in maestro.columns:
        maestro["variedad_canon"] = _canon_str(maestro["variedad"])
    else:
        maestro["variedad_canon"] = "UNKNOWN"

    if "area" in maestro.columns:
        maestro["area"] = _canon_str(maestro["area"])
    if "tipo_sp" in maestro.columns:
        maestro["tipo_sp"] = _canon_str(maestro["tipo_sp"])

    m_take = [c for c in [
        "ciclo_id", "bloque", "bloque_padre", "bloque_base",
        "variedad", "variedad_canon", "tipo_sp", "area", "estado", "tallos_proy"
    ] if c in maestro.columns]
    m2 = maestro[m_take].drop_duplicates(subset=["ciclo_id"])

    base = hw.merge(m2, on="ciclo_id", how="left", suffixes=("", "_m"))

    # Coalesce de algunos campos si vienen duplicados (por suffixes)
    for c in ["bloque_base", "variedad_canon", "area", "tipo_sp"]:
        cx, cy = f"{c}_x", f"{c}_y"
        if cx in base.columns or cy in base.columns:
            x = base[cx] if cx in base.columns else pd.Series([pd.NA] * len(base))
            y = base[cy] if cy in base.columns else pd.Series([pd.NA] * len(base))
            base[c] = x.combine_first(y)
            base = base.drop(columns=[k for k in [cx, cy] if k in base.columns])

    # Grid EXACTO usando start + n
    grid = _build_grid_from_start_n(base, "harvest_start_pred", "n_harvest_days_pred_final")
    if grid.empty:
        raise ValueError("Grid quedó vacío. Revisa harvest_start_pred / n_harvest_days_pred_final.")

    # Recalcular end_pred consistente
    grid["n_harvest_days_pred"] = pd.to_numeric(grid["n_harvest_days_pred_final"], errors="coerce").astype("Int64")
    grid["harvest_end_pred"] = grid["harvest_start_pred"] + pd.to_timedelta(grid["n_harvest_days_pred"].astype(int) - 1, unit="D")
    grid["harvest_end_pred"] = _to_date(grid["harvest_end_pred"])

    grid["rel_pos_pred"] = np.where(
        grid["n_harvest_days_pred"].notna()
        & (grid["n_harvest_days_pred"].astype(float) > 0)
        & grid["day_in_harvest_pred"].notna(),
        grid["day_in_harvest_pred"].astype(float) / grid["n_harvest_days_pred"].astype(float),
        np.nan,
    )

    grid["stage"] = "HARVEST"

    grid["dow"] = grid["fecha"].dt.dayofweek
    grid["month"] = grid["fecha"].dt.month
    grid["weekofyear"] = grid["fecha"].dt.isocalendar().week.astype(int)

    keep = [
        "ciclo_id",
        "fecha",
        "bloque_base",
        "variedad_canon",
        "area",
        "tipo_sp",
        "estado" if "estado" in grid.columns else None,
        "tallos_proy" if "tallos_proy" in grid.columns else None,
        "fecha_sp",
        "harvest_start_pred",
        "harvest_end_pred",
        "n_harvest_days_pred",
        "day_in_harvest_pred",
        "rel_pos_pred",
        "stage",
        "dow",
        "month",
        "weekofyear",
        "ml1_version" if "ml1_version" in grid.columns else None,
    ]
    keep = [c for c in keep if c is not None and c in grid.columns]

    out = grid[keep].sort_values(["bloque_base", "variedad_canon", "fecha", "ciclo_id"]).reset_index(drop=True)
    out["created_at"] = created_at

    write_parquet(out, OUT_GRID)

    fmin = pd.to_datetime(out["fecha"].min()).date()
    fmax = pd.to_datetime(out["fecha"].max()).date()
    print(f"OK -> {OUT_GRID} | rows={len(out):,} | fecha_min={fmin} fecha_max={fmax}")

    # sanity: ahora debería ser ~0%
    chk = out.groupby("ciclo_id", dropna=False).agg(
        n=("fecha", "count"),
        n_pred=("n_harvest_days_pred", "first"),    
    )
    diff = (chk["n"].astype(float) - chk["n_pred"].astype(float)).abs()
    print(f"[CHECK] % ciclos donde count!=n_harvest_days_pred: {float((diff > 0).mean()):.2%}")


if __name__ == "__main__":
    main()
    