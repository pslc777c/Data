from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet

IN_DIST = Path("data/gold/pred_dist_grado_ml1.parquet")
IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")

OUT_AUDIT = Path("data/audit/audit_dist_grado_ml1_checks.parquet")
OUT_SUMMARY = Path("data/audit/audit_dist_grado_ml1_summary.json")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    dist = read_parquet(IN_DIST).copy()
    grid = read_parquet(IN_GRID).copy()

    dist.columns = [str(c).strip() for c in dist.columns]
    grid.columns = [str(c).strip() for c in grid.columns]

    _require(dist, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1"], "pred_dist_grado_ml1")
    _require(grid, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")

    # Canon
    dist["ciclo_id"] = dist["ciclo_id"].astype(str)
    dist["fecha"] = _to_date(dist["fecha"])
    dist["bloque_base"] = _canon_int(dist["bloque_base"])
    dist["variedad_canon"] = _canon_str(dist["variedad_canon"])
    dist["grado"] = _canon_int(dist["grado"])
    dist["share_grado_ml1"] = pd.to_numeric(dist["share_grado_ml1"], errors="coerce")

    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_int(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    dist_key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]
    grp_dist = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]

    dup_count = int(dist.duplicated(subset=dist_key).sum())
    dup_rate = float(dup_count / max(len(dist), 1))

    # sum shares
    shares_sum = dist.groupby(grp_dist, dropna=False)["share_grado_ml1"].sum()
    # tolerancia: 1% (ajustable)
    bad_sum = (shares_sum - 1.0).abs() > 0.01
    bad_sum_rate = float(bad_sum.mean()) if len(shares_sum) else float("nan")

    # cobertura dist vs grid
    dist_groups = dist[grp_dist].drop_duplicates()
    grid_groups = grid[grp_dist].drop_duplicates()
    cov = float(
        grid_groups.merge(dist_groups.assign(has=1), on=grp_dist, how="left")["has"].fillna(0).mean()
    )

    rows = [
        {"metric": "rows_dist", "value": int(len(dist)), "level": "INFO", "hint": "", "created_at": created_at},
        {"metric": "dup_count_dist_key", "value": dup_count, "level": "WARN" if dup_count > 0 else "OK", "hint": "Duplicados por llave real (ciclo,fecha,bloque,variedad,grado).", "created_at": created_at},
        {"metric": "dup_rate_dist_key", "value": dup_rate, "level": "WARN" if dup_rate > 0 else "OK", "hint": "", "created_at": created_at},
        {"metric": "coverage_dist_vs_grid_groups", "value": cov, "level": "OK" if cov >= 0.999 else "WARN", "hint": "Grupos día del universe cubiertos por dist.", "created_at": created_at},
        {"metric": "bad_share_sum_rate_abs_gt_0.01", "value": bad_sum_rate, "level": "OK" if (np.isfinite(bad_sum_rate) and bad_sum_rate <= 0.01) else "WARN", "hint": "Proporción de grupos donde sum(shares) se aleja de 1.", "created_at": created_at},
    ]
    audit = pd.DataFrame(rows)

    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(audit, OUT_AUDIT)

    summary = {
        "created_at_utc": created_at.isoformat(),
        "dist_key": dist_key,
        "grp_dist": grp_dist,
        "rows_dist": int(len(dist)),
        "dup_count": dup_count,
        "dup_rate": dup_rate,
        "coverage_dist_vs_grid_groups": cov,
        "bad_share_sum_rate_abs_gt_0.01": bad_sum_rate,
        "horizon": {
            "min_fecha": str(pd.to_datetime(dist["fecha"].min()).date()) if len(dist) else None,
            "max_fecha": str(pd.to_datetime(dist["fecha"].max()).date()) if len(dist) else None,
        },
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"OK -> {OUT_AUDIT}")
    print(f"OK -> {OUT_SUMMARY}")
    print(f"[AUDIT] dup_count={dup_count:,} dup_rate={dup_rate:.6f} coverage={cov:.6f} bad_sum_rate={bad_sum_rate:.6f}")


if __name__ == "__main__":
    main()
