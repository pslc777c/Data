from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from common.io import read_parquet, write_parquet

IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")
OUT_AUDIT = Path("data/audit/audit_universe_harvest_grid_ml1_checks.parquet")
OUT_SUMMARY = Path("data/audit/audit_universe_harvest_grid_ml1_summary.json")

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def main() -> None:
    created_at = pd.Timestamp.utcnow()
    grid = read_parquet(IN_GRID).copy()
    grid.columns = [str(c).strip() for c in grid.columns]

    need = {"ciclo_id","fecha","bloque_base","variedad_canon"}
    miss = need - set(grid.columns)
    if miss:
        raise ValueError(f"universe_harvest_grid_ml1 sin columnas: {sorted(miss)}")

    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])

    key_dia = ["ciclo_id","fecha","bloque_base","variedad_canon"]
    dup = int(grid.duplicated(subset=key_dia).sum())
    dup_rate = float(dup / max(len(grid), 1))

    rows = [
        {"metric":"rows_grid","value":int(len(grid)),"level":"INFO","hint":"","created_at":created_at},
        {"metric":"dup_count_key_dia","value":dup,"level":"WARN" if dup>0 else "OK","hint":"Duplicados por (ciclo_id,fecha,bloque,variedad).","created_at":created_at},
        {"metric":"dup_rate_key_dia","value":dup_rate,"level":"WARN" if dup_rate>0 else "OK","hint":"","created_at":created_at},
    ]
    audit = pd.DataFrame(rows)
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(audit, OUT_AUDIT)

    summary = {
        "created_at_utc": created_at.isoformat(),
        "rows_grid": int(len(grid)),
        "dup_count_key_dia": dup,
        "dup_rate_key_dia": dup_rate,
        "horizon": {
            "min_fecha": str(pd.to_datetime(grid["fecha"].min()).date()) if len(grid) else None,
            "max_fecha": str(pd.to_datetime(grid["fecha"].max()).date()) if len(grid) else None,
        },
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"OK -> {OUT_AUDIT}")
    print(f"OK -> {OUT_SUMMARY}")
    print(f"[AUDIT] dup_count_key_dia={dup:,} dup_rate={dup_rate:.6f}")

if __name__ == "__main__":
    main()
