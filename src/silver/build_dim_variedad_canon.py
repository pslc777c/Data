from __future__ import annotations

from pathlib import Path
import pandas as pd

from common.io import write_parquet


MAP = {
    "XLENCE": "XL",
    "XL": "XL",
    "CLOUD": "CLO",
    "CLO": "CLO",
}


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    rows = []
    for k, v in MAP.items():
        rows.append({"variedad_raw": k, "variedad_canon": v})

    df = pd.DataFrame(rows).drop_duplicates().sort_values(["variedad_canon", "variedad_raw"])
    df["created_at"] = created_at

    write_parquet(df, Path("data/silver/dim_variedad_canon.parquet"))
    print(f"OK -> data/silver/dim_variedad_canon.parquet | rows={len(df):,}")


if __name__ == "__main__":
    main()
