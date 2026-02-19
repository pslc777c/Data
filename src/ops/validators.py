from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_exists(outputs: list[Path]) -> None:
    missing = [p for p in outputs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")


def validate_parquet_nonempty(outputs: list[Path]) -> None:
    for p in outputs:
        if p.suffix.lower() != ".parquet":
            continue
        df = pd.read_parquet(p)
        if len(df) == 0:
            raise ValueError(f"Parquet has 0 rows: {p}")
