from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    # Pandas decide engine (pyarrow/fastparquet) según instalación.
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    df.to_parquet(path, index=False)


def read_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)
