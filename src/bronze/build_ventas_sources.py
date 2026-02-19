# src/bronze/build_ventas_sources.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

from common.io import write_parquet


# -------------------------
# Helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _read_excel_raw(path: str, sheet_name: str) -> pd.DataFrame:
    """
    BRONZE ventas:
    - header=None
    - no inferir tipos
    - columnas col_0..col_n
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    raw.columns = [f"col_{i}" for i in range(raw.shape[1])]
    return raw


def _force_all_string(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].astype("string")
    return df


def main() -> None:
    cfg = load_settings()
    bronze_dir = Path(cfg["paths"]["bronze"])
    bronze_dir.mkdir(parents=True, exist_ok=True)

    ventas_cfg = cfg.get("ventas", {})
    sheet = ventas_cfg.get("ventas_sheet", "FULLES 10K")

    paths = {
        "ventas_2025": ventas_cfg.get("ventas_2025_path", ""),
        "ventas_2026": ventas_cfg.get("ventas_2026_path", ""),
    }

    for tag, path in paths.items():
        if not path:
            _info(f"{tag}: no definido, se omite.")
            continue

        df = _read_excel_raw(path, sheet_name=sheet)
        df = _force_all_string(df)

        df["bronze_source"] = tag
        df["bronze_extracted_at"] = datetime.now().isoformat(timespec="seconds")

        out = bronze_dir / f"{tag}_raw.parquet"
        write_parquet(df, out)

        _info(f"OK: {tag}_raw={len(df)} filas -> {out}")


if __name__ == "__main__":
    main()
