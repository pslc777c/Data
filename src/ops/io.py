from __future__ import annotations

from pathlib import Path
import yaml


def load_settings(path: str = "config/settings.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"settings not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
