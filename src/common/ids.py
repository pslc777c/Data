from __future__ import annotations

import re
import hashlib
import pandas as pd

def norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def make_bloque_id(bloque: object) -> str:
    return norm_text(bloque)

def make_variedad_id(variedad: object) -> str:
    return norm_text(variedad)

def make_ciclo_id(bloque_id: str, variedad_id: str, tipo_sp: str, fecha_sp) -> str:
    # fecha_sp se espera como Timestamp/date.
    f = pd.to_datetime(fecha_sp).date().isoformat()
    raw = f"{bloque_id}|{variedad_id}|{tipo_sp}|{f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()
