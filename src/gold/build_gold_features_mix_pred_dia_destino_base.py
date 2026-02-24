from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet

# Fuente segura de llaves (ya existe en tu pipeline ML1):
# build_pred_cajas_postcosecha_seed_mix_grado_dia.py produce gold/pred_poscosecha_seed_dia_destino.parquet
IN_KEYS = Path("data/gold/pred_poscosecha_seed_dia_destino.parquet")

OUT = Path("data/gold/features_mix_pred_dia_destino.parquet")


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main() -> None:
    df = read_parquet(IN_KEYS).copy()

    # Intentamos inferir columnas de llaves típicas
    # Requisito mínimo para downstream: fecha_post + destino
    if "fecha_post" not in df.columns:
        # fallback: algunos outputs usan fecha_dia / fecha / fecha_post_dia
        for alt in ["fecha", "fecha_dia", "fecha_post_dia", "dia"]:
            if alt in df.columns:
                df["fecha_post"] = pd.to_datetime(df[alt], errors="coerce").dt.normalize()
                break
        else:
            raise KeyError(f"IN_KEYS no tiene columna fecha_post ni alternativas. cols={list(df.columns)}")

    df["fecha_post"] = pd.to_datetime(df["fecha_post"], errors="coerce").dt.normalize()

    if "destino" not in df.columns:
        # fallback común: destino_final / destino_canon / destino_nom
        for alt in ["destino_final", "destino_canon", "destino_nom"]:
            if alt in df.columns:
                df["destino"] = df[alt]
                break
        else:
            raise KeyError(f"IN_KEYS no tiene columna destino ni alternativas. cols={list(df.columns)}")

    df["destino"] = _canon_destino(df["destino"])

    # Base keyspace
    out = df[["fecha_post", "destino"]].dropna().drop_duplicates().sort_values(["fecha_post", "destino"])

    # Columnas b2_* mínimas (extiende aquí si tus modelos exigen otras)
    # Usamos float y NaN/0 para que downstream no reviente por dtype.
    out["b2_share_56_75"] = np.nan  # el patch lo usa directo
    # Puedes precrear otras columnas b2_* si luego el modelo las exige:
    # out["b2_share_..."] = np.nan

    write_parquet(out, OUT)
    print(f"[OK] {OUT} | rows={len(out):,} | cols={len(out.columns)}")
    print("[INFO] fechas:", out["fecha_post"].min(), out["fecha_post"].max())
    print("[INFO] destinos:", sorted(out["destino"].unique().tolist()))


if __name__ == "__main__":
    main()