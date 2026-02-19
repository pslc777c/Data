from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


FEATURES_PATH = Path("data/features/features_peso_tallo_grado_bloque_dia.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/peso_tallo_grado")
OUT_PATH = Path("data/gold/pred_peso_tallo_grado_ml1.parquet")


NUM_COLS = [
    "pct_avance_real",
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    "dow",
    "month",
    "weekofyear",
    "peso_tallo_baseline_g",
]
CAT_COLS = ["variedad_canon", "tipo_sp", "area"]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir() if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df = read_parquet(FEATURES_PATH).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "peso_tallo_baseline_g"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"FEATURES sin columnas necesarias: {sorted(miss)}")

    # Canonización llaves para merges aguas abajo
    df["fecha"] = _to_date(df["fecha"])
    df["bloque_base"] = _canon_int(df["bloque_base"])
    df["grado"] = _canon_int(df["grado"])
    df["variedad_canon"] = _canon_str(df["variedad_canon"])

    # Asegurar columnas
    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    # IMPORTANTE: predecir todos los grados presentes (no solo los del meta)
    grados_features = sorted([int(x) for x in df["grado"].dropna().unique().tolist()])
    grados_meta = meta.get("grades", [])
    grados_meta = [int(x) for x in grados_meta] if grados_meta else []
    grades_all = sorted(set(grados_features) | set(grados_meta))

    preds = []

    for g in grades_all:
        sub = df[df["grado"] == g].copy()
        if sub.empty:
            continue

        model_path = ver_dir / f"model_grade_{g}.joblib"
        if model_path.exists():
            model = load(model_path)
            X = sub[NUM_COLS + CAT_COLS]
            sub["factor_peso_tallo_ml1_raw"] = model.predict(X)
            sub["peso_model_fallback_const1"] = 0
        else:
            sub["factor_peso_tallo_ml1_raw"] = 1.0
            sub["peso_model_fallback_const1"] = 1

        preds.append(
            sub[
                [
                    "fecha",
                    "bloque_base",
                    "variedad_canon",
                    "grado",
                    "peso_tallo_baseline_g",
                    "factor_peso_tallo_ml1_raw",
                    "peso_model_fallback_const1",
                ]
            ]
        )

    out = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=[
            "fecha",
            "bloque_base",
            "variedad_canon",
            "grado",
            "peso_tallo_baseline_g",
            "factor_peso_tallo_ml1_raw",
            "peso_model_fallback_const1",
        ]
    )

    # Clip coherente con train
    out["factor_peso_tallo_ml1"] = (
        pd.to_numeric(out["factor_peso_tallo_ml1_raw"], errors="coerce")
        .clip(lower=0.60, upper=1.60)
    )

    base = pd.to_numeric(out["peso_tallo_baseline_g"], errors="coerce")
    out["peso_tallo_ml1_g"] = np.where(
        base.fillna(0) > 0,
        base * out["factor_peso_tallo_ml1"],
        np.nan,
    )

    out["ml1_version"] = ver_dir.name
    out["created_at"] = pd.Timestamp.utcnow()

    # Unicidad defensiva por llaves (por si features trae duplicados)
    keys = ["fecha", "bloque_base", "variedad_canon", "grado"]
    if out.duplicated(subset=keys).any():
        out = (
            out.groupby(keys, dropna=False, as_index=False)
            .agg(
                peso_tallo_baseline_g=("peso_tallo_baseline_g", "mean"),
                factor_peso_tallo_ml1_raw=("factor_peso_tallo_ml1_raw", "mean"),
                factor_peso_tallo_ml1=("factor_peso_tallo_ml1", "mean"),
                peso_tallo_ml1_g=("peso_tallo_ml1_g", "mean"),
                peso_model_fallback_const1=("peso_model_fallback_const1", "max"),
                ml1_version=("ml1_version", "first"),
                created_at=("created_at", "first"),
            )
        )

    out = out[
        [
            "fecha",
            "bloque_base",
            "variedad_canon",
            "grado",
            "peso_tallo_baseline_g",
            "factor_peso_tallo_ml1_raw",
            "factor_peso_tallo_ml1",
            "peso_tallo_ml1_g",
            "peso_model_fallback_const1",
            "ml1_version",
            "created_at",
        ]
    ].sort_values(["bloque_base", "variedad_canon", "fecha", "grado"]).reset_index(drop=True)

    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={ver_dir.name}")


if __name__ == "__main__":
    main(version=None)
