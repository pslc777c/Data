from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


FEATURES_PATH = Path("data/features/features_cosecha_bloque_fecha.parquet")
REGISTRY_ROOT = Path("models_registry/ml1/dist_grado")
OUT_PATH = Path("data/gold/pred_dist_grado_ml1.parquet")


# Debe calzar con train_dist_grado.py
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
    "share_grado_baseline",
    # features “futuro” útiles (si existen)
    "day_in_harvest",
    "rel_pos",
    "n_harvest_days",
    "n_dias_cosecha",
]

CAT_COLS = [
    "variedad_canon",
    "tipo_sp",
    "area",
]


def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # defaults numéricas
    for c in NUM_COLS:
        if c not in out.columns:
            out[c] = np.nan

    # defaults categóricas
    for c in CAT_COLS:
        if c not in out.columns:
            out[c] = "UNKNOWN"

    return out


def _renormalize(out: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura share>=0 y sum=1 por (ciclo_id,bloque_base,variedad_canon,fecha).
    """
    out["share_grado_ml1_raw"] = pd.to_numeric(out["share_grado_ml1_raw"], errors="coerce")
    out["share_grado_ml1_raw"] = out["share_grado_ml1_raw"].fillna(0.0).clip(lower=0.0)

    grp = ["ciclo_id", "bloque_base", "variedad_canon", "fecha"]
    s = out.groupby(grp, dropna=False)["share_grado_ml1_raw"].transform("sum")

    # si suma=0, fallback a baseline y renormaliza baseline
    if (s == 0).any():
        out["share_grado_ml1_raw"] = np.where(
            s.to_numpy() > 0,
            out["share_grado_ml1_raw"].to_numpy(),
            pd.to_numeric(out["share_grado_baseline"], errors="coerce").fillna(0.0).to_numpy(),
        )
        s = out.groupby(grp, dropna=False)["share_grado_ml1_raw"].transform("sum")

    out["share_grado_ml1"] = np.where(s > 0, out["share_grado_ml1_raw"] / s, np.nan)
    return out


def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir() if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {ver_dir}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    grades = meta.get("grades", [])
    if not grades:
        raise ValueError("metrics.json no trae 'grades'")

    df = read_parquet(FEATURES_PATH).copy()
    df = _ensure_cols(df)

    need = {"ciclo_id", "fecha", "bloque_base", "grado", "share_grado_baseline", "variedad_canon"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"FEATURES sin columnas necesarias: {sorted(miss)}")

    # Tipos sanos
    df["ciclo_id"] = df["ciclo_id"].astype(str)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["bloque_base"] = pd.to_numeric(df["bloque_base"], errors="coerce").astype("Int64")
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")
    df["variedad_canon"] = df["variedad_canon"].astype(str).str.upper().str.strip()

    # base_all asegura TODOS los grados en salida (incluye futuro)
    base_all = df[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_baseline"]].copy()

    preds_parts: list[pd.DataFrame] = []
    missing_models: list[int] = []

    for g in grades:
        g_int = int(g)
        model_path = ver_dir / f"model_grade_{g_int}.joblib"
        if not model_path.exists():
            missing_models.append(g_int)
            continue

        model = load(model_path)

        sub = df[df["grado"] == g_int].copy()
        if sub.empty:
            continue

        X = sub[NUM_COLS + CAT_COLS]
        pred = model.predict(X)

        sub_out = sub[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_baseline"]].copy()
        sub_out["ml1_version"] = ver_dir.name
        sub_out["share_grado_ml1_raw"] = pd.to_numeric(pred, errors="coerce")

        preds_parts.append(sub_out)

    pred_only = pd.concat(preds_parts, ignore_index=True) if preds_parts else pd.DataFrame(
        columns=["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_baseline", "ml1_version", "share_grado_ml1_raw"]
    )

    # Merge pred con base
    out = base_all.merge(
        pred_only[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1_raw", "ml1_version"]],
        on=["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"],
        how="left",
    )

    out["ml1_version"] = out["ml1_version"].fillna(ver_dir.name)

    # fallback: donde no hay pred (no modelo o NaN), usa baseline
    out["share_grado_ml1_raw"] = out["share_grado_ml1_raw"].where(
        out["share_grado_ml1_raw"].notna(),
        out["share_grado_baseline"],
    )

    # renormalizar válido
    out = _renormalize(out)

    out["created_at"] = pd.Timestamp.utcnow()

    out = out[
        [
            "ciclo_id",
            "fecha",
            "bloque_base",
            "variedad_canon",
            "grado",
            "share_grado_baseline",
            "share_grado_ml1",
            "ml1_version",
            "created_at",
        ]
    ].sort_values(["ciclo_id", "bloque_base", "variedad_canon", "fecha", "grado"]).reset_index(drop=True)

    write_parquet(out, OUT_PATH)

    fmin = pd.to_datetime(out["fecha"].min()).date() if len(out) else None
    fmax = pd.to_datetime(out["fecha"].max()).date() if len(out) else None
    fut_rate = float((out["fecha"] > pd.Timestamp.today().normalize()).mean()) if len(out) else 0.0
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={ver_dir.name} | fecha_min={fmin} fecha_max={fmax}")
    print(f"[CHECK] % filas con fecha > hoy: {fut_rate:.4f}")
    if missing_models:
        missing_models = sorted(set(missing_models))
        print(f"[WARN] grados sin modelo en {ver_dir.name}: {missing_models[:30]}{'...' if len(missing_models)>30 else ''}")

    # sanity: sum=1
    grp = ["ciclo_id", "bloque_base", "variedad_canon", "fecha"]
    s = out.groupby(grp, dropna=False)["share_grado_ml1"].sum()
    if len(s):
        print(f"[CHECK] share_grado_ml1 sum min/max: {float(s.min()):.6f} / {float(s.max()):.6f}")


if __name__ == "__main__":
    main(version=None)
