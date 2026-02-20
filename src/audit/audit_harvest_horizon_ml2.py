from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/audit/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
SILVER = DATA / "silver"
EVAL = DATA / "eval" / "ml2"
GOLD = DATA / "gold"

IN_CICLO = SILVER / "fact_ciclo_maestro.parquet"

# En tu pipeline algunas salidas backtest se guardan en EVAL/ml2
# pero a veces cambian nombres. Lo resolvemos por búsqueda.
PRED_CANDIDATES = [
    EVAL / "backtest_pred_harvest_horizon_final_ml2.parquet",
    EVAL / "backtest_pred_harvest_horizon_ml2.parquet",
    GOLD / "pred_harvest_horizon_final_ml2.parquet",
    GOLD / "pred_harvest_horizon_ml2.parquet",
]

OUT_GLOBAL = EVAL / f"audit_harvest_horizon_ml2_global_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
OUT_EXAMPLES = EVAL / f"audit_harvest_horizon_ml2_examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _find_pred_file() -> Path:
    for p in PRED_CANDIDATES:
        if p.exists():
            return p

    # fallback: buscar por patrón en EVAL y GOLD
    for folder in [EVAL, GOLD]:
        hits = sorted(folder.glob("*harvest_horizon*ml2*.parquet"))
        if hits:
            return hits[-1]

    raise FileNotFoundError(
        "No encuentro parquet de predicción harvest_horizon ML2.\n"
        f"Probé candidatos: {[str(p) for p in PRED_CANDIDATES]}\n"
        f"Y patrones: {EVAL}/*harvest_horizon*ml2*.parquet, {GOLD}/*harvest_horizon*ml2*.parquet"
    )


def _mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else float("nan")


def _bias(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x)) if len(x) else float("nan")


def _ensure_real_horizon_days(ciclo: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve ciclo con columna horizon_real_days.
    Regla:
      1) Si existe alguna columna candidata de horizonte real -> usarla.
      2) Si no, calcular: (fecha_fin_cosecha - fecha_inicio_cosecha).days + 1 (inclusivo), clip>=1.
    """
    out = ciclo.copy()

    # 1) intentar columna explícita si existiera (en otros entornos)
    cand_real = [
        "n_harvest_days_real",
        "n_harvest_days",
        "harvest_horizon_days",
        "harvest_horizon",
        "horizon_days",
        "window_days",
    ]
    c = _pick_first(out, cand_real)
    if c is not None:
        out["horizon_real_days"] = pd.to_numeric(out[c], errors="coerce")
        return out

    # 2) calcular desde fechas (tu caso actual)
    if "fecha_inicio_cosecha" not in out.columns or "fecha_fin_cosecha" not in out.columns:
        raise KeyError(
            "No encuentro horizonte real ni puedo calcularlo.\n"
            f"Busqué candidatos: {cand_real}\n"
            "Y esperaba fechas: fecha_inicio_cosecha y fecha_fin_cosecha\n"
            f"cols={list(out.columns)}"
        )

    out["fecha_inicio_cosecha"] = _to_date(out["fecha_inicio_cosecha"])
    out["fecha_fin_cosecha"] = _to_date(out["fecha_fin_cosecha"])

    delta = (out["fecha_fin_cosecha"] - out["fecha_inicio_cosecha"]).dt.days
    # Inclusivo: start y end cuentan como días de cosecha
    out["horizon_real_days"] = (delta + 1).astype("Float64")

    # sanity
    out.loc[out["horizon_real_days"] < 1, "horizon_real_days"] = 1
    return out


def main() -> None:
    print("\n=== AUDIT: HARVEST HORIZON ML2 ===")
    created_at = pd.Timestamp.utcnow()

    ciclo = read_parquet(IN_CICLO).copy()
    pred_path = _find_pred_file()
    pred = read_parquet(pred_path).copy()

    ciclo.columns = [str(c).strip() for c in ciclo.columns]
    pred.columns = [str(c).strip() for c in pred.columns]

    # --- keys
    if "ciclo_id" not in ciclo.columns:
        raise KeyError(f"fact_ciclo_maestro sin ciclo_id. cols={list(ciclo.columns)}")
    if "ciclo_id" not in pred.columns:
        raise KeyError(f"pred file sin ciclo_id. file={pred_path} cols={list(pred.columns)}")

    ciclo["ciclo_id"] = ciclo["ciclo_id"].astype(str)
    pred["ciclo_id"] = pred["ciclo_id"].astype(str)

    # --- real horizon
    ciclo = _ensure_real_horizon_days(ciclo)

    # --- pred columns (tolerante)
    # Muchos scripts usan horizon_pred / horizon_final como días (int/float)
    pred_col_ml1 = _pick_first(pred, ["harvest_horizon_pred", "horizon_pred", "n_harvest_days_pred", "window_days_pred"])
    pred_col_ml2 = _pick_first(pred, ["harvest_horizon_final", "horizon_final", "n_harvest_days_final", "window_days_final"])

    if pred_col_ml1 is None and pred_col_ml2 is None:
        raise KeyError(
            "No encuentro columnas de predicción de horizonte en el parquet.\n"
            "Candidatos ML1: harvest_horizon_pred / horizon_pred / n_harvest_days_pred / window_days_pred\n"
            "Candidatos ML2: harvest_horizon_final / horizon_final / n_harvest_days_final / window_days_final\n"
            f"file={pred_path.name} cols={list(pred.columns)}"
        )

    df = ciclo[["ciclo_id", "horizon_real_days"]].merge(pred, on="ciclo_id", how="inner")

    # numeric
    df["horizon_real_days"] = pd.to_numeric(df["horizon_real_days"], errors="coerce")
    if pred_col_ml1 is not None:
        df["horizon_pred_ml1"] = pd.to_numeric(df[pred_col_ml1], errors="coerce")
    else:
        df["horizon_pred_ml1"] = np.nan

    if pred_col_ml2 is not None:
        df["horizon_pred_ml2"] = pd.to_numeric(df[pred_col_ml2], errors="coerce")
    else:
        df["horizon_pred_ml2"] = np.nan

    # filtro válido
    m = df["horizon_real_days"].notna() & (df["horizon_real_days"] > 0)
    if pred_col_ml1 is not None:
        m = m & df["horizon_pred_ml1"].notna()
    if pred_col_ml2 is not None:
        m = m & df["horizon_pred_ml2"].notna()

    d = df.loc[m].copy()

    # errores
    if pred_col_ml1 is not None:
        d["err_ml1_days"] = d["horizon_real_days"].astype(float) - d["horizon_pred_ml1"].astype(float)
    else:
        d["err_ml1_days"] = np.nan

    if pred_col_ml2 is not None:
        d["err_ml2_days"] = d["horizon_real_days"].astype(float) - d["horizon_pred_ml2"].astype(float)
    else:
        d["err_ml2_days"] = np.nan

    mae_ml1 = _mae(d["err_ml1_days"]) if pred_col_ml1 is not None else np.nan
    mae_ml2 = _mae(d["err_ml2_days"]) if pred_col_ml2 is not None else np.nan

    out = pd.DataFrame([{
        "n": int(len(d)),
        "pred_file": pred_path.name,
        "pred_col_ml1": pred_col_ml1,
        "pred_col_ml2": pred_col_ml2,
        "mae_ml1_days": float(mae_ml1) if pd.notna(mae_ml1) else np.nan,
        "mae_ml2_days": float(mae_ml2) if pd.notna(mae_ml2) else np.nan,
        "bias_ml1_days": _bias(d["err_ml1_days"]) if pred_col_ml1 is not None else np.nan,
        "bias_ml2_days": _bias(d["err_ml2_days"]) if pred_col_ml2 is not None else np.nan,
        "improvement_abs_days": (float(mae_ml1) - float(mae_ml2)) if (pd.notna(mae_ml1) and pd.notna(mae_ml2)) else np.nan,
        "created_at": created_at.normalize(),
    }])

    # ejemplos top/bottom por mejora absoluta
    if (pred_col_ml1 is not None) and (pred_col_ml2 is not None) and len(d):
        d["abs_err_ml1"] = d["err_ml1_days"].abs()
        d["abs_err_ml2"] = d["err_ml2_days"].abs()
        d["improvement_abs"] = d["abs_err_ml1"] - d["abs_err_ml2"]

        top = d.sort_values("improvement_abs", ascending=False).head(10).copy()
        top["sample_group"] = "TOP_IMPROVE_10"
        worst = d.sort_values("improvement_abs", ascending=True).head(10).copy()
        worst["sample_group"] = "TOP_WORSE_10"
        ex = pd.concat([top, worst], ignore_index=True)
    else:
        ex = pd.DataFrame()

    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_GLOBAL)
    if len(ex):
        write_parquet(ex, OUT_EXAMPLES)

    print(f"[OK] Global:   {OUT_GLOBAL}")
    print(out.to_string(index=False))
    if len(ex):
        print(f"[OK] Examples: {OUT_EXAMPLES} rows={len(ex)}")
    else:
        print("[WARN] Examples no generados (faltan pred ML1/ML2 o no hay filas válidas).")


if __name__ == "__main__":
    main()