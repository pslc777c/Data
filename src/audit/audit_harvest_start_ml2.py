from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/audit/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
SILVER = DATA / "silver"
EVAL = DATA / "eval" / "ml2"
GOLD = DATA / "gold"


# -------------------------
# INPUTS (kpis output)
# -------------------------
IN_KPIS = EVAL / "ml2_harvest_start_eval.parquet"

# ciclo maestro (real)
IN_CICLO = SILVER / "fact_ciclo_maestro.parquet"

# backtest pred (puede estar en eval o gold según tu pipeline)
# (no fijamos 1 sola ruta: resolvemos por glob)
PRED_PREFIX_CANDS = [
    "backtest_pred_harvest_start_final_ml2*.parquet",
    "backtest_*harvest_start*final*ml2*.parquet",
    "pred_harvest_start*_final_ml2*.parquet",
    "pred_harvest_start*_ml2*.parquet",
]


# -------------------------
# OUTPUTS
# -------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_AUDIT_KPI = EVAL / f"audit_harvest_start_ml2_kpi_{ts}.parquet"
OUT_AUDIT_EX = EVAL / f"audit_harvest_start_ml2_examples_10x2_{ts}.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _resolve_pred_file() -> Path:
    """
    Busca el parquet de backtest/pred final en:
      - data/eval/ml2
      - data/gold
    Devuelve el "más reciente" por mtime.
    """
    hits: list[Path] = []
    for pat in PRED_PREFIX_CANDS:
        hits += list(EVAL.glob(pat))
        hits += list(GOLD.glob(pat))

    hits = [p for p in hits if p.exists()]
    if not hits:
        # Mensaje accionable (sin adivinar nombres)
        raise FileNotFoundError(
            "No encuentro el parquet de pred/backtest para harvest_start ML2.\n"
            f"Busqué en:\n  - {EVAL}\n  - {GOLD}\n"
            f"Con patrones:\n  - " + "\n  - ".join(PRED_PREFIX_CANDS)
        )

    hits.sort(key=lambda p: p.stat().st_mtime)
    return hits[-1]


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def main() -> None:
    print("\n=== AUDIT HARVEST_START ML2 ===")

    # 1) KPI global (si existe)
    kpi = read_parquet(IN_KPIS).copy() if IN_KPIS.exists() else pd.DataFrame()

    # 2) pred/backtest (ruta flexible)
    pred_path = _resolve_pred_file()
    pred = read_parquet(pred_path).copy()

    # 3) real
    ciclo = read_parquet(IN_CICLO).copy()

    # --------- canon real
    if "ciclo_id" not in ciclo.columns:
        raise KeyError(f"fact_ciclo_maestro sin 'ciclo_id'. cols={list(ciclo.columns)}")
    if "fecha_inicio_cosecha" not in ciclo.columns:
        raise KeyError(f"fact_ciclo_maestro sin 'fecha_inicio_cosecha'. cols={list(ciclo.columns)}")

    ciclo["ciclo_id"] = ciclo["ciclo_id"].astype(str)
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])

    # --------- canon pred
    if "ciclo_id" not in pred.columns:
        raise KeyError(f"Pred ({pred_path.name}) sin 'ciclo_id'. cols={list(pred.columns)}")

    pred["ciclo_id"] = pred["ciclo_id"].astype(str)

    col_pred_ml1 = _pick_first(pred, ["harvest_start_pred", "harvest_start_ml1", "harvest_start_baseline"])
    col_pred_ml2 = _pick_first(pred, ["harvest_start_final", "harvest_start_ml2_final", "harvest_start_ml2"])

    if col_pred_ml1 is None or col_pred_ml2 is None:
        raise KeyError(
            f"Pred ({pred_path.name}) no tiene columnas esperadas.\n"
            f"Espero ML1 en una de: harvest_start_pred / harvest_start_ml1 / harvest_start_baseline\n"
            f"Espero ML2 en una de: harvest_start_final / harvest_start_ml2_final / harvest_start_ml2\n"
            f"cols={list(pred.columns)}"
        )

    pred[col_pred_ml1] = _to_date(pred[col_pred_ml1])
    pred[col_pred_ml2] = _to_date(pred[col_pred_ml2])

    # --------- join
    df = ciclo.merge(pred[["ciclo_id", col_pred_ml1, col_pred_ml2]], on="ciclo_id", how="inner")

    df = df.loc[
        df["fecha_inicio_cosecha"].notna() & df[col_pred_ml1].notna() & df[col_pred_ml2].notna(),
        :,
    ].copy()

    # --------- errores
    df["err_ml1_days"] = (df["fecha_inicio_cosecha"] - df[col_pred_ml1]).dt.days.astype(int)
    df["err_ml2_days"] = (df["fecha_inicio_cosecha"] - df[col_pred_ml2]).dt.days.astype(int)
    df["abs_err_ml1"] = df["err_ml1_days"].abs().astype(float)
    df["abs_err_ml2"] = df["err_ml2_days"].abs().astype(float)
    df["improvement_abs_days"] = df["abs_err_ml1"] - df["abs_err_ml2"]

    # --------- examples (10 mejores + 10 peores por improvement)
    if len(df) > 0:
        best = df.sort_values("improvement_abs_days", ascending=False).head(10).copy()
        best["sample_group"] = "TOP_IMPROVE_10"
        worst = df.sort_values("improvement_abs_days", ascending=True).head(10).copy()
        worst["sample_group"] = "TOP_WORSE_10"
        ex = pd.concat([best, worst], ignore_index=True)
    else:
        ex = pd.DataFrame(columns=list(df.columns) + ["sample_group"])

    # --------- audit kpi (si no hay KPI previo, lo calculamos aquí)
    def _mae(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(np.abs(x))) if len(x) else float("nan")

    def _bias(x: pd.Series) -> float:
        x = pd.to_numeric(x, errors="coerce")
        return float(np.nanmean(x)) if len(x) else float("nan")

    if len(kpi) == 0:
        mae_ml1 = _mae(df["err_ml1_days"])
        mae_ml2 = _mae(df["err_ml2_days"])
        kpi = pd.DataFrame([{
            "n": int(len(df)),
            "mae_ml1_days": mae_ml1,
            "mae_ml2_days": mae_ml2,
            "bias_ml1_days": _bias(df["err_ml1_days"]),
            "bias_ml2_days": _bias(df["err_ml2_days"]),
            "improvement_abs_days": (mae_ml1 - mae_ml2) if (pd.notna(mae_ml1) and pd.notna(mae_ml2)) else float("nan"),
            "pred_file_used": pred_path.name,
            "created_at": pd.Timestamp(datetime.now()).normalize(),
        }])
    else:
        kpi = kpi.copy()
        kpi["pred_file_used"] = pred_path.name

    # --------- write
    EVAL.mkdir(parents=True, exist_ok=True)
    write_parquet(kpi, OUT_AUDIT_KPI)
    write_parquet(ex, OUT_AUDIT_EX)

    print(f"[OK] pred used: {pred_path}")
    print(f"[OK] wrote kpi : {OUT_AUDIT_KPI}")
    print(f"[OK] wrote ex  : {OUT_AUDIT_EX}")

    print("\n--- KPI ---")
    print(kpi.to_string(index=False))

    print("\n--- EXAMPLES (10x2) ---")
    keep = ["ciclo_id", "fecha_inicio_cosecha", col_pred_ml1, col_pred_ml2, "err_ml1_days", "err_ml2_days", "improvement_abs_days", "sample_group"]
    keep = [c for c in keep if c in ex.columns]
    if len(ex):
        print(ex[keep].to_string(index=False))
    else:
        print("[WARN] No hay filas para examples (join vacío o fechas NaT).")


if __name__ == "__main__":
    main()