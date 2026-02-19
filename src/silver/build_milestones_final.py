# src/silver/build_milestones_final.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import yaml

from common.io import read_parquet, write_parquet


# -------------------------
# Config / helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _ensure_cols(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name}: faltan columnas requeridas: {sorted(missing)}. Columnas={list(df.columns)}")


def _as_str(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()


def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))

    fact_path = silver_dir / "fact_milestones_ciclo.parquet"
    pred_path = preds_dir / "pred_milestones_ciclo.parquet"

    if not fact_path.exists():
        raise FileNotFoundError(f"No existe: {fact_path}")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No existe: {pred_path}. Ejecuta primero build_pred_milestones_baseline."
        )

    run_ts = datetime.now().isoformat(timespec="seconds")

    fact = read_parquet(fact_path).copy()
    pred = read_parquet(pred_path).copy()

    fact.columns = [str(c).strip() for c in fact.columns]
    pred.columns = [str(c).strip() for c in pred.columns]

    # -------------------------
    # Validaciones mínimas
    # -------------------------
    _ensure_cols(fact, {"ciclo_id", "milestone_code", "fecha"}, "fact_milestones_ciclo")
    # source puede no existir en algunos builds antiguos; si no está, lo llenamos
    if "source" not in fact.columns:
        fact["source"] = "FACT"

    _ensure_cols(pred, {"ciclo_id", "milestone_code", "fecha_pred"}, "pred_milestones_ciclo")
    # method/model_version/created_at pueden variar; creamos defaults si faltan
    if "method" not in pred.columns:
        pred["method"] = "BASELINE"
    if "model_version" not in pred.columns:
        pred["model_version"] = "unknown"
    if "created_at" not in pred.columns:
        pred["created_at"] = run_ts

    # Normalizar ids/códigos como string (estable)
    for c in ["ciclo_id", "milestone_code"]:
        _as_str(fact, c)
        _as_str(pred, c)

    # Fechas
    fact["fecha"] = _norm_date(fact["fecha"])
    pred["fecha_pred"] = _norm_date(pred["fecha_pred"])

    # Filtrar filas inválidas (evita que entren NaT y generen joins basura)
    fact = fact[fact["ciclo_id"].notna() & fact["milestone_code"].notna() & fact["fecha"].notna()].copy()
    pred = pred[pred["ciclo_id"].notna() & pred["milestone_code"].notna() & pred["fecha_pred"].notna()].copy()

    # -------------------------
    # Estandarizar a esquema común
    # -------------------------
    fact2 = fact[["ciclo_id", "milestone_code", "fecha", "source"]].copy()
    fact2["kind"] = "FACT"
    fact2["method"] = fact2["source"].astype(str)
    fact2["model_version"] = "FACT"
    fact2["created_at"] = run_ts

    pred2 = pred[["ciclo_id", "milestone_code", "fecha_pred", "method", "model_version", "created_at"]].copy()
    pred2 = pred2.rename(columns={"fecha_pred": "fecha"})
    pred2["kind"] = "PRED"
    # "source" para PRED = model_version (útil para auditoría)
    pred2["source"] = pred2["model_version"].astype(str)

    # -------------------------
    # Unir y priorizar
    #   1) FACT sobre PRED
    #   2) Entre PRED, el más reciente (created_at mayor) si hay duplicados
    # -------------------------
    allm = pd.concat(
        [
            fact2[["ciclo_id", "milestone_code", "fecha", "kind", "method", "model_version", "source", "created_at"]],
            pred2[["ciclo_id", "milestone_code", "fecha", "kind", "method", "model_version", "source", "created_at"]],
        ],
        ignore_index=True,
    )

    # Priorización
    allm["__prio_kind"] = (allm["kind"] == "FACT").astype(int)
    # Para ordenar created_at como datetime (si viene como string ISO)
    allm["__created_at_dt"] = pd.to_datetime(allm["created_at"], errors="coerce")

    final = (
        allm.sort_values(
            ["ciclo_id", "milestone_code", "__prio_kind", "__created_at_dt"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["ciclo_id", "milestone_code"], keep="first")
        .drop(columns=["__prio_kind", "__created_at_dt"])
        .reset_index(drop=True)
    )

    out_path = silver_dir / "milestones_ciclo_final.parquet"
    write_parquet(final, out_path)

    print(f"OK: milestones_ciclo_final={len(final)} filas -> {out_path}")
    print("Counts by kind:\n", final["kind"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
