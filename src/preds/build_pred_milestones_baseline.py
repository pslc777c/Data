from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _safe_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _choose_group_cols(fallback_group: str) -> list[str]:
    if fallback_group == "variedad":
        return ["variedad"]
    if fallback_group == "tipo_sp":
        return ["tipo_sp"]
    if fallback_group == "area":
        return ["area"]
    if fallback_group == "global":
        return []
    # default
    return ["variedad"]


def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    preds_dir.mkdir(parents=True, exist_ok=True)

    fact_path = silver_dir / "fact_ciclo_maestro.parquet"
    if not fact_path.exists():
        raise FileNotFoundError(f"No existe: {fact_path}")

    fact = read_parquet(fact_path).copy()

    # Normalizar fechas
    for c in ["fecha_sp", "fecha_inicio_cosecha", "fecha_fin_cosecha"]:
        if c in fact.columns:
            fact[c] = _norm_date(fact[c])

    # Parámetros
    dh_days = int(cfg.get("milestones", {}).get("dh_days", 7))
    post_tail_days = int(cfg.get("milestones", {}).get("post_tail_days", 2))

    fallback_group = str(cfg.get("pred_milestones", {}).get("fallback_group", "variedad")).strip().lower()
    min_hist_rows = int(cfg.get("pred_milestones", {}).get("min_hist_rows", 10))
    group_cols = _choose_group_cols(fallback_group)

    # 1) Entrenamiento baseline: solo CERRADOS con fechas completas
    estado_col = "estado" if "estado" in fact.columns else None
    if estado_col:
        hist = fact[fact[estado_col].astype(str).str.upper().eq("CERRADO")].copy()
    else:
        hist = fact.copy()

    hist = hist[
        hist["fecha_sp"].notna()
        & hist["fecha_inicio_cosecha"].notna()
        & hist["fecha_fin_cosecha"].notna()
    ].copy()

    if len(hist) == 0:
        raise ValueError("No hay historia CERRADA con fechas completas para construir baseline de milestones.")

    hist["dias_veg"] = (hist["fecha_inicio_cosecha"] - hist["fecha_sp"]).dt.days
    hist["dur_cosecha"] = (hist["fecha_fin_cosecha"] - hist["fecha_inicio_cosecha"]).dt.days

    # Remover outliers obvios (protección básica)
    hist = hist[(hist["dias_veg"] >= 0) & (hist["dias_veg"] <= 500)].copy()
    hist = hist[(hist["dur_cosecha"] >= 0) & (hist["dur_cosecha"] <= 500)].copy()

    # Medianas por grupo (o global)
    if group_cols:
        agg = (hist.groupby(group_cols, dropna=False)
               .agg(
                   n=("ciclo_id", "count"),
                   dias_veg_mediana=("dias_veg", "median"),
                   dur_cosecha_mediana=("dur_cosecha", "median"),
               )
               .reset_index())
    else:
        agg = pd.DataFrame([{
            "n": len(hist),
            "dias_veg_mediana": float(hist["dias_veg"].median()),
            "dur_cosecha_mediana": float(hist["dur_cosecha"].median()),
        }])

    # Global fallback siempre disponible
    global_dias_veg = int(round(float(hist["dias_veg"].median())))
    global_dur_cosecha = int(round(float(hist["dur_cosecha"].median())))

    # 2) Aplicación: ciclos con milestones faltantes
    need_pred = fact.copy()
    need_pred = need_pred[need_pred["fecha_sp"].notna()].copy()

    # Definimos “faltante” si no hay inicio cosecha o no hay fin cosecha
    miss_hs = need_pred["fecha_inicio_cosecha"].isna()
    miss_he = need_pred["fecha_fin_cosecha"].isna()

    need_pred = need_pred[miss_hs | miss_he].copy()
    if len(need_pred) == 0:
        print("No hay ciclos con milestones faltantes. No se generó pred_milestones.")
        return

    # Join con tabla de medianas
    if group_cols:
        need_pred = need_pred.merge(agg, on=group_cols, how="left")
        # aplicar min_hist_rows: si el grupo tiene poca historia, usamos global
        low_hist = need_pred["n"].fillna(0) < min_hist_rows
        need_pred.loc[low_hist, "dias_veg_mediana"] = np.nan
        need_pred.loc[low_hist, "dur_cosecha_mediana"] = np.nan
    else:
        need_pred["dias_veg_mediana"] = agg.loc[0, "dias_veg_mediana"]
        need_pred["dur_cosecha_mediana"] = agg.loc[0, "dur_cosecha_mediana"]

    # Fallback final global
    need_pred["dias_veg_fill"] = need_pred["dias_veg_mediana"].fillna(global_dias_veg).round().astype(int)
    need_pred["dur_cosecha_fill"] = need_pred["dur_cosecha_mediana"].fillna(global_dur_cosecha).round().astype(int)

    # Predicciones
    harvest_start_pred = need_pred["fecha_sp"] + pd.to_timedelta(need_pred["dias_veg_fill"], unit="D")
    harvest_end_pred = harvest_start_pred + pd.to_timedelta(need_pred["dur_cosecha_fill"], unit="D")

    post_start_pred = harvest_start_pred + pd.to_timedelta(dh_days, unit="D")
    post_end_pred = harvest_end_pred + pd.to_timedelta(dh_days + post_tail_days, unit="D")

    # Construir tabla larga pred_milestones
    pred_rows = []

    def add_pred(code: str, fecha: pd.Series, method: str):
        tmp = need_pred[["ciclo_id"]].copy()
        tmp["milestone_code"] = code
        tmp["fecha_pred"] = _norm_date(fecha)
        tmp["method"] = method
        tmp["model_version"] = f"baseline_median_{'_'.join(group_cols) if group_cols else 'global'}"
        tmp["created_at"] = datetime.now().isoformat(timespec="seconds")
        pred_rows.append(tmp)

    add_pred("HARVEST_START", harvest_start_pred, f"median_days_to_harvest({group_cols or ['global']})")
    add_pred("HARVEST_END", harvest_end_pred, f"median_harvest_duration({group_cols or ['global']})")
    add_pred("POST_START", post_start_pred, f"rule:harvest_start+{dh_days}")
    add_pred("POST_END", post_end_pred, f"rule:harvest_end+{dh_days}+{post_tail_days}")

    pred = pd.concat(pred_rows, ignore_index=True)
    pred = pred[pred["fecha_pred"].notna()].copy()

    out_path = preds_dir / "pred_milestones_ciclo.parquet"
    write_parquet(pred, out_path)

    print(f"OK: pred_milestones_ciclo={len(pred)} filas -> {out_path}")
    print(f"Global baseline: dias_veg={global_dias_veg}, dur_cosecha={global_dur_cosecha}, DH={dh_days}")


if __name__ == "__main__":
    main()
