from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from src.common.io import read_parquet, write_parquet


def _project_root() -> Path:
    # .../src/eval/file.py -> repo_root = parents[2]
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
EVAL_DIR = DATA_DIR / "eval" / "ml2"

IN_CICLO = SILVER_DIR / "fact_ciclo_maestro.parquet"
IN_FACTOR = GOLD_DIR / "factors" / "factor_ml2_harvest_start.parquet"
IN_DS = GOLD_DIR / "ml2_datasets" / "ds_harvest_start_ml2_v2.parquet"  # para features clima agregadas


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _mae(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(np.abs(x))) if len(x) else float("nan")


def _bias(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x)) if len(x) else float("nan")


def main() -> None:
    # ---------- Load ----------
    ciclo = read_parquet(IN_CICLO).copy()
    fact = read_parquet(IN_FACTOR).copy()

    # Dataset v2 (para features clima as_of); puede no existir en prod, así que lo protegemos
    ds = None
    if IN_DS.exists():
        ds = read_parquet(IN_DS).copy()

    # ---------- Canon / dates ----------
    ciclo["estado"] = _canon_str(ciclo["estado"])
    ciclo["bloque_base"] = _canon_str(ciclo["bloque_base"])
    ciclo["fecha_sp"] = _to_date(ciclo["fecha_sp"])
    ciclo["fecha_inicio_cosecha"] = _to_date(ciclo["fecha_inicio_cosecha"])

    fact["bloque_base"] = _canon_str(fact["bloque_base"])
    if "estado" in fact.columns:
        fact["estado"] = _canon_str(fact["estado"])

    fact["fecha_sp"] = _to_date(fact["fecha_sp"])
    fact["harvest_start_pred"] = _to_date(fact["harvest_start_pred"])
    fact["harvest_start_final"] = _to_date(fact["harvest_start_final"])

    # ---------- Join core audit frame ----------
    df = ciclo.merge(
        fact,
        on="ciclo_id",
        how="inner",
        suffixes=("_real", "_ml2"),
    )

    # usar estado de ciclo maestro como autoridad
    df["estado"] = df["estado_real"]

    # ---------- Errors ----------
    df["err_ml1_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_pred"]).dt.days
    df["err_ml2_days"] = (df["fecha_inicio_cosecha"] - df["harvest_start_final"]).dt.days

    # Algunas filas pueden no tener real (activos sin inicio real aún). Filtramos para KPI real.
    df_eval = df.loc[df["fecha_inicio_cosecha"].notna() & df["harvest_start_pred"].notna() & df["harvest_start_final"].notna(), :].copy()

    # ---------- KPI global ----------
    mae_ml1 = _mae(df_eval["err_ml1_days"])
    mae_ml2 = _mae(df_eval["err_ml2_days"])
    bias_ml1 = _bias(df_eval["err_ml1_days"])
    bias_ml2 = _bias(df_eval["err_ml2_days"])

    kpi_global = pd.DataFrame([{
        "n": int(len(df_eval)),
        "mae_ml1_days": mae_ml1,
        "mae_ml2_days": mae_ml2,
        "bias_ml1_days": bias_ml1,
        "bias_ml2_days": bias_ml2,
        "improvement_abs_days": (mae_ml1 - mae_ml2) if (pd.notna(mae_ml1) and pd.notna(mae_ml2)) else np.nan,
        "created_at": pd.Timestamp(datetime.now()).normalize(),
    }])

    # ---------- Adjustment distribution ----------
    adj = pd.to_numeric(df.get("pred_error_start_days"), errors="coerce")
    dist = pd.DataFrame([{
        "n_factor_rows": int(len(df)),
        "adj_min": float(np.nanmin(adj)) if len(adj) else np.nan,
        "adj_p25": float(np.nanpercentile(adj, 25)) if len(adj) else np.nan,
        "adj_median": float(np.nanmedian(adj)) if len(adj) else np.nan,
        "adj_p75": float(np.nanpercentile(adj, 75)) if len(adj) else np.nan,
        "adj_max": float(np.nanmax(adj)) if len(adj) else np.nan,
    }])

    # % clipping (asumimos guardrail ±21)
    CLIP_LO, CLIP_HI = -21, 21
    clip_mask = (adj <= CLIP_LO) | (adj >= CLIP_HI)
    dist["pct_clip"] = float(np.nanmean(clip_mask)) if len(adj) else np.nan

    # ---------- KPI por estado (ABIERTO/CERRADO) ----------
    by_estado_rows = []
    for est, g in df_eval.groupby("estado"):
        mae1 = _mae(g["err_ml1_days"])
        mae2 = _mae(g["err_ml2_days"])
        by_estado_rows.append({
            "estado": est,
            "n": int(len(g)),
            "mae_ml1_days": mae1,
            "mae_ml2_days": mae2,
            "bias_ml1_days": _bias(g["err_ml1_days"]),
            "bias_ml2_days": _bias(g["err_ml2_days"]),
            "improvement_abs_days": (mae1 - mae2) if (pd.notna(mae1) and pd.notna(mae2)) else np.nan,
        })
    kpi_by_estado = pd.DataFrame(by_estado_rows).sort_values(["estado"])

    # ---------- “Sentido agronómico” simple (si hay dataset con features) ----------
    # Join con ds (tiene gdc_cum_sp, rain_cum_sp, etc. a nivel ciclo_id+as_of_date).
    # Para audit, tomamos el último as_of por ciclo (más cercano al inicio real).
    agr = pd.DataFrame()
    if ds is not None and len(ds):
        ds = ds.copy()
        ds["as_of_date"] = _to_date(ds["as_of_date"])
        # último as_of por ciclo
        ds_last = ds.sort_values(["ciclo_id", "as_of_date"]).groupby("ciclo_id", as_index=False).tail(1)
        # join
        wanted = ["ciclo_id", "gdc_cum_sp", "gdc_14d", "rain_7d", "rain_cum_sp", "solar_7d", "temp_avg_7d"]
        cols = [c for c in wanted if c in ds_last.columns]
        agr = df.merge(ds_last[cols].copy(), on="ciclo_id", how="left")


        # Correlaciones simples (no causal proof, solo sanity)
        rows = []
        feat_list = [c for c in ["gdc_cum_sp", "gdc_14d", "rain_7d", "rain_cum_sp", "solar_7d", "temp_avg_7d"] if c in agr.columns]
        for col in feat_list:

            x = pd.to_numeric(agr[col], errors="coerce")
            y = pd.to_numeric(agr["pred_error_start_days"], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 30:
                corr = float(np.corrcoef(x[m], y[m])[0, 1])
            else:
                corr = np.nan
            rows.append({"feature": col, "corr_with_pred_error_days": corr, "n": int(m.sum())})
        agr_corr = pd.DataFrame(rows)
    else:
        agr_corr = pd.DataFrame([{"feature": None, "corr_with_pred_error_days": np.nan, "n": 0}])

    # ---------- 10 ejemplos ABIERTO y 10 CERRADO ----------
    # Muestra reproducible, y prioriza casos con mayor ajuste absoluto para que sea informativo.
    def _pick_examples(est: str, n: int = 10) -> pd.DataFrame:
        g = df.copy()
        g = g.loc[g["estado"] == est, :].copy()
        g["abs_adj"] = pd.to_numeric(g["pred_error_start_days"], errors="coerce").abs()
        # si no hay suficientes, devuelve lo que haya
        g = g.sort_values(["abs_adj"], ascending=False).head(max(n * 3, n))  # pool
        # sample dentro del pool para variedad
        if len(g) > n:
            g = g.sample(n=n, random_state=42)
        cols = [
            "ciclo_id", "bloque_base_real", "variedad", "area", "tipo_sp",
            "estado",
            "fecha_sp_real", "fecha_inicio_cosecha",
            "harvest_start_pred", "harvest_start_final",
            "pred_error_start_days",
            "err_ml1_days", "err_ml2_days",
            "ml1_version", "ml2_run_id",
        ]
        # Algunas columnas tienen sufijos por el merge; normalizamos nombres:
        rename = {
            "bloque_base_real": "bloque_base",
            "fecha_sp_real": "fecha_sp",
        }
        out = g[[c for c in cols if c in g.columns]].rename(columns=rename)
        return out

    ex_abierto = _pick_examples("ABIERTO", 10)
    ex_cerrado = _pick_examples("CERRADO", 10)

    examples = pd.concat(
        [
            ex_abierto.assign(sample_group="ABIERTO_10"),
            ex_cerrado.assign(sample_group="CERRADO_10"),
        ],
        ignore_index=True,
    )

    # ---------- Save outputs ----------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_kpi_global = EVAL_DIR / f"audit_harvest_start_ml2_kpi_global_{ts}.parquet"
    out_dist = EVAL_DIR / f"audit_harvest_start_ml2_adjust_dist_{ts}.parquet"
    out_by_estado = EVAL_DIR / f"audit_harvest_start_ml2_kpi_by_estado_{ts}.parquet"
    out_corr = EVAL_DIR / f"audit_harvest_start_ml2_agro_corr_{ts}.parquet"
    out_examples = EVAL_DIR / f"audit_harvest_start_ml2_examples_10x2_{ts}.parquet"

    write_parquet(kpi_global, out_kpi_global)
    write_parquet(dist, out_dist)
    write_parquet(kpi_by_estado, out_by_estado)
    write_parquet(agr_corr, out_corr)
    write_parquet(examples, out_examples)

    # CSV opcional (muy útil para revisar rápido)
    examples.to_csv(EVAL_DIR / f"audit_harvest_start_ml2_examples_10x2_{ts}.csv", index=False)

    # ---------- Print summary ----------
    print("\n=== ML2 HARVEST START AUDIT ===")
    print(f"KPI global parquet : {out_kpi_global}")
    print(f"Dist ajustes parquet: {out_dist}")
    print(f"KPI por estado      : {out_by_estado}")
    print(f"Corr agronómica     : {out_corr}")
    print(f"Examples 10x2       : {out_examples}")
    print("\n--- KPI GLOBAL ---")
    print(kpi_global.to_string(index=False))
    print("\n--- AJUSTE DIST ---")
    print(dist.to_string(index=False))
    print("\n--- KPI POR ESTADO ---")
    print(kpi_by_estado.to_string(index=False))
    print("\n--- CORR (sanity) ---")
    print(agr_corr.to_string(index=False))
    print("\n--- EXAMPLES (top) ---")
    print(examples.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
