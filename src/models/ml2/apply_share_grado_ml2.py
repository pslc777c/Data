from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_share_grado_ml2_v1.parquet"

# Tallos totales ya corregidos por ML2 curva diaria (backtest)
IN_TALLOS_TOTAL_BT = EVAL / "backtest_pred_tallos_dia_ml2_final.parquet"

# Outputs
OUT_FACTOR_BT = EVAL / "backtest_factor_ml2_share_grado.parquet"
OUT_FINAL_BT = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"

OUT_FACTOR_PROD = GOLD / "factors" / "factor_ml2_share_grado.parquet"
OUT_FINAL_PROD = GOLD / "pred_tallos_grado_dia_ml2_final.parquet"


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _load_latest_model() -> tuple[object, dict]:
    metas = sorted(MODELS.glob("share_grado_ml2_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"No ML2 meta found in {MODELS}")
    meta_path = metas[-1]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_path = MODELS / f"share_grado_ml2_{meta['run_id']}.pkl"
    import joblib
    model = joblib.load(model_path)
    return model, meta


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prod", "backtest"], default="backtest")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model, meta = _load_latest_model()

    df = read_parquet(IN_DS).copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    for c in ["bloque_base", "variedad_canon", "grado", "tipo_sp", "estado", "area"]:
        if c in df.columns:
            df[c] = _canon_str(df[c])

    # features from meta
    for c in meta.get("num_cols", []):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in meta.get("cat_cols", []):
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    X = df[meta["num_cols"] + meta["cat_cols"]].copy()

    pred_log = model.predict(X).astype(float)
    lo, hi = meta["guardrails"]["clip_log_error"]
    pred_log = np.clip(pred_log, lo, hi)

    df["pred_log_error_share"] = pred_log
    df["pred_ratio_share"] = np.exp(df["pred_log_error_share"])

    df["share_ml1"] = pd.to_numeric(df["share_grado_ml1"], errors="coerce").fillna(0.0)
    df["share_raw"] = df["share_ml1"] * df["pred_ratio_share"]

    # --- RENORMALIZACIÓN (clave) por ciclo_id + fecha ---
    grp = ["ciclo_id", "fecha"]
    denom = df.groupby(grp)["share_raw"].transform("sum")
    df["share_final"] = np.where(denom > 0, df["share_raw"] / denom, df["share_ml1"])

    # Tallos totales (backtest usa salida de ML2 curva)
    if args.mode == "backtest":
        tallos_total = read_parquet(IN_TALLOS_TOTAL_BT).copy()
        tallos_total["fecha"] = pd.to_datetime(tallos_total["fecha"], errors="coerce").dt.normalize()
        if "variedad_canon" in tallos_total.columns:
            tallos_total["variedad_canon"] = _canon_str(tallos_total["variedad_canon"])
        tallos_total["bloque_base"] = _canon_str(tallos_total["bloque_base"])
        # buscamos columna de tallos total
        if "tallos_pred_final" in tallos_total.columns:
            tot_col = "tallos_pred_final"
        elif "tallos_final" in tallos_total.columns:
            tot_col = "tallos_final"
        elif "tallos_pred_ml2" in tallos_total.columns:
            tot_col = "tallos_pred_ml2"
        else:
            # fallback: toma primera que parezca tallos
            cand = [c for c in tallos_total.columns if "tallos" in c and "pred" in c]
            if not cand:
                raise KeyError("No encuentro columna de tallos total en backtest_pred_tallos_dia_ml2_final.parquet")
            tot_col = cand[0]

        base_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
        opt_cols = ["area", "tipo_sp", "estado"]

        cols_keep = [c for c in base_cols if c in tallos_total.columns]
        cols_keep += [c for c in opt_cols if c in tallos_total.columns]
        cols_keep += [tot_col]

        tallos_total = tallos_total[cols_keep].copy()
        tallos_total = tallos_total.rename(columns={tot_col: "tallos_total_ml2"})


        df = df.merge(tallos_total, on=["ciclo_id", "fecha", "bloque_base", "variedad_canon"], how="left")
    else:
        # PROD: usa tallos ML1 día (si aún no tienes tallos_curve_ml2 prod)
        # Si ya tienes pred_tallos_dia_ml2_final en gold, lo cambiamos luego.
        df["tallos_total_ml2"] = df["tallos_pred_ml1_grado_dia"].groupby([df["ciclo_id"], df["fecha"]]).transform("sum")

    df["tallos_final_grado_dia"] = df["tallos_total_ml2"] * df["share_final"]

    df["ml2_run_id"] = meta["run_id"]
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    # Factor output
    out_factor = df[[
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "share_ml1", "share_raw", "share_final",
        "pred_log_error_share", "pred_ratio_share",
        "ml2_run_id", "created_at"
    ]].copy()

    # Final output
    out_final = df[[
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "tallos_total_ml2", "share_final", "tallos_final_grado_dia",
        "ml2_run_id", "created_at"
    ]].copy()

    if args.mode == "backtest":
        EVAL.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_BT)
        write_parquet(out_final, OUT_FINAL_BT)
        print(f"[OK] BACKTEST factor: {OUT_FACTOR_BT} rows={len(out_factor):,}")
        print(f"[OK] BACKTEST final : {OUT_FINAL_BT} rows={len(out_final):,}")
    else:
        OUT_FACTOR_PROD.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(out_factor, OUT_FACTOR_PROD)
        write_parquet(out_final, OUT_FINAL_PROD)
        print(f"[OK] PROD factor: {OUT_FACTOR_PROD} rows={len(out_factor):,}")
        print(f"[OK] PROD final : {OUT_FINAL_PROD} rows={len(out_final):,}")


if __name__ == "__main__":
    main()
