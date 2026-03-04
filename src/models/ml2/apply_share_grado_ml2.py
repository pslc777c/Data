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
SILVER = DATA / "silver"
EVAL = DATA / "eval" / "ml2"
MODELS = DATA / "models" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_share_grado_ml2_v1.parquet"
IN_REAL = SILVER / "fact_cosecha_real_grado_dia.parquet"

# Tallos totales ya corregidos por ML2 curva diaria
IN_TALLOS_TOTAL_BT = EVAL / "backtest_pred_tallos_dia_ml2_final.parquet"
IN_TALLOS_TOTAL_PROD = GOLD / "pred_tallos_dia_ml2_final.parquet"

# Outputs
OUT_FACTOR_BT = EVAL / "backtest_factor_ml2_share_grado.parquet"
OUT_FINAL_BT = EVAL / "backtest_pred_tallos_grado_dia_ml2_final.parquet"

OUT_FACTOR_PROD = GOLD / "factors" / "factor_ml2_share_grado.parquet"
OUT_FINAL_PROD = GOLD / "pred_tallos_grado_dia_ml2_final.parquet"


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _resolve_block_base(df: pd.DataFrame) -> pd.DataFrame:
    if "bloque_base" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_base"])
        return df
    if "bloque_padre" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque_padre"])
        return df
    if "bloque" in df.columns:
        df["bloque_base"] = _canon_str(df["bloque"])
        return df
    raise KeyError("No encuentro bloque_base/bloque_padre/bloque en fact real.")


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


def _pick_tallos_total_col(df: pd.DataFrame) -> str:
    for c in ["tallos_final_ml2_dia", "tallos_pred_ml2_dia", "tallos_ml2_dia", "tallos_pred_final", "tallos_final"]:
        if c in df.columns:
            return c
    cand = [c for c in df.columns if "tallos" in c]
    if not cand:
        raise KeyError("No encuentro columna de tallos total ML2 diario.")
    return cand[0]


def main() -> None:
    args = _parse_args()
    model, meta = _load_latest_model()
    as_of_date = (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).normalize()

    df = read_parquet(IN_DS).copy()
    df["fecha"] = _to_date(df["fecha"])
    for c in ["bloque_base", "variedad_canon", "grado", "tipo_sp", "estado", "area"]:
        if c in df.columns:
            df[c] = _canon_str(df[c])
    if "estado" in df.columns:
        df["is_closed_cycle"] = df["estado"].eq("CERRADO")
    else:
        df["is_closed_cycle"] = False
    df["is_active_cycle"] = ~df["is_closed_cycle"]

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

    # Share ML2 model (sin anclaje)
    grp = ["ciclo_id", "fecha"]
    denom = df.groupby(grp)["share_raw"].transform("sum")
    df["share_ml2_model"] = np.where(denom > 0, df["share_raw"] / denom, df["share_ml1"])

    # Tallos totales diarios ML2
    tallos_in = IN_TALLOS_TOTAL_BT if args.mode == "backtest" else IN_TALLOS_TOTAL_PROD
    if tallos_in.exists():
        tallos_total = read_parquet(tallos_in).copy()
        tallos_total["fecha"] = _to_date(tallos_total["fecha"])
        tallos_total["bloque_base"] = _canon_str(tallos_total["bloque_base"])
        if "variedad_canon" in tallos_total.columns:
            tallos_total["variedad_canon"] = _canon_str(tallos_total["variedad_canon"])

        tot_col = _pick_tallos_total_col(tallos_total)
        keep = [c for c in ["ciclo_id", "fecha", "bloque_base", "variedad_canon"] if c in tallos_total.columns] + [tot_col]
        tallos_total = tallos_total[keep].copy().rename(columns={tot_col: "tallos_total_ml2"})

        df = df.merge(tallos_total, on=["ciclo_id", "fecha", "bloque_base", "variedad_canon"], how="left")
    else:
        # fallback extremo: sumar baseline ML1
        df["tallos_total_ml2"] = df["tallos_pred_ml1_grado_dia"].groupby([df["ciclo_id"], df["fecha"]]).transform("sum")

    df["tallos_total_ml2"] = pd.to_numeric(df["tallos_total_ml2"], errors="coerce").fillna(0.0)

    # Real share (anclaje) por fecha+bloque+grado
    real = read_parquet(IN_REAL).copy()
    real["fecha"] = _to_date(real["fecha"])
    real = _resolve_block_base(real)
    real["grado"] = _canon_str(real["grado"])
    real["tallos_real"] = pd.to_numeric(real["tallos_real"], errors="coerce").fillna(0.0)

    real_g = (
        real.groupby(["fecha", "bloque_base", "grado"], as_index=False)
        .agg(tallos_real_grado=("tallos_real", "sum"))
    )
    real_t = (
        real_g.groupby(["fecha", "bloque_base"], as_index=False)
        .agg(tallos_real_dia=("tallos_real_grado", "sum"))
    )
    real_g = real_g.merge(real_t, on=["fecha", "bloque_base"], how="left")
    real_g["share_real"] = np.where(real_g["tallos_real_dia"] > 0, real_g["tallos_real_grado"] / real_g["tallos_real_dia"], np.nan)

    real_merge = real_g[
        ["fecha", "bloque_base", "grado", "tallos_real_grado", "tallos_real_dia", "share_real"]
    ].rename(columns={"tallos_real_dia": "tallos_real_dia_obs", "share_real": "share_real_obs"})
    df = df.merge(real_merge, on=["fecha", "bloque_base", "grado"], how="left")

    tallos_real_day_col = "tallos_real_dia_obs" if "tallos_real_dia_obs" in df.columns else "tallos_real_dia"
    df["has_real_share_day"] = (
        df["fecha"].le(as_of_date)
        & pd.to_numeric(df[tallos_real_day_col], errors="coerce").fillna(0.0).gt(0)
    )

    share_real_col = "share_real_obs" if "share_real_obs" in df.columns else "share_real"
    df["share_real"] = pd.to_numeric(df[share_real_col], errors="coerce")

    closed_use_real = df["is_closed_cycle"] & df["has_real_share_day"]
    active_use_real = df["is_active_cycle"] & df["has_real_share_day"]

    df["share_final"] = np.select(
        [
            closed_use_real,
            df["is_closed_cycle"] & (~df["has_real_share_day"]),
            active_use_real,
        ],
        [
            df[share_real_col].fillna(0.0),
            df["share_ml1"],  # cerrado sin real: no recalcular con ML2
            df[share_real_col].fillna(0.0),
        ],
        default=df["share_ml2_model"],  # activo sin real
    )
    df["share_source"] = np.select(
        [
            closed_use_real,
            df["is_closed_cycle"] & (~df["has_real_share_day"]),
            active_use_real,
        ],
        ["REAL_CLOSED", "ML1_CLOSED", "REAL_ACTIVE"],
        default="ML2_ACTIVE",
    )

    # Renormalizacion defensiva por (ciclo_id, fecha)
    denom_final = df.groupby(grp)["share_final"].transform("sum")
    df["share_final"] = np.where(denom_final > 0, df["share_final"] / denom_final, df["share_ml2_model"])

    df["tallos_final_grado_dia"] = df["tallos_total_ml2"] * df["share_final"]

    df["ml2_run_id"] = meta["run_id"]
    df["created_at"] = pd.Timestamp(datetime.now()).normalize()

    out_factor = df[[
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "share_ml1", "share_raw", "share_ml2_model", "share_real", "share_source", "share_final",
        "pred_log_error_share", "pred_ratio_share",
        "is_active_cycle", "is_closed_cycle",
        "ml2_run_id", "created_at"
    ]].copy()

    out_final = df[[
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "tallos_total_ml2", "share_final", "share_source", "tallos_final_grado_dia",
        "is_active_cycle", "is_closed_cycle",
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
