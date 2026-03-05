from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet
from models.ml2.ml2_nn_common import (
    ML2_CAT_COLS,
    ML2_NUM_COLS,
    ML2_TARGET_SPECS,
    canon_str,
    corr_from_real_and_ml1,
    valid_corr_mask,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
SILVER = DATA / "silver"

ML1_PRED_DIR = DATA / "gold" / "ml1_nn"
DEFAULT_OUT = DATA / "gold" / "ml2_nn" / "ds_ml2_nn_v1.parquet"
IN_CICLO = SILVER / "fact_ciclo_maestro.parquet"
RECENT_PESO_AR_COLS = [
    "ar_peso_tallo_real_lag1",
    "ar_peso_tallo_real_roll7",
    "ar_peso_tallo_real_roll14",
    "ar_ratio_peso_real_vs_base_lag1",
    "ar_ratio_peso_real_vs_base_roll7",
    "ar_ratio_peso_real_vs_base_roll14",
]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_ds_ml2_nn_v1")
    ap.add_argument("--ml1-input", default=None, help="Path to pred_ml1_multitask_nn_*.parquet")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD. Default: today-1 (local naive)")
    return ap.parse_args()


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _latest_ml1_pred() -> Path:
    files = []
    for p in ML1_PRED_DIR.glob("pred_ml1_multitask_nn_*.parquet"):
        name = p.name
        if "_post_" in name:
            continue
        files.append(p)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No ML1 multitask predictions found in {ML1_PRED_DIR}")
    return files[-1]


def _as_of_date(v: str | None) -> pd.Timestamp:
    if v:
        return pd.Timestamp(v).normalize()
    return pd.Timestamp.now().normalize() - pd.Timedelta(days=1)


def _load_cycle_status(as_of: pd.Timestamp) -> pd.DataFrame:
    if not IN_CICLO.exists():
        return pd.DataFrame(
            columns=[
                "ciclo_id",
                "estado_ciclo",
                "fecha_fin_cosecha_ciclo",
                "is_closed_cycle",
                "is_active_cycle",
            ]
        )

    c = read_parquet(IN_CICLO).copy()
    c.columns = [str(x).strip() for x in c.columns]
    if "ciclo_id" not in c.columns:
        return pd.DataFrame(
            columns=[
                "ciclo_id",
                "estado_ciclo",
                "fecha_fin_cosecha_ciclo",
                "is_closed_cycle",
                "is_active_cycle",
            ]
        )

    c["ciclo_id"] = c["ciclo_id"].astype("string")
    c["estado_ciclo"] = canon_str(c["estado"]) if "estado" in c.columns else "UNKNOWN"
    c["fecha_fin_cosecha_ciclo"] = _to_date(c["fecha_fin_cosecha"]) if "fecha_fin_cosecha" in c.columns else pd.NaT

    c = (
        c[["ciclo_id", "estado_ciclo", "fecha_fin_cosecha_ciclo"]]
        .drop_duplicates(subset=["ciclo_id"], keep="last")
        .copy()
    )

    c["is_closed_cycle"] = (
        c["estado_ciclo"].eq("CERRADO")
        | (c["fecha_fin_cosecha_ciclo"].notna() & (c["fecha_fin_cosecha_ciclo"] <= as_of))
    )
    c["is_active_cycle"] = ~c["is_closed_cycle"]
    return c


def main() -> None:
    args = _parse_args()
    created_at = pd.Timestamp(datetime.now(timezone.utc))
    as_of = _as_of_date(args.asof)

    in_path = Path(args.ml1_input) if args.ml1_input else _latest_ml1_pred()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    df = read_parquet(in_path).copy()
    df.columns = [str(x).strip() for x in df.columns]

    required = {"row_id", "fecha_evento", "ciclo_id", "stage", "row_source"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"ML1 prediction dataset missing columns: {sorted(miss)}")

    df["row_id"] = pd.to_numeric(df["row_id"], errors="coerce").astype("Int64")
    df["fecha_evento"] = _to_date(df["fecha_evento"])
    df["fecha_post"] = _to_date(df["fecha_post"]) if "fecha_post" in df.columns else pd.NaT
    df["ciclo_id"] = df["ciclo_id"].astype("string").fillna("UNKNOWN")
    df["as_of_date"] = as_of

    cyc = _load_cycle_status(as_of)
    df = df.merge(cyc, on="ciclo_id", how="left")
    df["estado_ciclo"] = df["estado_ciclo"].fillna("UNKNOWN")
    df["is_closed_cycle"] = df["is_closed_cycle"].fillna(False).astype(bool)
    df["is_active_cycle"] = df["is_active_cycle"].fillna(True).astype(bool)

    for c in ML2_CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = canon_str(df[c])

    for c in ML2_NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for spec in ML2_TARGET_SPECS:
        real_col = spec.original_target
        pred_col = spec.ml1_pred_col
        base_mask_col = f"mask_{real_col}"
        corr_col = spec.corr_target
        corr_mask_col = f"mask_{corr_col}"

        if real_col not in df.columns:
            df[real_col] = np.nan
        if pred_col not in df.columns:
            df[pred_col] = np.nan
        if base_mask_col not in df.columns:
            df[base_mask_col] = 0.0

        y_real = pd.to_numeric(df[real_col], errors="coerce").to_numpy(dtype=np.float32)
        y_ml1 = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=np.float32)
        base_w = pd.to_numeric(df[base_mask_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float32)

        valid = valid_corr_mask(y_real=y_real, y_ml1=y_ml1, mode=spec.mode)
        corr = np.full(len(df), np.nan, dtype=np.float32)
        if valid.any():
            corr[valid] = corr_from_real_and_ml1(
                y_real=y_real[valid],
                y_ml1=y_ml1[valid],
                mode=spec.mode,
            ).astype(np.float32)
        corr = np.clip(corr, spec.corr_clip[0], spec.corr_clip[1]).astype(np.float32)

        mask = np.where(valid, base_w, 0.0).astype(np.float32)

        df[corr_col] = corr
        df[corr_mask_col] = mask

    corr_mask_cols = [f"mask_{s.corr_target}" for s in ML2_TARGET_SPECS]
    df["has_any_ml2_target"] = (df[corr_mask_cols].sum(axis=1) > 0.0).astype("int8")
    df["created_at"] = created_at
    df["ml1_input_file"] = in_path.name

    base_cols = [
        "row_id",
        "stage",
        "row_source",
        "fecha_evento",
        "fecha_post",
        "as_of_date",
        "ciclo_id",
        "bloque_base",
        "variedad_canon",
        "area",
        "tipo_sp",
        "grado",
        "destino",
        "estado_ciclo",
        "is_active_cycle",
        "is_closed_cycle",
        "fecha_fin_cosecha_ciclo",
    ]
    orig_cols = []
    for s in ML2_TARGET_SPECS:
        orig_cols.extend([s.original_target, f"mask_{s.original_target}", s.ml1_pred_col])
    corr_cols = []
    for s in ML2_TARGET_SPECS:
        corr_cols.extend([s.corr_target, f"mask_{s.corr_target}"])

    ordered = list(
        dict.fromkeys(
            base_cols
            + ML2_CAT_COLS
            + ML2_NUM_COLS
            + orig_cols
            + corr_cols
            + ["has_any_ml2_target", "ml1_input_file", "created_at"]
        )
    )
    ordered = [c for c in ordered if c in df.columns]
    ds = df[ordered].copy()
    ds = ds.sort_values(["fecha_evento", "stage", "row_source", "row_id"], kind="mergesort").reset_index(drop=True)

    out_path = Path(args.output)
    write_parquet(ds, out_path)

    print(f"[OK] Wrote dataset: {out_path}")
    print(f"     source={in_path.name}")
    print(f"     rows={len(ds):,} rows_with_any_ml2_target={int(ds['has_any_ml2_target'].sum()):,}")
    print(f"     as_of_date={as_of.date()} active_rows={int(ds['is_active_cycle'].sum()):,}")
    print("     ml2 target coverage:")
    for s in ML2_TARGET_SPECS:
        mcol = f"mask_{s.corr_target}"
        cov = float((ds[mcol] > 0).mean()) if len(ds) else 0.0
        print(f"       - {s.corr_target}: {cov:.2%}")
    print("     recent peso AR coverage:")
    for c in RECENT_PESO_AR_COLS:
        cov = float(ds[c].notna().mean()) if (c in ds.columns and len(ds)) else 0.0
        print(f"       - {c}: {cov:.2%}")


if __name__ == "__main__":
    main()
