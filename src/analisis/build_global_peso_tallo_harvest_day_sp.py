from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold" / "ml2_nn"
EVAL = DATA / "eval" / "ml2_nn"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_global_peso_tallo_harvest_day_sp")
    ap.add_argument("--ml1", default=str(GOLD / "view_pipeline_ml1_global.parquet"))
    ap.add_argument("--ml2-global", default=str(GOLD / "view_pipeline_ml2_global.parquet"))
    ap.add_argument("--ml2-operativo", default=str(GOLD / "view_pipeline_ml2_operativo_global.parquet"))
    ap.add_argument("--real", default=str(GOLD / "view_pipeline_real_global.parquet"))
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-pdf", default=None)
    ap.add_argument("--include-otros", action="store_true")
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _sp_group(v: str) -> str | None:
    s = str(v).upper().strip()
    if s == "S":
        return "S"
    if s in {"P1", "P2", "P3", "P4"}:
        return "PODAS_P1_P4"
    return "OTROS"


def _resolve_peso_real(df: pd.DataFrame) -> pd.Series:
    cand = [
        "peso_tallo_real_g",
        "peso_tallo_real_g_y",
        "peso_tallo_real_g_x",
    ]
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in cand:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            out = out.where(out.notna(), v)
    if "gramos_verde" in df.columns and "tallos_grado_dia" in df.columns:
        g = pd.to_numeric(df["gramos_verde"], errors="coerce")
        t = pd.to_numeric(df["tallos_grado_dia"], errors="coerce")
        fallback = g / t.replace(0, np.nan)
        out = out.where(out.notna(), fallback)
    return out


def _resolve_peso_ml(df: pd.DataFrame) -> pd.Series:
    cand = [
        "peso_tallo_estimado_g",
        "peso_tallo_real_g",
        "peso_tallo_real_g_y",
        "peso_tallo_real_g_x",
    ]
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in cand:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            out = out.where(out.notna(), v)
    if "gramos_verde" in df.columns and "tallos_grado_dia" in df.columns:
        g = pd.to_numeric(df["gramos_verde"], errors="coerce")
        t = pd.to_numeric(df["tallos_grado_dia"], errors="coerce")
        fallback = g / t.replace(0, np.nan)
        out = out.where(out.notna(), fallback)
    return out


def _build_layer(path: Path, layer: str, include_otros: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"View not found: {path}")
    df = read_parquet(path).copy()
    if len(df) == 0:
        return pd.DataFrame()

    if "stage" not in df.columns:
        raise ValueError(f"Missing 'stage' in {path}")
    if "tipo_sp" not in df.columns:
        raise ValueError(f"Missing 'tipo_sp' in {path}")
    if "day_in_harvest" not in df.columns:
        raise ValueError(f"Missing 'day_in_harvest' in {path}")
    if "tallos_grado_dia" not in df.columns:
        raise ValueError(f"Missing 'tallos_grado_dia' in {path}")

    df["stage"] = _canon_str(df["stage"])
    df["tipo_sp"] = _canon_str(df["tipo_sp"])
    df = df[df["stage"] == "HARVEST_GRADE"].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["sp_group"] = df["tipo_sp"].map(_sp_group)
    if not include_otros:
        df = df[df["sp_group"].isin(["S", "PODAS_P1_P4"])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["day_in_harvest"] = pd.to_numeric(df["day_in_harvest"], errors="coerce")
    df["tallos_grado_dia"] = pd.to_numeric(df["tallos_grado_dia"], errors="coerce")
    df = df[df["day_in_harvest"].notna() & (df["tallos_grado_dia"] > 0)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["day_in_harvest"] = np.rint(df["day_in_harvest"]).astype("int32")
    if layer == "REAL":
        df["peso_tallo_g"] = _resolve_peso_real(df)
    else:
        df["peso_tallo_g"] = _resolve_peso_ml(df)
    df = df[df["peso_tallo_g"].notna() & np.isfinite(df["peso_tallo_g"]) & (df["peso_tallo_g"] > 0)].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df["gramos_calc"] = df["tallos_grado_dia"] * df["peso_tallo_g"]

    out = (
        df.groupby(["sp_group", "day_in_harvest"], as_index=False)
        .agg(
            tallos_grado_dia_sum=("tallos_grado_dia", "sum"),
            gramos_calc_sum=("gramos_calc", "sum"),
            rows=("day_in_harvest", "size"),
        )
        .sort_values(["sp_group", "day_in_harvest"], kind="mergesort")
        .reset_index(drop=True)
    )
    out["peso_tallo_prom_g"] = out["gramos_calc_sum"] / out["tallos_grado_dia_sum"].replace(0, np.nan)
    out["layer"] = layer
    return out[
        [
            "layer",
            "sp_group",
            "day_in_harvest",
            "peso_tallo_prom_g",
            "tallos_grado_dia_sum",
            "gramos_calc_sum",
            "rows",
        ]
    ]


def main() -> None:
    args = _parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)

    layers = [
        ("ML1", Path(args.ml1)),
        ("ML2_GLOBAL", Path(args.ml2_global)),
        ("ML2_OPERATIVO", Path(args.ml2_operativo)),
        ("REAL", Path(args.real)),
    ]
    parts: list[pd.DataFrame] = []
    for layer, path in layers:
        p = _build_layer(path=path, layer=layer, include_otros=bool(args.include_otros))
        if len(p):
            parts.append(p)

    if not parts:
        raise ValueError("No rows generated for any layer.")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["layer", "sp_group", "day_in_harvest"], kind="mergesort").reset_index(drop=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_parquet = Path(args.out_parquet) if args.out_parquet else EVAL / f"global_peso_tallo_harvest_day_sp_{stamp}.parquet"
    out_csv = Path(args.out_csv) if args.out_csv else EVAL / f"global_peso_tallo_harvest_day_sp_{stamp}.csv"
    out_pdf = Path(args.out_pdf) if args.out_pdf else EVAL / f"global_peso_tallo_harvest_day_sp_{stamp}.pdf"

    write_parquet(out, out_parquet)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    # Latest aliases
    alias_pq = EVAL / "global_peso_tallo_harvest_day_sp_latest.parquet"
    alias_csv = EVAL / "global_peso_tallo_harvest_day_sp_latest.csv"
    alias_pdf = EVAL / "global_peso_tallo_harvest_day_sp_latest.pdf"
    write_parquet(out, alias_pq)
    out.to_csv(alias_csv, index=False, encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    groups = ["S", "PODAS_P1_P4"]
    for i, grp in enumerate(groups):
        ax = axes[i]
        sub = out[out["sp_group"] == grp].copy()
        for layer in ["REAL", "ML1", "ML2_GLOBAL", "ML2_OPERATIVO"]:
            d = sub[sub["layer"] == layer].sort_values("day_in_harvest")
            if len(d) == 0:
                continue
            ax.plot(d["day_in_harvest"], d["peso_tallo_prom_g"], label=layer, linewidth=1.8)
        ax.set_title(grp)
        ax.set_xlabel("day_in_harvest")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("peso_tallo_prom_g")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Peso tallo promedio por dia de cosecha (S vs PODAS P1..P4)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    fig.savefig(alias_pdf, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Global peso tallo by harvest day: {out_parquet}")
    print(f"[OK] CSV                         : {out_csv}")
    print(f"[OK] PDF                         : {out_pdf}")
    print(f"[OK] Latest alias parquet       : {alias_pq}")
    print(f"[OK] Latest alias csv           : {alias_csv}")
    print(f"[OK] Latest alias pdf           : {alias_pdf}")
    print(f"     rows={len(out):,} layers={out['layer'].nunique()} groups={out['sp_group'].nunique()}")


if __name__ == "__main__":
    main()
