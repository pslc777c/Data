# src/audit/audit_ml1_curva_clipping_impact.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =============================================================================
# Paths
# =============================================================================
IN_CURVA = Path("data/gold/pred_factor_curva_ml1.parquet")
IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")  # para rel_pos_pred/day_in_harvest_pred/n_harvest_days_pred

OUT_REPORT = Path("data/audit/audit_ml1_curva_clipping_report.parquet")
OUT_RELPOS = Path("data/audit/audit_ml1_curva_clipping_by_relpos.parquet")
OUT_CYCLES = Path("data/audit/audit_ml1_curva_clipping_by_cycle.parquet")


# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _coalesce(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    base = df[out_col] if out_col in df.columns else pd.Series([pd.NA] * len(df), index=df.index)
    for c in candidates:
        if c in df.columns:
            base = base.where(base.notna(), df[c])
    df[out_col] = base


def _safe_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    x = s.copy()
    if x.dtype.kind in "if":
        return x.fillna(0).astype(float).ne(0)
    return x.astype(str).str.strip().str.lower().isin(["1", "true", "t", "yes", "y"])


def _bin_relpos(rel: pd.Series, nbins: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Devuelve:
      - bin_label (string)   e.g. "[0.00,0.05]"
      - bin_lo (float)
      - bin_hi (float)
    """
    r = pd.to_numeric(rel, errors="coerce").clip(lower=0.0, upper=1.0)

    bins = np.linspace(0.0, 1.0, nbins + 1)
    cut = pd.cut(r, bins=bins, include_lowest=True)

    # cut es Interval; lo convertimos a string para que pyarrow lo soporte
    lbl = cut.astype(str)

    # extraer bounds numéricos (para análisis)
    lo = cut.map(lambda x: float(x.left) if pd.notna(x) else np.nan)
    hi = cut.map(lambda x: float(x.right) if pd.notna(x) else np.nan)

    return lbl, lo, hi


def _q(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float(x.quantile(q)) if x.notna().any() else float("nan")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    curva = read_parquet(IN_CURVA).copy()
    _require(
        curva,
        ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "factor_curva_ml1", "factor_curva_ml1_raw"],
        "pred_factor_curva_ml1",
    )

    # canon keys
    curva["ciclo_id"] = curva["ciclo_id"].astype(str)
    curva["fecha"] = _to_date(curva["fecha"])
    curva["bloque_base"] = _canon_int(curva["bloque_base"])
    curva["variedad_canon"] = _canon_str(curva["variedad_canon"])

    # merge grid para rel_pos_pred y day_in_harvest_pred (si existen)
    if IN_GRID.exists():
        grid = read_parquet(IN_GRID).copy()
        _require(grid, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")

        grid["ciclo_id"] = grid["ciclo_id"].astype(str)
        grid["fecha"] = _to_date(grid["fecha"])
        grid["bloque_base"] = _canon_int(grid["bloque_base"])
        grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

        key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
        take = key + [c for c in [
            "rel_pos_pred", "day_in_harvest_pred", "n_harvest_days_pred",
            "rel_pos", "day_in_harvest", "n_harvest_days"
        ] if c in grid.columns]
        grid2 = grid[take].drop_duplicates(subset=key)

        curva = curva.merge(grid2, on=key, how="left", suffixes=("", "_grid"))

        _coalesce(curva, "rel_pos_eff", ["rel_pos_eff", "rel_pos_pred", "rel_pos"])
        _coalesce(curva, "day_in_harvest_eff", ["day_in_harvest_eff", "day_in_harvest_pred", "day_in_harvest"])
        _coalesce(curva, "n_harvest_days_eff", ["n_harvest_days_eff", "n_harvest_days_pred", "n_harvest_days"])
    else:
        curva["rel_pos_eff"] = pd.NA
        curva["day_in_harvest_eff"] = pd.NA
        curva["n_harvest_days_eff"] = pd.NA

    raw = pd.to_numeric(curva["factor_curva_ml1_raw"], errors="coerce")
    fac = pd.to_numeric(curva["factor_curva_ml1"], errors="coerce")

    eps = 1e-12
    was_clipped = raw.notna() & fac.notna() & (raw - fac).abs().gt(eps)

    was_capped_pre = _safe_bool(curva["was_capped_pre"]) if "was_capped_pre" in curva.columns else pd.Series([False] * len(curva))
    was_capped_post = _safe_bool(curva["was_capped_post"]) if "was_capped_post" in curva.columns else pd.Series([False] * len(curva))

    # hits a extremos típicos (no asumimos, solo detectamos)
    min_hits = fac.notna() & fac.le(0.2000000001)
    max_hits = fac.notna() & fac.ge(4.9999999999)

    # rel_pos bins (STRING + bounds)
    lbl, lo, hi = _bin_relpos(curva["rel_pos_eff"], nbins=20)
    curva["rel_pos_bin"] = lbl
    curva["rel_pos_bin_lo"] = lo
    curva["rel_pos_bin_hi"] = hi

    # =========================
    # Report global
    # =========================
    n = len(curva)
    report = {
        "created_at": created_at,
        "rows": int(n),
        "raw_notna_rate": float(raw.notna().mean()) if n else float("nan"),
        "factor_notna_rate": float(fac.notna().mean()) if n else float("nan"),
        "clipped_rate_raw_vs_factor": float(was_clipped.mean()) if n else float("nan"),
        "was_capped_pre_rate": float(was_capped_pre.mean()) if n else float("nan"),
        "was_capped_post_rate": float(was_capped_post.mean()) if n else float("nan"),
        "min_hit_rate_factor": float(min_hits.mean()) if n else float("nan"),
        "max_hit_rate_factor": float(max_hits.mean()) if n else float("nan"),
        "raw_p01": _q(raw, 0.01),
        "raw_p50": _q(raw, 0.50),
        "raw_p99": _q(raw, 0.99),
        "factor_p01": _q(fac, 0.01),
        "factor_p50": _q(fac, 0.50),
        "factor_p99": _q(fac, 0.99),
    }
    df_report = pd.DataFrame([report])

    # =========================
    # Relpos aggregation (group by STRING label)
    # =========================
    by_rel = (
        curva.assign(
            _raw=raw,
            _fac=fac,
            _clipped=was_clipped,
            _cap_pre=was_capped_pre,
            _cap_post=was_capped_post,
            _min_hit=min_hits,
            _max_hit=max_hits,
        )
        .groupby(["rel_pos_bin", "rel_pos_bin_lo", "rel_pos_bin_hi"], dropna=False)
        .agg(
            rows=("ciclo_id", "size"),
            clipped_rate=("_clipped", "mean"),
            cap_pre_rate=("_cap_pre", "mean"),
            cap_post_rate=("_cap_post", "mean"),
            min_hit_rate=("_min_hit", "mean"),
            max_hit_rate=("_max_hit", "mean"),
            raw_p50=("_raw", "median"),
            raw_p95=("_raw", lambda s: _q(s, 0.95)),
            factor_p50=("_fac", "median"),
            factor_p95=("_fac", lambda s: _q(s, 0.95)),
        )
        .reset_index()
        .sort_values(["rel_pos_bin_lo"], ascending=[True])
    )

    # =========================
    # Cycle aggregation (top offenders)
    # =========================
    by_cyc = (
        curva.assign(
            _raw=raw,
            _fac=fac,
            _clipped=was_clipped,
            _cap_pre=was_capped_pre,
            _cap_post=was_capped_post,
            _min_hit=min_hits,
            _max_hit=max_hits,
        )
        .groupby(["ciclo_id"], dropna=False)
        .agg(
            days=("fecha", "count"),
            clipped_days=("_clipped", "sum"),
            clipped_rate=("_clipped", "mean"),
            cap_pre_days=("_cap_pre", "sum"),
            cap_post_days=("_cap_post", "sum"),
            min_hit_days=("_min_hit", "sum"),
            max_hit_days=("_max_hit", "sum"),
            raw_max=("_raw", "max"),
            raw_p95=("_raw", lambda s: _q(s, 0.95)),
            factor_max=("_fac", "max"),
            factor_p95=("_fac", lambda s: _q(s, 0.95)),
        )
        .reset_index()
        .sort_values(["clipped_days", "clipped_rate", "raw_max"], ascending=[False, False, False])
    )

    # =========================
    # Optional: impact share_pred_in vs share_curva_ml1 (si existe)
    # =========================
    if "share_pred_in" in curva.columns and "share_curva_ml1" in curva.columns:
        sp = pd.to_numeric(curva["share_pred_in"], errors="coerce")
        sm = pd.to_numeric(curva["share_curva_ml1"], errors="coerce")

        tmp = curva[["ciclo_id"]].copy()
        tmp["_abs"] = (sp - sm).abs()
        l1 = tmp.groupby("ciclo_id", dropna=False)["_abs"].sum().rename("l1_share_pred_vs_final").reset_index()

        by_cyc = by_cyc.merge(l1, on="ciclo_id", how="left")
        df_report["l1_share_pred_vs_final_median"] = float(l1["l1_share_pred_vs_final"].median()) if len(l1) else np.nan
        df_report["l1_share_pred_vs_final_p90"] = float(l1["l1_share_pred_vs_final"].quantile(0.90)) if len(l1) else np.nan

    # =========================
    # Persist
    # =========================
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(df_report, OUT_REPORT)
    write_parquet(by_rel, OUT_RELPOS)
    write_parquet(by_cyc, OUT_CYCLES)

    # =========================
    # Print summary
    # =========================
    print("\n=== CLIPPING REPORT (global) ===")
    print(df_report.to_string(index=False))

    print("\n=== CLIPPING BY REL_POS (top 12 bins by clipped_rate) ===")
    show = by_rel.sort_values(["clipped_rate", "rows"], ascending=[False, False]).head(12)
    print(show.to_string(index=False))

    print("\n=== TOP 20 CYCLES (most clipped days) ===")
    print(by_cyc.head(20).to_string(index=False))

    print(f"\nOK -> {OUT_REPORT}")
    print(f"OK -> {OUT_RELPOS}")
    print(f"OK -> {OUT_CYCLES}")


if __name__ == "__main__":
    main()
