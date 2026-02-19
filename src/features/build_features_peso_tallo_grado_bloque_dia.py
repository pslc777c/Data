from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_UNIVERSE = Path("data/gold/universe_harvest_grid_ml1.parquet")
IN_DIST_GRADO = Path("data/gold/pred_dist_grado_ml1.parquet")
IN_FEATS_COSECHA = Path("data/features/features_cosecha_bloque_fecha.parquet")

OUT_PATH = Path("data/features/features_peso_tallo_grado_bloque_dia.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")

def _ensure_calendar(df: pd.DataFrame) -> pd.DataFrame:
    if "dow" not in df.columns:
        df["dow"] = df["fecha"].dt.dayofweek.astype("Int64")
    if "month" not in df.columns:
        df["month"] = df["fecha"].dt.month.astype("Int64")
    if "weekofyear" not in df.columns:
        df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype("Int64")
    return df

def _detect_dist_cols(dist: pd.DataFrame) -> tuple[str, str]:
    grado_cands = [c for c in ["grado", "grade"] if c in dist.columns]
    if not grado_cands:
        grado_cands = [c for c in dist.columns if "grad" in c.lower()]
    if not grado_cands:
        raise ValueError(f"No pude detectar columna de grado en pred_dist_grado_ml1. Cols={list(dist.columns)}")
    col_grado = grado_cands[0]

    share_cands = [c for c in ["share_grado_ml1", "share_grado", "share", "pct_grado_ml1", "pct_grado"] if c in dist.columns]
    if not share_cands:
        share_cands = [c for c in dist.columns if ("share" in c.lower()) or ("pct" in c.lower())]
    if not share_cands:
        raise ValueError(f"No pude detectar columna share/pct en pred_dist_grado_ml1. Cols={list(dist.columns)}")
    col_share = share_cands[0]

    return col_grado, col_share

def _print_dup_sample(df: pd.DataFrame, keys: list[str], name: str, n: int = 10) -> None:
    dup_mask = df.duplicated(subset=keys, keep=False)
    dup_n = int(dup_mask.sum())
    if dup_n == 0:
        print(f"[OK] {name}: sin duplicados por {keys}")
        return
    print(f"[WARN] {name}: duplicados por {keys} -> rows_dup={dup_n:,} (sobre {len(df):,})")
    print(df.loc[dup_mask, keys].head(n).to_string(index=False))


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    # -------------------------
    # Universe (fecha,bloque,variedad)
    # -------------------------
    uni = read_parquet(IN_UNIVERSE).copy()
    uni.columns = [str(c).strip() for c in uni.columns]
    _require(uni, ["fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")

    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    uni_keys = ["fecha", "bloque_base", "variedad_canon"]

    # Diagnóstico + dedupe duro (si hay duplicados, colapsamos)
    _print_dup_sample(uni, uni_keys, "universe_harvest_grid_ml1")
    if uni.duplicated(subset=uni_keys).any():
        # Elegimos first en columnas no numéricas y sum/mean en numéricas relevantes si existen
        agg = {}
        for c in uni.columns:
            if c in uni_keys:
                continue
            if c in ("area", "tipo_sp", "estado", "stage", "ml1_version"):
                agg[c] = "first"
            elif pd.api.types.is_numeric_dtype(uni[c]):
                # para universe, usualmente tallos_proy se suma si hubiera duplicado accidental,
                # pero si el duplicado es bug, sum puede inflar. Mejor mean para estabilidad.
                agg[c] = "mean"
            else:
                agg[c] = "first"

        uni = uni.groupby(uni_keys, dropna=False, as_index=False).agg(agg)
        print(f"[FIX] universe deduplicado -> rows={len(uni):,}")

    # -------------------------
    # Dist grado (bloque,variedad,grado)
    # -------------------------
    dist = read_parquet(IN_DIST_GRADO).copy()
    dist.columns = [str(c).strip() for c in dist.columns]
    _require(dist, ["bloque_base", "variedad_canon"], "pred_dist_grado_ml1")

    dist["bloque_base"] = _canon_int(dist["bloque_base"])
    dist["variedad_canon"] = _canon_str(dist["variedad_canon"])

    col_grado, col_share = _detect_dist_cols(dist)
    dist[col_grado] = _canon_int(dist[col_grado])
    dist[col_share] = pd.to_numeric(dist[col_share], errors="coerce")

    dist = dist.rename(columns={col_grado: "grado", col_share: "share_grado_ml1"})
    dist_keys = ["bloque_base", "variedad_canon", "grado"]

    _print_dup_sample(dist, dist_keys, "pred_dist_grado_ml1")
    if dist.duplicated(subset=dist_keys).any():
        # Si hay varias filas por mismo grado, promediamos share (y luego renormalizamos)
        dist = dist.groupby(dist_keys, dropna=False, as_index=False).agg(
            share_grado_ml1=("share_grado_ml1", "mean")
        )
        print(f"[FIX] dist deduplicado -> rows={len(dist):,}")

    # Renormaliza share por (bloque,variedad)
    sum_share = dist.groupby(["bloque_base", "variedad_canon"], dropna=False)["share_grado_ml1"].transform("sum")
    dist["share_grado_ml1"] = np.where(sum_share.fillna(0) > 0, dist["share_grado_ml1"] / sum_share, dist["share_grado_ml1"])

    # -------------------------
    # Expand: universe x dist
    # -------------------------
    grid = uni.merge(dist, on=["bloque_base", "variedad_canon"], how="left", validate="m:m")

    miss_dist = float(grid["grado"].isna().mean())
    if miss_dist > 0:
        raise ValueError(
            f"Falta distribución de grado para parte del universo. miss_rate={miss_dist:.4f}. "
            "Revisa coverage de pred_dist_grado_ml1 por bloque_base+variedad_canon."
        )

    grid["grado"] = _canon_int(grid["grado"])

    keys = ["fecha", "bloque_base", "variedad_canon", "grado"]

    # Diagnóstico: grid debería ser único por keys
    _print_dup_sample(grid, keys, "grid (universe x dist)")
    # Si aquí hay duplicados, casi seguro era universe o dist; pero ya los deduplicamos.
    # Aun así, colapsamos defensivo.
    if grid.duplicated(subset=keys).any():
        agg = {c: "first" for c in grid.columns if c not in keys}
        # share si se repite, promediamos
        if "share_grado_ml1" in grid.columns:
            agg["share_grado_ml1"] = "mean"
        grid = grid.groupby(keys, dropna=False, as_index=False).agg(agg)
        print(f"[FIX] grid deduplicado -> rows={len(grid):,}")

    # -------------------------
    # Enrichment desde features_cosecha_bloque_fecha
    # -------------------------
    feats = read_parquet(IN_FEATS_COSECHA).copy()
    feats.columns = [str(c).strip() for c in feats.columns]
    _require(feats, ["fecha", "bloque_base", "grado", "variedad_canon", "peso_tallo_baseline_g"], "features_cosecha_bloque_fecha")

    feats["fecha"] = _to_date(feats["fecha"])
    feats["bloque_base"] = _canon_int(feats["bloque_base"])
    feats["grado"] = _canon_int(feats["grado"])
    feats["variedad_canon"] = _canon_str(feats["variedad_canon"])

    maybe_cols = [
        "tipo_sp",
        "area",
        "peso_tallo_baseline_g",
        "peso_tallo_real_g",
        "pct_avance_real",
        "dia_rel_cosecha_real",
        "gdc_acum_real",
        "rainfall_mm_dia",
        "horas_lluvia",
        "en_lluvia_dia",
        "temp_avg_dia",
        "solar_energy_j_m2_dia",
        "wind_speed_avg_dia",
        "wind_run_dia",
        "gdc_dia",
        "dias_desde_sp",
        "gdc_acum_desde_sp",
        "dow",
        "month",
        "weekofyear",
    ]
    take = [c for c in maybe_cols if c in feats.columns]
    feats_take = feats[keys + take].copy()

    _print_dup_sample(feats_take, keys, "features_cosecha_bloque_fecha (subset)")
    if feats_take.duplicated(subset=keys).any():
        agg = {}
        for c in take:
            if c in ("tipo_sp", "area"):
                agg[c] = "first"
            else:
                agg[c] = "mean"
        feats_take = feats_take.groupby(keys, dropna=False, as_index=False).agg(agg)
        print(f"[FIX] feats_take deduplicado -> rows={len(feats_take):,}")

    df = grid.merge(feats_take, on=keys, how="left", validate="1:1")

    # -------------------------
    # Ensure cols + targets
    # -------------------------
    for c in maybe_cols:
        if c not in df.columns:
            if c in ("tipo_sp", "area"):
                df[c] = "UNKNOWN"
            else:
                df[c] = np.nan

    df = _ensure_calendar(df)

    df["peso_tallo_baseline_g"] = pd.to_numeric(df["peso_tallo_baseline_g"], errors="coerce")
    df["peso_tallo_real_g"] = pd.to_numeric(df["peso_tallo_real_g"], errors="coerce")

    df["factor_peso_tallo"] = np.where(
        df["peso_tallo_baseline_g"].fillna(0) > 0,
        df["peso_tallo_real_g"] / df["peso_tallo_baseline_g"],
        np.nan,
    )
    df["factor_peso_tallo_clipped"] = pd.to_numeric(df["factor_peso_tallo"], errors="coerce").clip(lower=0.60, upper=1.60)
    df["delta_peso_tallo_g"] = df["peso_tallo_real_g"] - df["peso_tallo_baseline_g"]

    out_cols = [
        "fecha",
        "bloque_base",
        "variedad_canon",
        "grado",
        "tipo_sp",
        "area",
        "share_grado_ml1",
        "peso_tallo_baseline_g",
        "peso_tallo_real_g",
        "factor_peso_tallo",
        "factor_peso_tallo_clipped",
        "delta_peso_tallo_g",
        "pct_avance_real",
        "dia_rel_cosecha_real",
        "gdc_acum_real",
        "rainfall_mm_dia",
        "horas_lluvia",
        "en_lluvia_dia",
        "temp_avg_dia",
        "solar_energy_j_m2_dia",
        "wind_speed_avg_dia",
        "wind_run_dia",
        "gdc_dia",
        "dias_desde_sp",
        "gdc_acum_desde_sp",
        "dow",
        "month",
        "weekofyear",
    ]
    out = df[out_cols].copy()
    out["created_at"] = created_at

    # -------------------------
    # FINAL DEDUPE (último seguro)
    # -------------------------
    _print_dup_sample(out, keys, "OUT features_peso_tallo_grado_bloque_dia (pre-final)")
    if out.duplicated(subset=keys).any():
        # Colapsa determinístico: numéricos mean, categóricos first
        agg = {}
        for c in out.columns:
            if c in keys:
                continue
            if c in ("tipo_sp", "area"):
                agg[c] = "first"
            elif pd.api.types.is_numeric_dtype(out[c]):
                agg[c] = "mean"
            else:
                agg[c] = "first"

        out = out.groupby(keys, dropna=False, as_index=False).agg(agg)
        out["created_at"] = created_at  # re-set
        print(f"[FIX] OUT deduplicado por keys -> rows={len(out):,}")

    out = out.sort_values(["bloque_base", "variedad_canon", "fecha", "grado"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, OUT_PATH)

    cov_real = float(out["peso_tallo_real_g"].notna().mean())
    print(f"OK -> {OUT_PATH} | rows={len(out):,}")
    print(f"[COVERAGE] peso_tallo_real_g notna: {cov_real:.4f}")
    print(f"[HORIZON] min_fecha={out['fecha'].min()} max_fecha={out['fecha'].max()}")


if __name__ == "__main__":
    main()
