from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _prep_dim_var(dim_var: pd.DataFrame) -> pd.DataFrame:
    need = {"variedad_raw", "variedad_canon"}
    miss = need - set(dim_var.columns)
    if miss:
        raise ValueError(f"dim_variedad_canon.parquet sin columnas: {sorted(miss)}")

    dv = dim_var.copy()
    dv["variedad_raw_norm"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    return dv[["variedad_raw_norm", "variedad_canon"]].drop_duplicates()


def _attach_variedad_canon(df: pd.DataFrame, dv: pd.DataFrame, col_raw: str) -> pd.DataFrame:
    out = df.copy()
    if col_raw not in out.columns:
        raise ValueError(f"DF no tiene columna '{col_raw}' para canonizar variedad.")
    out[col_raw] = _canon_str(out[col_raw])
    out = out.merge(dv, left_on=col_raw, right_on="variedad_raw_norm", how="left")
    out["variedad_canon"] = out["variedad_canon"].fillna(out[col_raw])
    return out.drop(columns=["variedad_raw_norm"], errors="ignore")


def _renormalize_share(df: pd.DataFrame, share_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[share_col] = pd.to_numeric(out[share_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    s = out.groupby(group_cols, dropna=False)[share_col].transform("sum")
    out[share_col] = np.where(s > 0, out[share_col] / s, 0.0)
    return out


def _attach_baseline_nearest_ndias(
    feat: pd.DataFrame,
    base: pd.DataFrame,
    *,
    ndias_feat_col: str,
    ndias_base_col: str = "n_dias",
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = feat.copy()
    out["_nd"] = pd.to_numeric(out[ndias_feat_col], errors="coerce")

    b = base.copy()
    b["_nd"] = pd.to_numeric(b[ndias_base_col], errors="coerce")

    out["grado"] = _canon_int(out["grado"])
    b["grado"] = _canon_int(b["grado"])

    b = b.dropna(subset=["variedad_canon", "grado", "_nd"]).copy()
    b = b[["variedad_canon", "grado", "_nd", value_col]].copy()
    b = b.sort_values(["variedad_canon", "grado", "_nd"], kind="mergesort").reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for (var, g), sub in out.groupby(["variedad_canon", "grado"], dropna=False):
        sub = sub.copy()
        ref = b[(b["variedad_canon"] == var) & (b["grado"] == g)]
        if ref.empty:
            sub[out_col] = np.nan
            parts.append(sub.drop(columns=["_nd"], errors="ignore"))
            continue

        sub_null = sub[sub["_nd"].isna()].copy()
        sub_ok = sub[sub["_nd"].notna()].copy()

        sub_null[out_col] = np.nan

        if not sub_ok.empty:
            sub_ok = sub_ok.sort_values(["_nd"], kind="mergesort").reset_index(drop=True)
            ref2 = ref[["_nd", value_col]].sort_values(["_nd"], kind="mergesort").reset_index(drop=True)

            m = pd.merge_asof(
                sub_ok,
                ref2.rename(columns={value_col: out_col}),
                on="_nd",
                direction="nearest",
                allow_exact_matches=True,
            )
            m = m.drop(columns=["_nd"], errors="ignore")
            parts.append(m)

        sub_null = sub_null.drop(columns=["_nd"], errors="ignore")
        parts.append(sub_null)

    out2 = pd.concat(parts, ignore_index=True) if parts else out.drop(columns=["_nd"], errors="ignore")
    return out2


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    # -------------------------
    # Inputs (BASE = UNIVERSO ML1)
    # -------------------------
    grid = read_parquet(Path("data/gold/universe_harvest_grid_ml1.parquet")).copy()
    df_prog = read_parquet(Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")).copy()
    df_clima = read_parquet(Path("data/silver/dim_clima_bloque_dia.parquet")).copy()
    df_term = read_parquet(Path("data/silver/dim_estado_termico_cultivo_bloque_fecha.parquet")).copy()

    df_real_cosecha_grado = read_parquet(Path("data/silver/fact_cosecha_real_grado_dia.parquet")).copy()
    df_real_peso = read_parquet(Path("data/silver/fact_peso_tallo_real_grado_dia.parquet")).copy()

    df_base_dist = read_parquet(Path("data/silver/dim_dist_grado_baseline.parquet")).copy()
    df_base_peso = read_parquet(Path("data/silver/dim_peso_tallo_baseline.parquet")).copy()

    df_maestro = read_parquet(Path("data/silver/fact_ciclo_maestro.parquet")).copy()
    dim_var = read_parquet(Path("data/silver/dim_variedad_canon.parquet")).copy()
    dv = _prep_dim_var(dim_var)

    # -------------------------
    # Canon universo
    # -------------------------
    need_u = {"ciclo_id", "fecha", "bloque_base", "variedad_canon"}
    miss_u = need_u - set(grid.columns)
    if miss_u:
        raise ValueError(f"universe_harvest_grid_ml1 sin columnas: {sorted(miss_u)}")

    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_int(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    if "stage" in grid.columns:
        grid = grid[_canon_str(grid["stage"]).eq("HARVEST")].copy()

    # Aliases esperados por apply_dist_grado.py
    if "day_in_harvest_pred" in grid.columns and "day_in_harvest" not in grid.columns:
        grid["day_in_harvest"] = pd.to_numeric(grid["day_in_harvest_pred"], errors="coerce").astype("Int64")
    if "rel_pos_pred" in grid.columns and "rel_pos" not in grid.columns:
        grid["rel_pos"] = pd.to_numeric(grid["rel_pos_pred"], errors="coerce")
    if "n_harvest_days_pred" in grid.columns and "n_harvest_days" not in grid.columns:
        grid["n_harvest_days"] = pd.to_numeric(grid["n_harvest_days_pred"], errors="coerce").astype("Int64")
    if "harvest_start_pred" in grid.columns and "harvest_start" not in grid.columns:
        grid["harvest_start"] = _to_date(grid["harvest_start_pred"])
    if "harvest_end_pred" in grid.columns and "harvest_end_eff" not in grid.columns:
        grid["harvest_end_eff"] = _to_date(grid["harvest_end_pred"])

    base = grid.drop_duplicates(subset=["ciclo_id", "fecha", "bloque_base", "variedad_canon"]).copy()

    # -------------------------
    # Maestro meta
    # -------------------------
    df_maestro["ciclo_id"] = df_maestro["ciclo_id"].astype(str)
    if "bloque_base" in df_maestro.columns:
        df_maestro["bloque_base"] = _canon_int(df_maestro["bloque_base"])
    if "variedad_canon" not in df_maestro.columns and "variedad" in df_maestro.columns:
        df_maestro = _attach_variedad_canon(df_maestro, dv, "variedad")
    if "variedad_canon" in df_maestro.columns:
        df_maestro["variedad_canon"] = _canon_str(df_maestro["variedad_canon"])
    for c in ["tipo_sp", "area", "estado"]:
        if c in df_maestro.columns:
            df_maestro[c] = _canon_str(df_maestro[c])

    m_take = [c for c in ["ciclo_id", "tipo_sp", "area", "estado"] if c in df_maestro.columns]
    m2 = df_maestro[m_take].drop_duplicates("ciclo_id") if m_take else pd.DataFrame({"ciclo_id": base["ciclo_id"].unique()})
    feat_day = base.merge(m2, on="ciclo_id", how="left")

    # -------------------------
    # Expand a GRADOS (desde baseline dist)
    # -------------------------
    need_bd = {"variedad", "grado", "n_dias", "pct_grado"}
    miss_bd = need_bd - set(df_base_dist.columns)
    if miss_bd:
        raise ValueError(f"dim_dist_grado_baseline sin columnas: {sorted(miss_bd)}")

    base_dist = _attach_variedad_canon(df_base_dist, dv, "variedad")
    base_dist["grado"] = _canon_int(base_dist["grado"])
    base_dist["n_dias"] = _canon_int(base_dist["n_dias"])
    base_dist = base_dist.rename(columns={"pct_grado": "share_grado_baseline"})
    base_dist = base_dist.dropna(subset=["variedad_canon", "grado", "n_dias"])

    # catálogo grados por variedad (si faltara una variedad, se quedará sin grados -> FATAL)
    vg = base_dist[["variedad_canon", "grado"]].drop_duplicates()

    feat = feat_day.merge(vg, on="variedad_canon", how="left")
    if feat["grado"].isna().any():
        bad_vars = feat.loc[feat["grado"].isna(), "variedad_canon"].value_counts().head(20)
        raise ValueError(f"[FATAL] No pude expandir grados para algunas variedades. Ejemplos:\n{bad_vars.to_string()}")

    feat["grado"] = _canon_int(feat["grado"])

    # -------------------------
    # Canon otros inputs
    # -------------------------
    for d in (df_prog, df_clima, df_term, df_real_cosecha_grado, df_real_peso):
        if "fecha" in d.columns:
            d["fecha"] = _to_date(d["fecha"])
    if "fecha_sp" in df_term.columns:
        df_term["fecha_sp"] = _to_date(df_term["fecha_sp"])

    # -------------------------
    # Progreso real (LEFT)
    # -------------------------
    prog = df_prog.copy()
    if "ciclo_id" in prog.columns:
        prog["ciclo_id"] = prog["ciclo_id"].astype(str)
    if "bloque_base" in prog.columns:
        prog["bloque_base"] = _canon_int(prog["bloque_base"])
    if "variedad_canon" not in prog.columns and "variedad" in prog.columns:
        prog = _attach_variedad_canon(prog, dv, "variedad")
    if "variedad_canon" in prog.columns:
        prog["variedad_canon"] = _canon_str(prog["variedad_canon"])
    else:
        prog["variedad_canon"] = "UNKNOWN"

    prog_key = [c for c in ["ciclo_id", "fecha", "bloque_base", "variedad_canon"] if c in prog.columns]
    if prog_key:
        prog2 = prog.drop_duplicates(subset=prog_key)
        feat = feat.merge(prog2, on=prog_key, how="left", suffixes=("", "_prog"))

    # -------------------------
    # Clima (LEFT por fecha+bloque)
    # -------------------------
    clima = df_clima.copy()
    if "bloque_base" in clima.columns:
        clima["bloque_base"] = _canon_int(clima["bloque_base"])
    clima2 = clima.drop_duplicates(subset=["fecha", "bloque_base"])
    feat = feat.merge(clima2, on=["fecha", "bloque_base"], how="left", suffixes=("", "_cl"))

    # -------------------------
    # Term (LEFT por ciclo+bloque+fecha)
    # -------------------------
    term = df_term.copy()
    if "bloque_base" in term.columns:
        term["bloque_base"] = _canon_int(term["bloque_base"])
    if "ciclo_id" in term.columns:
        term["ciclo_id"] = term["ciclo_id"].astype(str)
        term2 = term.drop_duplicates(subset=["ciclo_id", "bloque_base", "fecha"])
        feat = feat.merge(term2, on=["ciclo_id", "bloque_base", "fecha"], how="left", suffixes=("", "_term"))

    # -------------------------
    # Calendario
    # -------------------------
    feat["dow"] = feat["fecha"].dt.dayofweek
    feat["month"] = feat["fecha"].dt.month
    feat["weekofyear"] = feat["fecha"].dt.isocalendar().week.astype(int)

    # -------------------------
    # n_dias_cosecha (para baseline nearest)
    # -------------------------
    a = pd.to_numeric(feat["dia_rel_cosecha_real"], errors="coerce") if "dia_rel_cosecha_real" in feat.columns else pd.Series([np.nan] * len(feat))
    b = pd.to_numeric(feat["day_in_harvest"], errors="coerce") if "day_in_harvest" in feat.columns else pd.Series([np.nan] * len(feat))
    feat["n_dias_cosecha"] = a.combine_first(b)
    feat["n_dias_cosecha"] = pd.to_numeric(feat["n_dias_cosecha"], errors="coerce").round()
    feat.loc[feat["n_dias_cosecha"].isna(), "n_dias_cosecha"] = 1
    feat["n_dias_cosecha"] = feat["n_dias_cosecha"].astype("Int64")

    # -------------------------
    # Baseline dist por nearest n_dias + renorm por día
    # -------------------------
    nd_min = int(base_dist["n_dias"].min())
    nd_max = int(base_dist["n_dias"].max())
    feat["n_dias_cosecha"] = feat["n_dias_cosecha"].clip(lower=nd_min, upper=nd_max)

    feat = _attach_baseline_nearest_ndias(
        feat,
        base=base_dist,
        ndias_feat_col="n_dias_cosecha",
        ndias_base_col="n_dias",
        value_col="share_grado_baseline",
        out_col="share_grado_baseline",
    )

    grp_day = ["ciclo_id", "bloque_base", "variedad_canon", "fecha"]
    feat["share_grado_baseline"] = feat["share_grado_baseline"].fillna(0.0)
    feat = _renormalize_share(feat, "share_grado_baseline", grp_day)

    # -------------------------
    # Baseline peso por nearest n_dias (si existe)
    # -------------------------
    need_bp = {"variedad", "grado", "n_dias", "peso_tallo_mediana_g"}
    miss_bp = need_bp - set(df_base_peso.columns)
    if miss_bp:
        raise ValueError(f"dim_peso_tallo_baseline sin columnas: {sorted(miss_bp)}")

    base_peso = _attach_variedad_canon(df_base_peso, dv, "variedad")
    base_peso["grado"] = _canon_int(base_peso["grado"])
    base_peso["n_dias"] = _canon_int(base_peso["n_dias"])
    base_peso = base_peso.rename(columns={"peso_tallo_mediana_g": "peso_tallo_baseline_g"})
    base_peso = base_peso.dropna(subset=["variedad_canon", "grado", "n_dias"])

    feat = _attach_baseline_nearest_ndias(
        feat,
        base=base_peso.rename(columns={"peso_tallo_baseline_g": "val"}),
        ndias_feat_col="n_dias_cosecha",
        ndias_base_col="n_dias",
        value_col="val",
        out_col="peso_tallo_baseline_g",
    )

    # -------------------------
    # Targets reales (LEFT, no corta futuro)
    # -------------------------
    real_c = df_real_cosecha_grado.copy()
    real_c["fecha"] = _to_date(real_c["fecha"])
    if "bloque_padre" in real_c.columns and "bloque_base" not in real_c.columns:
        real_c = real_c.rename(columns={"bloque_padre": "bloque_base"})
    if "bloque_base" in real_c.columns:
        real_c["bloque_base"] = _canon_int(real_c["bloque_base"])
    if "variedad_canon" not in real_c.columns and "variedad" in real_c.columns:
        real_c = _attach_variedad_canon(real_c, dv, "variedad")
    if "grado" in real_c.columns:
        real_c["grado"] = _canon_int(real_c["grado"])

    if "tallos_real" in real_c.columns:
        rc = (
            real_c.groupby(["fecha", "bloque_base", "variedad_canon", "grado"], as_index=False)
            .agg(tallos_real_grado=("tallos_real", "sum"))
        )
        rc_tot = (
            rc.groupby(["fecha", "bloque_base", "variedad_canon"], as_index=False)
            .agg(tallos_real_total=("tallos_real_grado", "sum"))
        )
        rc = rc.merge(rc_tot, on=["fecha", "bloque_base", "variedad_canon"], how="left")
        rc["share_grado_real"] = np.where(
            rc["tallos_real_total"].fillna(0) > 0,
            rc["tallos_real_grado"] / rc["tallos_real_total"],
            np.nan,
        )
        feat = feat.merge(
            rc[["fecha", "bloque_base", "variedad_canon", "grado", "tallos_real_grado", "tallos_real_total", "share_grado_real"]],
            on=["fecha", "bloque_base", "variedad_canon", "grado"],
            how="left",
        )

    # -------------------------
    # Residuales (listos ML2)
    # -------------------------
    if "share_grado_real" in feat.columns:
        feat["resid_share_grado"] = feat["share_grado_real"] - feat["share_grado_baseline"]

    feat["created_at"] = created_at

    # -------------------------
    # Output + checks
    # -------------------------
    out_path = Path("data/features/features_cosecha_bloque_fecha.parquet")
    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"]
    feat = feat.sort_values(key).reset_index(drop=True)

    if feat.duplicated(subset=key).any():
        raise ValueError("[FATAL] Duplicados en features_cosecha_bloque_fecha por key completa (día+grado).")

    write_parquet(feat, out_path)

    fmin = pd.to_datetime(feat["fecha"].min()).date() if len(feat) else None
    fmax = pd.to_datetime(feat["fecha"].max()).date() if len(feat) else None
    print(f"OK -> {out_path} | rows={len(feat):,} | fecha_min={fmin} fecha_max={fmax}")

    sums = feat.groupby(["ciclo_id", "bloque_base", "variedad_canon", "fecha"])["share_grado_baseline"].sum()
    if len(sums):
        print(f"[CHECK] baseline share sum min/max: {float(sums.min()):.6f} / {float(sums.max()):.6f}")

    cov_nd = float(feat["n_dias_cosecha"].notna().mean()) if len(feat) else float("nan")
    print(f"[CHECK] n_dias_cosecha notna coverage: {cov_nd:.4f}")


if __name__ == "__main__":
    main()
