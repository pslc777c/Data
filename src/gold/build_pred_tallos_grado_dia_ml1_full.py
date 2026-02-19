from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =============================================================================
# Paths
# =============================================================================
IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")
IN_MAESTRO = Path("data/silver/fact_ciclo_maestro.parquet")
IN_CURVA = Path("data/gold/pred_factor_curva_ml1.parquet")
IN_DIST = Path("data/gold/pred_dist_grado_ml1.parquet")
DIM_VAR = Path("data/silver/dim_variedad_canon.parquet")

OUT_PATH = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")


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


def _prep_dim_var(dim_var: pd.DataFrame) -> pd.DataFrame:
    need = {"variedad_raw", "variedad_canon"}
    miss = need - set(dim_var.columns)
    if miss:
        raise ValueError(f"dim_variedad_canon.parquet sin columnas: {sorted(miss)}")

    dv = dim_var.copy()
    dv["variedad_raw_norm"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon_norm"] = _canon_str(dv["variedad_canon"])
    return dv[["variedad_raw_norm", "variedad_canon_norm"]].drop_duplicates()


def _attach_variedad_canon_always(df: pd.DataFrame, dv: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "variedad" in out.columns:
        raw = out["variedad"]
    elif "variedad_canon" in out.columns:
        raw = out["variedad_canon"]
    else:
        raise ValueError("DF no tiene 'variedad' ni 'variedad_canon' para canonizar.")

    out["__var_raw_norm"] = _canon_str(raw)

    out = out.merge(
        dv,
        left_on="__var_raw_norm",
        right_on="variedad_raw_norm",
        how="left",
    )

    # fallback mínimo (debería coincidir con dim)
    fallback = {
        "XLENCE": "XL",
        "XL": "XL",
        "CLOUD": "CLO",
        "CLO": "CLO",
    }

    canon = out["variedad_canon_norm"]
    canon = canon.fillna(out["__var_raw_norm"].map(fallback))
    canon = canon.fillna(out["__var_raw_norm"])

    out["variedad_canon"] = canon

    return out.drop(
        columns=["variedad_raw_norm", "variedad_canon_norm", "__var_raw_norm"],
        errors="ignore",
    )


def _renormalize_positive(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out[col] = out[col].clip(lower=0.0)
    s = out.groupby(group_cols, dropna=False)[col].transform("sum")
    out[col] = np.where(s > 0, out[col] / s, 0.0)
    return out


def _anti_join(left: pd.DataFrame, right: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    rkeys = right[on].drop_duplicates()
    tmp = left[on].merge(rkeys, on=on, how="left", indicator=True)
    return left.loc[tmp["_merge"].eq("left_only").to_numpy()].copy()


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    grid = read_parquet(IN_GRID).copy()
    maestro = read_parquet(IN_MAESTRO).copy()
    curva = read_parquet(IN_CURVA).copy()
    dist = read_parquet(IN_DIST).copy()
    dim_var = read_parquet(DIM_VAR).copy()

    dv = _prep_dim_var(dim_var)

    # -------------------------
    # Requisitos mínimos
    # -------------------------
    _require(grid, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")
    _require(maestro, ["ciclo_id", "tallos_proy"], "fact_ciclo_maestro")
    _require(curva, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "factor_curva_ml1"], "pred_factor_curva_ml1")
    _require(dist, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_baseline", "share_grado_ml1"], "pred_dist_grado_ml1")

    # -------------------------
    # Canon
    # -------------------------
    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_int(grid["bloque_base"])
    grid = _attach_variedad_canon_always(grid, dv)

    maestro["ciclo_id"] = maestro["ciclo_id"].astype(str)
    maestro["tallos_proy"] = pd.to_numeric(maestro["tallos_proy"], errors="coerce").fillna(0.0)

    curva["ciclo_id"] = curva["ciclo_id"].astype(str)
    curva["fecha"] = _to_date(curva["fecha"])
    curva["bloque_base"] = _canon_int(curva["bloque_base"])
    curva = _attach_variedad_canon_always(curva, dv)

    dist["ciclo_id"] = dist["ciclo_id"].astype(str)
    dist["fecha"] = _to_date(dist["fecha"])
    dist["bloque_base"] = _canon_int(dist["bloque_base"])
    dist["grado"] = _canon_int(dist["grado"])
    dist = _attach_variedad_canon_always(dist, dv)

    # -------------------------
    # Definir universo base (grid) + tallos_proy (driver absoluto)
    # -------------------------
    key_dia = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]

    base = grid[key_dia].drop_duplicates().copy()
    if base.duplicated(subset=key_dia).any():
        raise ValueError("[FATAL] El grid tiene duplicados por key_dia; eso no debe ocurrir.")

    # Join tallos_proy por ciclo
    m2 = maestro[["ciclo_id", "tallos_proy"]].drop_duplicates("ciclo_id")
    base = base.merge(m2, on="ciclo_id", how="left")

    if base["tallos_proy"].isna().any():
        n = int(base["tallos_proy"].isna().sum())
        raise ValueError(f"[FATAL] {n:,} filas del grid quedaron sin tallos_proy (maestro incompleto).")

    # -------------------------
    # Curva: unir y renormalizar por ciclo -> w(d)
    # -------------------------
    curva_take = curva[key_dia + [c for c in ["factor_curva_ml1", "factor_curva_ml1_raw", "ml1_version"] if c in curva.columns]].drop_duplicates(subset=key_dia)

    dia = base.merge(curva_take, on=key_dia, how="left")

    # Diagnóstico de cobertura curva
    miss_curva = dia["factor_curva_ml1"].isna()
    if miss_curva.any():
        nmiss = int(miss_curva.sum())
        anti = _anti_join(base, curva_take, key_dia)
        print(f"[FATAL] Curva no cubre el universo: faltan {nmiss:,} filas.")
        print("Ejemplos faltantes (top 20):")
        print(anti[key_dia].head(20).to_string(index=False))
        raise ValueError("Cobertura de curva incompleta. Arregla llaves/universo, no uses fallback.")

    dia["factor_curva_ml1"] = pd.to_numeric(dia["factor_curva_ml1"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # w(d) por ciclo (usamos key por ciclo_id, porque el universo ya es por ciclo)
    # Si quieres más estricto: agrupar por (ciclo_id) solamente.
    s = dia.groupby(["ciclo_id"], dropna=False)["factor_curva_ml1"].transform("sum")
    zero = s <= 0
    if zero.any():
        nbad = int(zero.sum())
        bad_cycles = dia.loc[zero, "ciclo_id"].drop_duplicates().head(20).tolist()
        raise ValueError(f"[FATAL] Curva suma 0 en {nbad:,} filas (ciclos con curva inválida). Ej ciclos: {bad_cycles}")

    dia["w_d"] = dia["factor_curva_ml1"] / s

    # Tallos diarios ML1 (mass-balance garantizado)
    dia["tallos_pred_ml1_dia"] = dia["tallos_proy"].astype(float) * dia["w_d"].astype(float)

    # Baseline diario “determinístico razonable” (uniforme) solo para comparación
    cnt = dia.groupby(["ciclo_id"], dropna=False)["fecha"].transform("count").astype(float)
    dia["tallos_pred_baseline_dia"] = np.where(cnt > 0, dia["tallos_proy"].astype(float) / cnt, 0.0)

    # Check mass-balance por ciclo
    cyc = dia.groupby("ciclo_id", dropna=False).agg(
        proy=("tallos_proy", "max"),
        ml1_sum=("tallos_pred_ml1_dia", "sum"),
        base_sum=("tallos_pred_baseline_dia", "sum"),
    ).reset_index()
    cyc["abs_diff_ml1"] = (cyc["proy"] - cyc["ml1_sum"]).abs()
    max_abs = float(cyc["abs_diff_ml1"].max()) if len(cyc) else float("nan")
    print(f"[CHECK] ciclo mass-balance ML1 vs tallos_proy | max abs diff: {max_abs:.12f}")
    if max_abs > 1e-6:
        raise ValueError("[FATAL] Mass-balance ML1 no cierra (no debería pasar con w(d)).")

    # -------------------------
    # Dist grado: asegurar shares y renormalizar por día
    # -------------------------
    dist["share_grado_baseline"] = pd.to_numeric(dist["share_grado_baseline"], errors="coerce")
    dist["share_grado_ml1"] = pd.to_numeric(dist["share_grado_ml1"], errors="coerce")
    dist["share_grado_ml1"] = dist["share_grado_ml1"].where(dist["share_grado_ml1"].notna(), dist["share_grado_baseline"])

    grp_dist = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    dist = _renormalize_positive(dist, "share_grado_baseline", grp_dist)
    dist = _renormalize_positive(dist, "share_grado_ml1", grp_dist)

    # Diagnóstico de cobertura dist
    dist_keys = dist[grp_dist].drop_duplicates()
    dia_keys = dia[grp_dist].drop_duplicates()
    miss_dist = _anti_join(dia_keys, dist_keys, grp_dist)
    if len(miss_dist) > 0:
        print(f"[FATAL] Dist grado no cubre el universo: faltan {len(miss_dist):,} grupos día.")
        print("Ejemplos faltantes (top 20):")
        print(miss_dist.head(20).to_string(index=False))
        raise ValueError("Cobertura de dist_grado incompleta. Arregla llaves/universo.")

    # -------------------------
    # Expand a grado
    # -------------------------
    out = dist.merge(
        dia[grp_dist + ["tallos_proy", "tallos_pred_baseline_dia", "tallos_pred_ml1_dia", "factor_curva_ml1"] + [c for c in ["factor_curva_ml1_raw", "ml1_version"] if c in dia.columns]],
        on=grp_dist,
        how="left",
    )

    if out["tallos_pred_ml1_dia"].isna().any():
        raise ValueError("[FATAL] merge dist x dia dejó NaNs en tallos_pred_ml1_dia. No debe pasar.")

    out["tallos_pred_baseline_grado_dia"] = out["tallos_pred_baseline_dia"] * out["share_grado_baseline"]
    out["tallos_pred_ml1_grado_dia"] = out["tallos_pred_ml1_dia"] * out["share_grado_ml1"]

    # Check por día: sum grado = tallos_dia
    g = grp_dist
    sum_ml1 = out.groupby(g, dropna=False)["tallos_pred_ml1_grado_dia"].sum().rename("sum_grado_ml1").reset_index()
    chk = dia[g + ["tallos_pred_ml1_dia"]].merge(sum_ml1, on=g, how="left")
    eps = 1e-6
    mismatch = (chk["sum_grado_ml1"] - chk["tallos_pred_ml1_dia"]).abs() > eps
    print(f"[CHECK] % grupos mismatch (sum_grado_ml1 vs dia): {float(mismatch.mean()):.4%}")

    out["created_at"] = created_at

    final_cols = [
        "ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado",
        "tallos_proy",
        "tallos_pred_baseline_dia",
        "tallos_pred_ml1_dia",
        "factor_curva_ml1",
        "factor_curva_ml1_raw",
        "ml1_version",
        "share_grado_baseline",
        "share_grado_ml1",
        "tallos_pred_baseline_grado_dia",
        "tallos_pred_ml1_grado_dia",
        "created_at",
    ]
    final_cols = [c for c in final_cols if c in out.columns]

    out = out[final_cols].sort_values(["ciclo_id", "bloque_base", "variedad_canon", "fecha", "grado"]).reset_index(drop=True)

    write_parquet(out, OUT_PATH)
    fmin = pd.to_datetime(out["fecha"].min()).date() if len(out) else None
    fmax = pd.to_datetime(out["fecha"].max()).date() if len(out) else None
    print(f"OK -> {OUT_PATH} | rows={len(out):,} | fecha_min={fmin} fecha_max={fmax}")


if __name__ == "__main__":
    main()
