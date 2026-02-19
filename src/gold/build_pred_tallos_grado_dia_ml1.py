from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


IN_CURVA = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
IN_DIST_ML1 = Path("data/gold/pred_dist_grado_ml1.parquet")
OUT_PATH = Path("data/gold/pred_tallos_grado_dia_ml1.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    curva = read_parquet(IN_CURVA).copy()
    dist = read_parquet(IN_DIST_ML1).copy()

    # -------------------------
    # Normalización llaves/fechas
    # -------------------------
    curva["fecha"] = _to_date(curva["fecha"])
    curva["bloque_base"] = _canon_int(curva["bloque_base"])
    curva["variedad_canon"] = curva["variedad_canon"].astype(str)

    need_curva = {"fecha", "bloque_base", "variedad_canon", "tallos_pred_baseline_dia"}
    miss = need_curva - set(curva.columns)
    if miss:
        raise ValueError(f"features_curva_cosecha_bloque_dia sin columnas: {sorted(miss)}")

    # ⚠️ Universo operativo: SOLO donde existe baseline diario (curva)
    curva = curva[curva["tallos_pred_baseline_dia"].notna()].copy()
    curva = curva.drop_duplicates(subset=["fecha", "bloque_base", "variedad_canon"])

    # Dist (por grado)
    dist["fecha"] = _to_date(dist["fecha"])
    dist["bloque_base"] = _canon_int(dist["bloque_base"])
    dist["grado"] = _canon_int(dist["grado"])
    dist["variedad_canon"] = dist["variedad_canon"].astype(str)

    need_dist = {"ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1", "share_grado_baseline"}
    miss2 = need_dist - set(dist.columns)
    if miss2:
        raise ValueError(f"pred_dist_grado_ml1 sin columnas: {sorted(miss2)}")

    dist = dist.drop_duplicates(subset=["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"])

    # -------------------------
    # 1) Determinar ciclo_id por llave diaria de curva
    # -------------------------
    # dist trae ciclo_id; curva puede traerlo o no
    # Creamos un map (fecha,bloque,variedad)->ciclo_id usando dist (first)
    map_ciclo = (
        dist.groupby(["fecha", "bloque_base", "variedad_canon"], as_index=False)
            .agg(ciclo_id=("ciclo_id", "first"))
    )

    curva2 = curva.merge(
        map_ciclo,
        on=["fecha", "bloque_base", "variedad_canon"],
        how="left",
    )

    # Si hay días sin ciclo_id, no abortamos, pero los marcamos (igual se puede operar sin ciclo_id)
    if "ciclo_id" in curva2.columns:
        if curva2["ciclo_id"].isna().any():
            miss_c = float(curva2["ciclo_id"].isna().mean())
            print(f"[WARN] curva sin ciclo_id en {miss_c:.4f} de filas. Se mantiene, pero revisa si esperas ciclo_id completo.")
    else:
        print("[WARN] curva no tiene ciclo_id tras el mapeo. Se generará salida sin ciclo_id (operable igual).")


    # -------------------------
    # 2) Expandir curva a grados usando dist como universo de grados por variedad
    # -------------------------
    # Universo de grados por variedad_canon (desde dist)
    vg = dist[["variedad_canon", "grado"]].drop_duplicates()

    # Expandir curva (día/bloque/variedad) a (día/bloque/variedad/grado)
    base = curva2.merge(vg, on="variedad_canon", how="left")

    # -------------------------
    # 3) Traer shares (baseline y ml1) desde dist
    # -------------------------
    key_cols = ["fecha", "bloque_base", "variedad_canon", "grado"]

    # Para no perder ciclo_id, lo traemos a nivel key+grado desde dist.
    dist_take = dist[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_baseline", "share_grado_ml1"]].copy()

    out = base.merge(
        dist_take,
        on=["fecha", "bloque_base", "variedad_canon", "grado"],
        how="left",
        suffixes=("", "_d"),
    )

    # Si ciclo_id en out quedó NaN y venía en curva2, lo completamos
    if "ciclo_id_x" in out.columns and "ciclo_id_y" in out.columns:
        out["ciclo_id"] = out["ciclo_id_x"].combine_first(out["ciclo_id_y"])
        out = out.drop(columns=["ciclo_id_x", "ciclo_id_y"])
    elif "ciclo_id" not in out.columns and "ciclo_id_d" in out.columns:
        out = out.rename(columns={"ciclo_id_d": "ciclo_id"})

    # -------------------------
    # 4) Fallback + renormalización de shares por día/bloque/variedad
    # -------------------------
    out["share_grado_baseline"] = pd.to_numeric(out["share_grado_baseline"], errors="coerce")
    out["share_grado_ml1"] = pd.to_numeric(out["share_grado_ml1"], errors="coerce")

    # Si falta ML1 -> usar baseline
    out["share_grado_ml1_eff"] = out["share_grado_ml1"].where(out["share_grado_ml1"].notna(), out["share_grado_baseline"])

    # Si baseline falta (raro) -> 0
    out["share_grado_baseline_eff"] = out["share_grado_baseline"].fillna(0)
    out["share_grado_ml1_eff"] = out["share_grado_ml1_eff"].fillna(0)

    # Clip >=0
    out["share_grado_baseline_eff"] = out["share_grado_baseline_eff"].clip(lower=0)
    out["share_grado_ml1_eff"] = out["share_grado_ml1_eff"].clip(lower=0)

    grp = ["fecha", "bloque_base", "variedad_canon"]
    s_b = out.groupby(grp, dropna=False)["share_grado_baseline_eff"].transform("sum")
    s_m = out.groupby(grp, dropna=False)["share_grado_ml1_eff"].transform("sum")
    eps = 1e-12

    out["share_grado_baseline_eff"] = np.where(s_b > eps, out["share_grado_baseline_eff"] / s_b, np.nan)
    out["share_grado_ml1_eff"] = np.where(s_m > eps, out["share_grado_ml1_eff"] / s_m, np.nan)

    # -------------------------
    # 5) Tallos por grado día
    # -------------------------
    out["tallos_pred_baseline_dia"] = pd.to_numeric(out["tallos_pred_baseline_dia"], errors="coerce")

    out["tallos_pred_baseline_grado_dia"] = (out["tallos_pred_baseline_dia"] * out["share_grado_baseline_eff"]).clip(lower=0)
    out["tallos_pred_ml1_grado_dia"] = (out["tallos_pred_baseline_dia"] * out["share_grado_ml1_eff"]).clip(lower=0)

    out["created_at"] = created_at

    # -------------------------
    # Output
    # -------------------------
    cols = [
        "ciclo_id" if "ciclo_id" in out.columns else None,
        "fecha",
        "bloque_base",
        "variedad_canon",
        "grado",
        "tallos_pred_baseline_dia",
        "share_grado_baseline",   # original (puede venir NaN si no matcheó dist)
        "share_grado_ml1",        # original (puede venir NaN si no matcheó dist)
        "tallos_pred_baseline_grado_dia",
        "tallos_pred_ml1_grado_dia",
        "ml1_version" if "ml1_version" in out.columns else None,
        "created_at",
    ]
    cols = [c for c in cols if c is not None and c in out.columns]

    out = out[cols].sort_values(["bloque_base", "variedad_canon", "fecha", "grado"]).reset_index(drop=True)

    write_parquet(out, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(out):,}")

    cov = float(out["tallos_pred_baseline_dia"].notna().mean())
    print(f"coverage tallos_pred_baseline_dia notna: {cov:.4f}")

    # sanity: sumas por día ≈ total
    grp2 = ["fecha", "bloque_base", "variedad_canon"]
    s_ml1 = out.groupby(grp2, dropna=False)["tallos_pred_ml1_grado_dia"].sum()
    t = out.groupby(grp2, dropna=False)["tallos_pred_baseline_dia"].first()
    diff = (s_ml1 - t).abs()
    print(f"sanity | abs(sum(grado)-total) p50/p95/max: {diff.quantile(0.50):.6f} / {diff.quantile(0.95):.6f} / {diff.max():.6f}")


if __name__ == "__main__":
    main()
