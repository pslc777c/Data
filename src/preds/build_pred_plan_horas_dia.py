from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _norm_date(s: pd.Series) -> pd.Series:
    return _to_dt(s).dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _norm_station_from_area(area: pd.Series) -> pd.Series:
    a = area.astype(str).str.upper().str.strip()
    is_a4 = a.isin(["A-4", "A4", "SJP", "SAN JUAN"])
    return pd.Series(np.where(is_a4, "A4", "MAIN"), index=area.index)


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = list(df.columns)
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        cl = cand.lower()
        if cl in cols_l:
            return cols_l[cl]
    if required:
        raise ValueError(f"No encontré columna. Candidatos={candidates}. Disponibles={cols}")
    return None


def _collapse_capacidad(cap: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura grano único por (fecha, station, area_trabajada, variedad).
    Suma tallos/horas/personas; y promedia ponderado UPH por tallos_proy.
    """
    cap = cap.copy()
    keys = ["fecha", "station", "area_trabajada", "variedad"]

    dup = int(cap.duplicated(subset=keys).sum())
    if dup == 0:
        return cap

    _warn(f"capacidad_cosecha_dia: {dup} duplicados por {keys}. Se colapsará a grano único.")

    def wavg_group(g: pd.DataFrame, xcol: str, wcol: str) -> float:
        x = pd.to_numeric(g[xcol], errors="coerce")
        w = pd.to_numeric(g[wcol], errors="coerce").fillna(0.0)
        m = x.notna() & w.gt(0)
        if not m.any():
            return np.nan
        return float((x[m] * w[m]).sum() / w[m].sum())

    sum_cols = [c for c in ["tallos_proy", "horas_req", "personas_req"] if c in cap.columns]
    first_cols = [c for c in ["uph_base_zcsp_equiv", "n_base", "n_horas"] if c in cap.columns]
    agg = {c: "sum" for c in sum_cols}
    agg.update({c: "first" for c in first_cols})

    cap_g = cap.groupby(keys, dropna=False).agg(agg).reset_index()

    if "tallos_proy" in cap.columns:
        if "uph_eff" in cap.columns:
            cap_g["uph_eff"] = cap.groupby(keys, dropna=False).apply(lambda g: wavg_group(g, "uph_eff", "tallos_proy")).to_numpy()
        if "factor_uph_dia" in cap.columns:
            cap_g["factor_uph_dia"] = cap.groupby(keys, dropna=False).apply(lambda g: wavg_group(g, "factor_uph_dia", "tallos_proy")).to_numpy()
    else:
        if "uph_eff" in cap.columns:
            cap_g["uph_eff"] = cap.groupby(keys, dropna=False)["uph_eff"].median().to_numpy()
        if "factor_uph_dia" in cap.columns:
            cap_g["factor_uph_dia"] = cap.groupby(keys, dropna=False)["factor_uph_dia"].median().to_numpy()

    if "created_at" in cap.columns:
        cap_g["created_at"] = cap.groupby(keys, dropna=False)["created_at"].max().to_numpy()

    dup2 = int(cap_g.duplicated(subset=keys).sum())
    if dup2 > 0:
        ex = cap_g[cap_g.duplicated(subset=keys, keep=False)].head(20)
        raise ValueError(f"Aún hay {dup2} duplicados en capacidad después de colapsar. Ejemplos:\n{ex.to_string(index=False)}")

    return cap_g


def main() -> None:
    cfg = load_settings()
    preds_dir = Path(cfg["paths"]["preds"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    path_cap_cosecha = preds_dir / "capacidad_cosecha_dia.parquet"
    path_pos = preds_dir / "pred_horas_poscosecha_dia.parquet"
    path_peso_grado = preds_dir / "pred_peso_grado.parquet"

    for p in [path_cap_cosecha, path_pos, path_peso_grado]:
        if not p.exists():
            raise FileNotFoundError(f"No existe input requerido: {p}")

    cap = pd.read_parquet(path_cap_cosecha)
    pos = pd.read_parquet(path_pos)
    pg = pd.read_parquet(path_peso_grado)

    cap.columns = [str(c).strip() for c in cap.columns]
    pos.columns = [str(c).strip() for c in pos.columns]
    pg.columns = [str(c).strip() for c in pg.columns]

    # -------------------------
    # COSECHA (capacidad)
    # -------------------------
    _require_cols(cap, ["fecha", "station", "variedad", "tallos_proy", "uph_eff", "horas_req", "personas_req"], "capacidad_cosecha_dia")

    col_area_cap = _pick_col(cap, ["area_trabajada", "area", "Area"], required=False)
    if col_area_cap is None:
        raise ValueError("capacidad_cosecha_dia no trae area_trabajada/area; ajusta build_capacidad_cosecha_dia.py para incluirla.")

    cap = cap.copy()
    cap["fecha"] = _norm_date(cap["fecha"])
    cap["station"] = cap["station"].astype(str).str.upper().str.strip()
    cap["variedad"] = cap["variedad"].astype(str).str.upper().str.strip()
    cap["area_trabajada"] = cap[col_area_cap].astype(str).str.upper().str.strip()
    cap = cap[cap["fecha"].notna()].copy()

    # asegurar llaves sin nulos críticos
    cap["area_trabajada"] = cap["area_trabajada"].replace({"": "ALL"}).fillna("ALL")
    cap["variedad"] = cap["variedad"].replace({"": "ALL"}).fillna("ALL")

    cap = _collapse_capacidad(cap)
    plan_cosecha = cap.copy()
    plan_cosecha["tipo_flujo"] = "COSECHA"

    # -------------------------
    # POSCOSECHA (diario)
    # -------------------------
    _require_cols(pos, ["fecha_post", "kg_total", "horas_req_total", "personas_req_total"], "pred_horas_poscosecha_dia")

    pos = pos.copy()
    pos["fecha"] = _norm_date(pos["fecha_post"])
    pos = pos[pos["fecha"].notna()].copy()

    keep_pos = [
        "fecha",
        "kg_total", "cajas_total",
        "W_Blanco", "W_Arcoiris", "W_Tinturado",
        "horas_req_blanco", "horas_req_arcoiris", "horas_req_tinturado",
        "horas_req_total",
        "personas_req_blanco", "personas_req_arcoiris", "personas_req_tinturado",
        "personas_req_total",
    ]
    keep_pos = [c for c in keep_pos if c in pos.columns]
    pos2 = pos[keep_pos].copy()
    pos2["station"] = "PLANTA"
    pos2["area_trabajada"] = "ALL"
    pos2["variedad"] = "ALL"
    pos2["tipo_flujo"] = "POSCOSECHA"

    # -------------------------
    # pred_peso_grado (resumen por llave de cosecha)
    # -------------------------
    _require_cols(pg, ["fecha", "tallos_pred_grado", "peso_pred_g"], "pred_peso_grado")

    pg = pg.copy()
    pg["fecha"] = _norm_date(pg["fecha"])
    pg = pg[pg["fecha"].notna()].copy()

    # variedad: preferir variedad_std, si no existe usar variedad
    col_var = _pick_col(pg, ["variedad_std", "variedad"], required=True)
    pg["variedad_std"] = (
        pg[col_var].astype(str).str.upper().str.strip()
          .replace({"XLENCE": "XL", "CLOUD": "CLO"})
    )

    # station/area_trabajada: preferir station/area_trabajada si existen
    if "station" in pg.columns:
        pg["station"] = pg["station"].astype(str).str.upper().str.strip().replace({"A-4": "A4", "SJP": "A4"})
    elif "area" in pg.columns:
        pg["station"] = _norm_station_from_area(pg["area"])
    else:
        pg["station"] = "MAIN"

    if "area_trabajada" in pg.columns:
        pg["area_trabajada"] = pg["area_trabajada"].astype(str).str.upper().str.strip()
    elif "area" in pg.columns:
        pg["area_trabajada"] = pg["area"].astype(str).str.upper().str.strip()
    else:
        pg["area_trabajada"] = "ALL"

    pg["area_trabajada"] = pg["area_trabajada"].replace({"": "ALL"}).fillna("ALL")

    pg["tallos_pred_grado"] = _to_num(pg["tallos_pred_grado"]).fillna(0.0).astype(float)
    pg["peso_pred_g"] = _to_num(pg["peso_pred_g"]).astype(float)

    # promedio ponderado robusto: sum(peso * tallos) / sum(tallos)
    pg["peso_x_tallos"] = pg["peso_pred_g"] * pg["tallos_pred_grado"]

    pg_agg = (
        pg.groupby(["fecha", "station", "area_trabajada", "variedad_std"], dropna=False)
          .agg(
              tallos_pred_total=("tallos_pred_grado", "sum"),
              peso_x_tallos_sum=("peso_x_tallos", "sum"),
              n_grados=("tallos_pred_grado", "size"),
          )
          .reset_index()
          .rename(columns={"variedad_std": "variedad"})
    )

    pg_agg["peso_pred_prom_g"] = np.where(
        pg_agg["tallos_pred_total"] > 0,
        pg_agg["peso_x_tallos_sum"] / pg_agg["tallos_pred_total"],
        np.nan,
    )
    pg_agg = pg_agg.drop(columns=["peso_x_tallos_sum"])

    dup_pg = int(pg_agg.duplicated(subset=["fecha", "station", "area_trabajada", "variedad"]).sum())
    if dup_pg > 0:
        ex = pg_agg[pg_agg.duplicated(subset=["fecha", "station", "area_trabajada", "variedad"], keep=False)].head(20)
        raise ValueError(f"pred_peso_grado agregado NO es único por llave (dup={dup_pg}). Ejemplos:\n{ex.to_string(index=False)}")

    # merge cosecha + resumen peso
    plan_cosecha = plan_cosecha.merge(
        pg_agg,
        on=["fecha", "station", "area_trabajada", "variedad"],
        how="left",
        validate="one_to_one",
    )

    # -------------------------
    # Unión final
    # -------------------------
    common_cols = [
        "fecha", "station", "area_trabajada", "variedad", "tipo_flujo",
        "tallos_pred_total", "peso_pred_prom_g", "n_grados",
        "tallos_proy", "uph_base_zcsp_equiv", "factor_uph_dia", "uph_eff",
        "horas_req", "personas_req", "n_base", "n_horas",
        "kg_total", "cajas_total",
        "W_Blanco", "W_Arcoiris", "W_Tinturado",
        "horas_req_blanco", "horas_req_arcoiris", "horas_req_tinturado", "horas_req_total",
        "personas_req_blanco", "personas_req_arcoiris", "personas_req_tinturado", "personas_req_total",
    ]

    for c in common_cols:
        if c not in plan_cosecha.columns:
            plan_cosecha[c] = np.nan
        if c not in pos2.columns:
            pos2[c] = np.nan

    out = pd.concat([plan_cosecha[common_cols], pos2[common_cols]], ignore_index=True)
    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # -------------------------
    # Validaciones / warnings
    # -------------------------
    dup_out = int(out.duplicated(subset=["fecha", "station", "area_trabajada", "variedad", "tipo_flujo"]).sum())
    if dup_out > 0:
        _warn(f"Hay {dup_out} duplicados por (fecha,station,area,variedad,tipo_flujo). Revisa upstream.")

    miss_pg = int((out["tipo_flujo"].eq("COSECHA") & out["tallos_pred_total"].isna()).sum())
    if miss_pg > 0:
        _warn(f"COSECHA: {miss_pg} filas sin match en pred_peso_grado (tallos_pred_total NaN).")

    bad_a4_clo = out[(out["tipo_flujo"] == "COSECHA") & (out["station"] == "A4") & (out["variedad"] == "CLO")]
    if len(bad_a4_clo) > 0:
        _warn(f"[DATA QUALITY] Detecté {len(bad_a4_clo)} filas COSECHA con A4+CLO (no esperado).")

    out_path = preds_dir / "pred_plan_horas_dia.parquet"
    write_parquet(out, out_path)

    _info(f"OK: pred_plan_horas_dia={len(out)} filas -> {out_path}")
    _info(f"Rango fechas: {out['fecha'].min()} -> {out['fecha'].max()}")
    _info(f"Flujos: {out['tipo_flujo'].value_counts(dropna=False).to_dict()}")
    _info(f"Stations: {out['station'].value_counts(dropna=False).to_dict()}")
    _info(
        "Áreas (COSECHA, top): "
        f"{out[out['tipo_flujo']=='COSECHA']['area_trabajada'].value_counts().head(10).to_dict()}"
    )


if __name__ == "__main__":
    main()
