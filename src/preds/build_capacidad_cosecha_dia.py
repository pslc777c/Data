# src/preds/build_capacidad_cosecha_dia.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from common.io import write_parquet


# -------------------------
# Config / helpers
# -------------------------
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


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")


def _norm_station_from_area(area: pd.Series) -> pd.Series:
    """
    Regla negocio:
      - A-4 / A4 / SJP / SAN JUAN => A4
      - todo lo demás => MAIN
    """
    a = area.astype(str).str.upper().str.strip()
    is_a4 = a.isin(["A-4", "A4", "SJP", "SAN JUAN"])
    return pd.Series(np.where(is_a4, "A4", "MAIN"), index=area.index)


def _ensure_area_from_inputs(pred: pd.DataFrame, pg: pd.DataFrame | None) -> pd.DataFrame:
    """
    Retorna pred con columna canónica area_trabajada.
    Estrategia:
      1) si pred trae area/area_trabajada => usarla
      2) si no trae, y existe pred_peso_grado con área => intentar mapear por (fecha,bloque,variedad)
      3) else => ALL (con warning)
    """
    pred = pred.copy()

    col_area = _pick_col(pred, ["area_trabajada", "area", "Area"], required=False)
    if col_area is not None:
        pred["area_trabajada"] = pred[col_area].astype(str).str.upper().str.strip()
        return pred

    # intentar mapear con pred_peso_grado si existe y si pred trae bloque
    col_bloque = _pick_col(pred, ["bloque", "Bloque", "bloque_norm", "bloque_padre"], required=False)
    if col_bloque is not None and pg is not None and len(pg):
        pred["_bloque_key"] = pred[col_bloque].astype(str).str.extract(r"^(\d+)", expand=False)
        pred["_bloque_key"] = pd.to_numeric(pred["_bloque_key"], errors="coerce").astype("Int64")

        pg = pg.copy()
        if "fecha" in pg.columns:
            pg["fecha"] = _norm_date(pg["fecha"])

        col_pg_area = _pick_col(pg, ["area", "area_trabajada", "Area"], required=False)
        col_pg_bloque = _pick_col(pg, ["bloque", "Bloque", "bloque_padre"], required=False)
        col_pg_var = _pick_col(pg, ["variedad_std", "variedad", "variedad_estandar"], required=False)

        if col_pg_area and col_pg_bloque and col_pg_var:
            pg["_bloque_key"] = pg[col_pg_bloque].astype(str).str.extract(r"^(\d+)", expand=False)
            pg["_bloque_key"] = pd.to_numeric(pg["_bloque_key"], errors="coerce").astype("Int64")

            pg["area_trabajada"] = pg[col_pg_area].astype(str).str.upper().str.strip()
            pg["variedad"] = (
                pg[col_pg_var].astype(str).str.upper().str.strip()
                  .replace({"XLENCE": "XL", "CLOUD": "CLO"})
            )

            # mapping por (fecha, bloque, variedad) -> área (modo)
            m = (
                pg.dropna(subset=["fecha", "_bloque_key", "area_trabajada", "variedad"])
                  .groupby(["fecha", "_bloque_key", "variedad"], dropna=False)["area_trabajada"]
                  .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
                  .reset_index()
            )

            pred = pred.merge(
                m,
                left_on=["fecha", "_bloque_key", "variedad"],
                right_on=["fecha", "_bloque_key", "variedad"],
                how="left",
            )

            miss = int(pred["area_trabajada"].isna().sum())
            if miss > 0:
                _warn(f"No pude mapear área para {miss} filas (se asigna 'ALL').")
                pred["area_trabajada"] = pred["area_trabajada"].fillna("ALL")

            pred = pred.drop(columns=["_bloque_key"])
            return pred

    _warn("pred_tallos_cosecha_dia no trae área y no pude mapearla: se asigna area_trabajada='ALL'.")
    pred["area_trabajada"] = "ALL"
    return pred


def _validate_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = int(df.duplicated(subset=keys).sum())
    if dup > 0:
        # muestra ejemplos
        ex = df.loc[df.duplicated(subset=keys, keep=False), keys].head(20)
        raise ValueError(
            f"{name}: llaves no únicas (dup={dup}) para keys={keys}. Ejemplos:\n{ex.to_string(index=False)}"
        )


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg["paths"]["preds"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    path_pred_tallos = preds_dir / "pred_tallos_cosecha_dia.parquet"
    path_factor_uph = silver_dir / "dim_factor_uph_cosecha_clima.parquet"
    path_base_uph = silver_dir / "dim_baseline_uph_cosecha.parquet"
    path_weather = silver_dir / "weather_hour_wide.parquet"
    path_pg = preds_dir / "pred_peso_grado.parquet"

    for p in [path_pred_tallos, path_factor_uph, path_base_uph, path_weather]:
        if not p.exists():
            raise FileNotFoundError(f"No existe input requerido: {p}")

    pred = pd.read_parquet(path_pred_tallos)
    f = pd.read_parquet(path_factor_uph)
    b = pd.read_parquet(path_base_uph)
    w = pd.read_parquet(path_weather)
    pg = pd.read_parquet(path_pg) if path_pg.exists() else None

    pred.columns = [str(c).strip() for c in pred.columns]
    f.columns = [str(c).strip() for c in f.columns]
    b.columns = [str(c).strip() for c in b.columns]
    w.columns = [str(c).strip() for c in w.columns]
    if pg is not None:
        pg.columns = [str(c).strip() for c in pg.columns]

    # =========================================================
    # 1) pred_tallos: normalizar + asegurar área + SELLAR GRANO
    # =========================================================
    col_fecha = _pick_col(pred, ["fecha", "Fecha", "dt", "date"])
    pred["fecha"] = _norm_date(pred[col_fecha])

    col_station = _pick_col(pred, ["station", "estacion"], required=False)
    if col_station is None:
        col_area_tmp = _pick_col(pred, ["area_trabajada", "area", "Area"], required=False)
        if col_area_tmp is not None:
            pred["station"] = _norm_station_from_area(pred[col_area_tmp])
        else:
            raise ValueError("pred_tallos_cosecha_dia: falta 'station' y no hay 'area' para inferirlo.")
    else:
        pred["station"] = pred[col_station].astype(str).str.upper().str.strip()

    col_var = _pick_col(pred, ["variedad", "variedad_std", "variedad_estandar"])
    pred["variedad"] = (
        pred[col_var].astype(str).str.upper().str.strip()
            .replace({"XLENCE": "XL", "CLOUD": "CLO"})
    )

    col_tallos = _pick_col(pred, ["tallos_pred", "tallos_proy", "tallos", "tallos_pred_dia", "tallos_pred_total"])
    pred["tallos_proy"] = _to_num(pred[col_tallos]).fillna(0.0)

    pred = pred[pred["fecha"].notna()].copy()
    pred = pred[pred["station"].isin(["MAIN", "A4"])].copy()
    pred = pred[pred["variedad"].isin(["XL", "CLO"])].copy()

    # área
    pred = _ensure_area_from_inputs(pred, pg)
    pred["area_trabajada"] = pred["area_trabajada"].astype(str).str.upper().str.strip()

    # SELLAR grano (evita cualquier duplicado upstream)
    pred = (
        pred.groupby(["fecha", "station", "area_trabajada", "variedad"], dropna=False)
            .agg(tallos_proy=("tallos_proy", "sum"))
            .reset_index()
    )
    _validate_unique(pred, ["fecha", "station", "area_trabajada", "variedad"], "pred_tallos_cosecha_dia (sellado)")

    # =========================================================
    # 2) Baseline UPH (ZCSP equiv) por station+variedad
    # =========================================================
    if "uph_base_mediana" not in b.columns:
        raise ValueError("dim_baseline_uph_cosecha: falta uph_base_mediana.")
    b2 = b.copy()
    b2["station"] = b2["station"].astype(str).str.upper().str.strip()
    b2["variedad"] = b2["variedad"].astype(str).str.upper().str.strip()
    b2["uph_base_zcsp_equiv"] = _to_num(b2["uph_base_mediana"])

    base_keep = ["station", "variedad", "uph_base_zcsp_equiv"]
    if "n_base" in b2.columns:
        base_keep.append("n_base")
    else:
        b2["n_base"] = np.nan
        base_keep.append("n_base")

    _validate_unique(b2[base_keep], ["station", "variedad"], "dim_baseline_uph_cosecha")

    # =========================================================
    # 3) Clima por hora -> factor_uph por hora (pelado=1) -> factor diario por jornada
    # =========================================================
    # 3.1 weather_hour_wide mínimo
    col_w_dt = _pick_col(w, ["dt_hora", "fecha", "datetime", "timestamp"])
    # si viene "fecha" con hora, también sirve
    w["dt_hora"] = _to_dt(w[col_w_dt]).dt.floor("h")
    w = w[w["dt_hora"].notna()].copy()
    w["fecha"] = w["dt_hora"].dt.normalize()
    w["hora_n"] = w["dt_hora"].dt.hour.astype(int)

    col_w_station = _pick_col(w, ["station", "estacion", "station_code"])
    w["station"] = w[col_w_station].astype(str).str.upper().str.strip().replace({"A-4": "A4", "SJP": "A4"})
    w = w[w["station"].isin(["MAIN", "A4"])].copy()

    col_estado = _pick_col(w, ["Estado_Kardex", "estado_kardex"])
    col_lluvia = _pick_col(w, ["En_Lluvia", "en_lluvia"])
    w["Estado_Kardex"] = w[col_estado].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})
    w["En_Lluvia"] = _to_num(w[col_lluvia]).fillna(0).astype(int)
    w["estado_final"] = np.where(w["En_Lluvia"].eq(1), "LLUVIA", w["Estado_Kardex"])
    w["estado_final"] = w["estado_final"].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})

    # reducir a llaves únicas por hora+station
    w2 = (
        w[["dt_hora", "fecha", "hora_n", "station", "estado_final"]]
        .dropna(subset=["dt_hora", "station"])
        .drop_duplicates(subset=["dt_hora", "station"], keep="last")
        .copy()
    )
    _validate_unique(w2, ["dt_hora", "station"], "weather_hour_wide (w2)")

    # 3.2 dim_factor_uph_cosecha_clima: usar pelado=1 como estándar (ZCSP)
    _require_cols(f, ["station", "variedad", "pelado", "estado_final", "factor_uph"], "dim_factor_uph_cosecha_clima")

    f2 = f.copy()
    f2["station"] = f2["station"].astype(str).str.upper().str.strip()
    f2["variedad"] = f2["variedad"].astype(str).str.upper().str.strip().replace({"XLENCE": "XL", "CLOUD": "CLO"})
    f2["estado_final"] = f2["estado_final"].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})
    f2["pelado"] = _to_num(f2["pelado"]).fillna(0).astype(int)
    f2["factor_uph"] = _to_num(f2["factor_uph"])

    keep_cols = ["station", "variedad", "estado_final", "factor_uph"]
    if "n_obs" in f2.columns:
        keep_cols.append("n_obs")

    f_p1 = f2[f2["pelado"] == 1][keep_cols].copy()
    f_p0 = f2[f2["pelado"] == 0][keep_cols].copy()


    # --- Resolver duplicados en dim_factor: colapsar a 1 fila por (station,variedad,estado_final)
    def _collapse_factor(df_in: pd.DataFrame, name: str) -> pd.DataFrame:
        dfc = df_in.copy()

        # Si existe n_obs (lo usual), usamos una mediana ponderada aproximada repitiendo por bins no es eficiente,
        # así que usamos una aproximación robusta: promedio ponderado si hay pesos, si no mediana simple.
        # Para planificación, promedio ponderado por n_obs suele ser estable.
        if "n_obs" in dfc.columns:
            dfc["n_obs"] = _to_num(dfc["n_obs"]).fillna(0.0)
            g = (
                dfc.groupby(["station", "variedad", "estado_final"], dropna=False)
                   .apply(lambda x: pd.Series({
                       "factor_uph": float(
                           np.average(
                               _to_num(x["factor_uph"]).fillna(1.0).to_numpy(),
                               weights=x["n_obs"].to_numpy()
                           )
                       ) if x["n_obs"].sum() > 0 else float(_to_num(x["factor_uph"]).median())
                   }))
                   .reset_index()
            )
        else:
            g = (
                dfc.groupby(["station", "variedad", "estado_final"], dropna=False)
                   .agg(factor_uph=("factor_uph", "median"))
                   .reset_index()
            )

        # clip de seguridad (misma filosofía que en dim_factor)
        g["factor_uph"] = _to_num(g["factor_uph"]).fillna(1.0).clip(lower=0.30, upper=1.20)

        # validación final: ahora sí debe ser único
        _validate_unique(g, ["station", "variedad", "estado_final"], f"{name} (colapsado)")
        return g

    # colapsar duplicados (LLUVIA y otros)
    f_p1 = _collapse_factor(f_p1, "dim_factor_uph pelado=1")
    f_p0 = _collapse_factor(f_p0, "dim_factor_uph pelado=0")


    # 3.3 Expandir a factor por hora y variedad:
    #      Para cada (dt_hora, station) se replica por variedad usando join con f_p1.
    variedades = pd.DataFrame({"variedad": ["XL", "CLO"]})
    wh = w2.merge(variedades, how="cross")  # (hora,station) x variedad

    # Join factor pelado=1; fallback a pelado=0; fallback final 1.0
    wh = wh.merge(
        f_p1.rename(columns={"factor_uph": "factor_p1"}),
        on=["station", "variedad", "estado_final"],
        how="left",
    ).merge(
        f_p0.rename(columns={"factor_uph": "factor_p0"}),
        on=["station", "variedad", "estado_final"],
        how="left",
    )

    wh["factor_uph"] = _to_num(wh["factor_p1"]).fillna(_to_num(wh["factor_p0"])).fillna(1.0)
    wh = wh.drop(columns=["factor_p1", "factor_p0"])

    # 3.4 Agregar a día por horas de jornada
    horas_turno = float(cfg.get("cosecha", {}).get("horas_turno_cosecha", 9.5))
    # horas jornada: por defecto 6..14 (9 horas). Ajusta en settings si quieres.
    shift_hours = cfg.get("cosecha", {}).get("shift_hours", list(range(6, 15)))  # 6..14
    shift_hours = [int(x) for x in shift_hours]

    wh_j = wh[wh["hora_n"].isin(shift_hours)].copy()

    factor_dia = (
        wh_j.groupby(["fecha", "station", "variedad"], dropna=False)
            .agg(
                factor_uph_dia=("factor_uph", "median"),
                n_horas=("factor_uph", "size"),
            )
            .reset_index()
    )
    _validate_unique(factor_dia, ["fecha", "station", "variedad"], "factor_uph_dia")

    # =========================================================
    # 4) Join final: pred (día-área-variedad) + factor_dia + baseline
    # =========================================================
    m = pred.merge(factor_dia, on=["fecha", "station", "variedad"], how="left")
    miss_factor = int(m["factor_uph_dia"].isna().sum())
    if miss_factor > 0:
        _warn(f"Faltan {miss_factor} factores diarios (se imputan a 1.0). Revisa cobertura weather_hour_wide.")
    m["factor_uph_dia"] = _to_num(m["factor_uph_dia"]).fillna(1.0)

    m = m.merge(
        b2[base_keep],
        on=["station", "variedad"],
        how="left",
    )
    miss_base = int(m["uph_base_zcsp_equiv"].isna().sum())
    if miss_base > 0:
        raise ValueError(f"Faltan {miss_base} baselines por station/variedad. Revisa dim_baseline_uph_cosecha.")

    # UPH efectiva
    m["uph_eff"] = m["uph_base_zcsp_equiv"] * m["factor_uph_dia"]

    # Horas requeridas y personas (jornada semilla)
    m["horas_req"] = np.where(m["uph_eff"] > 0, m["tallos_proy"] / m["uph_eff"], np.nan)
    m["personas_req"] = np.where(m["horas_req"].notna(), m["horas_req"] / horas_turno, np.nan)

    # Si por cualquier razón alguien metió area_trabajada adicional, sellamos grano final:
    out = m[[
        "fecha", "station", "area_trabajada", "variedad",
        "tallos_proy",
        "uph_base_zcsp_equiv",
        "factor_uph_dia",
        "uph_eff",
        "horas_req",
        "personas_req",
        "n_base",
        "n_horas",
    ]].copy()

    out["n_horas"] = _to_num(out["n_horas"]).fillna(0).astype(int)
    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Validación final de unicidad
    _validate_unique(out, ["fecha", "station", "area_trabajada", "variedad"], "capacidad_cosecha_dia (output)")

    # Guardar
    out_path = preds_dir / "capacidad_cosecha_dia.parquet"
    write_parquet(out, out_path)

    _info(f"OK: capacidad_cosecha_dia={len(out)} filas -> {out_path}")
    _info(f"Rango fechas: {out['fecha'].min()} -> {out['fecha'].max()}")
    _info(f"Stations: {out['station'].value_counts(dropna=False).to_dict()}")
    _info(f"Áreas (top): {out['area_trabajada'].value_counts(dropna=False).head(10).to_dict()}")
    _info(f"Variedades: {out['variedad'].value_counts(dropna=False).to_dict()}")

    # Extra: check rápido de multiplicación típica (tu prueba del 9)
    vc = out.groupby(["fecha", "station", "area_trabajada", "variedad"]).size().value_counts().head(10)
    _info(f"Check duplicados (size por llave) -> {vc.to_dict()}")


if __name__ == "__main__":
    main()
