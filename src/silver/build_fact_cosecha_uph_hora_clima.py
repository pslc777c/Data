# src/silver/build_fact_cosecha_uph_hora_clima.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import warnings
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


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def bloque_norm_from_raw(b: pd.Series) -> pd.Series:
    x = b.astype(str).str.strip()
    out = x.str.extract(r"^(\d+)", expand=False)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def map_variedad(codigo_actividad: pd.Series) -> pd.Series:
    c = codigo_actividad.astype(str).str.upper().str.strip()
    xl = {"ZCS", "ZCSP", "ZXL", "ZVX", "ZPPX", "ZCX"}  # XLENCE
    clo = {"ZMP", "ZVP", "ZPC", "ZCC"}                 # CLOUD
    out = np.where(
        c.isin(list(xl)),
        "XLENCE",
        np.where(c.isin(list(clo)), "CLOUD", None),
    )
    return pd.Series(out, index=c.index)


def map_station(area_trabajada: pd.Series) -> pd.Series:
    a = area_trabajada.astype(str).str.upper().str.strip()
    is_a4 = a.isin(["A-4", "A4", "SJP", "SAN JUAN"])
    return pd.Series(np.where(is_a4, "A4", "MAIN"), index=a.index)


def build_hours_grid() -> pd.DataFrame:
    rows = []
    for h in range(24):
        ini = pd.to_timedelta(h, unit="h")
        fin = (
            pd.to_timedelta(h + 1, unit="h")
            if h < 23
            else pd.to_timedelta(23, unit="h")
            + pd.to_timedelta(59, unit="m")
            + pd.to_timedelta(59, unit="s")
        )
        rows.append({"hora_n": h, "td_ini": ini, "td_fin": fin})
    return pd.DataFrame(rows)


def overlap_seconds(a_start, a_end, b_start, b_end) -> np.ndarray:
    start = np.maximum(a_start, b_start)
    end = np.minimum(a_end, b_end)
    sec = (end - start) / np.timedelta64(1, "s")
    sec = np.where(sec < 0, 0, sec)
    return sec.astype(float)


def load_kardex_from_weather_wide(weather_path: Path, fecha_min: pd.Timestamp) -> pd.DataFrame:
    if not weather_path.exists():
        raise FileNotFoundError(f"No existe {weather_path}")

    k = pd.read_parquet(weather_path)
    k.columns = [str(c).strip() for c in k.columns]

    # dt_hora
    if "dt_hora" not in k.columns:
        if "fecha" in k.columns:
            k = k.rename(columns={"fecha": "dt_hora"})
        else:
            raise ValueError("weather_hour_wide: falta dt_hora (o 'fecha' con hora).")

    # station
    if "station" not in k.columns:
        if "estacion" in k.columns:
            k = k.rename(columns={"estacion": "station"})
        else:
            raise ValueError("weather_hour_wide: falta station (MAIN/A4).")

    # Estado_Kardex
    if "Estado_Kardex" not in k.columns:
        if "estado_kardex" in k.columns:
            k = k.rename(columns={"estado_kardex": "Estado_Kardex"})
        else:
            raise ValueError("weather_hour_wide: falta Estado_Kardex.")

    # En_Lluvia
    if "En_Lluvia" not in k.columns:
        if "en_lluvia" in k.columns:
            k = k.rename(columns={"en_lluvia": "En_Lluvia"})
        else:
            raise ValueError("weather_hour_wide: falta En_Lluvia.")

    k["dt_hora"] = pd.to_datetime(k["dt_hora"], errors="coerce")
    k = k[k["dt_hora"].notna()].copy()
    k = k[k["dt_hora"] >= pd.to_datetime(fecha_min)].copy()

    k["station"] = k["station"].astype(str).str.upper().str.strip().replace({"A-4": "A4", "SJP": "A4"})
    k["fecha"] = k["dt_hora"].dt.normalize()
    k["hora_n"] = k["dt_hora"].dt.hour.astype(int)

    k["Estado_Kardex"] = k["Estado_Kardex"].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})
    k["En_Lluvia"] = _to_num(k["En_Lluvia"]).fillna(0).astype(int)

    return k[["station", "fecha", "hora_n", "Estado_Kardex", "En_Lluvia"]].drop_duplicates()


def _robust_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.median())


def compute_factor_codigo_zcsp(
    df_hourly: pd.DataFrame,
    *,
    min_n_codigo: int = 200,
    clip_low: float = 0.60,
    clip_high: float = 1.60,
    min_horas_tramo: float = 0.10,
) -> pd.DataFrame:
    """
    Aprende factor por station/variedad/codigo_actividad para llevar UPH -> escala ZCSP
    en contexto base: SECO y En_Lluvia=0.
    factor = median(uph_ZCSP) / median(uph_codigo)
    """
    d = df_hourly.copy()

    base = d[
        (d["Estado_Kardex"].astype(str).str.upper().str.strip() == "SECO")
        & (pd.to_numeric(d["En_Lluvia"], errors="coerce").fillna(0).astype(int) == 0)
        & (pd.to_numeric(d["horas_tramo"], errors="coerce").fillna(0.0) >= float(min_horas_tramo))
    ].copy()

    if base.empty:
        raise ValueError("No hay data base (SECO sin lluvia) para calibrar factor_codigo_zcsp.")

    z = base[base["codigo_actividad"] == "ZCSP"].copy()
    ref = (
        z.groupby(["station", "variedad"], dropna=False)
        .agg(uph_zcsp_mediana=("uph_tramo", _robust_median), n_zcsp=("uph_tramo", "size"))
        .reset_index()
    )

    cod = (
        base.groupby(["station", "variedad", "codigo_actividad"], dropna=False)
        .agg(uph_cod_mediana=("uph_tramo", _robust_median), n_cod=("uph_tramo", "size"))
        .reset_index()
    )

    out = cod.merge(ref, on=["station", "variedad"], how="left")
    out["factor_codigo_zcsp"] = out["uph_zcsp_mediana"] / out["uph_cod_mediana"]
    out["factor_codigo_zcsp"] = out["factor_codigo_zcsp"].replace([np.inf, -np.inf], np.nan)

    low_n = out["n_cod"].fillna(0).astype(int) < int(min_n_codigo)
    out.loc[low_n, "factor_codigo_zcsp"] = np.nan

    out.loc[out["codigo_actividad"] == "ZCSP", "factor_codigo_zcsp"] = 1.0
    out["factor_codigo_zcsp"] = out["factor_codigo_zcsp"].fillna(1.0).astype(float)
    out["factor_codigo_zcsp"] = out["factor_codigo_zcsp"].clip(lower=float(clip_low), upper=float(clip_high))

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    cols = [
        "station", "variedad", "codigo_actividad",
        "uph_cod_mediana", "n_cod",
        "uph_zcsp_mediana", "n_zcsp",
        "factor_codigo_zcsp",
        "created_at",
    ]
    return out[cols].sort_values(["station", "variedad", "codigo_actividad"]).reset_index(drop=True)


def _require_cols(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: faltan columnas requeridas: {missing}. Columnas disponibles={list(df.columns)}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # seeds clima
    seeds = cfg.get("uph_clima_seed", {})
    f_seco = float(seeds.get("f_seco", 1.00))
    f_humedo = float(seeds.get("f_humedo", 0.88))
    f_mojado = float(seeds.get("f_mojado", 0.80))
    f_lluvia = float(seeds.get("f_lluvia", 0.72))

    # cosecha cfg
    cosecha_cfg = cfg.get("uph_cosecha", {})
    fecha_min = pd.to_datetime(cosecha_cfg.get("fecha_min", "2024-01-01"))

    station_by_area_only = bool(cosecha_cfg.get("station_by_area_trabajada_only", True))
    drop_cloud_in_a4 = bool(cosecha_cfg.get("drop_cloud_in_a4", True))

    valid_main = set(s.upper() for s in cosecha_cfg.get("valid_areas_main", []) if str(s).strip())
    valid_a4 = set(s.upper() for s in cosecha_cfg.get("valid_areas_a4", []) if str(s).strip())

    # calibración factor código
    cal_cfg = cosecha_cfg.get("cal_factor_codigo_zcsp", {})
    min_n_codigo = int(cal_cfg.get("min_n_codigo", 200))
    min_horas_tramo = float(cal_cfg.get("min_horas_tramo", 0.10))
    clip_low = float(cal_cfg.get("clip_low", 0.60))
    clip_high = float(cal_cfg.get("clip_high", 1.60))

    # clima (silver)
    weather_path = silver_dir / "weather_hour_wide.parquet"
    kardex = load_kardex_from_weather_wide(weather_path, fecha_min)

    # -------------------------
    # INPUT BRONZE: ghu_maestro_horas
    # -------------------------
    ghu_fname = cfg.get("ghu", {}).get("ghu_maestro_horas_file", "ghu_maestro_horas.parquet")
    ghu_path = bronze_dir / ghu_fname
    if not ghu_path.exists():
        raise FileNotFoundError(
            f"No existe input Bronze: {ghu_path}\n"
            "Ejecuta src/bronze/build_ghu_maestro_horas.py primero "
            "o define ghu.ghu_maestro_horas_file en settings.yaml."
        )

    ghu = pd.read_parquet(ghu_path)
    ghu.columns = [str(c).strip() for c in ghu.columns]

    _require_cols(
        ghu,
        [
            "fecha",
            "codigo_personal",
            "nombres",
            "actividad",
            "codigo_actividad",
            "tipo_actividad",
            "area_trabajada",
            "area_original",
            "horas_acumula",
            "bloque",
            "hora_ingreso",
            "hora_salida",
            "horas_presenciales",
            "unidades_producidas",
        ],
        "ghu_maestro_horas (bronze)",
    )

    # Filtros equivalentes a tu SQL
    ghu["fecha"] = _norm_date(ghu["fecha"])
    ghu = ghu[ghu["fecha"].notna()].copy()
    ghu = ghu[ghu["fecha"] >= pd.to_datetime(fecha_min)].copy()

    ghu["tipo_actividad"] = ghu["tipo_actividad"].astype(str).str.strip()
    ghu = ghu[ghu["tipo_actividad"].eq("COS-1")].copy()
    ghu = ghu[ghu["bloque"].notna()].copy()

    # -------------------------
    # Preparación GHU
    # -------------------------
    ghu["codigo_actividad"] = ghu["codigo_actividad"].astype(str).str.upper().str.strip()
    ghu["bloque_norm"] = bloque_norm_from_raw(ghu["bloque"])
    ghu = ghu[ghu["bloque_norm"].notna()].copy()

    ghu["area_trabajada"] = ghu["area_trabajada"].astype(str).str.upper().str.strip()
    if station_by_area_only:
        ghu = ghu[ghu["area_trabajada"].notna() & (ghu["area_trabajada"] != "")].copy()

    ghu["station"] = map_station(ghu["area_trabajada"])

    if valid_main:
        mask_main = (ghu["station"] == "MAIN") & (ghu["area_trabajada"].isin(valid_main))
    else:
        mask_main = (ghu["station"] == "MAIN")

    if valid_a4:
        mask_a4 = (ghu["station"] == "A4") & (ghu["area_trabajada"].isin(valid_a4))
    else:
        mask_a4 = (ghu["station"] == "A4")

    ghu = ghu[mask_main | mask_a4].copy()

    # variedad
    ghu["variedad"] = map_variedad(ghu["codigo_actividad"])
    ghu = ghu[ghu["variedad"].notna()].copy()

    if drop_cloud_in_a4:
        bad = (ghu["station"] == "A4") & (ghu["variedad"].astype(str).str.upper().str.contains("CLOUD"))
        n_bad = int(bad.sum())
        if n_bad > 0:
            warnings.warn(f"[DATA QUALITY] Eliminando {n_bad} registros CLOUD en A4 (no debería existir).")
            ghu = ghu[~bad].copy()

    # horas de ingreso/salida (en horas decimales)
    hi = _to_num(ghu["hora_ingreso"]).fillna(0.0)
    hf = _to_num(ghu["hora_salida"]).fillna(0.0)

    ghu["mins_ini"] = np.round(hi * 60.0).astype(int).clip(0, 24 * 60 - 1)
    ghu["mins_fin"] = np.round(hf * 60.0).astype(int).clip(0, 24 * 60 - 1)

    ghu["dt_ini"] = ghu["fecha"] + pd.to_timedelta(ghu["mins_ini"], unit="m")
    ghu["dt_fin"] = ghu["fecha"] + pd.to_timedelta(ghu["mins_fin"], unit="m")

    bad_time = ghu["dt_fin"] <= ghu["dt_ini"]
    if bad_time.any():
        warnings.warn(f"Registros con hora_salida <= hora_ingreso: {int(bad_time.sum())} (se ajustan +1 minuto)")
        ghu.loc[bad_time, "dt_fin"] = ghu.loc[bad_time, "dt_ini"] + pd.to_timedelta(1, unit="m")

    ghu["horas_presenciales"] = _to_num(ghu["horas_presenciales"]).fillna(0.0)
    ghu["unidades_producidas"] = _to_num(ghu["unidades_producidas"]).fillna(0.0)
    ghu = ghu[(ghu["horas_presenciales"] > 0) & (ghu["unidades_producidas"] > 0)].copy()

    # rec_id
    ghu["rec_id"] = (
        ghu.groupby(
            ["fecha", "codigo_personal", "bloque_norm", "mins_ini", "mins_fin", "horas_presenciales", "unidades_producidas"],
            dropna=False,
        )
        .cumcount()
        + 1
    )

    # -------------------------
    # Explosión por hora (e)
    # -------------------------
    horas = build_hours_grid()
    ghu["_key"] = 1
    horas["_key"] = 1
    e = ghu.merge(horas, on="_key", how="inner").drop(columns=["_key"])

    e["tramo_ini"] = e["fecha"] + e["td_ini"]
    e["tramo_fin"] = e["fecha"] + e["td_fin"]

    e["seg_tramo"] = overlap_seconds(
        e["dt_ini"].values.astype("datetime64[ns]"),
        e["dt_fin"].values.astype("datetime64[ns]"),
        e["tramo_ini"].values.astype("datetime64[ns]"),
        e["tramo_fin"].values.astype("datetime64[ns]"),
    )
    e = e[e["seg_tramo"] > 0].copy()

    keys = ["fecha", "codigo_personal", "bloque_norm", "rec_id"]
    seg_total = e.groupby(keys, dropna=False)["seg_tramo"].sum().rename("seg_total").reset_index()
    e = e.merge(seg_total, on=keys, how="left")
    e["W"] = np.where(e["seg_total"] > 0, e["seg_tramo"] / e["seg_total"], 0.0)

    # join clima
    e["station"] = e["station"].astype(str).str.upper().str.strip().replace({"A-4": "A4", "SJP": "A4"})
    e = e.merge(kardex, on=["station", "fecha", "hora_n"], how="left")

    miss = int(e["Estado_Kardex"].isna().sum())
    if miss > 0:
        warnings.warn(f"Faltan {miss} joins de clima (Estado_Kardex null). Se asume SECO y En_Lluvia=0.")
        e["Estado_Kardex"] = e["Estado_Kardex"].fillna("SECO")
        e["En_Lluvia"] = e["En_Lluvia"].fillna(0)

    e["Estado_Kardex"] = e["Estado_Kardex"].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})
    e["En_Lluvia"] = _to_num(e["En_Lluvia"]).fillna(0).astype(int)
    e["estado_final"] = np.where(e["En_Lluvia"].eq(1), "LLUVIA", e["Estado_Kardex"])

    # base UPH por tramo (SIN normalizar aún, solo para calibrar factor por código)
    e["horas_tramo"] = e["seg_tramo"] / 3600.0
    e["UP_total_reg_real"] = e["unidades_producidas"].astype(float)
    e["UP_tramo_real"] = e["UP_total_reg_real"] * e["W"]
    e["uph_tramo"] = np.where(e["horas_tramo"] > 0, e["UP_tramo_real"] / e["horas_tramo"], np.nan)

    # -------------------------
    # 1) Calibrar factor_codigo_zcsp (y guardarlo como dim)
    # -------------------------
    dim_factor = compute_factor_codigo_zcsp(
        e[[
            "station", "variedad", "codigo_actividad",
            "Estado_Kardex", "En_Lluvia",
            "horas_tramo", "uph_tramo"
        ]].copy(),
        min_n_codigo=min_n_codigo,
        min_horas_tramo=min_horas_tramo,
        clip_low=clip_low,
        clip_high=clip_high,
    )

    dim_path = silver_dir / "dim_factor_codigo_zcsp_cosecha.parquet"
    write_parquet(dim_factor, dim_path)

    # merge factor al hourly
    e = e.merge(
        dim_factor[["station", "variedad", "codigo_actividad", "factor_codigo_zcsp"]],
        on=["station", "variedad", "codigo_actividad"],
        how="left",
    )

    miss_fac = int(e["factor_codigo_zcsp"].isna().sum())
    if miss_fac > 0:
        warnings.warn(f"[WARN] {miss_fac} filas sin factor_codigo_zcsp. Se imputan a 1.0.")
        e["factor_codigo_zcsp"] = e["factor_codigo_zcsp"].fillna(1.0)

    # -------------------------
    # 2) Aplicar factor => UP_total_reg_equiv (ZCSP scale)
    # -------------------------
    e["UP_total_reg_equiv"] = e["UP_total_reg_real"] * e["factor_codigo_zcsp"].astype(float)

    # -------------------------
    # 3) Ajuste clima (sobre el equivalente)
    # -------------------------
    state_factor = {"SECO": f_seco, "HUMEDO": f_humedo, "MOJADO": f_mojado, "LLUVIA": f_lluvia}
    e["factor_clima"] = e["estado_final"].map(state_factor).fillna(f_seco).astype(float)

    e["factor_total"] = e["factor_clima"]

    e["WF"] = e["W"] * e["factor_total"]
    sum_wf = e.groupby(keys, dropna=False)["WF"].sum().rename("sum_WF").reset_index()
    e = e.merge(sum_wf, on=keys, how="left")

    e["UP_total_reg"] = e["UP_total_reg_equiv"]
    e["HP_total_reg"] = e["horas_presenciales"].astype(float)

    e["UP_tramo_raw"] = e["UP_total_reg"] * e["W"]
    e["UP_tramo_ajustada"] = np.where(e["sum_WF"] > 0, e["UP_total_reg"] * (e["WF"] / e["sum_WF"]), 0.0)

    e["uph_raw_tramo_equiv"] = np.where(e["horas_tramo"] > 0, e["UP_tramo_raw"] / e["horas_tramo"], np.nan)
    e["uph_ajustada_tramo_equiv"] = np.where(e["horas_tramo"] > 0, e["UP_tramo_ajustada"] / e["horas_tramo"], np.nan)

    e["dt_hora"] = e["fecha"] + pd.to_timedelta(e["hora_n"].astype(int), unit="h")

    out = e[[
        "fecha", "hora_n", "dt_hora",
        "codigo_personal", "nombres",
        "area_trabajada", "area_original",
        "bloque_norm", "rec_id",
        "codigo_actividad", "actividad",
        "variedad", "station",

        "Estado_Kardex", "En_Lluvia", "estado_final",

        "seg_tramo", "seg_total", "horas_tramo",
        "UP_total_reg_real", "UP_total_reg_equiv",
        "HP_total_reg",

        "W",
        "factor_codigo_zcsp",
        "factor_clima", "factor_total",
        "WF", "sum_WF",

        "UP_tramo_raw", "UP_tramo_ajustada",
        "uph_raw_tramo_equiv", "uph_ajustada_tramo_equiv",
    ]].copy()

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "fact_cosecha_uph_hora_clima.parquet"
    write_parquet(out, out_path)

    print(f"OK: dim_factor_codigo_zcsp_cosecha={len(dim_factor)} filas -> {dim_path}")
    print(f"OK: fact_cosecha_uph_hora_clima={len(out)} filas -> {out_path}")
    print("Rango fechas:", out["fecha"].min(), "->", out["fecha"].max())
    print("Stations:", out["station"].value_counts(dropna=False).to_dict())
    print("Variedades:", out["variedad"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
