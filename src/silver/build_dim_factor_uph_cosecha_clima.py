# src/silver/build_dim_factor_uph_cosecha_clima.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
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


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hour_floor(dt: pd.Series) -> pd.Series:
    # pandas: "H" deprecado -> usar "h"
    return _to_dt(dt).dt.floor("h")


def _norm_station_from_area(area: pd.Series) -> pd.Series:
    """
    Regla de negocio:
      - A4/SJP/SAN JUAN => station=A4
      - else => MAIN
    """
    a = area.astype(str).str.upper().str.strip()
    is_a4 = a.isin(["A-4", "A4", "SJP", "SAN JUAN"])
    return pd.Series(np.where(is_a4, "A4", "MAIN"), index=area.index)


def _derive_variedad_from_codigo(cod: pd.Series) -> pd.Series:
    c = cod.astype(str).str.upper().str.strip()
    xl = c.isin(["ZCS", "ZCSP", "ZXL", "ZVX", "ZPPX", "ZCX"])
    clo = c.isin(["ZMP", "ZVP", "ZPC", "ZCC"])
    out = np.where(xl, "XL", np.where(clo, "CLO", np.nan))
    return pd.Series(out, index=cod.index)


def _derive_pelado_from_codigo(cod: pd.Series) -> pd.Series:
    c = cod.astype(str).str.upper().str.strip()
    return (c == "ZCSP").astype(int)


def _q(x: pd.Series, q: float) -> float:
    x = x.dropna()
    return float(x.quantile(q)) if len(x) else np.nan


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """
    Devuelve el nombre real de la columna que existe en df, probando candidatos.
    Si required=True y no encuentra, lanza error con columnas disponibles.
    """
    cols = list(df.columns)
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        cl = cand.lower()
        if cl in cols_l:
            return cols_l[cl]
    if required:
        raise ValueError(
            "No se encontró ninguna columna válida. "
            f"Candidatos={candidates}. Columnas disponibles={cols}"
        )
    return None


def _normalize_area_trabajada(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza area_trabajada y aplica whitelist para MAIN.
    Reglas negocio:
      - A4 puede venir como A-4 / A4 / SJP / SAN JUAN
      - MAIN (cosecha) debe estar en {MH1, MH2, CULTIVOS VARIOS}
      - Otros valores se marcan como NaN (y se excluyen por baseline/agrupación)
    """
    df = df.copy()

    if "area_trabajada" not in df.columns:
        return df

    a = df["area_trabajada"].astype(str).str.upper().str.strip()

    # normalizaciones comunes
    a = a.replace(
        {
            "A-4": "A4",
            "SANJUAN": "SAN JUAN",
            "SAN_JUAN": "SAN JUAN",
        }
    )

    df["area_trabajada"] = a

    # whitelist MAIN
    main_ok = {"MH1", "MH2", "CULTIVOS VARIOS"}
    is_main = df.get("station", pd.Series(index=df.index, dtype="object")).astype(str).str.upper().eq("MAIN")
    is_a4 = df.get("station", pd.Series(index=df.index, dtype="object")).astype(str).str.upper().eq("A4")

    # Para MAIN: si no está en whitelist, poner NaN (evita sesgos tipo EMP/ORN)
    df.loc[is_main & (~df["area_trabajada"].isin(list(main_ok))), "area_trabajada"] = np.nan

    # Para A4: dejarlo como venga (A4/SJP/SAN JUAN ya normalizados), no necesitamos whitelist aquí
    # (si quisieras, podrías forzar df.loc[is_a4, "area_trabajada"]="A4")

    return df


def _ensure_station_from_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Station debe venir de area_trabajada (estricto).
    NO usamos area_original para evitar contaminación.
    """
    df = df.copy()

    if "area_trabajada" not in df.columns:
        raise ValueError(
            "Falta columna 'area_trabajada' en el input de cosecha horaria. "
            "No puedo inferir station sin esa columna."
        )

    df["station"] = _norm_station_from_area(df["area_trabajada"])
    df["station"] = df["station"].astype(str).str.upper().str.strip()
    df = df[df["station"].isin(["MAIN", "A4"])].copy()

    # normalizar area_trabajada + whitelist MAIN
    df = _normalize_area_trabajada(df)

    return df


def _baseline_from_areas(
    m: pd.DataFrame,
    q_baseline_area: float = 0.75,
    min_n_area: int = 500,
) -> pd.DataFrame:
    """
    Baseline robusto:
      1) Filtra condiciones ideales: SECO, sin lluvia, no pelado
      2) Calcula mediana por (station,variedad,area_trabajada)
      3) Se queda con áreas con n>=min_n_area
      4) Baseline final por (station,variedad) = cuantil q_baseline_area de esas medianas de área

    Esto evita que la "mediana global" se deprima por colas/heterogeneidad.
    """
    base_mask = (m["Estado_Kardex"] == "SECO") & (m["En_Lluvia"] == 0) & (m["pelado"] == 0)

    ideal = m[base_mask].copy()

    if "area_trabajada" not in ideal.columns:
        raise ValueError("Para baseline por áreas, falta area_trabajada en el input (post-merge).")

    # Excluir áreas no válidas (NaN) para baseline por área
    ideal = ideal[ideal["area_trabajada"].notna()].copy()

    base_area = (
        ideal.groupby(["station", "variedad", "area_trabajada"], dropna=False)
        .agg(
            uph_mediana_area=("uph", "median"),
            n_area=("uph", "size"),
        )
        .reset_index()
    )

    # Filtrar áreas con volumen mínimo para que no dominen áreas raras o con poca data
    base_area_ok = base_area[base_area["n_area"] >= int(min_n_area)].copy()

    if base_area_ok.empty:
        # fallback conservador: usar la base_area sin filtro n
        base_area_ok = base_area.copy()

    # Baseline final: cuantil alto de las medianas por área
    base = (
        base_area_ok.groupby(["station", "variedad"], dropna=False)
        .agg(
            uph_base_mediana=("uph_mediana_area", lambda s: float(s.quantile(q_baseline_area)) if len(s) else np.nan),
            n_base=("n_area", "sum"),
            n_areas=("area_trabajada", "nunique"),
        )
        .reset_index()
    )

    return base


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    uph_fname = cfg.get("cosecha", {}).get("uph_hour_file", "fact_cosecha_uph_hora_clima.parquet")
    uph_path = silver_dir / uph_fname
    weather_path = silver_dir / "weather_hour_wide.parquet"

    if not uph_path.exists():
        raise FileNotFoundError(
            f"No existe input de UPH cosecha por hora: {uph_path}\n"
            f"Config sugerida: cosecha.uph_hour_file en settings.yaml"
        )
    if not weather_path.exists():
        raise FileNotFoundError(f"No existe weather_hour_wide: {weather_path}")

    df = pd.read_parquet(uph_path)
    w = pd.read_parquet(weather_path)

    df.columns = [str(c).strip() for c in df.columns]
    w.columns = [str(c).strip() for c in w.columns]

    # -------------------------
    # Normalización df (cosecha)
    # -------------------------
    if "dt_hora" in df.columns:
        df["dt_hora"] = _hour_floor(df["dt_hora"])
    elif "fecha_hora" in df.columns:
        df["dt_hora"] = _hour_floor(df["fecha_hora"])
    elif "fecha" in df.columns and "hora_n" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
        df["hora_n"] = _to_num(df["hora_n"]).astype("Int64")
        df["dt_hora"] = df["fecha"] + pd.to_timedelta(df["hora_n"].fillna(0).astype(int), unit="h")
        df["dt_hora"] = _hour_floor(df["dt_hora"])
    else:
        raise ValueError("El input de cosecha debe traer 'dt_hora' o ('fecha' y 'hora_n') o 'fecha_hora'.")

    # station (estricto desde area_trabajada)
    df = _ensure_station_from_hours(df)

    # actividad -> variedad/pelado
    if "codigo_actividad" not in df.columns:
        raise ValueError("Falta columna codigo_actividad en input de cosecha horaria.")
    df["codigo_actividad"] = df["codigo_actividad"].astype(str).str.upper().str.strip()

    if "variedad" in df.columns:
        df["variedad"] = df["variedad"].astype(str).str.upper().str.strip()
        df["variedad"] = df["variedad"].replace({"XXLENCE": "XL", "XLENCE": "XL", "CLOUD": "CLO"})
    else:
        df["variedad"] = _derive_variedad_from_codigo(df["codigo_actividad"])

    if "pelado" in df.columns:
        df["pelado"] = _to_num(df["pelado"]).fillna(0).astype(int)
    else:
        df["pelado"] = _derive_pelado_from_codigo(df["codigo_actividad"])

    df = df[df["variedad"].isin(["XL", "CLO"])].copy()

    # Validación negocio: A4 NO puede tener CLO
    bad_a4_clo = df[(df["station"] == "A4") & (df["variedad"] == "CLO")]
    if len(bad_a4_clo) > 0:
        ex = bad_a4_clo[["dt_hora", "codigo_actividad", "variedad", "station", "area_trabajada"]].head(10)
        raise ValueError(
            "Detectado CLO en station=A4 (esto no debería ocurrir según negocio). "
            "Revisa mapping de area_trabajada y/o códigos. Ejemplos:\n"
            f"{ex.to_string(index=False)}"
        )

    # UPH
    if "UP_por_hora_ajustada" in df.columns:
        df["uph"] = _to_num(df["UP_por_hora_ajustada"])
    elif "uph_ajustada_tramo" in df.columns:
        df["uph"] = _to_num(df["uph_ajustada_tramo"])
    else:
        if "UP_tramo_ajustada" not in df.columns:
            raise ValueError("Falta UP_por_hora_ajustada/uph_ajustada_tramo y UP_tramo_ajustada en input.")
        if "horas_tramo" not in df.columns:
            raise ValueError("Falta horas_tramo para calcular UPH.")
        df["UP_tramo_ajustada"] = _to_num(df["UP_tramo_ajustada"])
        df["horas_tramo"] = _to_num(df["horas_tramo"])
        df["uph"] = np.where(df["horas_tramo"] > 0, df["UP_tramo_ajustada"] / df["horas_tramo"], np.nan)

    df["uph"] = _to_num(df["uph"])
    df = df[df["uph"].notna()].copy()
    df = df[(df["uph"] > 10) & (df["uph"] < 2000)].copy()

    # -------------------------
    # Normalización clima (weather_hour_wide)
    # -------------------------
    col_w_dt = _pick_col(w, ["fecha", "dt_hora", "datetime", "timestamp"])
    w["dt_hora"] = _hour_floor(w[col_w_dt])

    col_w_station = _pick_col(w, ["station", "estacion", "ws_station", "station_code"])
    w["station"] = w[col_w_station].astype(str).str.upper().str.strip()
    w = w[w["station"].isin(["MAIN", "A4"])].copy()

    col_estado = _pick_col(
        w,
        ["Estado_Kardex", "estado_kardex", "estado", "estado_clima", "kardex_estado", "Estado"],
        required=True,
    )
    col_lluvia = _pick_col(
        w,
        ["En_Lluvia", "en_lluvia", "lluvia", "is_rain", "rain_flag"],
        required=True,
    )

    w["Estado_Kardex"] = w[col_estado].astype(str).str.upper().str.strip().replace({"HÚMEDO": "HUMEDO"})
    w["En_Lluvia"] = _to_num(w[col_lluvia]).fillna(0).astype(int)

    w2 = (
        w[["dt_hora", "station", "Estado_Kardex", "En_Lluvia"]]
        .dropna(subset=["dt_hora", "station"])
        .drop_duplicates(subset=["dt_hora", "station"], keep="last")
        .copy()
    )

    # -------------------------
    # Join por hora + station
    # -------------------------
    for c in ["Estado_Kardex", "En_Lluvia", "estado_final"]:
        if c in df.columns:
            df = df.rename(columns={c: f"{c}_src"})

    m = df.merge(w2, on=["dt_hora", "station"], how="left")

    if "Estado_Kardex" not in m.columns:
        raise ValueError(f"Post-merge: no existe Estado_Kardex. Columnas={list(m.columns)}")
    if "En_Lluvia" not in m.columns:
        raise ValueError(f"Post-merge: no existe En_Lluvia. Columnas={list(m.columns)}")

    m["__miss_weather"] = m["Estado_Kardex"].isna().astype(int)
    m["Estado_Kardex"] = m["Estado_Kardex"].fillna("SECO").astype(str).str.upper().str.strip()
    m["En_Lluvia"] = pd.to_numeric(m["En_Lluvia"], errors="coerce").fillna(0).astype(int)
    m["estado_final"] = np.where(m["En_Lluvia"].eq(1), "LLUVIA", m["Estado_Kardex"])

    # -------------------------
    # Baseline robusto (POR ÁREA) en condiciones ideales
    # -------------------------
    # Parámetros (opcionales) desde settings.yaml:
    #   cosecha:
    #     baseline_area_quantile: 0.75
    #     baseline_area_min_n: 500
    q_baseline = float(cfg.get("cosecha", {}).get("baseline_area_quantile", 0.75))
    min_n_area = int(cfg.get("cosecha", {}).get("baseline_area_min_n", 500))

    base = _baseline_from_areas(m, q_baseline_area=q_baseline, min_n_area=min_n_area)

    if base.empty or base["uph_base_mediana"].isna().all():
        raise ValueError(
            "No se pudo calcular baseline robusto por áreas. "
            "Revisa si existen registros SECO/sin lluvia/no pelado y áreas válidas."
        )

    # -------------------------
    # Agregación por celda clima
    # -------------------------
    g = (
        m.groupby(["station", "variedad", "pelado", "Estado_Kardex", "En_Lluvia", "estado_final"], dropna=False)
        .agg(
            uph_mediana=("uph", "median"),
            uph_p25=("uph", lambda s: _q(s, 0.25)),
            uph_p75=("uph", lambda s: _q(s, 0.75)),
            n_obs=("uph", "size"),
            n_miss_weather=("__miss_weather", "sum"),
        )
        .reset_index()
    )

    out = g.merge(base[["station", "variedad", "uph_base_mediana", "n_base", "n_areas"]], on=["station", "variedad"], how="left")
    out["factor_uph"] = out["uph_mediana"] / out["uph_base_mediana"]
    out["factor_uph"] = out["factor_uph"].clip(lower=0.30, upper=1.20)

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    cols = [
        "station", "variedad", "pelado",
        "Estado_Kardex", "En_Lluvia", "estado_final",
        "uph_base_mediana", "n_base", "n_areas",
        "uph_mediana", "uph_p25", "uph_p75",
        "factor_uph",
        "n_obs", "n_miss_weather",
        "created_at",
    ]
    out = (
        out[cols]
        .sort_values(["station", "variedad", "pelado", "En_Lluvia", "Estado_Kardex"])
        .reset_index(drop=True)
    )

    out_path = silver_dir / "dim_factor_uph_cosecha_clima.parquet"
    write_parquet(out, out_path)

    base2 = base.copy()
    base2["created_at"] = datetime.now().isoformat(timespec="seconds")
    base_path = silver_dir / "dim_baseline_uph_cosecha.parquet"
    write_parquet(base2, base_path)

    # Logs
    print(f"OK: dim_factor_uph_cosecha_clima={len(out)} filas -> {out_path}")
    print(f"OK: dim_baseline_uph_cosecha={len(base2)} filas -> {base_path}")
    print("Baseline (uph_base_mediana) por station/variedad (robusto por áreas):")
    print(base2.sort_values(["station", "variedad"]).to_string(index=False))
    print("Resumen factor_uph:")
    print(out["factor_uph"].describe().to_string())

    miss = int(out["n_miss_weather"].sum())
    if miss > 0:
        print(f"[WARN] Hay {miss} tramos con clima faltante (imputados a SECO/0). Revisa cobertura de weather_hour_wide.")


if __name__ == "__main__":
    main()
