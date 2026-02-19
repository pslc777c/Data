from __future__ import annotations

from pathlib import Path
import pandas as pd

from common.io import read_parquet, write_parquet


BASE_GDC = 7.1  # umbral estándar empresa para Gypsophila


def _normalize_area(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.upper()
         .str.replace(r"\s+", " ", regex=True)
    )


def _area_to_station(area_norm: pd.Series) -> pd.Series:
    station = pd.Series(index=area_norm.index, dtype="object")

    is_main = area_norm.isin(["MH1", "MH2", "CV"])
    is_a4 = area_norm.isin(["A-4", "A4"])

    station.loc[is_main] = "MAIN"
    station.loc[is_a4] = "A4"
    return station


def main() -> None:
    created_at = pd.Timestamp.utcnow()

    df_weather = read_parquet(Path("data/silver/weather_hour_wide.parquet"))
    df_maestro = read_parquet(Path("data/silver/fact_ciclo_maestro.parquet"))

    # -------------------------
    # 1) Map bloque_base -> area -> station
    # -------------------------
    need_cols = {"bloque_base", "area"}
    missing = need_cols - set(df_maestro.columns)
    if missing:
        raise ValueError(f"fact_ciclo_maestro.parquet no tiene columnas requeridas: {sorted(missing)}")

    map_blk = df_maestro[["bloque_base", "area"]].copy()
    map_blk["area_norm"] = _normalize_area(map_blk["area"])
    map_blk["station"] = _area_to_station(map_blk["area_norm"])

    bad = map_blk[map_blk["station"].isna()][["bloque_base", "area"]].drop_duplicates()
    if len(bad) > 0:
        raise ValueError(f"No se pudo mapear area -> station. Ejemplos: {bad.head(20).to_dict('records')}")

    map_blk = (
        map_blk.groupby(["bloque_base", "area_norm", "station"], as_index=False)
               .size()
               .sort_values(["bloque_base", "size"], ascending=[True, False])
               .drop_duplicates(subset=["bloque_base"], keep="first")
    )
    map_blk = map_blk.rename(columns={"area_norm": "area"})
    map_blk = map_blk[["bloque_base", "area", "station"]]

    # -------------------------
    # 2) Weather hourly -> daily by station (AGRONÓMICO)
    # -------------------------
    need_wcols = {
        "dt_hora", "station",
        "rainfall_mm", "temp_avg", "solar_energy_j_m2",
        "En_Lluvia",
        "wind_speed_avg", "wind_run",   # <-- viento
    }
    missing_w = need_wcols - set(df_weather.columns)
    if missing_w:
        raise ValueError(f"weather_hour_wide.parquet no tiene columnas requeridas: {sorted(missing_w)}")

    w = df_weather[list(need_wcols)].copy()
    w["dt_hora"] = pd.to_datetime(w["dt_hora"], errors="coerce")
    if w["dt_hora"].isna().any():
        raise ValueError("dt_hora contiene valores inválidos")

    w["fecha"] = w["dt_hora"].dt.normalize()
    w["station"] = w["station"].astype(str).str.strip().str.upper()

    w["En_Lluvia"] = pd.to_numeric(w["En_Lluvia"], errors="coerce").fillna(0)
    w["En_Lluvia"] = (w["En_Lluvia"] > 0).astype(int)

    # Agregación diaria por estación
    daily = (
        w.groupby(["station", "fecha"], as_index=False)
         .agg(
             rainfall_mm_dia=("rainfall_mm", "sum"),
             temp_avg_dia=("temp_avg", "mean"),
             solar_energy_j_m2_dia=("solar_energy_j_m2", "sum"),
             horas_lluvia=("En_Lluvia", "sum"),
             en_lluvia_dia=("En_Lluvia", "max"),
             wind_speed_avg_dia=("wind_speed_avg", "mean"),
             wind_run_dia=("wind_run", "sum"),
         )
    )

    # GDC diario (normalizado)
    daily["gdc_base"] = BASE_GDC
    daily["gdc_dia"] = (daily["temp_avg_dia"] - BASE_GDC).clip(lower=0)

    # -------------------------
    # 3) Broadcast clima a bloque_base
    # -------------------------
    out = map_blk.merge(daily, on="station", how="left")

    if out["fecha"].isna().all():
        raise ValueError("Join bloque_base -> station con clima diario no produjo fechas. Revisa station.")

    out = out.dropna(subset=["fecha"]).copy()
    out["created_at"] = created_at

    out = out[
        [
            "fecha",
            "bloque_base",
            "area",
            "station",
            "rainfall_mm_dia",
            "horas_lluvia",
            "en_lluvia_dia",
            "temp_avg_dia",
            "solar_energy_j_m2_dia",
            "wind_speed_avg_dia",
            "wind_run_dia",
            "gdc_base",
            "gdc_dia",
            "created_at",
        ]
    ].sort_values(["fecha", "bloque_base"]).reset_index(drop=True)

    write_parquet(out, Path("data/silver/dim_clima_bloque_dia.parquet"))
    print(f"OK -> data/silver/dim_clima_bloque_dia.parquet | rows={len(out):,}")


if __name__ == "__main__":
    main()
