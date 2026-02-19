from __future__ import annotations

from dataclasses import dataclass
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


def _hour_floor(s: pd.Series) -> pd.Series:
    return _to_dt(s).dt.floor("h")  # "H" deprecated


@dataclass(frozen=True)
class KardexParams:
    th_lluvia: float = 0.3   # mm/h
    th_seco: float = 0.3     # K <= th_seco => SECO
    th_humedo: float = 2.0   # K <= th_humedo => HUMEDO; > => MOJADO
    kardex_cap: float = 3.0  # tope mm


def compute_kardex_estado(df: pd.DataFrame, p: KardexParams) -> pd.DataFrame:
    """
    df: estación única, ordenada por fecha_hora
    salida: agrega kardex_prev_mm, kardex_mm, Estado_Kardex, En_Lluvia
    """
    out = df.copy()

    out["lluvia_mm"] = pd.to_numeric(out.get("lluvia_mm", 0), errors="coerce").fillna(0.0).astype(float)
    out["et_mm"] = pd.to_numeric(out.get("et_mm", 0), errors="coerce").fillna(0.0).astype(float)

    out = out.sort_values("fecha_hora").reset_index(drop=True)
    out["En_Lluvia"] = (out["lluvia_mm"] >= float(p.th_lluvia)).astype("int8")

    # Recurrencia: Kt = clip(Kt-1 + lluvia - ET, 0, cap)
    k = np.zeros(len(out), dtype=float)
    for i in range(len(out)):
        delta = float(out.loc[i, "lluvia_mm"]) - float(out.loc[i, "et_mm"])
        if i == 0:
            k[i] = np.clip(delta, 0.0, float(p.kardex_cap))
        else:
            k[i] = np.clip(k[i - 1] + delta, 0.0, float(p.kardex_cap))

    out["kardex_prev_mm"] = np.r_[0.0, k[:-1]]
    out["kardex_mm"] = k

    def estado(x: float) -> str:
        if x <= float(p.th_seco):
            return "SECO"
        if x <= float(p.th_humedo):
            return "HUMEDO"
        return "MOJADO"

    out["Estado_Kardex"] = out["kardex_mm"].map(estado)
    return out


def _standardize_bronze_weather(df: pd.DataFrame, station_expected: str | None = None) -> pd.DataFrame:
    """
    Estandariza los BRONZE existentes:
      - fecha -> fecha_hora (floor hora)
      - rainfall_mm -> lluvia_mm
      - et -> et_mm
      - station upper
    Mantiene el resto de columnas (temp_avg, solar_energy, etc.) para usarlas después si quieres.
    """
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    if "fecha" not in d.columns:
        raise ValueError(f"Bronze weather: falta columna 'fecha'. Columnas={list(d.columns)}")

    d = d.rename(columns={"fecha": "fecha_hora"})
    d["fecha_hora"] = _hour_floor(d["fecha_hora"])
    d = d[d["fecha_hora"].notna()].copy()

    # Renombres a esquema canónico
    if "rainfall_mm" in d.columns and "lluvia_mm" not in d.columns:
        d = d.rename(columns={"rainfall_mm": "lluvia_mm"})
    if "et" in d.columns and "et_mm" not in d.columns:
        d = d.rename(columns={"et": "et_mm"})

    # Asegurar columnas mínimas
    if "lluvia_mm" not in d.columns:
        d["lluvia_mm"] = np.nan
    if "et_mm" not in d.columns:
        d["et_mm"] = np.nan

    d["lluvia_mm"] = pd.to_numeric(d["lluvia_mm"], errors="coerce").fillna(0.0)
    d["et_mm"] = pd.to_numeric(d["et_mm"], errors="coerce").fillna(0.0)

    if "station" not in d.columns:
        # Por seguridad (aunque tus scripts BRONZE sí la ponen)
        d["station"] = station_expected if station_expected else "UNKNOWN"

    d["station"] = d["station"].astype(str).str.upper().str.strip().replace({"A-4": "A4", "SJP": "A4"})
    if station_expected:
        d = d[d["station"].eq(station_expected.upper())].copy()

    # Deduplicación técnica (si existen duplicados por hora)
    d = d.sort_values("fecha_hora").drop_duplicates(subset=["station", "fecha_hora"], keep="last")
    return d


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    p_main = bronze_dir / "weather_hour_main.parquet"
    p_a4 = bronze_dir / "weather_hour_a4.parquet"

    if not p_main.exists():
        raise FileNotFoundError(f"No existe Bronze: {p_main}. Ejecuta src/bronze/build_weather_hour_main.py")
    if not p_a4.exists():
        raise FileNotFoundError(f"No existe Bronze: {p_a4}. Ejecuta src/bronze/build_weather_hour_a4.py")

    df_main = pd.read_parquet(p_main)
    df_a4 = pd.read_parquet(p_a4)

    main_std = _standardize_bronze_weather(df_main, station_expected="MAIN")
    a4_std = _standardize_bronze_weather(df_a4, station_expected="A4")

    weather = pd.concat([main_std, a4_std], ignore_index=True)
    weather = weather[weather["station"].isin(["MAIN", "A4"])].copy()

    # Parámetros kardex
    p = KardexParams(
        th_lluvia=float(cfg.get("weather", {}).get("th_lluvia", 0.3)),
        th_seco=float(cfg.get("weather", {}).get("th_seco", 0.3)),
        th_humedo=float(cfg.get("weather", {}).get("th_humedo", 2.0)),
        kardex_cap=float(cfg.get("weather", {}).get("kardex_cap", 3.0)),
    )

    out_frames = []
    for station, g in weather.groupby("station", dropna=False):
        out_frames.append(compute_kardex_estado(g, p))

    out = pd.concat(out_frames, ignore_index=True)
    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = silver_dir / "weather_hour_estado.parquet"
    write_parquet(out, out_path)

    print(f"OK: weather_hour_estado={len(out)} filas -> {out_path}")
    print("Stations:", out["station"].value_counts(dropna=False).to_dict())
    print("Rango:", out["fecha_hora"].min(), "->", out["fecha_hora"].max())
    print("Estado_Kardex:", out["Estado_Kardex"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
