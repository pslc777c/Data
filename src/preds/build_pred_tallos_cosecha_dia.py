# src/preds/build_pred_tallos_cosecha_dia.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

# OJO: para aislar problemas, escribimos con pandas.to_parquet (no con write_parquet)


# -------------------------
# Config / helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _pick_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _std_variedad(v: pd.Series) -> pd.Series:
    x = v.astype(str).str.upper().str.strip()
    return x.replace({"XXLENCE": "XL", "XLENCE": "XL", "CLOUD": "CLO"})


def _infer_station_from_area(area: pd.Series) -> pd.Series:
    """
    Regla negocio (clima):
      - A-4 / A4 / SJP / SAN JUAN => A4
      - resto => MAIN
    """
    a = area.astype(str).str.upper().str.strip()
    is_a4 = a.isin(["A-4", "A4", "SJP", "SAN JUAN"])
    return pd.Series(np.where(is_a4, "A4", "MAIN"), index=area.index)


def _bloque_key_from_any(df: pd.DataFrame) -> pd.Series | None:
    """
    Intenta inferir bloque_key si existe alguna columna compatible.
    """
    for c in ["bloque", "Bloque", "bloque_norm", "Bloque_norm", "bloque_padre", "block", "Block"]:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            k = s.str.extract(r"^(\d+)", expand=False)
            return pd.to_numeric(k, errors="coerce").astype("Int64")
    return None


def _ensure_area_trabajada(
    df: pd.DataFrame,
    preds_dir: Path,
) -> pd.DataFrame:
    """
    Asegura area_trabajada:
      1) Si viene en el insumo (area_trabajada/area/Area) => usarla
      2) Si no viene => intenta mapear desde pred_peso_grado.parquet por (fecha, bloque_key, variedad)
      3) Si no se puede => ALL (warning)
    """
    df = df.copy()

    # 1) directa
    for cand in ["area_trabajada", "area", "Area"]:
        if cand in df.columns:
            df["area_trabajada"] = df[cand].astype(str).str.upper().str.strip()
            return df

    # 2) mapping desde pred_peso_grado
    pg_path = preds_dir / "pred_peso_grado.parquet"
    if not pg_path.exists():
        _warn("No existe pred_peso_grado.parquet para mapear área -> se asigna ALL.")
        df["area_trabajada"] = "ALL"
        return df

    bloque_key = _bloque_key_from_any(df)
    if bloque_key is None:
        _warn("El insumo no trae bloque/bloque_norm para mapear área -> se asigna ALL.")
        df["area_trabajada"] = "ALL"
        return df

    df["bloque_key"] = bloque_key

    pg = pd.read_parquet(pg_path)
    pg.columns = [str(c).strip() for c in pg.columns]

    # checks mínimos
    req = {"fecha", "bloque", "area"}
    miss = req - set(pg.columns)
    if miss:
        raise ValueError(f"pred_peso_grado.parquet no tiene columnas requeridas {sorted(miss)}. Columnas={list(pg.columns)}")
    if ("variedad_std" not in pg.columns) and ("variedad" not in pg.columns):
        raise ValueError(f"pred_peso_grado.parquet: falta variedad_std/variedad. Columnas={list(pg.columns)}")

    pg["fecha"] = _norm_dt(pg["fecha"])
    pg["bloque_key"] = pd.to_numeric(pg["bloque"].astype(str).str.extract(r"^(\d+)", expand=False), errors="coerce").astype("Int64")

    if "variedad_std" in pg.columns:
        pg["variedad"] = _std_variedad(pg["variedad_std"])
    else:
        pg["variedad"] = _std_variedad(pg["variedad"])

    pg["area_trabajada"] = pg["area"].astype(str).str.upper().str.strip()

    # mapa (fecha,bloque_key,variedad) -> area_trabajada (mode)
    map_area = (
        pg.dropna(subset=["fecha", "bloque_key", "variedad", "area_trabajada"])
          .groupby(["fecha", "bloque_key", "variedad"], dropna=False)["area_trabajada"]
          .agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0])
          .reset_index()
    )

    # join
    if "fecha" not in df.columns:
        raise ValueError("No puedo mapear área: df no tiene columna 'fecha' normalizada.")

    if "variedad" not in df.columns:
        raise ValueError("No puedo mapear área: df no tiene columna 'variedad' estandarizada.")

    df = df.merge(map_area, on=["fecha", "bloque_key", "variedad"], how="left")

    miss_n = int(df["area_trabajada"].isna().sum())
    if miss_n > 0:
        _warn(f"No se pudo mapear area_trabajada para {miss_n} filas -> se asigna ALL.")
        df["area_trabajada"] = df["area_trabajada"].fillna("ALL")

    # limpieza
    df = df.drop(columns=["bloque_key"])
    df["area_trabajada"] = df["area_trabajada"].astype(str).str.upper().str.strip()
    return df


def _validate_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    d = df.duplicated(subset=keys).sum()
    if int(d) > 0:
        ex = df.loc[df.duplicated(subset=keys, keep=False), keys].head(20)
        raise ValueError(f"{name}: llaves no únicas dup={int(d)} keys={keys}. Ejemplos:\n{ex.to_string(index=False)}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg["paths"]["preds"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    # --- DEBUG anti-“no actualiza” ---
    _info(f"running file: {__file__}")
    _info(f"cwd: {Path.cwd()}")
    _info(f"preds_dir: {preds_dir.resolve()}")

    # Insumos (semillas)
    candidates = [
        preds_dir / "pred_oferta_grado.parquet",
        preds_dir / "pred_oferta_dia.parquet",
    ]
    src_path = _pick_first_existing(candidates)
    if src_path is None:
        raise FileNotFoundError(
            "No encontré insumo para construir pred_tallos_cosecha_dia.\n"
            f"Esperaba alguno de:\n- {candidates[0]}\n- {candidates[1]}\n"
            "Ejecuta primero el script que genera pred_oferta_grado o pred_oferta_dia."
        )
    _info(f"source selected: {src_path.resolve()}")

    df = pd.read_parquet(src_path)
    df.columns = [str(c).strip() for c in df.columns]
    _info(f"input rows: {len(df)} cols: {len(df.columns)}")

    # -------- Fecha --------
    if "fecha" in df.columns:
        df["fecha"] = _norm_dt(df["fecha"])
    elif "Fecha" in df.columns:
        df["fecha"] = _norm_dt(df["Fecha"])
    elif "Fecha_Cosecha" in df.columns:
        df["fecha"] = _norm_dt(df["Fecha_Cosecha"])
    elif "harvest_start" in df.columns:
        df["fecha"] = _norm_dt(df["harvest_start"])
    else:
        raise ValueError(f"No pude inferir columna de fecha en {src_path.name}. Columnas={list(df.columns)[:80]}")
    df = df[df["fecha"].notna()].copy()

    # -------- Tallos --------
    # -------- Tallos (fix: evitar duplicación por grado) --------
    src_name = src_path.name.lower()

    if "pred_oferta_grado" in src_name:
        if "tallos_pred_grado" not in df.columns:
            raise ValueError(
                "Insumo pred_oferta_grado.parquet requiere columna 'tallos_pred_grado'. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        df["tallos_pred"] = (
            _to_num(df["tallos_pred_grado"])
            .fillna(0.0)
            .astype(float)
        )
        _info("Tallos fuente: tallos_pred_grado (pred_oferta_grado)")
    else:
        # comportamiento original para oferta diaria
        if "tallos_proy" in df.columns:
            df["tallos_pred"] = _to_num(df["tallos_proy"]).fillna(0.0).astype(float)
        elif "tallos_pred" in df.columns:
            df["tallos_pred"] = _to_num(df["tallos_pred"]).fillna(0.0).astype(float)
        elif "tallos" in df.columns:
            df["tallos_pred"] = _to_num(df["tallos"]).fillna(0.0).astype(float)
        elif "Tallos" in df.columns:
            df["tallos_pred"] = _to_num(df["Tallos"]).fillna(0.0).astype(float)
        else:
            raise ValueError(
                f"No pude inferir columna de tallos en {src_path.name}. "
                f"Columnas={list(df.columns)}"
            )
        _info("Tallos fuente: total diario (pred_oferta_dia)")


    # -------- Variedad (semilla: XL/CLO) --------
    if "variedad" in df.columns:
        df["variedad"] = _std_variedad(df["variedad"])
    elif "Variedad" in df.columns:
        df["variedad"] = _std_variedad(df["Variedad"])
    elif "variedad_std" in df.columns:
        df["variedad"] = _std_variedad(df["variedad_std"])
    else:
        df["variedad"] = "UNKNOWN"

    # Normalización final esperada
    df["variedad"] = df["variedad"].replace({"XLENCE": "XL", "CLOUD": "CLO"})
    # NO filtramos a XL/CLO aquí obligatoriamente; pero avisamos
    if int((df["variedad"] == "UNKNOWN").sum()) > 0:
        _warn("Hay filas con variedad=UNKNOWN. Revisa el insumo si debería traer variedad.")

    # -------- Área canónica (area_trabajada) --------
    df = _ensure_area_trabajada(df, preds_dir=preds_dir)

    # -------- Station canónica (para clima y joins posteriores) --------
    df["station"] = _infer_station_from_area(df["area_trabajada"])

    # -------- Sellado de grano (nivel correcto para capacidad) --------
    out = (
        df.groupby(["fecha", "station", "area_trabajada", "variedad"], dropna=False)
          .agg(tallos_proy=("tallos_pred", "sum"))
          .reset_index()
    )
    out["tallos_proy"] = _to_num(out["tallos_proy"]).fillna(0.0).clip(lower=0.0).astype(float)

    # Validación unicidad (debe ser 1 fila por llave)
    _validate_unique(out, ["fecha", "station", "area_trabajada", "variedad"], "pred_tallos_cosecha_dia")

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # -------- Escritura robusta (sin helper) --------
    out_path = preds_dir / "pred_tallos_cosecha_dia.parquet"
    _info(f"out_path: {out_path.resolve()}")

    before = datetime.fromtimestamp(out_path.stat().st_mtime) if out_path.exists() else None
    if before:
        _info(f"mtime BEFORE: {before.isoformat(timespec='seconds')}")

    out.to_parquet(out_path, index=False)

    after = datetime.fromtimestamp(out_path.stat().st_mtime) if out_path.exists() else None
    if after:
        _info(f"mtime AFTER:  {after.isoformat(timespec='seconds')}")

    # -------- Resumen --------
    _info(f"OK: pred_tallos_cosecha_dia={len(out)} filas -> {out_path.name}")
    _info(f"Rango fechas: {out['fecha'].min()} -> {out['fecha'].max()}")
    _info(f"Stations: {out['station'].value_counts(dropna=False).to_dict()}")
    _info(f"Areas (top): {out['area_trabajada'].value_counts(dropna=False).head(15).to_dict()}")
    _info(f"Variedades: {out['variedad'].value_counts(dropna=False).head(10).to_dict()}")

    n_all = int((out["area_trabajada"] == "ALL").sum())
    if n_all > 0:
        _warn(f"Hay {n_all} filas con area_trabajada=ALL (no se pudo inferir área).")


if __name__ == "__main__":
    main()
