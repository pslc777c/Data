from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


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


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")


def _validate_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = int(df.duplicated(subset=keys).sum())
    if dup > 0:
        ex = df.loc[df.duplicated(subset=keys, keep=False), keys].head(30)
        raise ValueError(
            f"{name}: llaves no únicas dup={dup} para keys={keys}. Ejemplos:\n{ex.to_string(index=False)}"
        )


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    preds_dir.mkdir(parents=True, exist_ok=True)

    in_path = preds_dir / "pred_peso_final_ajustado_grado.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}. Ejecuta build_pred_peso_final_ajustado_grado primero.")

    df = read_parquet(in_path).copy()
    df.columns = [str(c).strip() for c in df.columns]

    # --- Requeridos mínimos para este pred ---
    # Nota: el resto de columnas las pasamos "si existen", pero no deben romper.
    required_min = ["peso_final_g", "fecha_post", "grado"]
    _require_cols(df, required_min, "pred_peso_final_ajustado_grado")

    # Normalización
    df["fecha_post"] = _norm_date(df["fecha_post"])
    df["grado"] = _to_num(df["grado"]).astype("Int64")

    # OJO: NO llenar NaN con 0; si no hay peso es problema upstream.
    df["peso_final_g"] = _to_num(df["peso_final_g"])

    # filtros mínimos
    n0 = len(df)
    df = df[df["fecha_post"].notna()].copy()
    df = df[df["grado"].notna()].copy()

    miss_peso = int(df["peso_final_g"].isna().sum())
    if miss_peso > 0:
        _warn(f"Hay {miss_peso} filas sin peso_final_g (NaN). Se excluyen para cálculo de cajas.")
        df = df[df["peso_final_g"].notna()].copy()

    # sanity: pesos no negativos
    neg = int((df["peso_final_g"] < 0).sum())
    if neg > 0:
        _warn(f"Hay {neg} filas con peso_final_g negativo. Se excluyen.")
        df = df[df["peso_final_g"] >= 0].copy()

    _info(f"Input: {n0} filas; después filtros mínimos: {len(df)} filas")

    # Conversión a kg y cajas (10 kg/caja)
    df["peso_final_kg"] = df["peso_final_g"] / 1000.0
    df["cajas_pred_grado"] = df["peso_final_kg"] / 10.0  # 10 kg / caja

    df["created_at"] = datetime.now().isoformat(timespec="seconds")

    # -------------------------
    # Output 1: pred_cajas_grado (detalle)
    # -------------------------
    # Columnas “deseadas” (si faltan, no revienta; solo se omiten)
    desired_cols = [
        "ciclo_id",
        "fecha", "fecha_post", "dh_dias",
        "destino",
        "bloque", "bloque_padre",
        "variedad", "variedad_std",
        "tipo_sp", "area", "estado",
        "stage",
        "grado",
        "tallos_pred_grado",
        "peso_final_g",
        "peso_final_kg",
        "cajas_pred_grado",
        "created_at",
    ]
    out_grado_cols = [c for c in desired_cols if c in df.columns]

    out_grado = df[out_grado_cols].copy()

    # Validación grano (elige el set de llaves más estable según tu upstream)
    # Preferimos: ciclo_id + fecha_post + bloque_padre + destino + grado
    keys_candidates = [
        ["ciclo_id", "fecha_post", "bloque_padre", "destino", "grado"],
        ["fecha_post", "bloque_padre", "destino", "grado"],
        ["fecha_post", "destino", "grado"],
    ]
    chosen_keys = None
    for k in keys_candidates:
        if all(col in out_grado.columns for col in k):
            chosen_keys = k
            break
    if chosen_keys is None:
        _warn("No pude validar unicidad de pred_cajas_grado: faltan columnas de llaves. (No es fatal, pero revísalo).")
    else:
        _validate_unique(out_grado, chosen_keys, "pred_cajas_grado")

    out_grado_path = preds_dir / "pred_cajas_grado.parquet"
    write_parquet(out_grado, out_grado_path)

    # -------------------------
    # Output 2: pred_cajas_dia (agregado)
    # -------------------------
    group_keys = ["ciclo_id", "fecha_post", "bloque", "bloque_padre", "variedad", "variedad_std", "destino"]
    group_keys = [k for k in group_keys if k in out_grado.columns]

    if "fecha_post" not in group_keys:
        raise ValueError("No puedo construir pred_cajas_dia: falta 'fecha_post' en el dataset.")

    daily = (
        out_grado.groupby(group_keys, dropna=False)
        .agg(
            peso_final_g=("peso_final_g", "sum"),
            peso_final_kg=("peso_final_kg", "sum"),
            cajas_pred=("cajas_pred_grado", "sum"),
        )
        .reset_index()
    )
    daily["created_at"] = datetime.now().isoformat(timespec="seconds")

    _validate_unique(daily, group_keys, "pred_cajas_dia")

    out_daily_path = preds_dir / "pred_cajas_dia.parquet"
    write_parquet(daily, out_daily_path)

    # -------------------------
    # Logs
    # -------------------------
    _info(f"OK: pred_cajas_grado={len(out_grado)} -> {out_grado_path}")
    _info(f"OK: pred_cajas_dia={len(daily)} -> {out_daily_path}")

    if "cajas_pred_grado" in out_grado.columns and len(out_grado):
        _info("cajas_pred_grado describe:\n" + out_grado["cajas_pred_grado"].describe().to_string())
    if len(daily):
        _info("cajas_pred (daily) describe:\n" + daily["cajas_pred"].describe().to_string())


if __name__ == "__main__":
    main()
