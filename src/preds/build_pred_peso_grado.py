from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    # robusto con strings y coma decimal
    ss = s.copy()
    if ss.dtype == object or pd.api.types.is_string_dtype(ss):
        ss = (
            ss.astype("string")
              .str.replace(" ", "", regex=False)
              .str.replace(",", ".", regex=False)
        )
    return pd.to_numeric(ss, errors="coerce")


def _validate_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = int(df.duplicated(subset=keys).sum())
    if dup > 0:
        ex = df.loc[df.duplicated(subset=keys, keep=False), keys].head(20)
        raise ValueError(
            f"{name}: llaves no únicas (dup={dup}) para keys={keys}. Ejemplos:\n{ex.to_string(index=False)}"
        )


def _get_use_stage(cfg: dict) -> str:
    # pred_oferta.use_stage = "HARVEST" por ejemplo
    use_stage = (cfg.get("pred_oferta", {}) or {}).get("use_stage", "")
    return str(use_stage).strip().upper()


def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    oferta_path = preds_dir / "pred_oferta_grado.parquet"
    peso_dim_path = silver_dir / "dim_peso_tallo_baseline.parquet"

    if not oferta_path.exists():
        raise FileNotFoundError(f"No existe: {oferta_path}. Ejecuta build_pred_oferta_grado primero.")
    if not peso_dim_path.exists():
        raise FileNotFoundError(f"No existe: {peso_dim_path}. Ejecuta build_dim_peso_tallo_baseline primero.")

    oferta = read_parquet(oferta_path).copy()
    dim = read_parquet(peso_dim_path).copy()

    oferta.columns = [str(c).strip() for c in oferta.columns]
    dim.columns = [str(c).strip() for c in dim.columns]

    # -------------------------
    # Normalizar oferta
    # -------------------------
    if "fecha" not in oferta.columns:
        raise ValueError("pred_oferta_grado: falta columna 'fecha'")
    oferta["fecha"] = _norm_date(oferta["fecha"])

    req = ["variedad", "grado", "tallos_pred_grado"]
    miss = [c for c in req if c not in oferta.columns]
    if miss:
        raise ValueError(f"pred_oferta_grado: faltan columnas requeridas: {miss}")

    oferta["variedad"] = oferta["variedad"].astype(str).str.strip().str.upper()
    oferta["grado"] = _to_num(oferta["grado"]).astype("Int64")
    oferta["tallos_pred_grado"] = _to_num(oferta["tallos_pred_grado"]).fillna(0.0).astype(float)

    # Mapping de variedad pred -> estándar
    var_map = (cfg.get("mappings", {}).get("variedad_map", {}) or {})
    var_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in var_map.items()}
    oferta["variedad_std"] = oferta["variedad"].map(lambda x: var_map.get(x, x))

    # -------------------------
    # FILTRO CLAVE: stage (HARVEST vs HARVEST_POST)
    # -------------------------
    use_stage = _get_use_stage(cfg)
    if "stage" in oferta.columns:
        oferta["stage"] = oferta["stage"].astype(str).str.strip().str.upper()

        if use_stage:
            n0 = len(oferta)
            oferta = oferta[oferta["stage"].eq(use_stage)].copy()
            print(f"[INFO] pred_oferta_grado filtrado por stage='{use_stage}': {len(oferta):,}/{n0:,}")

            if oferta.empty:
                raise ValueError(
                    f"pred_oferta_grado quedó vacío tras filtrar stage='{use_stage}'. "
                    f"Revisa pred_oferta.use_stage en settings.yaml y/o valores reales en columna 'stage'."
                )
        else:
            print("[WARN] pred_oferta.use_stage vacío; se mantienen todos los stages (puede haber duplicados).")
    else:
        # si no existe stage, no podemos resolver la duplicación por stage desde acá
        if use_stage:
            print("[WARN] pred_oferta.use_stage está definido, pero pred_oferta_grado NO trae columna 'stage'. "
                  "Se ignora el filtro. (Si hay duplicados, revisa build_pred_oferta_grado.)")

    # -------------------------
    # Validar grano mínimo de oferta (después del filtro)
    # -------------------------
    grain_keys = [k for k in ["ciclo_id", "fecha", "bloque", "bloque_padre", "variedad_std", "grado"] if k in oferta.columns]
    if len(grain_keys) >= 4:
        _validate_unique(oferta, grain_keys, "pred_oferta_grado (input)")

    # -------------------------
    # Normalizar dim peso tallo
    # -------------------------
    req_dim = ["variedad", "grado", "peso_tallo_mediana_g"]
    miss_dim = [c for c in req_dim if c not in dim.columns]
    if miss_dim:
        raise ValueError(f"dim_peso_tallo_baseline: faltan columnas requeridas: {miss_dim}")

    dim["variedad"] = dim["variedad"].astype(str).str.strip().str.upper()
    dim["grado"] = _to_num(dim["grado"]).astype("Int64")
    dim["peso_tallo_mediana_g"] = _to_num(dim["peso_tallo_mediana_g"])

    dim["variedad_std"] = dim["variedad"]  # dim ya está estándar

    # Colapsar a 1 fila por llave (robusto)
    dim2 = (
        dim.groupby(["variedad_std", "grado"], dropna=False)
           .agg(peso_tallo_mediana_g=("peso_tallo_mediana_g", "median"))
           .reset_index()
    )
    _validate_unique(dim2, ["variedad_std", "grado"], "dim_peso_tallo_baseline (colapsado)")

    # -------------------------
    # Join + cálculo
    # -------------------------
    merged = oferta.merge(dim2, on=["variedad_std", "grado"], how="left")

    miss_peso = int(merged["peso_tallo_mediana_g"].isna().sum())
    if miss_peso > 0:
        ex = (
            merged.loc[merged["peso_tallo_mediana_g"].isna(), ["variedad_std", "grado"]]
                  .drop_duplicates()
                  .head(20)
        )
        raise ValueError(
            "Falta peso_tallo_mediana_g para algunas combinaciones variedad_std+grado. Ejemplos:\n"
            + ex.to_string(index=False)
        )

    merged["peso_pred_g"] = merged["tallos_pred_grado"] * merged["peso_tallo_mediana_g"]

    # columnas de salida (solo si existen)
    want_cols = [
        "ciclo_id", "fecha",
        "bloque", "bloque_padre",
        "variedad", "variedad_std",
        "tipo_sp", "area", "estado",
        "stage",
        "grado",
        "tallos_pred_grado",
        "peso_tallo_mediana_g",
        "peso_pred_g",
    ]
    out_cols = [c for c in want_cols if c in merged.columns]
    out = merged[out_cols].copy()

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Validación final de unicidad (sin stage, porque ya filtraste stage)
    out_keys = ["ciclo_id", "fecha", "bloque", "bloque_padre", "variedad_std", "grado"]
    out_keys = [k for k in out_keys if k in out.columns]
    _validate_unique(out, out_keys, "pred_peso_grado (output)")

    out_path = preds_dir / "pred_peso_grado.parquet"
    write_parquet(out, out_path)

    print(f"OK: pred_peso_grado={len(out):,} filas -> {out_path}")
    if "peso_pred_g" in out.columns:
        print("peso_pred_g describe:\n", out["peso_pred_g"].describe().to_string())


if __name__ == "__main__":
    main()
