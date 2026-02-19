from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import read_parquet, write_parquet


# -------------------------
# Settings / helpers
# -------------------------
def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    oferta_path = preds_dir / "pred_oferta_dia.parquet"
    dist_path = silver_dir / "dim_dist_grado_baseline.parquet"

    if not oferta_path.exists():
        raise FileNotFoundError(f"No existe: {oferta_path}. Ejecuta build_pred_oferta_dia primero.")
    if not dist_path.exists():
        raise FileNotFoundError(f"No existe: {dist_path}. Ejecuta build_dim_dist_grado_baseline primero.")

    oferta = read_parquet(oferta_path).copy()
    dist = read_parquet(dist_path).copy()

    # --- Validaciones mínimas
    _require(oferta, ["ciclo_id", "fecha", "variedad", "tallos_pred"], "pred_oferta_dia")
    _require(dist, ["variedad", "grado", "pct_grado"], "dim_dist_grado_baseline")

    # --- Canonicalizar
    oferta["ciclo_id"] = oferta["ciclo_id"].astype(str)
    oferta["fecha"] = _to_date(oferta["fecha"])
    oferta["variedad"] = _canon_str(oferta["variedad"])
    oferta["tallos_pred"] = pd.to_numeric(oferta["tallos_pred"], errors="coerce").fillna(0.0)

    # llaves opcionales (no forzamos, pero estandarizamos si existen)
    for col in ["bloque", "bloque_padre"]:
        if col in oferta.columns:
            oferta[col] = _canon_int(oferta[col])

    dist["variedad"] = _canon_str(dist["variedad"])
    dist["grado"] = pd.to_numeric(dist["grado"], errors="coerce").astype("Int64")
    dist["pct_grado"] = pd.to_numeric(dist["pct_grado"], errors="coerce").astype(float)

    # Mapeo de variedades (mismo estándar en oferta y dist)
    var_map = (cfg.get("mappings", {}).get("variedad_map", {}) or {})
    var_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in var_map.items()}

    oferta["variedad_std"] = oferta["variedad"].map(lambda x: var_map.get(x, x))
    dist["variedad_std"] = dist["variedad"].map(lambda x: var_map.get(x, x))

    # --- Validación: unicidad por (variedad_std, grado)
    dup = int(dist.duplicated(subset=["variedad_std", "grado"]).sum())
    if dup > 0:
        ex = dist.loc[
            dist.duplicated(subset=["variedad_std", "grado"], keep=False),
            ["variedad", "variedad_std", "grado", "pct_grado"],
        ].sort_values(["variedad_std", "grado"]).head(50)
        raise ValueError(
            "dim_dist_grado_baseline: duplicados por (variedad_std, grado). "
            f"dup={dup}. Ejemplos:\n{ex.to_string(index=False)}"
        )

    # --- Renormalizar distribución para que sume 1 por variedad_std
    # (esto evita que la masa “se pierda” si la dim viene con redondeos o faltantes pequeños)
    sum_by_var = dist.groupby("variedad_std", dropna=False)["pct_grado"].sum().reset_index(name="sum_pct")
    dist = dist.merge(sum_by_var, on="variedad_std", how="left")
    bad = dist["sum_pct"].isna() | (dist["sum_pct"] <= 0)
    if bad.any():
        bad_vars = dist.loc[bad, "variedad_std"].dropna().unique()[:20]
        raise ValueError(f"Distribución inválida (sum_pct <= 0) para variedad_std. Ejemplos: {bad_vars}")

    dist["pct_grado_norm"] = dist["pct_grado"] / dist["sum_pct"]

    # --- Join oferta x grados
    merged = oferta.merge(
        dist[["variedad_std", "grado", "pct_grado_norm"]],
        on="variedad_std",
        how="left",
    )

    # Si faltan variedades en dim => error (no silenciar)
    if merged["pct_grado_norm"].isna().any():
        miss = merged.loc[merged["pct_grado_norm"].isna(), "variedad_std"].value_counts().head(20)
        raise ValueError(
            "Falta distribución por grado para algunas variedad_std. "
            "Corrige dim_dist_grado_baseline o variedad_map. Ejemplos:\n"
            f"{miss.to_string()}"
        )

    merged["tallos_pred_grado"] = merged["tallos_pred"].astype(float) * merged["pct_grado_norm"].astype(float)

    # --- Output (mantener compatibilidad downstream)
    cols_out = [
        "ciclo_id", "fecha",
        "bloque", "bloque_padre", "variedad", "variedad_std", "tipo_sp", "area", "estado",
        "stage",
        "grado", "pct_grado_norm",
        "tallos_pred", "tallos_pred_grado",
    ]
    # crear columnas faltantes opcionales como NA (por compatibilidad)
    for c in cols_out:
        if c not in merged.columns:
            merged[c] = pd.NA

    out = merged[cols_out].copy()
    out = out.rename(columns={"pct_grado_norm": "pct_grado"})  # mantener nombre esperado

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_path = preds_dir / "pred_oferta_grado.parquet"
    write_parquet(out, out_path)

    # --- Auditoría masa: sum(tallos_pred_grado) ~= tallos_pred por ciclo/fecha
    audit = (
        out.groupby(["ciclo_id", "fecha"], dropna=False)["tallos_pred_grado"].sum()
           .reset_index(name="sum_grado")
    )
    audit = audit.merge(
        oferta[["ciclo_id", "fecha", "tallos_pred"]],
        on=["ciclo_id", "fecha"],
        how="left"
    )
    audit["diff"] = audit["sum_grado"] - audit["tallos_pred"]
    audit["abs_diff"] = audit["diff"].abs()

    print(f"OK: pred_oferta_grado={len(out):,} filas -> {out_path}")
    print("Audit diff describe:\n", audit["diff"].describe().to_string())
    print("Audit abs_diff describe:\n", audit["abs_diff"].describe().to_string())
    print("Audit max abs_diff:", float(audit["abs_diff"].max() if len(audit) else 0.0))


if __name__ == "__main__":
    main()
