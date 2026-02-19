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
    return pd.to_numeric(s, errors="coerce")


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas {missing}. Disponibles={list(df.columns)}")


def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    in_path = preds_dir / "pred_peso_hidratado_grado.parquet"
    dim_path = silver_dir / "dim_mermas_ajuste_fecha_post_destino.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"No existe: {in_path}. Ejecuta build_pred_peso_hidratado_grado primero.")
    if not dim_path.exists():
        raise FileNotFoundError(f"No existe: {dim_path}. Ejecuta build_dim_mermas_ajuste_fecha_post primero.")

    df = read_parquet(in_path).copy()
    dim = read_parquet(dim_path).copy()

    df.columns = [str(c).strip() for c in df.columns]
    dim.columns = [str(c).strip() for c in dim.columns]

    # Requeridos mínimos
    _require_cols(df, ["fecha_post", "destino", "peso_hidratado_g"], "pred_peso_hidratado_grado")
    _require_cols(dim, ["fecha_post", "destino", "factor_desp", "ajuste"], "dim_mermas_ajuste_fecha_post_destino")

    # Normalización
    df["fecha_post"] = _norm_date(df["fecha_post"])
    df["destino"] = df["destino"].astype(str).str.strip().str.upper()
    df["peso_hidratado_g"] = _to_num(df["peso_hidratado_g"]).fillna(0.0)

    dim["fecha_post"] = _norm_date(dim["fecha_post"])
    dim["destino"] = dim["destino"].astype(str).str.strip().str.upper()
    dim["factor_desp"] = _to_num(dim["factor_desp"])
    dim["ajuste"] = _to_num(dim["ajuste"])

    dim = dim[dim["fecha_post"].notna() & dim["destino"].notna()].copy()

    # --- Blindaje clave: colapsar dim a 1 fila por (fecha_post, destino)
    # Si vinieran duplicados, usamos mediana robusta.
    dim2 = (
        dim.groupby(["fecha_post", "destino"], dropna=False)
           .agg(
               factor_desp=("factor_desp", "median"),
               ajuste=("ajuste", "median"),
           )
           .reset_index()
    )

    # Merge
    out = df.merge(
        dim2[["fecha_post", "destino", "factor_desp", "ajuste"]],
        on=["fecha_post", "destino"],
        how="left",
    )

    miss = int(out["factor_desp"].isna().sum())
    if miss > 0:
        pct = miss / max(len(out), 1)
        print(f"[WARN] Sin match en dim para {miss} filas ({pct:.2%}). Se imputan medianas globales.")

    # Fallbacks globales si faltan días
    fd_med = float(np.nanmedian(dim2["factor_desp"].values)) if dim2["factor_desp"].notna().any() else 1.0
    aj_med = float(np.nanmedian(dim2["ajuste"].values)) if dim2["ajuste"].notna().any() else 1.0
    out["factor_desp"] = out["factor_desp"].fillna(fd_med)
    out["ajuste"] = out["ajuste"].fillna(aj_med)

    # Caps de seguridad
    out["factor_desp"] = out["factor_desp"].clip(0.05, 1.0)
    out["ajuste"] = out["ajuste"].clip(0.5, 2.0)

    # 1) aplicar desperdicio (factor)
    out["peso_post_desp_g"] = out["peso_hidratado_g"] * out["factor_desp"]

    # 2) aplicar ajuste: peso_final / ajuste (tu regla)
    out["peso_final_g"] = out["peso_post_desp_g"] / out["ajuste"]

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    # Mantener esquema actual; si falta alguna columna “informativa”, la creamos NaN para no reventar
    keep = [
        "ciclo_id",
        "fecha", "fecha_post", "dh_dias",
        "destino",
        "bloque", "bloque_padre",
        "variedad", "variedad_std",
        "tipo_sp", "area", "estado",
        "stage",
        "grado",
        "tallos_pred_grado",
        "peso_pred_g",
        "factor_hidr",
        "peso_hidratado_g",
        "factor_desp",
        "ajuste",
        "peso_post_desp_g",
        "peso_final_g",
        "created_at",
    ]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[keep].copy()

    out_path = preds_dir / "pred_peso_final_ajustado_grado.parquet"
    write_parquet(out, out_path)

    print(f"OK: pred_peso_final_ajustado_grado={len(out)} filas -> {out_path}")
    print("factor_desp describe:\n", out["factor_desp"].describe().to_string())
    print("ajuste describe:\n", out["ajuste"].describe().to_string())
    print("peso_final_g describe:\n", out["peso_final_g"].describe().to_string())


if __name__ == "__main__":
    main()
