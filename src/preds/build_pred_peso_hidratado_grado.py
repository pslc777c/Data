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


def _validate_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = int(df.duplicated(subset=keys).sum())
    if dup > 0:
        ex = df.loc[df.duplicated(subset=keys, keep=False), keys].head(20)
        raise ValueError(
            f"{name}: llaves no únicas (dup={dup}) para keys={keys}. Ejemplos:\n{ex.to_string(index=False)}"
        )


def main() -> None:
    cfg = load_settings()

    preds_dir = Path(cfg.get("paths", {}).get("preds", "data/preds"))
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    in_pred = preds_dir / "pred_peso_grado.parquet"
    dim_hidr = silver_dir / "dim_hidratacion_dh_grado_destino.parquet"
    fact_hidr = silver_dir / "fact_hidratacion_real_post_grado_destino.parquet"

    if not in_pred.exists():
        raise FileNotFoundError(f"No existe: {in_pred}. Ejecuta build_pred_peso_grado primero.")
    if not dim_hidr.exists():
        raise FileNotFoundError(f"No existe: {dim_hidr}. Ejecuta build_hidratacion_real_from_balanza2 primero.")

    pred = read_parquet(in_pred).copy()
    dim = read_parquet(dim_hidr).copy()

    pred.columns = [str(c).strip() for c in pred.columns]
    dim.columns = [str(c).strip() for c in dim.columns]

    # -------------------------
    # Normalizar pred
    # -------------------------
    if "fecha" not in pred.columns or "grado" not in pred.columns or "peso_pred_g" not in pred.columns:
        raise ValueError("pred_peso_grado: faltan columnas requeridas (fecha, grado, peso_pred_g)")

    pred["fecha"] = _norm_date(pred["fecha"])
    pred["grado"] = _to_num(pred["grado"]).astype("Int64")
    pred["peso_pred_g"] = _to_num(pred["peso_pred_g"]).fillna(0.0).astype(float)

    # -------------------------
    # Normalizar / colapsar dim hidratación
    # -------------------------
    req = {"dh_dias", "grado", "destino", "hidr_pct", "factor_hidr"}
    missing = req - set(dim.columns)
    if missing:
        raise ValueError(f"dim_hidratacion_dh_grado_destino: faltan columnas {sorted(missing)}")

    dim["dh_dias"] = _to_num(dim["dh_dias"]).astype("Int64")
    dim["grado"] = _to_num(dim["grado"]).astype("Int64")
    dim["destino"] = dim["destino"].astype(str).str.strip().str.upper()
    dim["hidr_pct"] = _to_num(dim["hidr_pct"])
    dim["factor_hidr"] = _to_num(dim["factor_hidr"])

    # colapsar por si hay duplicados
    dim2 = (
        dim.groupby(["dh_dias", "grado", "destino"], dropna=False)
           .agg(
               factor_hidr=("factor_hidr", "median"),
               hidr_pct=("hidr_pct", "median"),
           )
           .reset_index()
    )
    _validate_unique(dim2, ["dh_dias", "grado", "destino"], "dim_hidratacion_dh_grado_destino (colapsado)")

    # -------------------------
    # Definir DH y destino default
    # -------------------------
    post_cfg = cfg.get("post", {})
    dh_default = int(post_cfg.get("dh_default", 7))
    destino_default = str(post_cfg.get("destino_default", "APERTURA")).strip().upper()

    dh_mediana = None
    if fact_hidr.exists():
        fact = read_parquet(fact_hidr).copy()
        if "dh_dias" in fact.columns:
            fact["dh_dias"] = _to_num(fact["dh_dias"])
            if fact["dh_dias"].notna().any():
                dh_mediana = int(np.nanmedian(fact["dh_dias"].to_numpy()))

    dh_pred = dh_mediana if dh_mediana is not None else dh_default
    # clamp de seguridad para evitar locuras
    dh_pred = int(np.clip(dh_pred, 0, 30))

    # destino default: si no existe en dim, escoger el destino más frecuente
    destinos_dim = set(dim2["destino"].dropna().unique().tolist())
    if destino_default not in destinos_dim:
        # toma el modo (destino con más filas)
        destino_default = (
            dim2["destino"].value_counts(dropna=True).index[0]
            if len(dim2) else "APERTURA"
        )

    # -------------------------
    # Enriquecer pred con DH/destino/fecha_post
    # -------------------------
    pred["dh_dias"] = dh_pred
    pred["fecha_post"] = _norm_date(pred["fecha"] + pd.to_timedelta(pred["dh_dias"], unit="D"))
    pred["destino"] = destino_default

    # -------------------------
    # Join hidratación por dh+grado+destino
    # -------------------------
    out = pred.merge(
        dim2[["dh_dias", "grado", "destino", "hidr_pct", "factor_hidr"]],
        on=["dh_dias", "grado", "destino"],
        how="left",
    )

    # Fallback: mediana por (dh_dias, destino) ignorando grado
    if out["factor_hidr"].isna().any():
        fb = (
            dim2.groupby(["dh_dias", "destino"], dropna=False)["factor_hidr"]
               .median()
               .reset_index(name="factor_hidr_fb")
        )
        out = out.merge(fb, on=["dh_dias", "destino"], how="left")
        out["factor_hidr"] = out["factor_hidr"].fillna(out["factor_hidr_fb"])
        out["hidr_pct"] = out["hidr_pct"].fillna(out["factor_hidr"] - 1.0)
        out = out.drop(columns=["factor_hidr_fb"])

    # Fallback extremo
    out["factor_hidr"] = out["factor_hidr"].fillna(1.0).clip(0.5, 3.0)
    out["hidr_pct"] = out["hidr_pct"].fillna(out["factor_hidr"] - 1.0)

    out["peso_hidratado_g"] = out["peso_pred_g"] * out["factor_hidr"]

    out["created_at"] = datetime.now().isoformat(timespec="seconds")
    out["dh_source"] = "mediana_real" if dh_mediana is not None else "default"

    keep = [
        "ciclo_id",
        "fecha",
        "fecha_post",
        "dh_dias",
        "destino",
        "bloque", "bloque_padre",
        "variedad", "variedad_std",
        "tipo_sp", "area", "estado",
        "stage",
        "grado",
        "tallos_pred_grado",
        "peso_tallo_mediana_g",
        "peso_pred_g",
        "hidr_pct",
        "factor_hidr",
        "peso_hidratado_g",
        "dh_source",
        "created_at",
    ]
    keep = [c for c in keep if c in out.columns]
    out = out[keep].copy()

    out_path = preds_dir / "pred_peso_hidratado_grado.parquet"
    write_parquet(out, out_path)

    print(f"OK: pred_peso_hidratado_grado={len(out)} filas -> {out_path}")
    print(f"DH usado (pred): {dh_pred} dias (source={out['dh_source'].iloc[0]})")
    print(f"Destino default usado: {destino_default}")
    print("factor_hidr describe:\n", out["factor_hidr"].describe().to_string())
    print("peso_hidratado_g describe:\n", out["peso_hidratado_g"].describe().to_string())


if __name__ == "__main__":
    main()
