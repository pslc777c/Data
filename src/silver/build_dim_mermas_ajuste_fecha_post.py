# src/silver/build_dim_mermas_ajuste_fecha_post.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

from common.io import write_parquet


def load_settings() -> dict:
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    p2a = bronze_dir / "balanza_2a_raw.parquet"
    p2 = bronze_dir / "balanza_2_raw.parquet"

    if not p2a.exists():
        raise FileNotFoundError(f"No existe Bronze: {p2a}. Ejecuta src/bronze/build_balanza_mermas_sources.py")
    if not p2.exists():
        raise FileNotFoundError(f"No existe Bronze: {p2}. Ejecuta src/bronze/build_balanza_mermas_sources.py")

    df2a = pd.read_parquet(p2a)
    df2 = pd.read_parquet(p2)

    df2a.columns = [str(c).strip() for c in df2a.columns]
    df2.columns = [str(c).strip() for c in df2.columns]

    # -------------------------
    # BALANZA 2A (AgrupadoFinal por fecha_post+destino)
    # -------------------------
    df2a["Fecha"] = _norm_date(df2a["Fecha"])
    df2a["Origen"] = df2a["Origen"].astype(str).str.strip()
    df2a["Seccion"] = df2a["Seccion"].astype(str).str.strip().str.upper()
    df2a["Variedad"] = df2a["Variedad"].astype(str).str.strip().str.upper()
    df2a["codigo_actividad"] = df2a["codigo_actividad"].astype(str).str.strip().str.lower()
    df2a["Grado"] = df2a["Grado"].astype(str).str.strip()

    base = df2a[
        (df2a["Fecha"].notna())
        & (df2a["Origen"] != "GV PELADO")
        & (df2a["Seccion"] == "CLASIFICACION")
        & (df2a["Variedad"] == "XLENCE")
        & (df2a["Fecha"] >= pd.Timestamp("2025-01-01"))
    ].copy()

    # ReemplazoActividad + filtro psmc
    base = base[base["codigo_actividad"] != "psmc"].copy()

    rep = {
        "cbx": "BLANCO",
        "cxlta1": "TINTURADO",
        "05cts": "TINTURADO",
        "0504gufdgu": "GUIRNALDA",
        "cxltarh": "ARCOIRIS",
    }
    base["codigo_actividad_reemplazo"] = base["codigo_actividad"].map(rep).fillna(base["codigo_actividad"].astype(str).str.upper())

    base["peso_balanza"] = _to_num(base["peso_balanza"])
    base["tallos"] = _to_num(base["tallos"])
    base["num_bunches"] = _to_num(base["num_bunches"])

    base["PESOKG"] = base["peso_balanza"] / 1000.0
    base["TALLOSTOTALES"] = base["tallos"] * base["num_bunches"]

    agr0 = (
        base.groupby(["Fecha", "codigo_actividad_reemplazo", "Grado"], dropna=False)
            .agg(
                Peso=("PESOKG", "sum"),
                Tallos=("TALLOSTOTALES", "sum"),
                bunches=("num_bunches", "sum"),
            )
            .reset_index()
            .rename(columns={"codigo_actividad_reemplazo": "codigo_actividad"})
    )

    grado = agr0["Grado"].astype(str).str.strip()
    is_pet = grado.eq("PET")
    is_bqt = grado.eq("BQT")
    has_gr = grado.str.contains("GR", na=False)

    agr0["Grado2"] = np.where(
        is_pet, "PET",
        np.where(is_bqt, "BQT",
                 np.where(has_gr, grado, grado + "GR"))
    )

    g2 = agr0["Grado2"].astype(str)
    num_gr = pd.to_numeric(
        g2.where(g2.str.contains("GR", na=False), np.nan)
          .str.split("GR").str[0],
        errors="coerce"
    )

    agr0["PESO_IDEAL"] = np.where(
        g2.str.contains("BQT", na=False), (agr0["Tallos"].astype(float) / 800.0) * 10.0,
        np.where(
            g2.str.contains("PET", na=False), (agr0["Tallos"].astype(float) / 1000.0) * 10.0,
            np.where(
                g2.str.contains("GUIRNALDA", na=False), agr0["Peso"],
                np.where(g2.str.contains("GR", na=False), num_gr, np.nan)
            )
        )
    )

    agr0["PESOIDEALTOTALKG"] = np.where(
        agr0["Grado2"].isin(["BQT", "PET", "GUIRNALDAGR"]),
        agr0["PESO_IDEAL"],
        (agr0["PESO_IDEAL"] * agr0["bunches"]) / 1000.0
    )

    a = (
        agr0.groupby(["Fecha", "codigo_actividad"], dropna=False)
            .agg(
                w2a_kg=("Peso", "sum"),
                wideal_kg=("PESOIDEALTOTALKG", "sum"),
            )
            .reset_index()
            .rename(columns={"Fecha": "fecha_post", "codigo_actividad": "destino"})
    )
    a["destino"] = a["destino"].astype(str).str.strip().str.upper()

    # -------------------------
    # BALANZA 2 (w2_kg por fecha_post+destino)
    # -------------------------
    df2["fecha_entrega"] = _norm_date(df2["fecha_entrega"])
    df2["Destino"] = df2["Destino"].astype(str).str.strip().str.upper()
    df2["variedad"] = df2["variedad"].astype(str).str.strip().str.upper()
    df2["tipo_pelado"] = df2["tipo_pelado"].astype(str).str.strip()
    df2["Origen"] = df2["Origen"].astype(str).str.strip().str.upper()

    prod = df2.get("producto")
    if prod is None:
        df2["producto"] = np.nan
        prod = df2["producto"]
    prod = prod.astype("string")

    b2 = df2[
        (df2["fecha_entrega"].notna())
        & (df2["variedad"] == "GYPXLE")
        & (df2["tipo_pelado"] == "Sin Pelar")
        & (df2["Origen"] == "APERTURA")
        & (prod.isna() | (prod.astype(str).str.upper().eq("GUIRNALDA")))
    ].copy()

    b2["destino"] = np.where(b2["Destino"].isin(["GUIRNALDA", "CLASIFICACION"]), "BLANCO", b2["Destino"])
    b2["peso_neto"] = _to_num(b2["peso_neto"])

    b = (
        b2.groupby(["fecha_entrega", "destino"], dropna=False)
           .agg(w2_kg=("peso_neto", "sum"))
           .reset_index()
           .rename(columns={"fecha_entrega": "fecha_post"})
    )
    b["destino"] = b["destino"].astype(str).str.strip().str.upper()

    # -------------------------
    # Join por fecha_post+destino y cálculo por destino
    # -------------------------
    m = a.merge(b, on=["fecha_post", "destino"], how="left")

    out0 = (
        m.groupby(["fecha_post", "destino"], dropna=False)
         .agg(
             w2_kg=("w2_kg", "sum"),
             w2a_kg=("w2a_kg", "sum"),
             wideal_kg=("wideal_kg", "sum"),
         )
         .reset_index()
         .sort_values(["fecha_post", "destino"], ascending=[False, True])
    )

    out0["fecha_post"] = _norm_date(out0["fecha_post"])
    for c in ["w2_kg", "w2a_kg", "wideal_kg"]:
        out0[c] = _to_num(out0[c])

    out0 = out0[out0["fecha_post"].notna()].copy()
    out0 = out0[out0["w2_kg"].notna() & (out0["w2_kg"] > 0)].copy()
    out0 = out0[out0["w2a_kg"].notna() & (out0["w2a_kg"] > 0)].copy()
    out0 = out0[out0["wideal_kg"].notna() & (out0["wideal_kg"] > 0)].copy()

    out0["desp_pct"] = (1.0 - (out0["w2a_kg"] / out0["w2_kg"])).clip(0.0, 0.95)
    out0["factor_desp"] = (out0["w2a_kg"] / out0["w2_kg"]).clip(0.05, 1.0)

    out0["ajuste"] = (out0["wideal_kg"] / out0["w2a_kg"]).clip(0.5, 2.0)
    out0["factor_ajuste"] = (out0["w2a_kg"] / out0["wideal_kg"]).clip(0.5, 2.0)

    # Mantener solo procesos relevantes
    keep_proc = {"BLANCO", "ARCOIRIS", "TINTURADO"}
    out0 = out0[out0["destino"].isin(sorted(keep_proc))].copy()

    out0["created_at"] = datetime.now().isoformat(timespec="seconds")

    out = out0[[
        "fecha_post", "destino",
        "w2_kg", "w2a_kg", "wideal_kg",
        "desp_pct", "factor_desp",
        "ajuste", "factor_ajuste",
        "created_at",
    ]].copy()

    out_path = silver_dir / "dim_mermas_ajuste_fecha_post_destino.parquet"
    write_parquet(out, out_path)

    print(f"OK: dim_mermas_ajuste_fecha_post_destino={len(out)} filas -> {out_path}")
    print("destinos:", out["destino"].value_counts().to_dict())
    print("desp_pct describe:\n", out["desp_pct"].describe().to_string())
    print("ajuste describe:\n", out["ajuste"].describe().to_string())


if __name__ == "__main__":
    main()
