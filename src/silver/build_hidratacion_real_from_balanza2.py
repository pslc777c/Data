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


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _require_cols(df: pd.DataFrame, required: list[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name}: faltan columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def _normalize_grado_to_int(s: pd.Series) -> pd.Series:
    # Soporta "60", 60, "60GR", "60 GR", "BQT", "PET" -> PET/BQT quedarán NaN
    x = s.astype(str).str.upper().str.strip()
    # extrae primer número
    n = x.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(n, errors="coerce").astype("Int64")


def main() -> None:
    cfg = load_settings()

    bronze_dir = Path(cfg["paths"]["bronze"])
    silver_dir = Path(cfg["paths"]["silver"])
    silver_dir.mkdir(parents=True, exist_ok=True)

    p2 = bronze_dir / "balanza_2_raw.parquet"
    p1c = bronze_dir / "balanza_1c_raw.parquet"

    if not p2.exists():
        raise FileNotFoundError(f"No existe Bronze: {p2}")
    if not p1c.exists():
        raise FileNotFoundError(f"No existe Bronze: {p1c}. Ejecuta build_balanza_1c_raw.py")

    b2 = pd.read_parquet(p2)
    b1c = pd.read_parquet(p1c)

    b2.columns = [str(c).strip() for c in b2.columns]
    b1c.columns = [str(c).strip() for c in b1c.columns]

    _require_cols(
        b2,
        required=["fecha_entrega", "Lote", "Grado", "Destino", "Tallos", "peso_neto", "variedad", "tipo_pelado", "Origen"],
        df_name="balanza_2_raw",
    )
    _require_cols(
        b1c,
        required=["Fecha", "Destino", "Variedad", "Grado", "peso_balanza", "tallos"],
        df_name="balanza_1c_raw",
    )

    fecha_min = pd.to_datetime(cfg.get("hidratacion", {}).get("fecha_min", "2025-01-01"))

    # =========================
    # 1) BALANZA 2 (post)
    # =========================
    _info(f"Input b2: {len(b2)} filas")
    b2["fecha_post"] = _norm_date(b2["fecha_entrega"])
    b2["fecha_cosecha"] = _norm_date(b2["Lote"])

    b2["Destino"] = b2["Destino"].astype(str).str.strip().str.upper()
    b2["destino"] = np.where(b2["Destino"].isin(["GUIRNALDA", "CLASIFICACION"]), "BLANCO", b2["Destino"])

    b2["variedad"] = b2["variedad"].astype(str).str.strip().str.upper()
    b2["tipo_pelado"] = b2["tipo_pelado"].astype(str).str.strip()
    b2["Origen"] = b2["Origen"].astype(str).str.strip().str.upper()

    if "producto" not in b2.columns:
        b2["producto"] = np.nan
    prod = b2["producto"].astype("string")

    b2["grado"] = _normalize_grado_to_int(b2["Grado"])
    b2["tallos"] = _to_num(b2["Tallos"]).fillna(0.0)
    b2["peso_post_raw"] = _to_num(b2["peso_neto"])

    # Diagnóstico de dominios
    _info("b2 dominios:")
    _info(f"  fecha_post: {b2['fecha_post'].min()} -> {b2['fecha_post'].max()}")
    _info(f"  fecha_cosecha(Lote): {b2['fecha_cosecha'].min()} -> {b2['fecha_cosecha'].max()}")
    _info(f"  variedad top: {b2['variedad'].value_counts(dropna=False).head(5).to_dict()}")
    _info(f"  tipo_pelado top: {b2['tipo_pelado'].value_counts(dropna=False).head(5).to_dict()}")
    _info(f"  Origen top: {b2['Origen'].value_counts(dropna=False).head(5).to_dict()}")

    # Filtros negocio (como tu SQL)
    m2 = (
        (b2["fecha_post"].notna())
        & (b2["fecha_cosecha"].notna())
        & (b2["fecha_post"] >= fecha_min)
        & (b2["grado"].notna())
        & (b2["variedad"].eq("GYPXLE"))
        & (b2["tipo_pelado"].eq("Sin Pelar"))
        & (b2["Origen"].eq("APERTURA"))
        & (prod.isna() | (prod.astype(str).str.upper().eq("GUIRNALDA")))
        & (b2["tallos"] > 0)
        & (b2["peso_post_raw"].notna() & (b2["peso_post_raw"] > 0))
    )

    b2f = b2[m2].copy()
    _info(f"b2 filtrado (SQL-like): {len(b2f)} filas")

    if b2f.empty:
        _warn("b2f quedó vacío. Revisa filtros: variedad/tipo_pelado/Origen/producto.")
        # No seguimos para que no te dé todo NaN sin explicación.
        return

    # --- Auto-unidad peso_neto ---
    # Heurística: si el "peso por tallo" implícito queda demasiado alto, es probable que esté en gramos.
    # (peso_post_raw / tallos) debería ser ~0.005 a 0.1 kg/tallo típico (5g a 100g)
    ratio = (b2f["peso_post_raw"].astype(float) / b2f["tallos"].astype(float)).replace([np.inf, -np.inf], np.nan)
    med_ratio = float(np.nanmedian(ratio.values))

    # Si med_ratio > 1.0 => probablemente gramos (porque sería >1 kg por tallo, absurdo)
    # Si med_ratio < 0.001 => también raro
    if med_ratio > 1.0:
        _warn(f"peso_neto parece estar en GRAMOS (med peso/tallo={med_ratio:.3f}). Se convierte a kg (/1000).")
        b2f["peso_post_kg"] = b2f["peso_post_raw"] / 1000.0
    else:
        b2f["peso_post_kg"] = b2f["peso_post_raw"]

    _info(f"peso_post_kg describe:\n{b2f['peso_post_kg'].describe().to_string()}")

    b2g = (
        b2f.groupby(["fecha_post", "fecha_cosecha", "grado", "destino"], dropna=False)
           .agg(
               tallos=("tallos", "sum"),
               peso_post_kg=("peso_post_kg", "sum"),
           )
           .reset_index()
    )
    _info(f"b2 agregado: {len(b2g)} filas")

    # =========================
    # 2) BALANZA 1C (base cosecha)
    # =========================
    _info(f"Input b1c: {len(b1c)} filas")

    b1c["fecha_cosecha"] = _norm_date(b1c["Fecha"])
    b1c["Destino"] = b1c["Destino"].astype(str).str.strip().str.upper()
    b1c["Variedad"] = b1c["Variedad"].astype(str).str.strip().str.upper()

    b1c["grado"] = _normalize_grado_to_int(b1c["Grado"])
    b1c["peso_balanza_raw"] = _to_num(b1c["peso_balanza"])
    b1c["tallos"] = _to_num(b1c["tallos"]).fillna(0.0)

    _info("b1c dominios:")
    _info(f"  fecha_cosecha: {b1c['fecha_cosecha'].min()} -> {b1c['fecha_cosecha'].max()}")
    _info(f"  Destino top: {b1c['Destino'].value_counts(dropna=False).head(5).to_dict()}")
    _info(f"  Variedad top: {b1c['Variedad'].value_counts(dropna=False).head(5).to_dict()}")

    m1 = (
        (b1c["fecha_cosecha"].notna())
        & (b1c["fecha_cosecha"] >= fecha_min)
        & (b1c["Destino"].eq("APERTURA"))
        & (b1c["Variedad"].isin(["GYPXLE", "XLENCE"]))
        & (b1c["grado"].notna())
        & (b1c["peso_balanza_raw"].notna() & (b1c["peso_balanza_raw"] > 0))
        & (b1c["tallos"] > 0)
    )
    b1cf = b1c[m1].copy()
    _info(f"b1c filtrado (SQL-like): {len(b1cf)} filas")

    if b1cf.empty:
        _warn("b1cf quedó vacío. Revisa Destino='APERTURA' y Variedad in (GYPXLE, XLENCE).")
        return

    b1g = (
        b1cf.groupby(["fecha_cosecha", "grado"], dropna=False)
            .agg(
                peso_balanza_sum=("peso_balanza_raw", "sum"),
                tallos_sum=("tallos", "sum"),
            )
            .reset_index()
    )

    # --- Auto-unidad peso_balanza ---
    # Si peso_balanza es KG: (kg*1000)/tallos ~ 5-80 g/tallo típico.
    # Si ya es GR: (gr)/tallos ~ 5-80 g/tallo típico.
    peso_g_as_kg = (b1g["peso_balanza_sum"] * 1000.0) / b1g["tallos_sum"]
    peso_g_as_g = (b1g["peso_balanza_sum"]) / b1g["tallos_sum"]

    med_as_kg = float(np.nanmedian(peso_g_as_kg.values))
    med_as_g = float(np.nanmedian(peso_g_as_g.values))

    # Elegimos la opción cuyo valor mediano cae en rango plausible (2g a 200g)
    def in_plausible(x: float) -> bool:
        return (x >= 2.0) and (x <= 200.0)

    if in_plausible(med_as_g) and not in_plausible(med_as_kg):
        _warn(f"peso_balanza parece estar en GRAMOS (med g/tallo={med_as_g:.2f}). No se multiplica por 1000.")
        b1g["peso_por_tallo_g"] = peso_g_as_g
    elif in_plausible(med_as_kg) and not in_plausible(med_as_g):
        _info(f"peso_balanza parece estar en KILOS (med g/tallo={med_as_kg:.2f}). Se multiplica por 1000.")
        b1g["peso_por_tallo_g"] = peso_g_as_kg
    else:
        # Ambiguo: por defecto asumimos kg (tu supuesto original), pero lo avisamos
        _warn(
            f"Unidad de peso_balanza ambigua. med_as_kg={med_as_kg:.2f}, med_as_g={med_as_g:.2f}. "
            "Se asume KILOS por defecto. Si sale mal, fuerza en settings.yaml."
        )
        b1g["peso_por_tallo_g"] = peso_g_as_kg

    _info(f"peso_por_tallo_g describe:\n{b1g['peso_por_tallo_g'].describe().to_string()}")
    b1g = b1g[["fecha_cosecha", "grado", "peso_por_tallo_g"]].copy()

    # =========================
    # 3) JOIN + HIDR
    # =========================
    # Diagnóstico de cobertura antes del join: cuántas llaves coinciden
    keys_b2 = b2g[["fecha_cosecha", "grado"]].drop_duplicates()
    keys_b1 = b1g[["fecha_cosecha", "grado"]].drop_duplicates()

    inter = keys_b2.merge(keys_b1, on=["fecha_cosecha", "grado"], how="inner")
    _info(f"Keys b2 (fecha_cosecha,grado): {len(keys_b2)}; Keys b1: {len(keys_b1)}; Intersección: {len(inter)}")

    if inter.empty:
        _warn(
            "No hay intersección de llaves (fecha_cosecha, grado) entre balanza_2 y balanza_1c.\n"
            "Revisa: (1) Lote realmente es la fecha de cosecha, (2) grado viene comparable, (3) rangos de fechas."
        )
        # Te dejo muestras rápidas para inspección
        _info("Muestras b2 keys:\n" + keys_b2.head(10).to_string(index=False))
        _info("Muestras b1 keys:\n" + keys_b1.head(10).to_string(index=False))
        return

    m = b2g.merge(b1g, on=["fecha_cosecha", "grado"], how="left")
    miss = int(m["peso_por_tallo_g"].isna().sum())
    _info(f"Post-join: filas={len(m)}; sin match peso_por_tallo_g={miss}")

    m = m[m["peso_por_tallo_g"].notna() & (m["peso_por_tallo_g"] > 0)].copy()
    if m.empty:
        _warn("Después del join, todo quedó sin peso_por_tallo_g. Es mismatch total o nulls.")
        return

    m["peso_base_kg"] = (m["peso_por_tallo_g"] * m["tallos"]) / 1000.0
    m["hidr_pct"] = (m["peso_post_kg"] / m["peso_base_kg"]) - 1.0
    m["dh_dias"] = (m["fecha_post"] - m["fecha_cosecha"]).dt.days.astype("Int64")

    _info(f"hidr_pct BEFORE caps:\n{m['hidr_pct'].describe().to_string()}")
    _info(f"dh_dias BEFORE caps:\n{m['dh_dias'].dropna().astype(int).describe().to_string()}")

    # Caps seguridad
    m = m[m["dh_dias"].between(0, 30)].copy()
    m = m[m["hidr_pct"].between(-0.2, 3.0)].copy()

    _info(f"Después caps: {len(m)} filas")

    if m.empty:
        _warn(
            "Todo se eliminó por caps de dh_dias/hidr_pct. "
            "Probable problema de unidades (peso) o Lote incorrecto."
        )
        return

    m["peso_base_g"] = m["peso_base_kg"] * 1000.0
    m["peso_post_g"] = m["peso_post_kg"] * 1000.0

    fact = m[[
        "fecha_cosecha", "fecha_post", "dh_dias",
        "grado", "destino",
        "tallos",
        "peso_base_g", "peso_post_g",
        "hidr_pct",
    ]].copy()

    fact["created_at"] = datetime.now().isoformat(timespec="seconds")
    out_fact = silver_dir / "fact_hidratacion_real_post_grado_destino.parquet"
    write_parquet(fact, out_fact)

    # =========================
    # 4) Dims (ponderado) sin apply
    # =========================
    fact["_w"] = _to_num(fact["peso_base_g"]).fillna(0.0)
    fact["_hydr_w"] = _to_num(fact["hidr_pct"]) * fact["_w"]

    dim_dh = (
        fact.groupby(["dh_dias", "grado", "destino"], dropna=False)
            .agg(
                n=("hidr_pct", "size"),
                tallos=("tallos", "sum"),
                peso_base_g=("peso_base_g", "sum"),
                sum_hydr_w=("_hydr_w", "sum"),
                sum_w=("_w", "sum"),
            )
            .reset_index()
    )
    dim_dh["hidr_pct"] = np.where(dim_dh["sum_w"] > 0, dim_dh["sum_hydr_w"] / dim_dh["sum_w"], np.nan)
    dim_dh = dim_dh.drop(columns=["sum_hydr_w", "sum_w"])
    dim_dh["factor_hidr"] = 1.0 + dim_dh["hidr_pct"]
    dim_dh["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_dim_dh = silver_dir / "dim_hidratacion_dh_grado_destino.parquet"
    write_parquet(dim_dh, out_dim_dh)

    dim_fecha = (
        fact.groupby(["fecha_post", "grado", "destino"], dropna=False)
            .agg(
                n=("hidr_pct", "size"),
                tallos=("tallos", "sum"),
                peso_base_g=("peso_base_g", "sum"),
                sum_hydr_w=("_hydr_w", "sum"),
                sum_w=("_w", "sum"),
            )
            .reset_index()
    )
    dim_fecha["hidr_pct"] = np.where(dim_fecha["sum_w"] > 0, dim_fecha["sum_hydr_w"] / dim_fecha["sum_w"], np.nan)
    dim_fecha = dim_fecha.drop(columns=["sum_hydr_w", "sum_w"])
    dim_fecha["factor_hidr"] = 1.0 + dim_fecha["hidr_pct"]
    dim_fecha["created_at"] = datetime.now().isoformat(timespec="seconds")

    out_dim_fecha = silver_dir / "dim_hidratacion_fecha_post_grado_destino.parquet"
    write_parquet(dim_fecha, out_dim_fecha)

    _info(f"OK: fact_hidratacion_real_post_grado_destino={len(fact)} -> {out_fact}")
    _info(f"OK: dim_hidratacion_dh_grado_destino={len(dim_dh)} -> {out_dim_dh}")
    _info(f"OK: dim_hidratacion_fecha_post_grado_destino={len(dim_fecha)} -> {out_dim_fecha}")
    _info("hidr_pct FINAL describe:\n" + fact["hidr_pct"].describe().to_string())


if __name__ == "__main__":
    main()
