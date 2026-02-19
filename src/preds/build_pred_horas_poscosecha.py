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


def _week_id(fecha: pd.Series) -> pd.Series:
    # Tu lógica: +2 días y semana Sunday-first (%U) => YYWW
    fa = _norm_date(fecha) + pd.to_timedelta(2, unit="D")
    yy = fa.dt.year.astype(str).str[-2:]
    ww = fa.dt.strftime("%U").astype(int)
    return yy + ww.astype(str).str.zfill(2)


def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    return np.where(b == 0, np.nan, a / b)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _pick_pred_file(preds_dir: Path) -> Path:
    """
    Escoge el mejor candidato existente como input de demanda.
    (No depende de features; solo de preds ya construidos).
    """
    candidates = [
        preds_dir / "pred_peso_final_ajustado_grado.parquet",
        preds_dir / "pred_peso_final_grado_dia.parquet",
        preds_dir / "pred_peso_grado.parquet",
        preds_dir / "pred_oferta_grado.parquet",
        preds_dir / "pred_oferta_dia.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No encuentro parquet de pred en data/preds. Busqué:\n- "
        + "\n- ".join(str(x) for x in candidates)
    )


def _ensure_peso_final_g(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera/estandariza 'peso_final_g' en el dataframe de pred.
    Reglas:
      1) si existe peso_final_g => listo
      2) si existe peso_final_kg => peso_final_g = *1000
      3) si existe peso_hidratado_g + factor_desp + ajuste => peso_final_g = peso_hidratado_g * factor_desp / ajuste
      4) fallback a peso_pred_g / peso_real_g (si existe)
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def has(name: str) -> bool:
        return name.lower() in cols

    if has("peso_final_g"):
        df["peso_final_g"] = pd.to_numeric(df[cols["peso_final_g"]], errors="coerce").fillna(0.0)
        return df

    if has("peso_final_kg"):
        x = pd.to_numeric(df[cols["peso_final_kg"]], errors="coerce").fillna(0.0)
        df["peso_final_g"] = x * 1000.0
        return df

    if has("peso_hidratado_g") and has("factor_desp") and has("ajuste"):
        ph = pd.to_numeric(df[cols["peso_hidratado_g"]], errors="coerce").fillna(0.0)
        fd = pd.to_numeric(df[cols["factor_desp"]], errors="coerce").fillna(1.0)
        aj = pd.to_numeric(df[cols["ajuste"]], errors="coerce").replace(0, np.nan)
        df["peso_final_g"] = (ph * fd) / aj
        df["peso_final_g"] = df["peso_final_g"].fillna(0.0)
        return df

    if has("peso_pred_g"):
        df["peso_final_g"] = pd.to_numeric(df[cols["peso_pred_g"]], errors="coerce").fillna(0.0)
        return df

    if has("peso_real_g"):
        df["peso_final_g"] = pd.to_numeric(df[cols["peso_real_g"]], errors="coerce").fillna(0.0)
        return df

    raise ValueError("Pred: no pude derivar peso_final_g. Columnas disponibles: " + ", ".join(df.columns))


def _ensure_fecha_post(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    if "post_start" in cols:
        df["fecha_post"] = _norm_date(df[cols["post_start"]])
        return df
    if "fecha_post" in cols:
        df["fecha_post"] = _norm_date(df[cols["fecha_post"]])
        return df
    if "fecha" in cols:
        df["fecha_post"] = _norm_date(df[cols["fecha"]])
        return df
    raise ValueError("Pred: no encuentro post_start/fecha_post/fecha para derivar fecha_post")


def _pick_first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _get_baseline_kg_h_blanco(silver_dir: Path) -> tuple[float | None, str]:
    """
    Busca baseline kg/h persona para BLANCO en silver, con compatibilidad de nombres.
    Retorna (valor, source_name). Si no existe, (None, 'missing').
    """
    candidates = [
        silver_dir / "dim_baseline_capacidad_kg_h_persona.parquet",
        silver_dir / "dim_capacidad_baseline_proceso.parquet",
        silver_dir / "dim_capacidad_baseline_kg_h_persona.parquet",
        silver_dir / "dim_baseline_capacidad_proceso_kg_h_persona.parquet",
    ]
    p = _pick_first_existing(candidates)
    if p is None:
        return None, "missing"

    df = read_parquet(p).copy()
    df.columns = df.columns.str.strip()
    if "proceso" not in df.columns:
        raise ValueError(f"{p.name}: falta columna 'proceso'.")

    df["proceso"] = df["proceso"].astype(str).str.upper().str.strip()

    kg_col = None
    for c in ["kg_h_persona_mediana", "kg_h_persona", "kg_h_mediana", "kg_h"]:
        if c in df.columns:
            kg_col = c
            break
    if kg_col is None:
        raise ValueError(f"{p.name}: no encuentro columna kg/h persona. Columnas={list(df.columns)}")

    s = df.loc[df["proceso"].eq("BLANCO"), kg_col]
    if s.empty:
        raise ValueError(f"{p.name}: no hay fila proceso=BLANCO")

    return float(pd.to_numeric(s, errors="coerce").iloc[0]), p.name


def _get_baseline_tallos_h(silver_dir: Path, proceso: str) -> float:
    """
    Devuelve baseline de tallos/h/persona para un proceso.
    - Compatible con varios nombres de archivo (porque tu pipeline ya genera otros nombres).
    - Si no encuentra el proceso, hace fallback: OTROS -> mediana global.
    """
    candidates = [
        silver_dir / "dim_baseline_capacidad_tallos_h_persona.parquet",
        silver_dir / "dim_capacidad_baseline_tallos_proceso.parquet",   # <-- el que tú generas
        silver_dir / "dim_capacidad_baseline_tallos_h_persona.parquet",
    ]
    p = _pick_first_existing(candidates)
    if p is None:
        raise FileNotFoundError(
            "No existe baseline tallos/h en silver. Busqué:\n- " + "\n- ".join(str(x) for x in candidates)
        )

    df = read_parquet(p).copy()
    df.columns = df.columns.str.strip()

    if "proceso" not in df.columns:
        raise ValueError(f"{p.name}: falta columna 'proceso'. Columnas={list(df.columns)}")

    df["proceso"] = df["proceso"].astype(str).str.upper().str.strip()

    # localizar columna tallos/h
    col = None
    for c in ["tallos_h_persona_mediana", "tallos_h_persona", "tallos_h_mediana", "tallos_h"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"{p.name}: no encuentro columna tallos/h. Columnas={list(df.columns)}")

    df[col] = pd.to_numeric(df[col], errors="coerce")

    proc = proceso.upper().strip()
    s = df.loc[df["proceso"].eq(proc), col].dropna()
    if not s.empty:
        return float(s.iloc[0])

    # fallback 1: OTROS
    s2 = df.loc[df["proceso"].eq("OTROS"), col].dropna()
    if not s2.empty:
        _warn(f"{p.name}: no hay baseline para {proc}. Fallback: OTROS.")
        return float(s2.iloc[0])

    # fallback 2: mediana global
    gmed = float(np.nanmedian(df[col].values))
    if np.isnan(gmed) or gmed <= 0:
        raise ValueError(f"{p.name}: no hay baseline válido ni para {proc} ni global (todo NaN/<=0).")
    _warn(f"{p.name}: no hay baseline para {proc}. Fallback: mediana global={gmed:.3f}.")
    return gmed



# -------------------------
# Main
# -------------------------
def main() -> None:
    cfg = load_settings()
    silver_dir = Path(cfg["paths"]["silver"])
    preds_dir = Path(cfg["paths"]["preds"])
    preds_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Input demanda (desde PREDS)
    # -------------------------
    pred_path = _pick_pred_file(preds_dir)
    pred = read_parquet(pred_path).copy()
    pred.columns = pred.columns.str.strip()

    pred = _ensure_fecha_post(pred)
    pred = _ensure_peso_final_g(pred)

    _info(
        f"Input pred: {pred_path.name} filas={len(pred)} "
        f"rango fecha_post={pred['fecha_post'].min()} -> {pred['fecha_post'].max()}"
    )

    # Agregar demanda a día
    dem = (pred.groupby(["fecha_post"], dropna=False)
              .agg(peso_final_g_total=("peso_final_g", "sum"))
              .reset_index())
    dem = dem[dem["fecha_post"].notna()].copy()
    dem["kg_total"] = dem["peso_final_g_total"] / 1000.0
    dem["cajas_total"] = dem["kg_total"] / 10.0
    dem["Semana_Ventas"] = _week_id(dem["fecha_post"])

    # -------------------------
    # 2) Mix semanal (SILVER)
    # -------------------------
    mix_path = silver_dir / "dim_mix_proceso_semana.parquet"
    if not mix_path.exists():
        raise FileNotFoundError(f"No existe: {mix_path} (requerido)")

    mix = read_parquet(mix_path).copy()
    mix.columns = mix.columns.str.strip()

    if "Semana_Ventas" not in mix.columns and "semana_ventas" in mix.columns:
        mix = mix.rename(columns={"semana_ventas": "Semana_Ventas"})

    for c in ["W_Blanco", "W_Arcoiris", "W_Tinturado"]:
        if c not in mix.columns:
            raise ValueError(f"Mix: falta columna {c} en {mix_path}")

    seed = {
        "W_Blanco": float(pd.to_numeric(mix["W_Blanco"], errors="coerce").median()),
        "W_Arcoiris": float(pd.to_numeric(mix["W_Arcoiris"], errors="coerce").median()),
        "W_Tinturado": float(pd.to_numeric(mix["W_Tinturado"], errors="coerce").median()),
    }

    dem = dem.merge(
        mix[["Semana_Ventas", "W_Blanco", "W_Arcoiris", "W_Tinturado"]],
        on="Semana_Ventas",
        how="left",
    )

    dem["W_Blanco"] = pd.to_numeric(dem["W_Blanco"], errors="coerce").fillna(seed["W_Blanco"])
    dem["W_Arcoiris"] = pd.to_numeric(dem["W_Arcoiris"], errors="coerce").fillna(seed["W_Arcoiris"])
    dem["W_Tinturado"] = pd.to_numeric(dem["W_Tinturado"], errors="coerce").fillna(seed["W_Tinturado"])

    s = dem["W_Blanco"] + dem["W_Arcoiris"] + dem["W_Tinturado"]
    dem["W_Blanco"] = safe_div(dem["W_Blanco"], s)
    dem["W_Arcoiris"] = safe_div(dem["W_Arcoiris"], s)
    dem["W_Tinturado"] = safe_div(dem["W_Tinturado"], s)

    dem["kg_blanco"] = dem["kg_total"] * dem["W_Blanco"]
    dem["kg_arcoiris"] = dem["kg_total"] * dem["W_Arcoiris"]
    dem["kg_tinturado"] = dem["kg_total"] * dem["W_Tinturado"]

    # -------------------------
    # 3) Peso promedio (SILVER)
    # -------------------------
    peso_prom_path = silver_dir / "dim_peso_tallo_promedio_dia.parquet"
    if not peso_prom_path.exists():
        raise FileNotFoundError(f"No existe: {peso_prom_path} (requerido)")

    peso_prom = read_parquet(peso_prom_path).copy()
    peso_prom.columns = peso_prom.columns.str.strip()

    if "fecha" in peso_prom.columns:
        peso_prom["fecha_post"] = _norm_date(peso_prom["fecha"])
    elif "fecha_post" in peso_prom.columns:
        peso_prom["fecha_post"] = _norm_date(peso_prom["fecha_post"])
    else:
        raise ValueError("dim_peso_tallo_promedio_dia: falta fecha/fecha_post")

    if "peso_tallo_prom_g" not in peso_prom.columns:
        raise ValueError("dim_peso_tallo_promedio_dia: falta peso_tallo_prom_g")

    peso_seed_g = float(pd.to_numeric(peso_prom["peso_tallo_prom_g"], errors="coerce").median())
    peso_prom = peso_prom[["fecha_post", "peso_tallo_prom_g"]].drop_duplicates()

    out = dem.merge(peso_prom, on="fecha_post", how="left")
    out["peso_tallo_prom_g"] = pd.to_numeric(out["peso_tallo_prom_g"], errors="coerce").fillna(peso_seed_g)

    # -------------------------
    # 4) Capacidades baseline (SILVER)
    # -------------------------
    # BLANCO: intentar kg/h directo, si no existe usar fallback desde tallos/h
    kg_h_blanco, kg_h_src = _get_baseline_kg_h_blanco(silver_dir)
    if kg_h_blanco is None:
        _warn("No existe baseline kg/h para BLANCO en silver. Fallback: BLANCO kg/h = tallos_h(BLANCO)*peso_prom_g/1000.")
        tallos_h_blanco = _get_baseline_tallos_h(silver_dir, "BLANCO")
        out["kg_h_persona_blanco"] = (tallos_h_blanco * out["peso_tallo_prom_g"]) / 1000.0
        out["kg_h_blanco_source"] = "fallback_from_tallos_h"
    else:
        out["kg_h_persona_blanco"] = float(kg_h_blanco)
        out["kg_h_blanco_source"] = kg_h_src

    tallos_h_tint = _get_baseline_tallos_h(silver_dir, "TINTURADO")
    tallos_h_arc = _get_baseline_tallos_h(silver_dir, "ARCOIRIS")

    out["kg_h_persona_tinturado"] = (tallos_h_tint * out["peso_tallo_prom_g"]) / 1000.0
    out["kg_h_persona_arcoiris"] = (tallos_h_arc * out["peso_tallo_prom_g"]) / 1000.0

    # -------------------------
    # 5) Horas y personas
    # -------------------------
    out["horas_req_blanco"] = safe_div(out["kg_blanco"], out["kg_h_persona_blanco"])
    out["horas_req_tinturado"] = safe_div(out["kg_tinturado"], out["kg_h_persona_tinturado"])
    out["horas_req_arcoiris"] = safe_div(out["kg_arcoiris"], out["kg_h_persona_arcoiris"])
    out["horas_req_total"] = out["horas_req_blanco"] + out["horas_req_tinturado"] + out["horas_req_arcoiris"]

    horas_turno = float(cfg.get("pipeline", {}).get("horas_turno_poscosecha", 8.0))
    out["personas_req_blanco"] = safe_div(out["horas_req_blanco"], horas_turno)
    out["personas_req_tinturado"] = safe_div(out["horas_req_tinturado"], horas_turno)
    out["personas_req_arcoiris"] = safe_div(out["horas_req_arcoiris"], horas_turno)
    out["personas_req_total"] = safe_div(out["horas_req_total"], horas_turno)

    out["created_at"] = datetime.now().isoformat(timespec="seconds")

    keep = [
        "fecha_post", "Semana_Ventas",
        "kg_total", "cajas_total",
        "W_Blanco", "W_Arcoiris", "W_Tinturado",
        "kg_blanco", "kg_arcoiris", "kg_tinturado",
        "peso_tallo_prom_g",
        "kg_h_persona_blanco", "kg_h_persona_arcoiris", "kg_h_persona_tinturado",
        "kg_h_blanco_source",
        "horas_req_blanco", "horas_req_arcoiris", "horas_req_tinturado", "horas_req_total",
        "personas_req_blanco", "personas_req_arcoiris", "personas_req_tinturado", "personas_req_total",
        "created_at",
    ]

    out = out[keep].sort_values("fecha_post").reset_index(drop=True)

    out_path = preds_dir / "pred_horas_poscosecha_dia.parquet"
    write_parquet(out, out_path)

    _info(f"OK: pred_horas_poscosecha_dia={len(out)} filas -> {out_path}")
    _info(f"input_pred={pred_path.name}")
    _info(f"Seed mix (mediana): {seed}")
    _info(f"peso_tallo_prom_g seed: {peso_seed_g}")
    _info(f"horas_turno_poscosecha: {horas_turno}")
    _info(f"BLANCO baseline source: {out['kg_h_blanco_source'].iloc[0] if len(out) else 'NA'}")


if __name__ == "__main__":
    main()
