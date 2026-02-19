from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from common.io import read_parquet, write_parquet


IN_PRED = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")
IN_MAESTRO = Path("data/silver/fact_ciclo_maestro.parquet")

OUT_DIA = Path("data/gold/view_planificacion_campo_tallos_dia.parquet")
OUT_SEM = Path("data/gold/view_planificacion_campo_tallos_semana.parquet")
OUT_SEM_BLOQUE = Path("data/gold/view_planificacion_campo_tallos_semana_bloque.parquet")
OUT_SEM_AREA = Path("data/gold/view_planificacion_campo_tallos_semana_area.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _coalesce_area(df: pd.DataFrame) -> pd.DataFrame:
    if "area" in df.columns:
        df["area"] = _canon_str(df["area"]).replace({"NAN": np.nan, "NONE": np.nan})
        return df
    if "area_x" in df.columns or "area_y" in df.columns:
        ax = df["area_x"] if "area_x" in df.columns else pd.Series([pd.NA] * len(df))
        ay = df["area_y"] if "area_y" in df.columns else pd.Series([pd.NA] * len(df))
        df["area"] = ay.combine_first(ax)
        df = df.drop(columns=[c for c in ["area_x", "area_y"] if c in df.columns])
        df["area"] = _canon_str(df["area"]).replace({"NAN": np.nan, "NONE": np.nan})
        return df
    df["area"] = pd.Series([pd.NA] * len(df), dtype="object")
    return df


def _pick_cols(pred: pd.DataFrame) -> dict[str, str]:
    """
    Decide qué columnas usar (ML1 vs baseline).
    Devuelve dict con keys:
      - tallos_grado: columna tallos por grado/día
      - tallos_dia: columna tallos total/día (puede ser None)
      - share: columna share por grado (para reconstrucción si hace falta)
    """
    cols = set(pred.columns)

    # Preferimos ML1
    tallos_grado_ml1 = None
    for c in ["tallos_pred_grado_dia_ml1", "tallos_pred_grado_dia_ml1_full", "tallos_pred_grado_dia"]:
        if c in cols:
            # OJO: tallos_pred_grado_dia es ambiguo; lo validamos luego
            tallos_grado_ml1 = c
            break

    tallos_dia_ml1 = None
    for c in ["tallos_pred_dia_ml1", "tallos_pred_ml1_dia", "tallos_pred_dia"]:
        if c in cols:
            tallos_dia_ml1 = c
            break

    share_ml1 = None
    for c in ["share_grado_ml1", "share_ml1", "share_grado_pred"]:
        if c in cols:
            share_ml1 = c
            break

    # Baseline (opcional)
    share_base = None
    for c in ["share_grado_baseline", "share_baseline"]:
        if c in cols:
            share_base = c
            break

    return {
        "tallos_grado": tallos_grado_ml1,
        "tallos_dia": tallos_dia_ml1,
        "share_ml1": share_ml1,
        "share_base": share_base,
    }


def main() -> None:
    if not IN_PRED.exists():
        raise FileNotFoundError(f"No existe: {IN_PRED}")
    if not IN_MAESTRO.exists():
        raise FileNotFoundError(f"No existe: {IN_MAESTRO}")

    pred = read_parquet(IN_PRED).copy()
    maestro = read_parquet(IN_MAESTRO).copy()

    # Requeridos mínimos
    _require(pred, ["fecha", "bloque_base", "variedad_canon", "grado"], "pred_tallos_grado_dia_ml1_full")
    _require(maestro, ["bloque_base", "area"], "fact_ciclo_maestro")

    # Normalización tipos
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_int(pred["bloque_base"])
    pred["grado"] = _canon_int(pred["grado"])
    pred["variedad_canon"] = _canon_str(pred["variedad_canon"])

    maestro["bloque_base"] = _canon_int(maestro["bloque_base"])
    maestro["area"] = _canon_str(maestro["area"])

    # Merge área
    map_area = (
        maestro[["bloque_base", "area"]]
        .dropna(subset=["bloque_base"])
        .sort_values(["bloque_base", "area"])
        .drop_duplicates(subset=["bloque_base"], keep="first")
    )
    pred = pred.merge(map_area, on="bloque_base", how="left")
    pred = _coalesce_area(pred)
    pred["area"] = pred["area"].fillna("UNKNOWN")

    # Selección de columnas ML1/baseline
    sel = _pick_cols(pred)
    tg = sel["tallos_grado"]
    td = sel["tallos_dia"]
    sh = sel["share_ml1"]

    if tg is None and (td is None or sh is None):
        raise ValueError(
            "No tengo suficientes columnas para construir tallos por grado.\n"
            "Necesito (tallos_pred_grado_dia_ml1) o (tallos_pred_dia_ml1 + share_grado_ml1).\n"
            f"Columnas disponibles: {list(pred.columns)}"
        )

    # Si NO existe tallos por grado, lo reconstruimos
    if tg is None:
        pred[td] = pd.to_numeric(pred[td], errors="coerce")
        pred[sh] = pd.to_numeric(pred[sh], errors="coerce")
        pred["tallos_pred_grado_dia_ml1"] = pred[td] * pred[sh]
        tg = "tallos_pred_grado_dia_ml1"

    # Forzar numérico
    pred[tg] = pd.to_numeric(pred[tg], errors="coerce")

    # ---- DIAGNÓSTICO CLAVE: detectar “duplicado por grado” (lo que te pasó) ----
    # Si para un mismo grupo (fecha,bloque,variedad) hay muchos grados y tg es casi constante => tg NO es por grado.
    gchk = pred.groupby(["fecha", "bloque_base", "variedad_canon"], dropna=False)[tg].agg(["nunique", "count"])
    # Consideramos sospechoso si hay >1 fila (varios grados) pero solo 1 valor único
    suspect = ((gchk["count"] > 3) & (gchk["nunique"] <= 1)).mean()
    if suspect > 0.01:
        print(
            f"[WARN] La columna '{tg}' parece CONSTANTE entre grados en ~{suspect:.2%} de grupos "
            f"(eso indica que NO es tallos por grado). "
            f"Revisa el gold upstream: {IN_PRED.name}"
        )

    # Vista diaria (por grado)
    out_dia = (
        pred.groupby(["fecha", "area", "bloque_base", "variedad_canon", "grado"], dropna=False, as_index=False)
            .agg(tallos_pred_grado_dia=(tg, "sum"))
    )

    # Total diario: SI existe columna total diaria confiable, la usamos; sino sumamos por grado
    if td is not None:
        # usamos la primera por grupo (debería ser la misma para todos los grados)
        pred[td] = pd.to_numeric(pred[td], errors="coerce")
        base_total = (
            pred.groupby(["fecha", "area", "bloque_base", "variedad_canon"], dropna=False, as_index=False)
                .agg(tallos_pred_dia=(td, "first"))
        )
        out_dia = out_dia.merge(base_total, on=["fecha", "area", "bloque_base", "variedad_canon"], how="left")
    else:
        tot = (
            out_dia.groupby(["fecha", "area", "bloque_base", "variedad_canon"], dropna=False, as_index=False)
                  .agg(tallos_pred_dia=("tallos_pred_grado_dia", "sum"))
        )
        out_dia = out_dia.merge(tot, on=["fecha", "area", "bloque_base", "variedad_canon"], how="left")

    out_dia["created_at"] = pd.Timestamp.utcnow()
    out_dia = out_dia.sort_values(["fecha", "area", "bloque_base", "variedad_canon", "grado"]).reset_index(drop=True)

    write_parquet(out_dia, OUT_DIA)
    print(f"OK -> {OUT_DIA} | rows={len(out_dia):,}")

    # Vista semanal ISO
    iso = out_dia["fecha"].dt.isocalendar()
    out_dia["anio"] = iso.year.astype(int)
    out_dia["semana_iso"] = iso.week.astype(int)

    sem_grado = (
        out_dia.groupby(["anio", "semana_iso", "area", "bloque_base", "variedad_canon", "grado"], dropna=False, as_index=False)
              .agg(tallos_pred_grado_semana=("tallos_pred_grado_dia", "sum"))
    )
    sem_total = (
        sem_grado.groupby(["anio", "semana_iso", "area", "bloque_base", "variedad_canon"], dropna=False, as_index=False)
                 .agg(tallos_pred_semana=("tallos_pred_grado_semana", "sum"))
    )
    out_sem = sem_grado.merge(
        sem_total,
        on=["anio", "semana_iso", "area", "bloque_base", "variedad_canon"],
        how="left",
    )
    out_sem["created_at"] = pd.Timestamp.utcnow()
    out_sem = out_sem.sort_values(["anio", "semana_iso", "area", "bloque_base", "variedad_canon", "grado"]).reset_index(drop=True)

    write_parquet(out_sem, OUT_SEM)
    print(f"OK -> {OUT_SEM} | rows={len(out_sem):,}")
    # -------------------------
    # EXTRA 1) Semana por bloque (SIN grado)
    # -------------------------
    out_sem_bloque = (
        out_sem.groupby(["anio", "semana_iso", "area", "bloque_base", "variedad_canon"], dropna=False, as_index=False)
              .agg(tallos_pred_semana=("tallos_pred_grado_semana", "sum"))
    )
    out_sem_bloque["created_at"] = pd.Timestamp.utcnow()
    out_sem_bloque = out_sem_bloque.sort_values(
        ["anio", "semana_iso", "area", "bloque_base", "variedad_canon"]
    ).reset_index(drop=True)

    write_parquet(out_sem_bloque, OUT_SEM_BLOQUE)
    print(f"OK -> {OUT_SEM_BLOQUE} | rows={len(out_sem_bloque):,}")

    # -------------------------
    # EXTRA 2) Semana por área (TOTAL, sin bloque/variedad/grado)
    # -------------------------
    out_sem_area = (
        out_sem_bloque.groupby(["anio", "semana_iso", "area"], dropna=False, as_index=False)
                     .agg(tallos_pred_semana_area=("tallos_pred_semana", "sum"))
    )
    out_sem_area["created_at"] = pd.Timestamp.utcnow()
    out_sem_area = out_sem_area.sort_values(["anio", "semana_iso", "area"]).reset_index(drop=True)

    write_parquet(out_sem_area, OUT_SEM_AREA)
    print(f"OK -> {OUT_SEM_AREA} | rows={len(out_sem_area):,}")

    # Check: sum(grados)=total
    chk = out_dia.groupby(["fecha", "area", "bloque_base", "variedad_canon"], dropna=False).agg(
        s_grados=("tallos_pred_grado_dia", "sum"),
        s_total=("tallos_pred_dia", "first"),
    )
    mismatch = float((np.abs(chk["s_grados"] - chk["s_total"]) > 1e-6).mean())
    print(f"[CHECK] % mismatch (día): {mismatch:.4f}")
    
    # Check semanal: suma por bloque = suma por área
    chk_sem = out_sem_bloque.groupby(["anio", "semana_iso", "area"], dropna=False)["tallos_pred_semana"].sum()
    chk_area = out_sem_area.set_index(["anio", "semana_iso", "area"])["tallos_pred_semana_area"]
    chk = (chk_sem - chk_area).abs()
    mismatch_sem = float((chk > 1e-6).mean()) if len(chk) else 0.0
    print(f"[CHECK] % mismatch (semana bloque vs area): {mismatch_sem:.4f}")


if __name__ == "__main__":
    main()
