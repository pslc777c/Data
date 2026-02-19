from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# -------------------------
# Paths (BASE = UNIVERSO ML1)
# -------------------------
IN_GRID = Path("data/gold/universe_harvest_grid_ml1.parquet")
IN_MAESTRO = Path("data/silver/fact_ciclo_maestro.parquet")

IN_PROG = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")
IN_CLIMA = Path("data/silver/dim_clima_bloque_dia.parquet")
IN_TERM = Path("data/silver/dim_estado_termico_cultivo_bloque_fecha.parquet")
IN_DIM_VAR = Path("data/silver/dim_variedad_canon.parquet")

OUT = Path("data/features/features_curva_cosecha_bloque_dia.parquet")


# -------------------------
# Helpers
# -------------------------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _prep_dim_var(dim_var: pd.DataFrame) -> pd.DataFrame:
    need = {"variedad_raw", "variedad_canon"}
    miss = need - set(dim_var.columns)
    if miss:
        raise ValueError(f"dim_variedad_canon.parquet sin columnas: {sorted(miss)}")
    dv = dim_var.copy()
    dv["variedad_raw_norm"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    return dv[["variedad_raw_norm", "variedad_canon"]].drop_duplicates()


def _attach_variedad_canon(df: pd.DataFrame, dv: pd.DataFrame, col_raw: str) -> pd.DataFrame:
    out = df.copy()
    out[col_raw] = _canon_str(out[col_raw])
    out = out.merge(dv, left_on=col_raw, right_on="variedad_raw_norm", how="left")
    out["variedad_canon"] = out["variedad_canon"].fillna(out[col_raw])
    return out.drop(columns=["variedad_raw_norm"], errors="ignore")


# -------------------------
# Main
# -------------------------
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    grid = read_parquet(IN_GRID).copy()
    maestro = read_parquet(IN_MAESTRO).copy()
    prog = read_parquet(IN_PROG).copy() if IN_PROG.exists() else pd.DataFrame()
    clima = read_parquet(IN_CLIMA).copy() if IN_CLIMA.exists() else pd.DataFrame()
    term = read_parquet(IN_TERM).copy() if IN_TERM.exists() else pd.DataFrame()
    dim_var = read_parquet(IN_DIM_VAR).copy()

    dv = _prep_dim_var(dim_var)

    # -------------------------
    # Base: universo ML1
    # -------------------------
    _require(grid, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")
    grid["ciclo_id"] = grid["ciclo_id"].astype(str)
    grid["fecha"] = _to_date(grid["fecha"])
    grid["bloque_base"] = _canon_int(grid["bloque_base"])
    grid["variedad_canon"] = _canon_str(grid["variedad_canon"])

    # quedarnos solo HARVEST por si acaso
    if "stage" in grid.columns:
        st = _canon_str(grid["stage"])
        grid = grid[st.eq("HARVEST")].copy()

    base_cols = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    # metadatos útiles si están en grid
    for c in ["area", "tipo_sp", "estado", "day_in_harvest_pred", "rel_pos_pred", "n_harvest_days_pred", "harvest_start_pred", "harvest_end_pred"]:
        if c in grid.columns:
            base_cols.append(c)

    base = grid[base_cols].drop_duplicates(subset=["ciclo_id", "fecha", "bloque_base", "variedad_canon"]).copy()

    # -------------------------
    # Maestro: tallos_proy (driver) + meta
    # -------------------------
    _require(maestro, ["ciclo_id", "tallos_proy"], "fact_ciclo_maestro")
    maestro["ciclo_id"] = maestro["ciclo_id"].astype(str)
    maestro["tallos_proy"] = pd.to_numeric(maestro["tallos_proy"], errors="coerce").fillna(0.0)

    m_take = ["ciclo_id", "tallos_proy"]
    for c in ["area", "tipo_sp", "estado", "variedad", "variedad_canon", "bloque_base"]:
        if c in maestro.columns:
            m_take.append(c)

    m2 = maestro[m_take].drop_duplicates("ciclo_id").copy()
    if "variedad_canon" not in m2.columns and "variedad" in m2.columns:
        m2 = _attach_variedad_canon(m2, dv, "variedad")
    if "variedad_canon" in m2.columns:
        m2["variedad_canon"] = _canon_str(m2["variedad_canon"])
    if "bloque_base" in m2.columns:
        m2["bloque_base"] = _canon_int(m2["bloque_base"])
    for c in ["area", "tipo_sp", "estado"]:
        if c in m2.columns:
            m2[c] = _canon_str(m2[c])

    df = base.merge(m2, on="ciclo_id", how="left", suffixes=("", "_m"))

    # coalesce meta desde maestro si grid no lo trae
    for c in ["area", "tipo_sp", "estado"]:
        if c not in df.columns and f"{c}_m" in df.columns:
            df[c] = df[f"{c}_m"]
        if f"{c}_m" in df.columns:
            df = df.drop(columns=[f"{c}_m"])

    # -------------------------
    # Baseline diario: uniforme por ciclo en el UNIVERSO
    # -------------------------
    cnt = df.groupby("ciclo_id", dropna=False)["fecha"].transform("count").astype(float)
    df["tallos_pred_baseline_dia"] = np.where(cnt > 0, df["tallos_proy"].astype(float) / cnt, 0.0)

    # -------------------------
    # Progreso real
    # -------------------------
    if not prog.empty:
        prog["fecha"] = _to_date(prog["fecha"])
        if "ciclo_id" in prog.columns:
            prog["ciclo_id"] = prog["ciclo_id"].astype(str)
        if "bloque_base" in prog.columns:
            prog["bloque_base"] = _canon_int(prog["bloque_base"])
        elif "bloque_padre" in prog.columns:
            prog["bloque_base"] = _canon_int(prog["bloque_padre"])
        elif "bloque" in prog.columns:
            prog["bloque_base"] = _canon_int(prog["bloque"])

        if "variedad_canon" not in prog.columns and "variedad" in prog.columns:
            prog = _attach_variedad_canon(prog, dv, "variedad")
        if "variedad_canon" in prog.columns:
            prog["variedad_canon"] = _canon_str(prog["variedad_canon"])
        else:
            prog["variedad_canon"] = "UNKNOWN"

        prog_take = [c for c in [
            "ciclo_id", "fecha", "bloque_base", "variedad_canon",
            "tallos_real_dia", "pct_avance_real", "dia_rel_cosecha_real",
            "en_ventana_cosecha_real", "gdc_acum_real",
        ] if c in prog.columns]

        prog2 = prog[prog_take].drop_duplicates(subset=["ciclo_id", "fecha", "bloque_base", "variedad_canon"])
        df = df.merge(prog2, on=["ciclo_id", "fecha", "bloque_base", "variedad_canon"], how="left")

    # -------------------------
    # Clima (por fecha + bloque_base)
    # -------------------------
    if not clima.empty:
        clima["fecha"] = _to_date(clima["fecha"])
        if "bloque_base" in clima.columns:
            clima["bloque_base"] = _canon_int(clima["bloque_base"])
        elif "bloque_padre" in clima.columns:
            clima["bloque_base"] = _canon_int(clima["bloque_padre"])
        elif "bloque" in clima.columns:
            clima["bloque_base"] = _canon_int(clima["bloque"])

        clima_take = [c for c in [
            "fecha", "bloque_base",
            "rainfall_mm_dia", "horas_lluvia", "en_lluvia_dia",
            "temp_avg_dia", "solar_energy_j_m2_dia",
            "wind_speed_avg_dia", "wind_run_dia", "gdc_dia",
        ] if c in clima.columns]

        clima2 = clima[clima_take].drop_duplicates(subset=["fecha", "bloque_base"])
        df = df.merge(clima2, on=["fecha", "bloque_base"], how="left")

    # -------------------------
    # Termal (por ciclo + fecha + bloque_base)
    # -------------------------
    if not term.empty:
        term["fecha"] = _to_date(term["fecha"])
        if "fecha_sp" in term.columns:
            term["fecha_sp"] = _to_date(term["fecha_sp"])
        if "ciclo_id" in term.columns:
            term["ciclo_id"] = term["ciclo_id"].astype(str)

        if "bloque_base" in term.columns:
            term["bloque_base"] = _canon_int(term["bloque_base"])
        elif "bloque_padre" in term.columns:
            term["bloque_base"] = _canon_int(term["bloque_padre"])
        elif "bloque" in term.columns:
            term["bloque_base"] = _canon_int(term["bloque"])

        term_take = [c for c in [
            "ciclo_id", "bloque_base", "fecha",
            "fecha_sp", "dias_desde_sp", "gdc_acum_desde_sp",
        ] if c in term.columns]

        term2 = term[term_take].drop_duplicates(subset=["ciclo_id", "bloque_base", "fecha"])
        if set(["ciclo_id", "bloque_base", "fecha"]).issubset(term2.columns):
            df = df.merge(term2, on=["ciclo_id", "bloque_base", "fecha"], how="left")

    # -------------------------
    # Calendario
    # -------------------------
    df["dow"] = df["fecha"].dt.dayofweek
    df["month"] = df["fecha"].dt.month
    df["weekofyear"] = df["fecha"].dt.isocalendar().week.astype(int)

    # -------------------------
    # Targets (si hay real)
    # -------------------------
    if "tallos_real_dia" in df.columns:
        df["tallos_real_dia"] = pd.to_numeric(df["tallos_real_dia"], errors="coerce")

    df["factor_tallos_dia"] = np.where(
        df["tallos_pred_baseline_dia"].fillna(0) > 0,
        df.get("tallos_real_dia", np.nan) / df["tallos_pred_baseline_dia"],
        np.nan,
    )
    df["factor_tallos_dia"] = pd.to_numeric(df["factor_tallos_dia"], errors="coerce")
    df["factor_tallos_dia_clipped"] = df["factor_tallos_dia"].clip(lower=0.2, upper=5.0)

    df["resid_tallos_dia"] = df.get("tallos_real_dia", np.nan) - df["tallos_pred_baseline_dia"]

    # -------------------------
    # Final + checks
    # -------------------------
    df["created_at"] = created_at

    k = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    dup_rate = float(df.duplicated(subset=k).mean()) if len(df) else 0.0
    if dup_rate > 0:
        raise ValueError(f"[FATAL] Duplicados en features_curva por {k}. dup_rate={dup_rate:.6f}")

    write_parquet(df.sort_values(["bloque_base", "variedad_canon", "fecha"]).reset_index(drop=True), OUT)
    print(f"OK -> {OUT} | rows={len(df):,} | fecha_min={df['fecha'].min().date()} fecha_max={df['fecha'].max().date()}")
    print(f"[CHECK] dup_rate features_curva por {k}: {dup_rate:.6f}")
    print(f"[COVERAGE] baseline notna: {float(df['tallos_pred_baseline_dia'].notna().mean()):.4f}")
    if "tallos_real_dia" in df.columns:
        print(f"[COVERAGE] real notna: {float(df['tallos_real_dia'].notna().mean()):.4f}")

    # Coverage vs universo (debe ser 1:1)
    n_univ = len(base)
    n_feat = len(df)
    if n_feat != n_univ:
        raise ValueError(f"[FATAL] features_curva rows != universe rows ({n_feat} != {n_univ}). Debe ser 1:1.")


if __name__ == "__main__":
    main()
