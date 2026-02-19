from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet

MAESTRO_PATH = Path("data/silver/fact_ciclo_maestro.parquet")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _describe_series(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().empty:
        return "EMPTY"
    q = x.quantile([0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    desc = x.describe()
    lines = []
    lines.append(desc.to_string())
    lines.append("")
    lines.append("quantiles:")
    lines.append(q.to_string())
    return "\n".join(lines)


def main() -> None:
    df = read_parquet(MAESTRO_PATH).copy()
    print(f"OK read -> {MAESTRO_PATH} | rows={len(df):,}")

    # --- Requeridos según tu schema ---
    req = [
        "ciclo_id",
        "bloque_base",
        "variedad",
        "tipo_sp",
        "area",
        "fecha_sp",
        "fecha_inicio_cosecha",
        "fecha_fin_cosecha",
    ]
    _require(df, req, "fact_ciclo_maestro")

    # --- Canon ---
    df["ciclo_id"] = df["ciclo_id"].astype(str)
    df["bloque_base"] = _canon_int(df["bloque_base"])
    df["variedad"] = _canon_str(df["variedad"])
    df["tipo_sp"] = _canon_str(df["tipo_sp"])
    df["area"] = _canon_str(df["area"])

    df["fecha_sp"] = _to_date(df["fecha_sp"])
    df["harvest_start_real"] = _to_date(df["fecha_inicio_cosecha"])
    df["harvest_end_real"] = _to_date(df["fecha_fin_cosecha"])

    # --- Duplicados clave ---
    key = ["ciclo_id"]
    dup = int(df.duplicated(subset=key).sum())
    if dup > 0:
        print(f"[WARN] duplicated ciclo_id: {dup:,} (esto puede existir si hay varias líneas por ciclo; revisa)")
    else:
        print("[OK] ciclo_id único (sin duplicados)")

    # --- Coberturas ---
    cov = {
        "fecha_sp": float(df["fecha_sp"].notna().mean()),
        "harvest_start_real": float(df["harvest_start_real"].notna().mean()),
        "harvest_end_real": float(df["harvest_end_real"].notna().mean()),
        "bloque_base": float(df["bloque_base"].notna().mean()),
        "variedad": float(df["variedad"].notna().mean()),
        "tipo_sp": float(df["tipo_sp"].notna().mean()),
        "area": float(df["area"].notna().mean()),
    }
    print("\n[COVERAGE]")
    for k, v in cov.items():
        print(f"- {k}: {v:.4f}")

    # --- Consistencia temporal ---
    # (a) start >= sp
    m1 = df["fecha_sp"].notna() & df["harvest_start_real"].notna()
    bad_start_before_sp = float((df.loc[m1, "harvest_start_real"] < df.loc[m1, "fecha_sp"]).mean()) if m1.any() else np.nan

    # (b) end >= start
    m2 = df["harvest_start_real"].notna() & df["harvest_end_real"].notna()
    bad_end_before_start = float((df.loc[m2, "harvest_end_real"] < df.loc[m2, "harvest_start_real"]).mean()) if m2.any() else np.nan

    print("\n[CONSISTENCY]")
    print(f"- % harvest_start_real < fecha_sp: {bad_start_before_sp:.4f}")
    print(f"- % harvest_end_real < harvest_start_real: {bad_end_before_start:.4f}")

    # --- Targets ML1 ---
    # d_start_real = start - sp
    df["d_start_real"] = (df["harvest_start_real"] - df["fecha_sp"]).dt.days
    # n_harvest_days_real = end - start + 1
    df["n_harvest_days_real"] = (df["harvest_end_real"] - df["harvest_start_real"]).dt.days + 1

    print("\n[TARGET DISTRIBUTIONS: d_start_real (days)]")
    print(_describe_series(df["d_start_real"]))

    print("\n[TARGET DISTRIBUTIONS: n_harvest_days_real (days)]")
    print(_describe_series(df["n_harvest_days_real"]))

    # --- Outliers / casos imposibles ---
    # define tolerancias duras (ajústalas si lo deseas)
    # start negativo o demasiado alto
    m3 = df["d_start_real"].notna()
    bad_d_start_neg = float((df.loc[m3, "d_start_real"] < 0).mean()) if m3.any() else np.nan
    bad_d_start_huge = float((df.loc[m3, "d_start_real"] > 365).mean()) if m3.any() else np.nan

    m4 = df["n_harvest_days_real"].notna()
    bad_n_days_le0 = float((df.loc[m4, "n_harvest_days_real"] <= 0).mean()) if m4.any() else np.nan
    bad_n_days_huge = float((df.loc[m4, "n_harvest_days_real"] > 120).mean()) if m4.any() else np.nan

    print("\n[OUTLIERS]")
    print(f"- % d_start_real < 0: {bad_d_start_neg:.4f}")
    print(f"- % d_start_real > 365: {bad_d_start_huge:.4f}")
    print(f"- % n_harvest_days_real <= 0: {bad_n_days_le0:.4f}")
    print(f"- % n_harvest_days_real > 120: {bad_n_days_huge:.4f}")

    # --- Segmentación (clave para ver si “medianas” de 30 rows era normal o no) ---
    # Para cada (area,variedad,tipo_sp): conteo, mediana, IQR
    seg_cols = ["area", "variedad", "tipo_sp"]
    seg = (
        df.groupby(seg_cols, dropna=False, as_index=False)
          .agg(
              n=("ciclo_id", "nunique"),
              d_start_med=("d_start_real", "median"),
              d_start_p25=("d_start_real", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.25)),
              d_start_p75=("d_start_real", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.75)),
              n_days_med=("n_harvest_days_real", "median"),
              n_days_p25=("n_harvest_days_real", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.25)),
              n_days_p75=("n_harvest_days_real", lambda x: pd.to_numeric(x, errors="coerce").quantile(0.75)),
          )
          .sort_values(["n"], ascending=False)
    )

    print("\n[SEGMENT SUMMARY TOP 30 by n]")
    print(seg.head(30).to_string(index=False))

    # --- Leakage sanity: ¿cuántos ciclos tienen real completo? ---
    full = df["fecha_sp"].notna() & df["harvest_start_real"].notna() & df["harvest_end_real"].notna()
    print("\n[TRAINABLE ROWS]")
    print(f"- rows with full (sp,start,end): {int(full.sum()):,} / {len(df):,} = {float(full.mean()):.4f}")

    # --- Recomendación automática de baseline vs ML ---
    # Si trainable es muy bajo por segmento, el ML puede colapsar; te lo marca.
    low = seg[seg["n"] < 20]
    print("\n[WARN] segmentos con n < 20 (ML1 puede generalizar peor; usar regularización / pooling):")
    print(low.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
