from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet

PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")
DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")

OUT_PATH = Path("data/gold/dim_cap_tallos_real_dia.parquet")

PCTL_MAIN = 0.99
PCTL_ALT = 0.95
MIN_CAP_FLOOR = 500.0     # seguridad mínima
CAP_FALLBACK_GLOBAL = 4000.0  # si todo falla (debería no usarse)

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")

def _load_var_map(dim_var: pd.DataFrame) -> dict[str, str]:
    _require(dim_var, ["variedad_raw", "variedad_canon"], "dim_variedad_canon")
    dv = dim_var.copy()
    dv["variedad_raw"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    dv = dv.dropna(subset=["variedad_raw", "variedad_canon"]).drop_duplicates(subset=["variedad_raw"])
    return dict(zip(dv["variedad_raw"], dv["variedad_canon"]))

def _q(s: pd.Series, q: float) -> float:
    a = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.quantile(a, q))

def main() -> None:
    for p in [PROG_PATH, DIM_VAR_PATH, FEATURES_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    prog = read_parquet(PROG_PATH).copy()
    dim_var = read_parquet(DIM_VAR_PATH).copy()
    feat = read_parquet(FEATURES_PATH).copy()

    var_map = _load_var_map(dim_var)

    _require(prog, ["ciclo_id", "fecha", "tallos_real_dia"], "prog")
    if ("variedad" not in prog.columns) and ("variedad_canon" not in prog.columns):
        raise ValueError("prog: falta variedad o variedad_canon")

    _require(feat, ["ciclo_id", "fecha", "variedad_canon"], "features")
    for c in ["area", "tipo_sp"]:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    # --- Canon prog
    prog["ciclo_id"] = prog["ciclo_id"].astype(str)
    prog["fecha"] = _to_date(prog["fecha"])
    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0)

    if "variedad" in prog.columns:
        prog["variedad_raw"] = _canon_str(prog["variedad"])
        prog["variedad_canon"] = prog["variedad_raw"].map(var_map).fillna(prog["variedad_raw"])
    else:
        prog["variedad_canon"] = _canon_str(prog["variedad_canon"])

    prog["variedad_canon"] = _canon_str(prog["variedad_canon"])

    # --- Canon features
    feat["ciclo_id"] = feat["ciclo_id"].astype(str)
    feat["fecha"] = _to_date(feat["fecha"])
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    feat["area"] = _canon_str(feat["area"].fillna("UNKNOWN"))
    feat["tipo_sp"] = _canon_str(feat["tipo_sp"].fillna("UNKNOWN"))

    # =============================================================================
    # 1) Agregar real por día (ciclo+fecha+variedad) para que el cap sea "físico"
    # =============================================================================
    real_day = (
        prog.groupby(["ciclo_id", "fecha", "variedad_canon"], dropna=False)["tallos_real_dia"]
        .sum()
        .reset_index()
    )

    # =============================================================================
    # 2) Adjuntar segmento (area/tipo_sp) sin usar bloque_base (evita el error)
    #    Si hay múltiples filas en features para el mismo (ciclo,fecha,variedad) tomamos el primero.
    # =============================================================================
    seg = (
        feat[["ciclo_id", "fecha", "variedad_canon", "area", "tipo_sp"]]
        .dropna(subset=["ciclo_id", "fecha", "variedad_canon"])
        .drop_duplicates(subset=["ciclo_id", "fecha", "variedad_canon"])
    )

    real_day = real_day.merge(seg, on=["ciclo_id", "fecha", "variedad_canon"], how="left")
    real_day["area"] = _canon_str(real_day["area"].fillna("UNKNOWN"))
    real_day["tipo_sp"] = _canon_str(real_day["tipo_sp"].fillna("UNKNOWN"))

    # Solo días con real>0
    x = real_day.loc[real_day["tallos_real_dia"] > 0, ["area", "tipo_sp", "variedad_canon", "tallos_real_dia"]].copy()
    if len(x) == 0:
        raise ValueError("No hay tallos_real_dia>0 para calcular caps.")

    # =============================================================================
    # 3) Caps por (area,tipo_sp,variedad) + fallback por variedad + fallback global
    # =============================================================================
    caps = (
        x.groupby(["area", "tipo_sp", "variedad_canon"], dropna=False)["tallos_real_dia"]
        .agg(
            n="size",
            cap_p99=lambda s: _q(s, PCTL_MAIN),
            cap_p95=lambda s: _q(s, PCTL_ALT),
            mean=lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )

    var_caps = (
        x.groupby(["variedad_canon"], dropna=False)["tallos_real_dia"]
        .agg(
            cap_var_p99=lambda s: _q(s, PCTL_MAIN),
            cap_var_p95=lambda s: _q(s, PCTL_ALT),
            mean_var=lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
    )

    global_cap = float(_q(x["tallos_real_dia"], PCTL_MAIN))
    if not np.isfinite(global_cap) or global_cap <= 0:
        global_cap = CAP_FALLBACK_GLOBAL

    caps = caps.merge(var_caps, on="variedad_canon", how="left")

    # Cap final: p99 del segmento; si no, p99 por variedad; si no, global; y piso
    cap_final = caps["cap_p99"]
    cap_final = cap_final.where(cap_final.notna(), caps["cap_var_p99"])
    cap_final = cap_final.fillna(global_cap)
    cap_final = pd.to_numeric(cap_final, errors="coerce").fillna(global_cap)
    cap_final = cap_final.clip(lower=MIN_CAP_FLOOR)

    caps["cap_dia"] = cap_final
    caps["cap_global_p99"] = global_cap
    caps["created_at"] = pd.Timestamp.utcnow()

    write_parquet(caps, OUT_PATH)
    print(f"OK -> {OUT_PATH} | rows={len(caps):,} | global_p99={global_cap:.1f}")

if __name__ == "__main__":
    main()
