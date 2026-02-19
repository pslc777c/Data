from __future__ import annotations

from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet

# =============================================================================
# Paths
# =============================================================================
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")  # real tallos dia
DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")  # <-- NUEVO (canon prog -> canon)

OUT_PATH = Path("data/features/trainset_curva_beta_params.parquet")
DBG_PANEL_PATH = Path("data/features/_debug_panel_beta_params.parquet")

# =============================================================================
# Config
# =============================================================================
EPS = 1e-12
REL_CLIP = 1e-4
MIN_REAL_TOTAL = 10.0  # <-- BAJADO para no matar el entrenamiento; ajusta luego con evidencia
GRID_K = 7
GRID_STEP = 0.12

# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _canon_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Index(df.columns.astype(str))
    if cols.is_unique:
        return df
    out = df.copy()
    seen: dict[str, list[int]] = {}
    for i, c in enumerate(out.columns.astype(str)):
        seen.setdefault(c, []).append(i)
    keep: dict[str, pd.Series] = {}
    for c, idxs in seen.items():
        s = out.iloc[:, idxs[0]]
        for j in idxs[1:]:
            s2 = out.iloc[:, j]
            s = s.where(s.notna(), s2)
        keep[c] = s
    ordered: list[str] = []
    for c in out.columns.astype(str):
        if c not in ordered:
            ordered.append(c)
    return pd.DataFrame({c: keep[c] for c in ordered})

def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    if out_col in df.columns:
        base = df[out_col]
    else:
        base = pd.Series([pd.NA] * len(df), index=df.index)
    for c in candidates:
        if c in df.columns:
            base = base.where(base.notna(), df[c])
    df[out_col] = base

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

def _log_beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - logB

def _beta_share_on_grid(rel: np.ndarray, a: float, b: float) -> np.ndarray:
    rel = np.clip(rel, REL_CLIP, 1.0 - REL_CLIP)
    lp = _log_beta_pdf(rel, a, b)
    lp = lp - np.max(lp)
    p = np.exp(lp)
    s = float(np.sum(p))
    if s <= 0:
        return np.zeros_like(p)
    return p / s

def _fit_beta_params_discrete(rel: np.ndarray, share_real: np.ndarray) -> tuple[float, float, dict]:
    w = np.clip(share_real.astype(float), 0.0, None)
    sw = float(np.sum(w))
    if sw <= 0:
        return (2.0, 2.0, {"fit_ok": False, "reason": "zero_share"})

    w = w / sw
    x = np.clip(rel.astype(float), REL_CLIP, 1.0 - REL_CLIP)

    mu = float(np.sum(w * x))
    var = float(np.sum(w * (x - mu) ** 2))
    var = max(var, 1e-6)

    k = mu * (1.0 - mu) / var - 1.0
    if not np.isfinite(k) or k <= 0:
        a0, b0 = 2.0, 2.0
    else:
        a0 = mu * k
        b0 = (1.0 - mu) * k

    a0 = float(max(a0, 1.05))
    b0 = float(max(b0, 1.05))

    def nll(a: float, b: float) -> float:
        pb = _beta_share_on_grid(x, a, b)
        pb = np.clip(pb, EPS, 1.0)
        return float(-np.sum(w * np.log(pb)))

    best_a, best_b = a0, b0
    best = nll(best_a, best_b)

    la0, lb0 = math.log(best_a), math.log(best_b)
    for ia in range(-GRID_K, GRID_K + 1):
        for ib in range(-GRID_K, GRID_K + 1):
            a = math.exp(la0 + ia * GRID_STEP)
            b = math.exp(lb0 + ib * GRID_STEP)
            if a <= 1.0 or b <= 1.0:
                continue
            val = nll(a, b)
            if val < best:
                best = val
                best_a, best_b = a, b

    info = {"fit_ok": True, "mu": mu, "var": var, "a0_mom": a0, "b0_mom": b0, "a": float(best_a), "b": float(best_b), "nll": float(best)}
    return float(best_a), float(best_b), info

def _agg_cycle_features(feat_h: pd.DataFrame) -> pd.Series:
    out = {}
    for c in [
        "rainfall_mm_dia",
        "horas_lluvia",
        "temp_avg_dia",
        "solar_energy_j_m2_dia",
        "wind_speed_avg_dia",
        "wind_run_dia",
        "gdc_dia",
    ]:
        if c in feat_h.columns:
            v = pd.to_numeric(feat_h[c], errors="coerce")
            out[f"{c}__mean"] = float(v.mean(skipna=True)) if len(v) else np.nan
            out[f"{c}__sum"] = float(v.sum(skipna=True)) if len(v) else np.nan

    for c in ["dias_desde_sp", "gdc_acum_desde_sp"]:
        if c in feat_h.columns:
            v = pd.to_numeric(feat_h[c], errors="coerce")
            out[f"{c}__last"] = float(v.dropna().iloc[-1]) if v.notna().any() else np.nan
            out[f"{c}__mean"] = float(v.mean(skipna=True)) if len(v) else np.nan

    if "n_harvest_days" in feat_h.columns:
        v = pd.to_numeric(feat_h["n_harvest_days"], errors="coerce")
        out["n_harvest_days"] = float(v.max(skipna=True)) if v.notna().any() else np.nan

    if "tallos_proy" in feat_h.columns:
        v = pd.to_numeric(feat_h["tallos_proy"], errors="coerce")
        out["tallos_proy"] = float(v.max(skipna=True)) if v.notna().any() else 0.0

    return pd.Series(out)

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    created_at = pd.Timestamp.utcnow()

    for p in [FEATURES_PATH, UNIVERSE_PATH, PROG_PATH, DIM_VAR_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    feat = _dedupe_columns(read_parquet(FEATURES_PATH).copy())
    uni = read_parquet(UNIVERSE_PATH).copy()
    prog = read_parquet(PROG_PATH).copy()
    dim_var = read_parquet(DIM_VAR_PATH).copy()
    var_map = _load_var_map(dim_var)

    # Reqs mínimos
    _require(feat, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "features")
    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe")
    _require(prog, ["ciclo_id", "fecha", "bloque_base"], "prog")
    if "tallos_real_dia" not in prog.columns:
        raise ValueError("prog: falta tallos_real_dia")
    if ("variedad" not in prog.columns) and ("variedad_canon" not in prog.columns):
        raise ValueError("prog: falta variedad o variedad_canon")

    # Canon ids
    for df in (feat, uni, prog):
        df["ciclo_id"] = df["ciclo_id"].astype(str)
        df["fecha"] = _to_date(df["fecha"])
        df["bloque_base"] = _canon_int(df["bloque_base"])

    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    # Canon prog variedad con dim_var (esta era la causa típica)
    if "variedad" in prog.columns:
        prog["variedad_raw"] = _canon_str(prog["variedad"])
        prog["variedad_canon"] = prog["variedad_raw"].map(var_map).fillna(prog["variedad_raw"])
    else:
        prog["variedad_canon"] = _canon_str(prog["variedad_canon"])

    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Harvest cols en features
    _coalesce_cols(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce_cols(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])
    feat["day_in_harvest"] = pd.to_numeric(feat["day_in_harvest"], errors="coerce")
    feat["n_harvest_days"] = pd.to_numeric(feat["n_harvest_days"], errors="coerce")

    # Universe keys
    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    # Panel universe + features + prog
    feat_cols = [
        "day_in_harvest", "n_harvest_days", "tallos_proy", "area", "tipo_sp",
        "rainfall_mm_dia", "horas_lluvia", "temp_avg_dia", "solar_energy_j_m2_dia",
        "wind_speed_avg_dia", "wind_run_dia", "gdc_dia", "dias_desde_sp", "gdc_acum_desde_sp",
    ]
    feat_take = [c for c in (key + feat_cols) if c in feat.columns]

    panel = (
        uni_k
        .merge(feat[feat_take], on=key, how="left")
        .merge(prog[key + ["tallos_real_dia"]].drop_duplicates(subset=key), on=key, how="left")
    )

    # Diagnóstico join prog
    panel["tallos_real_dia"] = pd.to_numeric(panel["tallos_real_dia"], errors="coerce").fillna(0.0)
    n_total = len(panel)
    n_pos = int((panel["tallos_real_dia"] > 0).sum())
    s_tot = float(panel["tallos_real_dia"].sum())
    print(f"[DBG] panel rows={n_total:,} | rows con tallos_real_dia>0: {n_pos:,} ({(n_pos/max(n_total,1))*100:.2f}%) | sum_real={s_tot:.1f}")

    # Diagnóstico harvest mask
    dih = pd.to_numeric(panel["day_in_harvest"], errors="coerce")
    nh = pd.to_numeric(panel["n_harvest_days"], errors="coerce")
    is_h = dih.notna() & nh.notna() & (dih >= 1) & (nh >= 1) & (dih <= nh)
    print(f"[DBG] is_harvest rows={int(is_h.sum()):,} ({(float(is_h.mean())*100):.2f}%) | day_in_harvest notna={(dih.notna().mean()*100):.2f}% | n_harvest_days notna={(nh.notna().mean()*100):.2f}%")

    panel = panel[is_h].copy()
    if len(panel) == 0:
        write_parquet(panel, DBG_PANEL_PATH)
        raise ValueError("No hay filas harvest (day_in_harvest/n_harvest_days). Revisa features_curva y universe.")

    # Totales reales por grupo
    grp = ["ciclo_id", "bloque_base", "variedad_canon"]
    tot_real = panel.groupby(grp, dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    n_groups_all = panel[grp].drop_duplicates().shape[0]
    n_groups_pos = panel.loc[tot_real > 0, grp].drop_duplicates().shape[0]
    print(f"[DBG] grupos harvest={n_groups_all:,} | grupos con real_total>0: {n_groups_pos:,}")

    panel = panel[tot_real >= MIN_REAL_TOTAL].copy()
    n_groups_train = panel[grp].drop_duplicates().shape[0]
    print(f"[DBG] MIN_REAL_TOTAL={MIN_REAL_TOTAL} => grupos para train: {n_groups_train:,}")

    if len(panel) == 0:
        # guarda debug para inspección
        write_parquet(panel, DBG_PANEL_PATH)
        raise ValueError("No hay suficientes ciclos con señal real para entrenar beta params. Revisa join PROG/universe o MIN_REAL_TOTAL.")

    # Rel pos
    dih = pd.to_numeric(panel["day_in_harvest"], errors="coerce").astype(float)
    nh = pd.to_numeric(panel["n_harvest_days"], errors="coerce").astype(float)
    panel["rel_pos"] = np.clip((dih - 0.5) / (nh + EPS), REL_CLIP, 1.0 - REL_CLIP)

    panel = panel.sort_values(grp + ["day_in_harvest", "fecha"], kind="mergesort").reset_index(drop=True)

    rows = []
    fit_meta = {}

    for (cid, bb, var), df in panel.groupby(grp, dropna=False):
        y = df["tallos_real_dia"].to_numpy(dtype=float)
        s = float(np.sum(y))
        if s <= 0:
            continue
        share_real = y / s
        rel = df["rel_pos"].to_numpy(dtype=float)

        a, b, info = _fit_beta_params_discrete(rel, share_real)
        f = _agg_cycle_features(df)

        row = {
            "ciclo_id": str(cid),
            "bloque_base": int(bb) if pd.notna(bb) else pd.NA,
            "variedad_canon": str(var),
            "area": str(df["area"].iloc[0]) if "area" in df.columns else "UNKNOWN",
            "tipo_sp": str(df["tipo_sp"].iloc[0]) if "tipo_sp" in df.columns else "UNKNOWN",
            "alpha": float(a),
            "beta": float(b),
            "real_total": float(s),
            "n_harvest_days": float(pd.to_numeric(df["n_harvest_days"], errors="coerce").max()),
        }
        row.update({k: (float(v) if pd.notna(v) else np.nan) for k, v in f.to_dict().items()})
        rows.append(row)
        fit_meta[f"{cid}|{bb}|{var}"] = info

    out = pd.DataFrame(rows)
    if len(out) == 0:
        write_parquet(panel, DBG_PANEL_PATH)
        raise ValueError("No se pudieron ajustar parámetros beta en ningún grupo. Revisa rel_pos/day_in_harvest/n_harvest_days.")

    out["created_at"] = created_at
    write_parquet(out, OUT_PATH)

    meta_path = OUT_PATH.with_suffix(".fitmeta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"created_at": str(created_at), "n_rows": int(len(out)), "MIN_REAL_TOTAL": MIN_REAL_TOTAL, "fit_info": fit_meta}, f, indent=2, ensure_ascii=False)

    print(f"OK -> {OUT_PATH} | rows={len(out):,} | grupos={out[['ciclo_id','bloque_base','variedad_canon']].drop_duplicates().shape[0]:,}")


if __name__ == "__main__":
    main()
