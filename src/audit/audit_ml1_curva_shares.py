from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


# =========================
# Paths
# =========================
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PRED_FULL_PATH = Path("data/gold/pred_tallos_grado_dia_ml1_full.parquet")

CURVA_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")
DIST_PATH = Path("data/gold/pred_dist_grado_ml1.parquet")
OFERTA_PATH = Path("data/preds/pred_oferta_dia.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")

DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")

AUDIT_DIR = Path("data/audit/ml1_status")


# =========================
# Helpers
# =========================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    # compat con pandas viejos: Int64 puede no existir como dtype en algunos entornos raros
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except TypeError:
        return pd.to_numeric(s, errors="coerce")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _ensure_bloque_base(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Normaliza bloque_base:
      - si existe bloque_base, ok
      - si no, intenta con bloque
      - si no, intenta con bloque_padre
    """
    out = df.copy()
    if "bloque_base" in out.columns:
        return out

    c = _pick_first(out, ["bloque", "bloque_padre"])
    if c is None:
        raise ValueError(f"{name}: no existe bloque_base ni bloque/bloque_padre. Cols={list(out.columns)}")

    out["bloque_base"] = out[c]
    return out


def _load_var_map() -> dict[str, str]:
    if not DIM_VAR_PATH.exists():
        # fallback duro (tu convención)
        return {"XLENCE": "XL", "XL": "XL", "CLOUD": "CLO", "CLO": "CLO"}
    dv = read_parquet(DIM_VAR_PATH).copy()
    _require(dv, ["variedad_raw", "variedad_canon"], "dim_variedad_canon")
    dv["raw"] = _canon_str(dv["variedad_raw"])
    dv["canon"] = _canon_str(dv["variedad_canon"])
    m = dict(zip(dv["raw"], dv["canon"]))
    # asegurar mínimos
    m.setdefault("XLENCE", "XL")
    m.setdefault("XL", "XL")
    m.setdefault("CLOUD", "CLO")
    m.setdefault("CLO", "CLO")
    return m


def _canon_var_from_col(df: pd.DataFrame, col: str, var_map: dict[str, str]) -> pd.Series:
    raw = _canon_str(df[col]) if col in df.columns else pd.Series(["UNKNOWN"] * len(df))
    return raw.map(var_map).fillna(raw)


def _ensure_relpos(uni: pd.DataFrame) -> pd.DataFrame:
    out = uni.copy()
    if "rel_pos" not in out.columns and "rel_pos_pred" in out.columns:
        out["rel_pos"] = pd.to_numeric(out["rel_pos_pred"], errors="coerce")
    if "day_in_harvest" not in out.columns and "day_in_harvest_pred" in out.columns:
        out["day_in_harvest"] = pd.to_numeric(out["day_in_harvest_pred"], errors="coerce")
    if "n_harvest_days" not in out.columns and "n_harvest_days_pred" in out.columns:
        out["n_harvest_days"] = pd.to_numeric(out["n_harvest_days_pred"], errors="coerce")
    return out


def _cycle_shape_metrics(sub: pd.DataFrame) -> dict:
    # requiere: fecha, share_ml1, share_real opcional, rel_pos opcional
    sub = sub.sort_values("fecha").copy()

    ml1 = pd.to_numeric(sub["share_ml1"], errors="coerce").fillna(0.0).to_numpy(float)
    ml1 = np.clip(ml1, 0.0, None)
    ml1 = ml1 / (ml1.sum() + 1e-12)

    out = {
        "n_days": int(len(sub)),
        "has_real": 0,
        "peak_idx_ml1": int(np.argmax(ml1)) if len(ml1) else pd.NA,
        "peak_share_ml1": float(np.max(ml1)) if len(ml1) else pd.NA,
        "l1_share": pd.NA,
        "ks_cdf": pd.NA,
        "peak_pos_err_days": pd.NA,
        "mass_early_ml1": pd.NA,
        "mass_tail_ml1": pd.NA,
        "mass_early_real": pd.NA,
        "mass_tail_real": pd.NA,
        "mass_early_diff": pd.NA,
        "mass_tail_diff": pd.NA,
    }

    if "rel_pos" in sub.columns:
        rel = pd.to_numeric(sub["rel_pos"], errors="coerce").to_numpy(float)
        out["mass_early_ml1"] = float(ml1[rel <= 0.15].sum())
        out["mass_tail_ml1"] = float(ml1[rel >= 0.85].sum())

    if "share_real" not in sub.columns or sub["share_real"].notna().sum() == 0:
        return out

    real = pd.to_numeric(sub["share_real"], errors="coerce").fillna(0.0).to_numpy(float)
    real = np.clip(real, 0.0, None)
    real = real / (real.sum() + 1e-12)

    out["has_real"] = 1
    peak_real = int(np.argmax(real)) if len(real) else pd.NA
    out["peak_pos_err_days"] = (out["peak_idx_ml1"] - peak_real) if pd.notna(peak_real) else pd.NA
    out["l1_share"] = float(np.abs(real - ml1).sum())
    out["ks_cdf"] = float(np.max(np.abs(np.cumsum(real) - np.cumsum(ml1))))

    if "rel_pos" in sub.columns:
        rel = pd.to_numeric(sub["rel_pos"], errors="coerce").to_numpy(float)
        m_early_real = float(real[rel <= 0.15].sum())
        m_tail_real = float(real[rel >= 0.85].sum())
        out["mass_early_real"] = m_early_real
        out["mass_tail_real"] = m_tail_real
        out["mass_early_diff"] = float(out["mass_early_ml1"] - m_early_real) if pd.notna(out["mass_early_ml1"]) else pd.NA
        out["mass_tail_diff"] = float(out["mass_tail_ml1"] - m_tail_real) if pd.notna(out["mass_tail_ml1"]) else pd.NA

    return out


# =========================
# Main
# =========================
def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    created_at = pd.Timestamp.now("UTC")  # ✅ reemplaza utcnow (warning pandas4)
    var_map = _load_var_map()

    # ---------- Load
    uni = read_parquet(UNIVERSE_PATH).copy()
    curva = read_parquet(CURVA_PATH).copy()
    dist = read_parquet(DIST_PATH).copy()
    oferta = read_parquet(OFERTA_PATH).copy()
    full = read_parquet(PRED_FULL_PATH).copy()
    prog = read_parquet(PROG_PATH).copy() if PROG_PATH.exists() else pd.DataFrame()

    # ---------- Canon UNIVERSE
    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")
    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])
    if "stage" in uni.columns:
        uni["stage"] = _canon_str(uni["stage"])
        uni = uni[uni["stage"].eq("HARVEST")].copy()
    uni = _ensure_relpos(uni)

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    # ---------- Canon CURVA
    _require(curva, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "pred_factor_curva_ml1")
    curva["ciclo_id"] = curva["ciclo_id"].astype(str)
    curva["fecha"] = _to_date(curva["fecha"])
    curva["bloque_base"] = _canon_int(curva["bloque_base"])
    curva["variedad_canon"] = _canon_str(curva["variedad_canon"])
    curva_k = curva[key].drop_duplicates()

    # ---------- Canon DIST (día-level coverage; grade not needed here)
    _require(dist, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado", "share_grado_ml1"], "pred_dist_grado_ml1")
    dist["ciclo_id"] = dist["ciclo_id"].astype(str)
    dist["fecha"] = _to_date(dist["fecha"])
    dist["bloque_base"] = _canon_int(dist["bloque_base"])
    dist["variedad_canon"] = _canon_str(dist["variedad_canon"])
    dist["grado"] = _canon_int(dist["grado"])
    dist_day_k = dist[key].drop_duplicates()

    # ---------- Canon OFERTA (baseline)  ✅ FIX bloque_base
    oferta = _ensure_bloque_base(oferta, "pred_oferta_dia")  # ✅ crea bloque_base desde bloque
    _require(oferta, ["ciclo_id", "fecha", "bloque_base", "tallos_pred"], "pred_oferta_dia")
    oferta["ciclo_id"] = oferta["ciclo_id"].astype(str)
    oferta["fecha"] = _to_date(oferta["fecha"])
    oferta["bloque_base"] = _canon_int(oferta["bloque_base"])
    if "stage" in oferta.columns:
        oferta["stage"] = _canon_str(oferta["stage"])
        oferta = oferta[oferta["stage"].eq("HARVEST")].copy()

    # variedad_canon
    if "variedad_canon" in oferta.columns:
        oferta["variedad_canon"] = _canon_str(oferta["variedad_canon"])
    elif "variedad" in oferta.columns:
        oferta["variedad_canon"] = _canon_var_from_col(oferta, "variedad", var_map)
    else:
        oferta["variedad_canon"] = "UNKNOWN"

    oferta["tallos_pred_baseline_dia"] = pd.to_numeric(oferta["tallos_pred"], errors="coerce").fillna(0.0)
    oferta_k = oferta[key + ["tallos_pred_baseline_dia"]].drop_duplicates(subset=key)

    # ---------- Canon PROG (real)
    has_prog = (len(prog) > 0)
    if has_prog:
        prog = _ensure_bloque_base(prog, "dim_cosecha_progress_bloque_fecha")  # ✅ por si prog viene como bloque
        _require(prog, ["ciclo_id", "fecha", "bloque_base", "variedad", "tallos_real_dia"], "dim_cosecha_progress_bloque_fecha")
        prog["ciclo_id"] = prog["ciclo_id"].astype(str)
        prog["fecha"] = _to_date(prog["fecha"])
        prog["bloque_base"] = _canon_int(prog["bloque_base"])
        prog["variedad_canon"] = _canon_var_from_col(prog, "variedad", var_map)
        prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce")
        extra_cols = ["pct_avance_real"] if "pct_avance_real" in prog.columns else []
        prog_k = prog[key + ["tallos_real_dia"] + extra_cols].drop_duplicates(subset=key)
        if "pct_avance_real" not in prog_k.columns:
            prog_k["pct_avance_real"] = np.nan
    else:
        prog_k = pd.DataFrame(columns=key + ["tallos_real_dia", "pct_avance_real"])

    # =========================
    # 1) COVERAGE
    # =========================
    miss_curva = uni_k.merge(curva_k, on=key, how="left", indicator=True)
    miss_curva = miss_curva[miss_curva["_merge"].eq("left_only")].drop(columns=["_merge"])

    miss_dist = uni_k.merge(dist_day_k, on=key, how="left", indicator=True)
    miss_dist = miss_dist[miss_dist["_merge"].eq("left_only")].drop(columns=["_merge"])

    miss_oferta = uni_k.merge(oferta_k[key].drop_duplicates(), on=key, how="left", indicator=True)
    miss_oferta = miss_oferta[miss_oferta["_merge"].eq("left_only")].drop(columns=["_merge"])

    miss_prog = uni_k.merge(prog_k[key].drop_duplicates(), on=key, how="left", indicator=True)
    miss_prog = miss_prog[miss_prog["_merge"].eq("left_only")].drop(columns=["_merge"])

    cov = pd.DataFrame([{
        "created_at": created_at,
        "universe_rows": int(len(uni_k)),
        "curva_rows": int(len(curva_k)),
        "dist_day_rows": int(len(dist_day_k)),
        "oferta_rows": int(len(oferta_k)),
        "prog_rows": int(len(prog_k)),
        "miss_curva_rows": int(len(miss_curva)),
        "miss_dist_rows": int(len(miss_dist)),
        "miss_oferta_rows": int(len(miss_oferta)),
        "miss_prog_rows": int(len(miss_prog)),
        "miss_curva_rate": float(len(miss_curva) / max(len(uni_k), 1)),
        "miss_dist_rate": float(len(miss_dist) / max(len(uni_k), 1)),
        "miss_oferta_rate": float(len(miss_oferta) / max(len(uni_k), 1)),
        "miss_prog_rate": float(len(miss_prog) / max(len(uni_k), 1)),
    }])
    write_parquet(cov, AUDIT_DIR / "coverage.parquet")

    # =========================
    # 2) INVARIANTS on FINAL
    # =========================
    _require(full, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "grado"], "pred_tallos_grado_dia_ml1_full")
    for c in ["tallos_pred_baseline_dia", "tallos_pred_ml1_dia",
              "tallos_pred_baseline_grado_dia", "tallos_pred_ml1_grado_dia"]:
        if c in full.columns:
            full[c] = pd.to_numeric(full[c], errors="coerce")

    full["ciclo_id"] = full["ciclo_id"].astype(str)
    full["fecha"] = _to_date(full["fecha"])
    full["bloque_base"] = _canon_int(full["bloque_base"])
    full["variedad_canon"] = _canon_str(full["variedad_canon"])
    full["grado"] = _canon_int(full["grado"])

    g_day = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    sum_base = full.groupby(g_day, dropna=False)["tallos_pred_baseline_grado_dia"].sum().rename("sum_grado_base")
    sum_ml1 = full.groupby(g_day, dropna=False)["tallos_pred_ml1_grado_dia"].sum().rename("sum_grado_ml1")

    day_first = full.drop_duplicates(subset=g_day)[g_day + ["tallos_pred_baseline_dia", "tallos_pred_ml1_dia"]].set_index(g_day)
    day_chk = day_first.join(sum_base).join(sum_ml1).reset_index()

    eps = 1e-6
    day_chk["mismatch_base"] = (day_chk["sum_grado_base"] - day_chk["tallos_pred_baseline_dia"]).abs() > eps
    day_chk["mismatch_ml1"] = (day_chk["sum_grado_ml1"] - day_chk["tallos_pred_ml1_dia"]).abs() > eps

    inv = pd.DataFrame([{
        "created_at": created_at,
        "pct_day_mismatch_base": float(day_chk["mismatch_base"].mean()) if len(day_chk) else np.nan,
        "pct_day_mismatch_ml1": float(day_chk["mismatch_ml1"].mean()) if len(day_chk) else np.nan,
        "pct_day_ml1_dia_nan": float(day_chk["tallos_pred_ml1_dia"].isna().mean()) if len(day_chk) else np.nan,
        "pct_day_ml1_dia_zero": float((day_chk["tallos_pred_ml1_dia"].fillna(0.0) == 0.0).mean()) if len(day_chk) else np.nan,
    }])
    write_parquet(inv, AUDIT_DIR / "invariants_day.parquet")

    # Mass-balance per cycle (needs tallos_proy; take from oferta if present)
    if "tallos_proy" in oferta.columns:
        oferta_tp = oferta[["ciclo_id", "tallos_proy"]].copy()
        oferta_tp["ciclo_id"] = oferta_tp["ciclo_id"].astype(str)
        oferta_tp["tallos_proy"] = pd.to_numeric(oferta_tp["tallos_proy"], errors="coerce")
        tproy = oferta_tp.groupby("ciclo_id", dropna=False)["tallos_proy"].max()
    else:
        tproy = pd.Series(dtype=float)

    cyc = day_chk.groupby("ciclo_id", dropna=False).agg(
        ml1_sum=("tallos_pred_ml1_dia", "sum"),
        base_sum=("tallos_pred_baseline_dia", "sum"),
        days=("ciclo_id", "size"),
    )
    if len(tproy):
        cyc = cyc.join(tproy.rename("tallos_proy"), how="left")
        cyc["abs_diff_ml1_vs_proy"] = (cyc["ml1_sum"] - cyc["tallos_proy"]).abs()
        cyc["rel_diff_ml1_vs_proy"] = np.where(
            cyc["tallos_proy"].fillna(0).astype(float) > 0,
            cyc["abs_diff_ml1_vs_proy"] / cyc["tallos_proy"].astype(float),
            np.nan,
        )
    else:
        cyc["tallos_proy"] = np.nan
        cyc["abs_diff_ml1_vs_proy"] = np.nan
        cyc["rel_diff_ml1_vs_proy"] = np.nan

    cyc = cyc.reset_index()
    write_parquet(cyc, AUDIT_DIR / "mass_balance_cycle.parquet")

    # =========================
    # 3) SHAPE vs REAL (curve)
    # =========================
    day_panel = day_chk.merge(
        uni.drop_duplicates(subset=g_day)[g_day + [c for c in ["rel_pos", "day_in_harvest", "n_harvest_days", "month"] if c in uni.columns]],
        on=g_day,
        how="left",
    ).merge(
        prog_k[g_day + ["tallos_real_dia", "pct_avance_real"]],
        on=g_day,
        how="left",
    )

    denom_ml1 = day_panel.groupby("ciclo_id", dropna=False)["tallos_pred_ml1_dia"].transform("sum")
    day_panel["share_ml1"] = np.where(denom_ml1 > 0, day_panel["tallos_pred_ml1_dia"] / denom_ml1, 0.0)

    denom_real = day_panel.groupby("ciclo_id", dropna=False)["tallos_real_dia"].transform(
        lambda s: np.nansum(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float))
    )
    day_panel["share_real"] = np.where(denom_real > 0, day_panel["tallos_real_dia"] / denom_real, np.nan)

    write_parquet(day_panel, AUDIT_DIR / "day_panel.parquet")

    cyc_metrics = []
    for cid, sub in day_panel.groupby("ciclo_id", dropna=False):
        m = _cycle_shape_metrics(sub)
        m["ciclo_id"] = cid
        v = sub["variedad_canon"].dropna()
        m["variedad_canon"] = v.iloc[0] if len(v) else pd.NA
        b = sub["bloque_base"].dropna()
        m["bloque_base"] = b.iloc[0] if len(b) else pd.NA
        mo = sub["month"].dropna() if "month" in sub.columns else pd.Series([], dtype=object)
        m["month"] = int(mo.iloc[0]) if len(mo) and pd.notna(mo.iloc[0]) else pd.NA
        cyc_metrics.append(m)

    cyc_shape = pd.DataFrame(cyc_metrics)
    write_parquet(cyc_shape, AUDIT_DIR / "shape_cycle.parquet")

    with_real = cyc_shape[cyc_shape["has_real"].eq(1)].copy()
    seg_rows = []
    if len(with_real):
        for by, tag in [
            (["variedad_canon"], "by_variedad"),
            (["bloque_base"], "by_bloque"),
            (["variedad_canon", "month"], "by_variedad_month"),
        ]:
            g = with_real.groupby(by, dropna=False).agg(
                n_cycles=("ciclo_id", "count"),
                l1_median=("l1_share", "median"),
                ks_median=("ks_cdf", "median"),
                peak_err_median=("peak_pos_err_days", "median"),
                early_diff_median=("mass_early_diff", "median"),
                tail_diff_median=("mass_tail_diff", "median"),
            ).reset_index()
            g["segment_tag"] = tag
            seg_rows.append(g.sort_values("n_cycles", ascending=False))
    seg = pd.concat(seg_rows, ignore_index=True) if seg_rows else pd.DataFrame()
    write_parquet(seg, AUDIT_DIR / "shape_segmentation.parquet")

    # =========================
    # Prints (state snapshot)
    # =========================
    print("\n=== COVERAGE ===")
    print(cov.to_string(index=False))

    print("\n=== INVARIANTS (day) ===")
    print(inv.to_string(index=False))

    if len(cyc):
        mx = float(pd.to_numeric(cyc["abs_diff_ml1_vs_proy"], errors="coerce").max())
        print("\n=== MASS BALANCE (cycle) ===")
        print(f"cycles={len(cyc):,} | max abs diff ml1 vs proy: {mx:.12f}")

    print("\n=== SHAPE (cycle) ===")
    print(f"cycles total={len(cyc_shape):,} | cycles with real={int(cyc_shape['has_real'].sum()):,}")
    if len(with_real):
        print("\nL1 share (lower is better):")
        print(with_real["l1_share"].describe().to_string())
        print("\nKS CDF (lower is better):")
        print(with_real["ks_cdf"].describe().to_string())
        print("\nPeak position error (days):")
        print(pd.to_numeric(with_real["peak_pos_err_days"], errors="coerce").describe().to_string())
        print("\nMass early diff (ML1 - real):")
        print(pd.to_numeric(with_real["mass_early_diff"], errors="coerce").describe().to_string())
        print("\nMass tail diff (ML1 - real):")
        print(pd.to_numeric(with_real["mass_tail_diff"], errors="coerce").describe().to_string())
    else:
        print("[WARN] No hay ciclos con real en el join (revisa PROG).")

    print(f"\nOK -> auditoría escrita en: {AUDIT_DIR}")


if __name__ == "__main__":
    main()