from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from common.io import read_parquet

# -------------------------
# Paths
# -------------------------
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")

# Baseline día (para comparar forma)
OFERTA_PATH = Path("data/preds/pred_oferta_dia.parquet")

# Predicciones nuevas por share (SALIDA del apply_curva_share_dia)
PRED_FACTOR_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")

# Para reconstruir tallos_ml1_dia: baseline * factor (compat downstream)
OUT_REPORT = Path("data/audit/audit_ml1_curva_share_report.parquet")


# -------------------------
# Helpers
# -------------------------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_int(s: pd.Series) -> pd.Series:
    # pandas nullable Int64 puede no estar disponible en algunos entornos viejos;
    # acá lo dejamos, pero si falla en tu ambiente ya sabes el patrón del fix.
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Cols={list(df.columns)}")


def _pick_first(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _cycle_metrics(df: pd.DataFrame) -> pd.Series:
    """
    df: rows de un ciclo con columnas:
      - tallos_real_dia (>=0)
      - tallos_ml1_dia (>=0)
      - tallos_base_dia (>=0)
    """
    r = df["tallos_real_dia"].to_numpy(dtype=float)
    m = df["tallos_ml1_dia"].to_numpy(dtype=float)

    sr = r.sum()
    sm = m.sum()

    if sr <= 0:
        return pd.Series(
            {
                "has_real": False,
                "l1_share": np.nan,
                "ks_cdf": np.nan,
                "peak_pos_err_days": np.nan,
                "mass_early_diff": np.nan,
                "mass_tail_diff": np.nan,
                "n_days": int(len(df)),
            }
        )

    pr = r / sr
    pm = np.where(sm > 0, m / sm, 0.0)

    l1 = float(np.abs(pm - pr).sum())

    cdf_r = np.cumsum(pr)
    cdf_m = np.cumsum(pm)
    ks = float(np.max(np.abs(cdf_m - cdf_r)))

    peak_r = int(np.argmax(pr))
    peak_m = int(np.argmax(pm))
    peak_err = peak_m - peak_r

    n = len(df)
    k = max(1, int(np.ceil(0.20 * n)))
    early_r = float(pr[:k].sum())
    early_m = float(pm[:k].sum())
    tail_r = float(pr[-k:].sum())
    tail_m = float(pm[-k:].sum())

    return pd.Series(
        {
            "has_real": True,
            "l1_share": l1,
            "ks_cdf": ks,
            "peak_pos_err_days": peak_err,
            "mass_early_diff": early_m - early_r,
            "mass_tail_diff": tail_m - tail_r,
            "n_days": int(n),
        }
    )


def main() -> None:
    created_at = pd.Timestamp.now("UTC")

    # -------------------------
    # Read
    # -------------------------
    uni = read_parquet(UNIVERSE_PATH).copy()
    prog = read_parquet(PROG_PATH).copy()
    oferta = read_parquet(OFERTA_PATH).copy()
    pred = read_parquet(PRED_FACTOR_PATH).copy()

    # -------------------------
    # Universe keys
    # -------------------------
    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe")
    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    # -------------------------
    # PROG (real)
    # -------------------------
    _require(prog, ["ciclo_id", "fecha", "bloque_base", "variedad", "tallos_real_dia"], "prog")
    prog["ciclo_id"] = prog["ciclo_id"].astype(str)
    prog["fecha"] = _to_date(prog["fecha"])
    prog["bloque_base"] = _canon_int(prog["bloque_base"])
    prog["variedad_raw"] = _canon_str(prog["variedad"])
    prog["variedad_canon"] = prog["variedad_raw"].replace({"XLENCE": "XL", "CLOUD": "CLO"})
    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0).astype(float)

    prog_k = prog[["ciclo_id", "fecha", "bloque_base", "variedad_canon", "tallos_real_dia"]].drop_duplicates(subset=key)

    # -------------------------
    # OFERTA baseline (FIX: bloque vs bloque_base)
    # -------------------------
    _require(oferta, ["ciclo_id", "fecha", "stage", "tallos_pred"], "oferta")

    # FIX PRINCIPAL: resolver columna de bloque (bloque_base o bloque)
    bloque_col = _pick_first(oferta, ["bloque_base", "bloque"])
    if bloque_col is None:
        raise ValueError(f"oferta: no encuentro bloque_base/bloque. Cols={list(oferta.columns)}")

    oferta["ciclo_id"] = oferta["ciclo_id"].astype(str)
    oferta["fecha"] = _to_date(oferta["fecha"])
    oferta["stage"] = _canon_str(oferta["stage"])

    # crear bloque_base canónico
    # (si 'bloque' ya es numérico, perfecto; si es string, se coerciona)
    oferta["bloque_base"] = _canon_int(oferta[bloque_col])

    # canon variedad
    if "variedad_canon" in oferta.columns:
        oferta["variedad_canon"] = _canon_str(oferta["variedad_canon"])
    elif "variedad" in oferta.columns:
        oferta["variedad_canon"] = _canon_str(oferta["variedad"]).replace({"XLENCE": "XL", "CLOUD": "CLO"})
    else:
        oferta["variedad_canon"] = "UNKNOWN"

    oferta = oferta[oferta["stage"].eq("HARVEST")].copy()
    oferta["tallos_base_dia"] = pd.to_numeric(oferta["tallos_pred"], errors="coerce").fillna(0.0).astype(float)

    # agrupar baseline por key
    oferta_k = (
        oferta.groupby(key, as_index=False)
        .agg(
            tallos_base_dia=("tallos_base_dia", "sum"),
            tallos_proy=("tallos_proy", "max") if "tallos_proy" in oferta.columns else ("tallos_base_dia", "sum"),
        )
    )

    # -------------------------
    # PRED factor / share
    # -------------------------
    _require(pred, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "factor_curva_ml1"], "pred_factor")
    pred["ciclo_id"] = pred["ciclo_id"].astype(str)
    pred["fecha"] = _to_date(pred["fecha"])
    pred["bloque_base"] = _canon_int(pred["bloque_base"])
    pred["variedad_canon"] = _canon_str(pred["variedad_canon"])
    pred["factor_curva_ml1"] = pd.to_numeric(pred["factor_curva_ml1"], errors="coerce").fillna(1.0).astype(float)

    pred_k = pred[key + ["factor_curva_ml1"]].drop_duplicates(subset=key)

    # -------------------------
    # Panel
    # -------------------------
    panel = (
        uni_k.merge(oferta_k, on=key, how="left")
             .merge(pred_k, on=key, how="left")
             .merge(prog_k, on=key, how="left")
    )

    panel["tallos_base_dia"] = pd.to_numeric(panel.get("tallos_base_dia"), errors="coerce").fillna(0.0).astype(float)
    panel["factor_curva_ml1"] = pd.to_numeric(panel.get("factor_curva_ml1"), errors="coerce").fillna(1.0).astype(float)

    panel["_f"] = panel["factor_curva_ml1"].clip(lower=0.0)
    s = panel.groupby(["ciclo_id"], dropna=False)["_f"].transform("sum")
    panel["_w"] = np.where(s > 0, panel["_f"] / s, 0.0)

    # reconstrucción: tallos_ml1_dia = share * tallos_proy
    panel["tallos_proy"] = pd.to_numeric(panel.get("tallos_proy"), errors="coerce").fillna(0.0).astype(float)
    panel["tallos_ml1_dia"] = panel["_w"] * panel["tallos_proy"]

    panel["tallos_real_dia"] = pd.to_numeric(panel.get("tallos_real_dia"), errors="coerce").fillna(0.0).astype(float)

    # -------------------------
    # Coverage / invariants
    # -------------------------
    universe_rows = int(len(uni_k))
    cov_real = float((panel["tallos_real_dia"] > 0).mean())

    print("=== COVERAGE ===")
    print(
        pd.DataFrame([{
            "created_at": created_at,
            "universe_rows": universe_rows,
            "coverage_real_gt0": cov_real,
        }]).to_string(index=False)
    )

    cyc = panel.groupby("ciclo_id", dropna=False).agg(
        proy=("tallos_proy", "max"),
        ml1_sum=("tallos_ml1_dia", "sum"),
    )
    cyc["abs_diff"] = (cyc["proy"].astype(float) - cyc["ml1_sum"].astype(float)).abs()
    print("\n=== MASS BALANCE (cycle) ===")
    print(f"cycles={len(cyc):,} | max abs diff ml1 vs proy: {float(cyc['abs_diff'].max()):.12f}")

    # -------------------------
    # Shape metrics (cycle-level)
    # -------------------------
    panel = panel.sort_values(["ciclo_id", "fecha"]).reset_index(drop=True)
    cyc_metrics = panel.groupby("ciclo_id", dropna=False).apply(_cycle_metrics).reset_index()

    n_total = int(cyc_metrics["ciclo_id"].nunique())
    n_real = int(cyc_metrics["has_real"].sum())
    print("\n=== SHAPE (cycle) ===")
    print(f"cycles total={n_total:,} | cycles with real={n_real:,}")

    if n_real > 0:
        cols_show = ["l1_share", "ks_cdf", "peak_pos_err_days", "mass_early_diff", "mass_tail_diff"]
        for c in cols_show:
            ss = cyc_metrics.loc[cyc_metrics["has_real"], c].astype(float)
            print(f"\n{c}:")
            print(ss.describe().to_string())

    # -------------------------
    # Save detailed report
    # -------------------------
    panel["created_at_audit"] = created_at
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(OUT_REPORT, index=False)

    cyc_out = cyc_metrics.copy()
    cyc_out["created_at_audit"] = created_at
    cyc_path = OUT_REPORT.with_name("audit_ml1_curva_share_cycle_metrics.parquet")
    cyc_out.to_parquet(cyc_path, index=False)

    print(f"\nOK -> {OUT_REPORT}")
    print(f"OK -> {cyc_path}")


if __name__ == "__main__":
    main()