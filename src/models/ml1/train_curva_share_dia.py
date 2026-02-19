from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

from common.io import read_parquet


FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")
PROG_PATH = Path("data/silver/dim_cosecha_progress_bloque_fecha.parquet")
DIM_VAR_PATH = Path("data/silver/dim_variedad_canon.parquet")

REGISTRY_ROOT = Path("models_registry/ml1/curva_share_dia")


# -------------------------
# Model columns
# -------------------------
NUM_COLS = [
    "day_in_harvest",
    "rel_pos",
    "n_harvest_days",
    "pct_avance_real",          # <- opcional en data, pero el modelo la puede usar si existe
    "dia_rel_cosecha_real",
    "gdc_acum_real",
    "rainfall_mm_dia",
    "horas_lluvia",
    "temp_avg_dia",
    "solar_energy_j_m2_dia",
    "wind_speed_avg_dia",
    "wind_run_dia",
    "gdc_dia",
    "dias_desde_sp",
    "gdc_acum_desde_sp",
    "dow",
    "month",
    "weekofyear",
]
CAT_COLS = ["variedad_canon", "area", "tipo_sp"]


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


def _make_version() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


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


def _load_var_map(dim_var: pd.DataFrame) -> dict[str, str]:
    _require(dim_var, ["variedad_raw", "variedad_canon"], "dim_variedad_canon")
    dv = dim_var.copy()
    dv["variedad_raw"] = _canon_str(dv["variedad_raw"])
    dv["variedad_canon"] = _canon_str(dv["variedad_canon"])
    dv = dv.dropna(subset=["variedad_raw", "variedad_canon"]).drop_duplicates(subset=["variedad_raw"])
    return dict(zip(dv["variedad_raw"], dv["variedad_canon"]))


def _relpos_bin(rel_pos: pd.Series) -> pd.Series:
    x = pd.to_numeric(rel_pos, errors="coerce").astype(float)
    bins = [-np.inf, 0.05, 0.15, 0.30, 0.60, 0.85, np.inf]
    labels = ["00-05", "05-15", "15-30", "30-60", "60-85", "85-100"]
    return pd.cut(x, bins=bins, labels=labels)


def _ndays_bucket(n: pd.Series) -> pd.Series:
    x = pd.to_numeric(n, errors="coerce")
    return pd.cut(
        x.astype(float),
        bins=[-np.inf, 30, 45, 60, np.inf],
        labels=["<=30", "31-45", "46-60", ">60"],
    )


def _safe_quantile(s: pd.Series, q: float) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    if len(v) < 30:
        return float("nan")
    return float(v.quantile(q))


def main() -> None:
    for p in [FEATURES_PATH, UNIVERSE_PATH, PROG_PATH, DIM_VAR_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"No existe: {p}")

    created_at = pd.Timestamp.utcnow()
    version = _make_version()
    out_dir = REGISTRY_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Dim variedad canon
    # -------------------------
    dim_var = read_parquet(DIM_VAR_PATH).copy()
    var_map = _load_var_map(dim_var)

    # -------------------------
    # Features
    # -------------------------
    feat = read_parquet(FEATURES_PATH).copy()
    feat = _dedupe_columns(feat)

    _require(
        feat,
        ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "tallos_pred_baseline_dia", "tallos_proy"],
        "features_curva",
    )

    feat["ciclo_id"] = feat["ciclo_id"].astype(str)
    feat["fecha"] = _to_date(feat["fecha"])
    feat["bloque_base"] = _canon_int(feat["bloque_base"])
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    for c in ["area", "tipo_sp"]:
        if c in feat.columns:
            feat[c] = _canon_str(feat[c])

    # asegurar columnas del modelo (num+cat) en FEAT
    for c in NUM_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in CAT_COLS:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    feat["tallos_pred_baseline_dia"] = pd.to_numeric(feat["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0)
    feat["tallos_proy"] = pd.to_numeric(feat["tallos_proy"], errors="coerce").fillna(0.0)

    # -------------------------
    # Universe (positional)
    # -------------------------
    uni = read_parquet(UNIVERSE_PATH).copy()
    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "rel_pos_pred", "n_harvest_days_pred"], "universe_harvest_grid_ml1")

    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key + ["rel_pos_pred", "day_in_harvest_pred", "n_harvest_days_pred"]].drop_duplicates(subset=key)

    # -------------------------
    # Prog real (solo real + opcional pct_avance_real)
    # -------------------------
    prog = read_parquet(PROG_PATH).copy()
    _require(prog, ["ciclo_id", "fecha", "bloque_base", "variedad", "tallos_real_dia"], "dim_cosecha_progress_bloque_fecha")

    prog["ciclo_id"] = prog["ciclo_id"].astype(str)
    prog["fecha"] = _to_date(prog["fecha"])
    prog["bloque_base"] = _canon_int(prog["bloque_base"])

    prog["variedad_raw"] = _canon_str(prog["variedad"])
    prog["variedad_canon"] = prog["variedad_raw"].map(var_map).fillna(prog["variedad_raw"])

    prog["tallos_real_dia"] = pd.to_numeric(prog["tallos_real_dia"], errors="coerce").fillna(0.0)

    prog_take = ["ciclo_id", "fecha", "bloque_base", "variedad_canon", "tallos_real_dia"]
    if "pct_avance_real" in prog.columns:
        prog_take.append("pct_avance_real")
    prog_k = prog[prog_take].drop_duplicates(subset=key)

    # -------------------------
    # Panel
    # -------------------------
    # Importante: traer SOLO lo que necesitamos de feat (evita duplicados raros)
    feat_take = key + ["tallos_pred_baseline_dia", "tallos_proy"] + NUM_COLS + ["area", "tipo_sp"]
    feat_take = [c for c in dict.fromkeys(feat_take) if c in feat.columns]

    panel = (
        uni_k.merge(feat[feat_take], on=key, how="left")
            .merge(prog_k, on=key, how="left")
    )

    # Asegurar TODAS las columnas del modelo en el panel (incl. pct_avance_real)
    for c in NUM_COLS:
        if c not in panel.columns:
            panel[c] = np.nan
    for c in CAT_COLS:
        if c not in panel.columns:
            panel[c] = "UNKNOWN"

    # Missing prog day => 0 real
    panel["tallos_real_dia"] = pd.to_numeric(panel["tallos_real_dia"], errors="coerce").fillna(0.0)

    # Position: usar *_pred como eje de forma
    panel["rel_pos"] = pd.to_numeric(panel["rel_pos"], errors="coerce")
    panel["rel_pos"] = panel["rel_pos"].where(panel["rel_pos"].notna(), pd.to_numeric(panel["rel_pos_pred"], errors="coerce"))

    panel["day_in_harvest"] = pd.to_numeric(panel["day_in_harvest"], errors="coerce")
    panel["day_in_harvest"] = panel["day_in_harvest"].where(panel["day_in_harvest"].notna(), pd.to_numeric(panel["day_in_harvest_pred"], errors="coerce"))

    panel["n_harvest_days"] = pd.to_numeric(panel["n_harvest_days"], errors="coerce")
    panel["n_harvest_days"] = panel["n_harvest_days"].where(panel["n_harvest_days"].notna(), pd.to_numeric(panel["n_harvest_days_pred"], errors="coerce"))

    # Canon categóricas
    panel["variedad_canon"] = _canon_str(panel["variedad_canon"])
    panel["area"] = _canon_str(panel.get("area", "UNKNOWN").fillna("UNKNOWN"))
    panel["tipo_sp"] = _canon_str(panel.get("tipo_sp", "UNKNOWN").fillna("UNKNOWN"))

    # -------------------------
    # Train set
    # -------------------------
    cyc_sum = panel.groupby("ciclo_id", dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    train = panel[cyc_sum > 0].copy()

    denom = train.groupby("ciclo_id", dropna=False)["tallos_real_dia"].transform("sum").astype(float)
    train["share_real"] = np.where(denom > 0, train["tallos_real_dia"].astype(float) / denom, np.nan)

    # sample_weight por avance si existe; si no, weight=1
    pav = pd.to_numeric(train.get("pct_avance_real", np.nan), errors="coerce").fillna(0.0).astype(float)
    sw = np.ones(len(train), dtype=float)
    if "pct_avance_real" in train.columns:
        sw = np.clip((pav - 0.02) / 0.03, 0.2, 1.0)
    train["sample_weight"] = sw

    # X/y
    X = train[NUM_COLS + CAT_COLS].copy()
    y = pd.to_numeric(train["share_real"], errors="coerce")
    ok = y.notna()

    X = X.loc[ok].copy()
    y = y.loc[ok].astype(float)
    sample_weight = train.loc[ok, "sample_weight"].astype(float).to_numpy()

    # one-hot + guardar feature names
    X = pd.get_dummies(X, columns=CAT_COLS, dummy_na=True)

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_depth=6,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1e-4,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    dump(model, out_dir / "model_curva_share_dia.joblib")

    # -------------------------
    # Calibration caps/floors (learned)
    # -------------------------
    cal = train[["ciclo_id", "variedad_canon", "area", "tipo_sp", "rel_pos", "n_harvest_days", "share_real"]].copy()
    cal["rel_pos_bin"] = _relpos_bin(cal["rel_pos"])
    cal["n_days_bucket"] = _ndays_bucket(cal["n_harvest_days"])

    seg_cols = ["variedad_canon", "area", "tipo_sp", "n_days_bucket", "rel_pos_bin"]
    g = cal.groupby(seg_cols, dropna=False)["share_real"]

    cap = g.apply(lambda s: _safe_quantile(s, 0.99)).rename("cap_share_p99")
    flo = g.apply(lambda s: _safe_quantile(s, 0.01)).rename("floor_share_p01")
    cal_tab = pd.concat([cap, flo], axis=1).reset_index()

    g2 = cal.groupby(["rel_pos_bin"], dropna=False)["share_real"]
    cal_glob = pd.DataFrame({
        "rel_pos_bin": g2.apply(lambda s: _safe_quantile(s, 0.99)).index.astype(str),
        "cap_share_p99_global": g2.apply(lambda s: _safe_quantile(s, 0.99)).to_numpy(),
        "floor_share_p01_global": g2.apply(lambda s: _safe_quantile(s, 0.01)).to_numpy(),
    })

    cal_tab["created_at"] = created_at
    cal_glob["created_at"] = created_at

    cal_tab_path = out_dir / "cap_floor_share_by_relpos.parquet"
    cal_glob_path = out_dir / "cap_floor_share_by_relpos_global.parquet"
    cal_tab.to_parquet(cal_tab_path, index=False)
    cal_glob.to_parquet(cal_glob_path, index=False)

    meta = {
        "created_at": str(created_at),
        "version": version,
        "best_model": "HistGradientBoostingRegressor",
        "target": "share_real per-cycle (panelized on universe; missing prog days filled with 0)",
        "feature_cols_numeric": NUM_COLS,
        "feature_cols_categorical": CAT_COLS,
        "feature_names": list(X.columns),
        "n_train_rows": int(len(X)),
        "n_cycles_train": int(train["ciclo_id"].nunique()),
        "calibration": {
            "rel_pos_bins": ["00-05", "05-15", "15-30", "30-60", "60-85", "85-100"],
            "segmentation": seg_cols,
            "cap_quantile": 0.99,
            "floor_quantile": 0.01,
            "min_samples_quantile": 30,
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK -> {out_dir} | n_train_rows={len(X):,} | n_cycles={meta['n_cycles_train']:,}")
    print(f"OK -> {cal_tab_path.name} | rows={len(cal_tab):,}")
    print(f"OK -> {cal_glob_path.name} | rows={len(cal_glob):,}")


if __name__ == "__main__":
    main()
