from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


# =============================================================================
# Paths
# =============================================================================
FEATURES_PATH = Path("data/features/features_curva_cosecha_bloque_dia.parquet")
UNIVERSE_PATH = Path("data/gold/universe_harvest_grid_ml1.parquet")

REGISTRY_ROOT = Path("models_registry/ml1/curva_share_dia")

OUT_PATH = Path("data/gold/pred_factor_curva_ml1.parquet")


# =============================================================================
# Model columns (must match TRAIN)
# =============================================================================
NUM_COLS = [
    "day_in_harvest",
    "rel_pos",
    "n_harvest_days",
    "pct_avance_real",
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


# =============================================================================
# Helpers
# =============================================================================
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _canon_int(s: pd.Series) -> pd.Series:
    # OJO: pandas nullable Int64 (capital I) existe; numpy int64 no sirve si hay NA.
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _latest_version_dir() -> Path:
    if not REGISTRY_ROOT.exists():
        raise FileNotFoundError(f"No existe {REGISTRY_ROOT}")
    dirs = [p for p in REGISTRY_ROOT.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {REGISTRY_ROOT}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas {miss}. Disponibles={list(df.columns)}")


def _coalesce_cols(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    if out_col in df.columns:
        base = df[out_col]
    else:
        base = pd.Series([pd.NA] * len(df), index=df.index)

    for c in candidates:
        if c in df.columns:
            base = base.where(base.notna(), df[c])
    df[out_col] = base


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
        if len(idxs) == 1:
            keep[c] = out.iloc[:, idxs[0]]
        else:
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


def _infer_is_harvest(df: pd.DataFrame) -> np.ndarray:
    """
    Define máscara de días dentro de ventana harvest.
    Preferimos:
      - harvest_start_pred / harvest_end_pred (si existen)
      - day_in_harvest (si existe y es >=1)
      - fallback: baseline > 0
    """
    f = pd.to_datetime(df["fecha"], errors="coerce")

    hs_col = None
    he_col = None
    for c in ["harvest_start_pred", "harvest_start", "fecha_inicio_real"]:
        if c in df.columns:
            hs_col = c
            break
    for c in ["harvest_end_pred", "harvest_end_eff", "harvest_end", "fecha_fin_real"]:
        if c in df.columns:
            he_col = c
            break

    if hs_col and he_col:
        hs = pd.to_datetime(df[hs_col], errors="coerce")
        he = pd.to_datetime(df[he_col], errors="coerce")
        m = hs.notna() & he.notna() & f.notna() & (f >= hs) & (f <= he)
        return m.to_numpy(dtype=bool)

    if "day_in_harvest" in df.columns:
        dih = pd.to_numeric(df["day_in_harvest"], errors="coerce")
        m = dih.notna() & (dih.astype(float) >= 1)
        return m.to_numpy(dtype=bool)

    base = pd.to_numeric(df.get("tallos_pred_baseline_dia", 0.0), errors="coerce").fillna(0.0)
    return (base > 0).to_numpy(dtype=bool)


def _make_relpos_bin(rel_pos: pd.Series) -> pd.Series:
    """
    Bins 0..1 en 6 tramos (consistente con lo que te está saliendo: global rows=6).
    """
    x = pd.to_numeric(rel_pos, errors="coerce")
    bins = [-np.inf, 0.0, 0.10, 0.25, 0.50, 0.75, np.inf]
    labels = ["B0", "B1", "B2", "B3", "B4", "B5"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True).astype("object")


def _safe_load_cap_tables(ver_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Intenta cargar:
      - cap_floor_share_by_relpos.parquet (segmentado)
      - cap_floor_share_by_relpos_global.parquet (fallback global)
    desde la carpeta de versión.
    """
    seg = ver_dir / "cap_floor_share_by_relpos.parquet"
    glb = ver_dir / "cap_floor_share_by_relpos_global.parquet"
    seg_df = read_parquet(seg) if seg.exists() else None
    glb_df = read_parquet(glb) if glb.exists() else None
    return seg_df, glb_df


def _apply_cap_floor(
    df: pd.DataFrame,
    share_col: str,
    seg_caps: pd.DataFrame | None,
    glb_caps: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Aplica cap/floor por (variedad_canon, area, tipo_sp, rel_pos_bin) y fallback global por rel_pos_bin.
    No hardcodea “punteos/colas”, solo limita extremos estadísticos.
    """
    out = df.copy()

    out[share_col] = pd.to_numeric(out[share_col], errors="coerce").fillna(0.0)
    out[share_col] = out[share_col].clip(lower=0.0)

    # preparar bins
    out["rel_pos_bin"] = _make_relpos_bin(out.get("rel_pos", np.nan))

    cap = pd.Series([np.nan] * len(out), index=out.index)
    floor = pd.Series([np.nan] * len(out), index=out.index)

    # segment caps
    if seg_caps is not None and len(seg_caps):
        s = seg_caps.copy()
        # normalizar nombres esperados mínimos
        for c in ["variedad_canon", "area", "tipo_sp"]:
            if c in s.columns:
                s[c] = _canon_str(s[c])
        if "rel_pos_bin" in s.columns:
            s["rel_pos_bin"] = s["rel_pos_bin"].astype("object")

        # detectar columnas cap/floor
        cap_col = "cap_share" if "cap_share" in s.columns else None
        floor_col = "floor_share" if "floor_share" in s.columns else None

        keys = [c for c in ["variedad_canon", "area", "tipo_sp", "rel_pos_bin"] if c in s.columns]
        if cap_col and keys:
            tmp = out.merge(
                s[keys + [cap_col] + ([floor_col] if floor_col else [])],
                on=keys,
                how="left",
                suffixes=("", "_cap"),
            )
            cap = pd.to_numeric(tmp[cap_col], errors="coerce")
            if floor_col:
                floor = pd.to_numeric(tmp[floor_col], errors="coerce")

    # global caps
    if glb_caps is not None and len(glb_caps):
        g = glb_caps.copy()
        if "rel_pos_bin" in g.columns:
            g["rel_pos_bin"] = g["rel_pos_bin"].astype("object")
        cap_col_g = "cap_share" if "cap_share" in g.columns else None
        floor_col_g = "floor_share" if "floor_share" in g.columns else None

        if cap_col_g and "rel_pos_bin" in g.columns:
            tmpg = out[["rel_pos_bin"]].merge(g[["rel_pos_bin", cap_col_g] + ([floor_col_g] if floor_col_g else [])],
                                             on="rel_pos_bin", how="left")
            cap_g = pd.to_numeric(tmpg[cap_col_g], errors="coerce")
            floor_g = pd.to_numeric(tmpg[floor_col_g], errors="coerce") if floor_col_g else pd.Series([np.nan]*len(out), index=out.index)

            cap = cap.where(cap.notna(), cap_g)
            floor = floor.where(floor.notna(), floor_g)

    out["cap_share"] = cap
    out["floor_share"] = floor

    # aplicar
    was_cap = pd.Series(False, index=out.index)
    was_floor = pd.Series(False, index=out.index)

    if out["cap_share"].notna().any():
        m = out[share_col] > out["cap_share"]
        was_cap = was_cap | m.fillna(False)
        out.loc[m.fillna(False), share_col] = out.loc[m.fillna(False), "cap_share"]

    if out["floor_share"].notna().any():
        m = out[share_col] < out["floor_share"]
        was_floor = was_floor | m.fillna(False)
        out.loc[m.fillna(False), share_col] = out.loc[m.fillna(False), "floor_share"]

    out["was_capped"] = was_cap
    out["was_floored"] = was_floor
    return out


def _smooth_and_renorm(df: pd.DataFrame, share_col: str, win: int = 5) -> pd.Series:
    """
    Suaviza share por ciclo con rolling mean centrado y renormaliza a sum=1.
    No impone “porcentajes fijos”; solo reduce picos espurios.
    """
    if win < 3:
        win = 3
    if win % 2 == 0:
        win += 1

    out = np.zeros(len(df), dtype=float)

    # asumimos df ya ordenado por ciclo/fecha
    grp = df.groupby("ciclo_id", dropna=False, sort=False)
    for _, sub in grp:
        idx = sub.index.to_numpy()
        s = pd.to_numeric(sub[share_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if len(s) == 0:
            continue
        # rolling mean centrado
        ss = pd.Series(s).rolling(window=win, center=True, min_periods=max(1, win // 2)).mean().to_numpy(dtype=float)
        ss = np.clip(ss, 0.0, None)
        tot = float(np.nansum(ss))
        if tot > 0:
            ss = ss / tot
        out[idx] = ss

    return pd.Series(out, index=df.index, dtype=float)


# =============================================================================
# Main
# =============================================================================
def main(version: str | None = None) -> None:
    ver_dir = _latest_version_dir() if version is None else (REGISTRY_ROOT / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe la versión: {ver_dir}")

    metrics_path = ver_dir / "metrics.json"
    model_path = ver_dir / "model_curva_share_dia.joblib"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No encontré metrics.json en {ver_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"No encontré modelo: {model_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names")
    if not feature_names:
        raise ValueError("metrics.json no trae feature_names (necesario para alinear dummies en apply).")

    model = load(model_path)

    # caps
    seg_caps, glb_caps = _safe_load_cap_tables(ver_dir)

    # inputs
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

    # alias posición
    _coalesce_cols(feat, "day_in_harvest", ["day_in_harvest", "day_in_harvest_pred", "day_in_harvest_pred_final"])
    _coalesce_cols(feat, "rel_pos", ["rel_pos", "rel_pos_pred", "rel_pos_pred_final"])
    _coalesce_cols(feat, "n_harvest_days", ["n_harvest_days", "n_harvest_days_pred", "n_harvest_days_pred_final"])

    # asegurar columnas
    for c in NUM_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in CAT_COLS:
        if c not in feat.columns:
            feat[c] = "UNKNOWN"

    # canon cats (importante antes de dummies)
    feat["variedad_canon"] = _canon_str(feat["variedad_canon"])
    feat["area"] = _canon_str(feat.get("area", "UNKNOWN")).fillna("UNKNOWN")
    feat["tipo_sp"] = _canon_str(feat.get("tipo_sp", "UNKNOWN")).fillna("UNKNOWN")

    # numerics
    for c in NUM_COLS:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    feat["tallos_pred_baseline_dia"] = pd.to_numeric(feat["tallos_pred_baseline_dia"], errors="coerce").fillna(0.0).astype(float)
    feat["tallos_proy"] = pd.to_numeric(feat["tallos_proy"], errors="coerce").fillna(0.0).astype(float)

    # universo
    uni = read_parquet(UNIVERSE_PATH).copy()
    _require(uni, ["ciclo_id", "fecha", "bloque_base", "variedad_canon"], "universe_harvest_grid_ml1")
    uni["ciclo_id"] = uni["ciclo_id"].astype(str)
    uni["fecha"] = _to_date(uni["fecha"])
    uni["bloque_base"] = _canon_int(uni["bloque_base"])
    uni["variedad_canon"] = _canon_str(uni["variedad_canon"])

    key = ["ciclo_id", "fecha", "bloque_base", "variedad_canon"]
    uni_k = uni[key].drop_duplicates()

    # panel = universe LEFT join feat
    panel = uni_k.merge(feat, on=key, how="left", suffixes=("", "_feat"))

    # completar cats defaults si quedaron NaN (por universe sin match)
    panel["variedad_canon"] = _canon_str(panel["variedad_canon"])
    panel["area"] = _canon_str(panel.get("area", "UNKNOWN")).fillna("UNKNOWN")
    panel["tipo_sp"] = _canon_str(panel.get("tipo_sp", "UNKNOWN")).fillna("UNKNOWN")

    # completar numerics faltantes
    for c in NUM_COLS:
        if c not in panel.columns:
            panel[c] = np.nan
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel["tallos_pred_baseline_dia"] = pd.to_numeric(panel.get("tallos_pred_baseline_dia", 0.0), errors="coerce").fillna(0.0).astype(float)
    panel["tallos_proy"] = pd.to_numeric(panel.get("tallos_proy", 0.0), errors="coerce").fillna(0.0).astype(float)

    # infer harvest mask
    is_h = _infer_is_harvest(panel)

    # X (dummies aligned)
    X = panel[NUM_COLS + CAT_COLS].copy()
    X = pd.get_dummies(X, columns=CAT_COLS, dummy_na=True)

    # IMPORTANT: align to training feature_names
    X = X.reindex(columns=feature_names, fill_value=0)

    # predict share raw
    share_pred = model.predict(X)
    share_pred = pd.to_numeric(pd.Series(share_pred, index=panel.index), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    share_pred = np.clip(share_pred, 0.0, None)
    share_pred = np.where(is_h, share_pred, 0.0)

    panel["share_pred_in"] = share_pred

    # cap/floor PRE
    tmp = panel.copy()
    tmp["_share_source"] = "model"
    tmp["share_curva_ml1_raw"] = tmp["share_pred_in"]

    tmp = _apply_cap_floor(tmp, "share_curva_ml1_raw", seg_caps, glb_caps)
    tmp["was_capped_pre"] = tmp["was_capped"]
    tmp["was_floored_pre"] = tmp["was_floored"]

    # renorm raw share por ciclo (si quedó todo 0, fallback baseline weights)
    base = tmp["tallos_pred_baseline_dia"].to_numpy(dtype=float)
    base_h = np.where(is_h, base, 0.0)

    # sum raw por ciclo
    sraw = tmp.groupby("ciclo_id", dropna=False)["share_curva_ml1_raw"].transform("sum").to_numpy(dtype=float)
    sb = pd.Series(base_h, index=tmp.index).groupby(tmp["ciclo_id"], dropna=False).transform("sum").to_numpy(dtype=float)

    share_raw = tmp["share_curva_ml1_raw"].to_numpy(dtype=float)
    share_ren = np.where(
        sraw > 0,
        share_raw / sraw,
        np.where(sb > 0, base_h / sb, 0.0),
    )
    tmp["share_curva_ml1"] = share_ren

    # smooth + renorm
    tmp = tmp.sort_values(["ciclo_id", "fecha"], kind="mergesort").reset_index(drop=True)
    tmp["share_smooth"] = _smooth_and_renorm(tmp, "share_curva_ml1", win=5)

    # cap/floor POST sobre share_smooth (opcional pero útil para evitar rebotes negativos/raros)
    tmp = _apply_cap_floor(tmp, "share_smooth", seg_caps, glb_caps)
    tmp["was_capped_post"] = tmp["was_capped"]
    tmp["was_floored_post"] = tmp["was_floored"]

    # renorm final post-cap
    s2 = tmp.groupby("ciclo_id", dropna=False)["share_smooth"].transform("sum").to_numpy(dtype=float)
    tmp["share_smooth"] = np.where(s2 > 0, tmp["share_smooth"].to_numpy(dtype=float) / s2, 0.0)

    # cyc_total = tallos_proy max por ciclo (fallback sum baseline)
    tproy_max = tmp.groupby("ciclo_id", dropna=False)["tallos_proy"].transform("max").to_numpy(dtype=float)
    tproy_max = np.where(np.isfinite(tproy_max), tproy_max, 0.0)
    cyc_total = np.where(tproy_max > 0, tproy_max, sb)

    tmp["tallos_pred_ml1_dia_smooth"] = cyc_total * tmp["share_smooth"].to_numpy(dtype=float)

    # factor compatible downstream
    eps = 1e-9
    tmp["factor_curva_ml1_raw_smooth"] = np.where(
        is_h,
        tmp["tallos_pred_ml1_dia_smooth"].to_numpy(dtype=float) / (base + eps),
        1.0,
    )

    FACTOR_MIN, FACTOR_MAX = 0.2, 5.0
    tmp["factor_curva_ml1_raw"] = tmp["factor_curva_ml1_raw_smooth"]
    tmp["factor_curva_ml1"] = np.clip(tmp["factor_curva_ml1_raw"].to_numpy(dtype=float), FACTOR_MIN, FACTOR_MAX)
    tmp["factor_curva_ml1"] = np.where(np.isfinite(tmp["factor_curva_ml1"]), tmp["factor_curva_ml1"], 1.0)

    # metadata
    tmp["ml1_version"] = ver_dir.name
    tmp["created_at"] = pd.Timestamp.utcnow()

    # outputs
    out_cols = [
        "ciclo_id",
        "fecha",
        "bloque_base",
        "variedad_canon",
        "factor_curva_ml1",
        "factor_curva_ml1_raw",
        "ml1_version",
        "created_at",
        "_share_source",
        "cap_share",
        "share_pred_in",
        "share_smooth",
        "share_curva_ml1",
        "tallos_pred_ml1_dia_smooth",
        "factor_curva_ml1_raw_smooth",
        "was_capped_pre",
        "was_capped_post",
        "was_floored_pre",
        "was_floored_post",
    ]
    out_cols = [c for c in out_cols if c in tmp.columns]

    out = tmp[out_cols].sort_values(["bloque_base", "variedad_canon", "fecha"], kind="mergesort").reset_index(drop=True)

    write_parquet(out, OUT_PATH)

    # quick audits
    # share sum by cycle
    ss = out.groupby("ciclo_id", dropna=False)["share_smooth"].sum()
    # mass balance
    tallos_sum = out.groupby("ciclo_id", dropna=False)["tallos_pred_ml1_dia_smooth"].sum()
    proy = tmp.groupby("ciclo_id", dropna=False)["tallos_proy"].max()
    max_abs = float((tallos_sum - proy).abs().max()) if len(tallos_sum) else float("nan")

    print(f"OK -> {OUT_PATH} | rows={len(out):,} | version={ver_dir.name}")
    print(f"[CHECK] share_smooth sum min/max: {float(ss.min()):.6f} / {float(ss.max()):.6f}")
    print(f"[CHECK] ciclo mass-balance ML1 vs tallos_proy | max abs diff: {max_abs:.12f}")


if __name__ == "__main__":
    main(version=None)
