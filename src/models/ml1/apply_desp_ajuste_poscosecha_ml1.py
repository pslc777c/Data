from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


IN_UNIVERSE = Path("data/gold/pred_poscosecha_ml1_hidr_grado_dia_bloque_destino.parquet")
FEATURES_MIX_PRED = Path("data/gold/features_mix_pred_dia_destino.parquet")

REG_GATE = Path("models_registry/ml1/desp_poscosecha_gate")
REG_DN = Path("models_registry/ml1/desp_poscosecha_reg_normal")
REG_DE = Path("models_registry/ml1/desp_poscosecha_reg_extremo")
REG_AJ = Path("models_registry/ml1/ajuste_poscosecha")

OUT_PATH = Path("data/gold/pred_poscosecha_ml1_full_grado_dia_bloque_destino.parquet")


def _latest_version_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"No existe {root}")
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No hay versiones dentro de {root}")
    return sorted(dirs, key=lambda p: p.name)[-1]


def _load_meta_and_model(root: Path, version: str | None):
    ver_dir = _latest_version_dir(root) if version is None else (root / version)
    if not ver_dir.exists():
        raise FileNotFoundError(f"No existe versión: {ver_dir}")
    with open(ver_dir / "metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load(ver_dir / "model.joblib")
    return ver_dir.name, meta, model


def _canon_destino(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def main(
    version_gate: str | None = None,
    version_desp: str | None = None,
    version_aj: str | None = None,
) -> None:
    v_gate, meta_gate, model_gate = _load_meta_and_model(REG_GATE, version_gate)
    v_dn, meta_dn, model_dn = _load_meta_and_model(REG_DN, version_desp)
    v_de, meta_de, model_de = _load_meta_and_model(REG_DE, version_desp)
    v_aj, meta_aj, model_aj = _load_meta_and_model(REG_AJ, version_aj)

    df0 = read_parquet(IN_UNIVERSE).copy()
    df0.columns = [str(c).strip() for c in df0.columns]

    need = {"fecha", "bloque_base", "variedad_canon", "grado", "destino"}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"Universe sin columnas: {sorted(miss)}")

    dh_col = None
    for c in ["dh_dias_ml1", "dh_dias_pred_ml1", "dh_dias_pred", "dh_dias"]:
        if c in df0.columns:
            dh_col = c
            break
    if dh_col is None:
        raise ValueError("No encontré columna dh (esperaba dh_dias_ml1 o similar).")

    df0["fecha"] = pd.to_datetime(df0["fecha"], errors="coerce").dt.normalize()
    df0["destino"] = _canon_destino(df0["destino"])
    df0[dh_col] = pd.to_numeric(df0[dh_col], errors="coerce").astype("Int64")

    df0["fecha_post_pred_ml1"] = df0["fecha"] + pd.to_timedelta(df0[dh_col].fillna(0).astype(int), unit="D")

    df0["dow"] = df0["fecha_post_pred_ml1"].dt.dayofweek.astype("Int64")
    df0["month"] = df0["fecha_post_pred_ml1"].dt.month.astype("Int64")
    df0["weekofyear"] = df0["fecha_post_pred_ml1"].dt.isocalendar().week.astype("Int64")

    if not FEATURES_MIX_PRED.exists():
        raise FileNotFoundError(f"No existe features pred: {FEATURES_MIX_PRED}")

    mix = read_parquet(FEATURES_MIX_PRED).copy()
    mix.columns = [str(c).strip() for c in mix.columns]
    if not {"fecha_post", "destino"} <= set(mix.columns):
        raise ValueError("features_mix_pred debe tener: fecha_post, destino")

    mix["fecha_post"] = pd.to_datetime(mix["fecha_post"], errors="coerce").dt.normalize()
    mix["destino"] = _canon_destino(mix["destino"])
    mix = mix.dropna(subset=["fecha_post", "destino"]).copy()

    df = df0.merge(
        mix,
        left_on=["fecha_post_pred_ml1", "destino"],
        right_on=["fecha_post", "destino"],
        how="left",
    )

    feat_cols_from_mix = [c for c in mix.columns if c not in {"fecha_post", "destino"}]
    if "fecha_post" in df.columns:
        df = df.drop(columns=["fecha_post"])

    # -----------------------------
    # X para GATE (meta_gate)
    # -----------------------------
    cat_gate = meta_gate.get("features_cat", ["destino"])
    num_gate = meta_gate.get("features_num", ["dow", "month", "weekofyear"])
    for c in cat_gate:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    for c in num_gate:
        if c not in df.columns:
            df[c] = np.nan

    X_gate = df[cat_gate + num_gate]

    gate_thr = float(meta_gate.get("gate_threshold", 0.5))
    p_ext = pd.Series(model_gate.predict_proba(X_gate)[:, 1], index=df.index)
    df["desp_extremo_prob_ml1"] = pd.to_numeric(p_ext, errors="coerce").fillna(0.0)
    df["desp_es_extremo_ml1"] = (df["desp_extremo_prob_ml1"] >= gate_thr).astype("Int64")

    # -----------------------------
    # X para DESP normal/extremo (meta_dn)
    # -----------------------------
    cat_d = meta_dn.get("features_cat", ["destino"])
    num_d = meta_dn.get("features_num", ["dow", "month", "weekofyear"])
    for c in cat_d:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    for c in num_d:
        if c not in df.columns:
            df[c] = np.nan

    X_d = df[cat_d + num_d]

    pred_n = pd.Series(model_dn.predict(X_d), index=df.index)
    pred_e = pd.Series(model_de.predict(X_d), index=df.index)
    desp_pred = np.where(df["desp_es_extremo_ml1"].fillna(0).astype(int).to_numpy() == 1, pred_e, pred_n)
    desp_pred = pd.to_numeric(pd.Series(desp_pred, index=df.index), errors="coerce")

    lo_d, hi_d = meta_dn.get("clip_range_apply", [0.0, 0.95])
    df["desp_pct_ml1"] = desp_pred.clip(lower=float(lo_d), upper=float(hi_d)).fillna(0.30)
    df["factor_desp_ml1"] = (1.0 - df["desp_pct_ml1"]).clip(lower=0.05, upper=1.0)

    # -----------------------------
    # AJUSTE (meta_aj)
    # -----------------------------
    cat_a = meta_aj.get("features_cat", ["destino"])
    num_a = meta_aj.get("features_num", ["dow", "month", "weekofyear"])
    for c in cat_a:
        if c not in df.columns:
            df[c] = "UNKNOWN"
    for c in num_a:
        if c not in df.columns:
            df[c] = np.nan

    X_a = df[cat_a + num_a]
    lo_a, hi_a = meta_aj.get("clip_range_apply", [0.80, 1.50])
    aj_raw = pd.Series(model_aj.predict(X_a), index=df.index)
    df["ajuste_ml1_raw"] = pd.to_numeric(aj_raw, errors="coerce")
    df["ajuste_ml1"] = df["ajuste_ml1_raw"].clip(lower=float(lo_a), upper=float(hi_a)).fillna(1.0)

    # -----------------------------
    # cajas_postcosecha
    # -----------------------------
    cajas_col = None
    for c in [
        "cajas_split_grado_dia",
        "cajas_split",
        "cajas_seed_split",
        "cajas_destino_grado_dia",
        "cajas_iniciales",
        "cajas_ml1_grado_dia",
        "cajas_ml1_grado_dia_in",
    ]:
        if c in df.columns:
            cajas_col = c
            break
    if cajas_col is None:
        raise ValueError("No encontré columna de cajas base para postcosecha.")

    df["cajas_post_base_col"] = cajas_col

    base = pd.to_numeric(df[cajas_col], errors="coerce").fillna(0.0)
    hidr = pd.to_numeric(df.get("factor_hidr_ml1"), errors="coerce").fillna(1.0)
    desp_factor = pd.to_numeric(df["factor_desp_ml1"], errors="coerce").fillna(1.0)
    aj = pd.to_numeric(df["ajuste_ml1"], errors="coerce").where(lambda s: s != 0, np.nan).fillna(1.0)

    df["cajas_postcosecha_ml1"] = base * hidr * desp_factor / aj

    # versions + limpieza
    df["ml1_gate_version"] = v_gate
    df["ml1_desp_version"] = v_dn
    df["ml1_ajuste_version"] = v_aj
    df["created_at"] = pd.Timestamp.now("UTC")

    keep_from_universe = list(df0.columns)
    keep_extra = [
        "fecha_post_pred_ml1", "dow", "month", "weekofyear",
        "desp_extremo_prob_ml1", "desp_es_extremo_ml1",
        "desp_pct_ml1", "factor_desp_ml1",
        "ajuste_ml1_raw", "ajuste_ml1",
        "cajas_post_base_col", "cajas_postcosecha_ml1",
        "ml1_gate_version", "ml1_desp_version", "ml1_ajuste_version",
        "created_at",
    ]

    keep = []
    for c in keep_from_universe + keep_extra:
        if c in df.columns and c not in keep:
            keep.append(c)

    out = df[keep].copy()
    write_parquet(out, OUT_PATH)

    print(f"OK -> {OUT_PATH} | rows={len(out):,} | desp={v_dn} (gate={v_gate}) | ajuste={v_aj} | cajas_base={cajas_col}")
    print(f"[INFO] features_pred={FEATURES_MIX_PRED} | dropped_features_cols={len(feat_cols_from_mix)}")


if __name__ == "__main__":
    main(version_gate=None, version_desp=None, version_aj=None)
