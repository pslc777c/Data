from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from joblib import load

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"

IN_UNIVERSE = GOLD / "pred_poscosecha_ml2_hidr_grado_dia_bloque_destino.parquet"

REG_DESP = ROOT / "models_registry" / "ml1" / "desp_poscosecha"
REG_AJ = ROOT / "models_registry" / "ml1" / "ajuste_poscosecha"

OUT_PATH = GOLD / "pred_poscosecha_ml2_full_grado_dia_bloque_destino.parquet"

NUM_COLS = ["dow", "month", "weekofyear"]
CAT_COLS = ["destino"]


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
        raise FileNotFoundError(f"No existe version: {ver_dir}")
    with open(ver_dir / "metrics.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    model = load(ver_dir / "model.joblib")
    return ver_dir.name, meta, model


def _as_list(cols) -> list[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return []


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in items:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _split_num_cat(cols: list[str]) -> tuple[list[str], list[str]]:
    cat = [c for c in cols if c in CAT_COLS]
    num = [c for c in cols if c not in set(cat)]
    return num, cat


def _feature_lists_from_preprocessor(model: object) -> tuple[list[str], list[str]]:
    named_steps = getattr(model, "named_steps", None)
    if not named_steps or "pre" not in named_steps:
        return [], []

    pre = named_steps["pre"]
    transformers = getattr(pre, "transformers_", None) or getattr(pre, "transformers", None)
    if transformers is None:
        return [], []

    num_cols: list[str] = []
    cat_cols: list[str] = []

    for name, _transformer, cols in transformers:
        if name == "remainder":
            continue
        cols_list = _as_list(cols)
        if not cols_list:
            continue
        if "cat" in str(name).lower():
            cat_cols.extend(cols_list)
        elif "num" in str(name).lower():
            num_cols.extend(cols_list)
        else:
            for c in cols_list:
                if c in CAT_COLS:
                    cat_cols.append(c)
                else:
                    num_cols.append(c)

    return _dedupe_keep_order(num_cols), _dedupe_keep_order(cat_cols)


def _resolve_feature_lists(meta: dict, model: object) -> tuple[list[str], list[str]]:
    # 1) Prefer explicit lists from metrics when present.
    num_meta = _as_list(meta.get("features_num") or meta.get("num_cols"))
    cat_meta = _as_list(meta.get("features_cat") or meta.get("cat_cols"))
    if num_meta or cat_meta:
        num = _dedupe_keep_order(num_meta)
        cat = _dedupe_keep_order(cat_meta)
        # If metrics accidentally put categorical columns inside num, move them.
        moved = [c for c in num if c in CAT_COLS]
        if moved:
            num = [c for c in num if c not in set(moved)]
            cat = _dedupe_keep_order(cat + moved)
        return num, cat

    # 2) Fallback to feature_names_in_ from fitted sklearn pipeline.
    feat_in = getattr(model, "feature_names_in_", None)
    if feat_in is not None:
        num, cat = _split_num_cat([str(c) for c in feat_in])
        if num or cat:
            return _dedupe_keep_order(num), _dedupe_keep_order(cat)

    # 3) Fallback to ColumnTransformer column assignments.
    num_pre, cat_pre = _feature_lists_from_preprocessor(model)
    if num_pre or cat_pre:
        return num_pre, cat_pre

    # 4) Last-resort defaults.
    return list(NUM_COLS), list(CAT_COLS)


def _build_X(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    x = df.copy()

    for c in num_cols:
        if c not in x.columns:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    for c in cat_cols:
        if c not in x.columns:
            x[c] = "UNKNOWN"
        x[c] = x[c].astype(str).fillna("UNKNOWN")
        if c == "destino":
            x[c] = x[c].str.upper().str.strip()

    return x[num_cols + cat_cols].copy()


def main(version_desp: str | None = None, version_aj: str | None = None) -> None:
    v_desp, meta_desp, model_desp = _load_meta_and_model(REG_DESP, version_desp)
    v_aj, meta_aj, model_aj = _load_meta_and_model(REG_AJ, version_aj)

    df = read_parquet(IN_UNIVERSE).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {
        "fecha",
        "bloque_base",
        "variedad_canon",
        "grado",
        "destino",
        "fecha_post_pred_ml1",
        "factor_hidr_ml1",
        "cajas_split_grado_dia",
    }
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"Universe Hidr ML2 sin columnas: {sorted(miss)}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["fecha_post_pred_ml1"] = pd.to_datetime(df["fecha_post_pred_ml1"], errors="coerce").dt.normalize()
    df["destino"] = df["destino"].astype(str).str.upper().str.strip()
    df["grado"] = pd.to_numeric(df["grado"], errors="coerce").astype("Int64")

    # Calendar features on postharvest day.
    df["dow"] = df["fecha_post_pred_ml1"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post_pred_ml1"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post_pred_ml1"].dt.isocalendar().week.astype("Int64")

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = "UNKNOWN"

    # DESP
    num_desp, cat_desp = _resolve_feature_lists(meta_desp, model_desp)
    lo_d, hi_d = meta_desp.get("clip_range_apply", [0.05, 1.00])
    X_desp = _build_X(df, num_desp, cat_desp)
    df["factor_desp_ml1_raw"] = pd.to_numeric(pd.Series(model_desp.predict(X_desp)), errors="coerce")
    df["factor_desp_ml1"] = df["factor_desp_ml1_raw"].clip(lower=float(lo_d), upper=float(hi_d))

    # AJUSTE
    num_aj, cat_aj = _resolve_feature_lists(meta_aj, model_aj)
    lo_a, hi_a = meta_aj.get("clip_range_apply", [0.80, 1.50])
    X_aj = _build_X(df, num_aj, cat_aj)
    df["ajuste_ml1_raw"] = pd.to_numeric(pd.Series(model_aj.predict(X_aj)), errors="coerce")
    df["ajuste_ml1"] = df["ajuste_ml1_raw"].clip(lower=float(lo_a), upper=float(hi_a))

    df["ml1_desp_version"] = v_desp
    df["ml1_ajuste_version"] = v_aj
    df["created_at"] = pd.Timestamp.now(tz="UTC")

    # Final postharvest estimate seeded by ML2 field output.
    df["cajas_postcosecha_ml1"] = (
        pd.to_numeric(df["cajas_split_grado_dia"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["factor_hidr_ml1"], errors="coerce").fillna(1.0)
        * pd.to_numeric(df["factor_desp_ml1"], errors="coerce").fillna(1.0)
        / pd.to_numeric(df["ajuste_ml1"], errors="coerce").replace(0, np.nan).fillna(1.0)
    )

    write_parquet(df, OUT_PATH)
    print(
        f"[OK] Wrote: {OUT_PATH} rows={len(df):,} desp={v_desp} ajuste={v_aj} "
        f"nfeat_desp={len(num_desp) + len(cat_desp)} nfeat_aj={len(num_aj) + len(cat_aj)}"
    )


if __name__ == "__main__":
    main(version_desp=None, version_aj=None)
