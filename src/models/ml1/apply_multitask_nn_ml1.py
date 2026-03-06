from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA_DIR = ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "gold" / "ml1_nn" / "ds_ml1_nn_v1.parquet"
MODELS_DIR = DATA_DIR / "models" / "ml1_nn"
OUT_DIR = DATA_DIR / "gold" / "ml1_nn"
IN_FACT_PESO = DATA_DIR / "silver" / "fact_peso_tallo_real_grado_dia.parquet"
IN_FEATURES_COSECHA = DATA_DIR / "features" / "features_cosecha_bloque_fecha.parquet"

_STEM_WEIGHT_CAPS_CACHE: dict[str, object] | None = None
_SHARE_GRADE_PRIORS_CACHE: dict[str, object] | None = None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("apply_multitask_nn_ml1")
    ap.add_argument("--run-id", default=None, help="Model run_id. If omitted, latest model is used.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=None)
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _safe_div(a: pd.Series, b: pd.Series, default: float = np.nan) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    out = aa / bb.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return np.nan
    v = v[m]
    w = w[m]
    if len(v) == 0:
        return np.nan
    if len(v) == 1:
        return float(v[0])
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    tot = float(cw[-1])
    if not np.isfinite(tot) or tot <= 0.0:
        return np.nan
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.interp(q * tot, cw, v))


def _build_stem_weight_caps() -> dict[str, object]:
    if not IN_FACT_PESO.exists():
        return {}
    fact = read_parquet(IN_FACT_PESO).copy()
    need = {"grado", "peso_tallo_real_g", "tallos_real"}
    if not need.issubset(set(fact.columns)):
        return {}

    # Prefer reliable recent operational history.
    if "fecha" in fact.columns:
        fecha = pd.to_datetime(fact["fecha"], errors="coerce").dt.normalize()
        fact = fact.loc[fecha >= pd.Timestamp("2025-05-01")].copy()

    if "variedad" in fact.columns:
        fact["variedad"] = _canon_str(fact["variedad"])
    else:
        fact["variedad"] = "UNKNOWN"
    fact["grado"] = pd.to_numeric(fact["grado"], errors="coerce").round().astype("Int64")
    fact["peso_tallo_real_g"] = pd.to_numeric(fact["peso_tallo_real_g"], errors="coerce")
    fact["tallos_real"] = pd.to_numeric(fact["tallos_real"], errors="coerce")

    # Remove extreme data glitches before computing quantile caps.
    fact = fact.loc[
        fact["grado"].notna()
        & fact["peso_tallo_real_g"].notna()
        & fact["tallos_real"].notna()
        & (fact["tallos_real"] > 0.0)
        & (fact["peso_tallo_real_g"] > 1.0)
        & (fact["peso_tallo_real_g"] < 120.0)
    ].copy()
    if fact.empty:
        return {}

    def _caps_for_group(g: pd.DataFrame, q_lo: float = 0.05, q_hi: float = 0.95) -> tuple[float, float, int, float]:
        x = pd.to_numeric(g["peso_tallo_real_g"], errors="coerce").to_numpy(dtype=np.float64)
        w = pd.to_numeric(g["tallos_real"], errors="coerce").to_numpy(dtype=np.float64)
        lo = _weighted_quantile(x, w, q_lo)
        hi = _weighted_quantile(x, w, q_hi)
        n = int(np.isfinite(x).sum())
        wsum = float(np.nansum(np.where(np.isfinite(w), w, 0.0)))
        return lo, hi, n, wsum

    rows_g: list[dict] = []
    for g, sub in fact.groupby("grado", dropna=False):
        lo, hi, n, wsum = _caps_for_group(sub)
        if np.isfinite(lo) and np.isfinite(hi) and (hi > lo) and n >= 120 and wsum >= 10000.0:
            rows_g.append({"grado": int(g), "lo": float(lo), "hi": float(hi), "n": n, "wsum": wsum})
    caps_g = pd.DataFrame(rows_g)

    rows_vg: list[dict] = []
    for (v, g), sub in fact.groupby(["variedad", "grado"], dropna=False):
        lo, hi, n, wsum = _caps_for_group(sub)
        if np.isfinite(lo) and np.isfinite(hi) and (hi > lo) and n >= 80 and wsum >= 5000.0:
            rows_vg.append(
                {"variedad": str(v), "grado": int(g), "lo": float(lo), "hi": float(hi), "n": n, "wsum": wsum}
            )
    caps_vg = pd.DataFrame(rows_vg)

    x_all = fact["peso_tallo_real_g"].to_numpy(dtype=np.float64)
    w_all = fact["tallos_real"].to_numpy(dtype=np.float64)
    g_lo = _weighted_quantile(x_all, w_all, 0.02)
    g_hi = _weighted_quantile(x_all, w_all, 0.98)
    if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_hi <= g_lo:
        g_lo, g_hi = 5.0, 70.0

    out: dict[str, object] = {"global_lo": float(g_lo), "global_hi": float(g_hi)}
    if not caps_g.empty:
        out["g_lo"] = pd.Series(caps_g["lo"].to_numpy(dtype=np.float64), index=caps_g["grado"].astype("Int64"))
        out["g_hi"] = pd.Series(caps_g["hi"].to_numpy(dtype=np.float64), index=caps_g["grado"].astype("Int64"))
    if not caps_vg.empty:
        idx_vg = pd.MultiIndex.from_frame(caps_vg[["variedad", "grado"]])
        out["vg_lo"] = pd.Series(caps_vg["lo"].to_numpy(dtype=np.float64), index=idx_vg)
        out["vg_hi"] = pd.Series(caps_vg["hi"].to_numpy(dtype=np.float64), index=idx_vg)
    return out


def _get_stem_weight_caps() -> dict[str, object]:
    global _STEM_WEIGHT_CAPS_CACHE
    if _STEM_WEIGHT_CAPS_CACHE is None:
        _STEM_WEIGHT_CAPS_CACHE = _build_stem_weight_caps()
    return _STEM_WEIGHT_CAPS_CACHE


def _build_share_grade_priors() -> dict[str, object]:
    if not IN_FEATURES_COSECHA.exists():
        return {}
    src = read_parquet(IN_FEATURES_COSECHA).copy()
    need_any = {"grado", "tallos_real_grado"} | {"grado", "share_grado_real", "tallos_real_dia"}
    if not ({"grado", "tallos_real_grado"} <= set(src.columns) or {"grado", "share_grado_real", "tallos_real_dia"} <= set(src.columns)):
        return {}

    src["grado"] = pd.to_numeric(src["grado"], errors="coerce").round().astype("Int64")
    if "variedad_canon" in src.columns:
        src["variedad_canon"] = _canon_str(src["variedad_canon"])
    else:
        src["variedad_canon"] = "UNKNOWN"

    tallos = pd.to_numeric(src.get("tallos_real_grado"), errors="coerce")
    if not tallos.notna().any():
        share = pd.to_numeric(src.get("share_grado_real"), errors="coerce")
        tallos_day = pd.to_numeric(src.get("tallos_real_dia"), errors="coerce")
        tallos = share * tallos_day
    src["tallos_real_grado"] = pd.to_numeric(tallos, errors="coerce")

    src = src.loc[
        src["grado"].notna()
        & src["tallos_real_grado"].notna()
        & (src["tallos_real_grado"] > 0.0)
    ].copy()
    if src.empty:
        return {}

    g = (
        src.groupby("grado", dropna=False)["tallos_real_grado"]
        .sum()
        .astype("float64")
    )
    gtot = float(g.sum())
    if not np.isfinite(gtot) or gtot <= 0.0:
        return {}
    g_share = (g / gtot).astype("float64")

    vg_src = (
        src.groupby(["variedad_canon", "grado"], dropna=False)["tallos_real_grado"]
        .sum()
        .reset_index()
    )
    vtot = vg_src.groupby("variedad_canon", dropna=False)["tallos_real_grado"].transform("sum")
    vg_src["share"] = np.where(vtot > 0.0, vg_src["tallos_real_grado"] / vtot, np.nan)
    vg_src = vg_src.loc[vg_src["share"].notna()].copy()
    vg_idx = pd.MultiIndex.from_frame(vg_src[["variedad_canon", "grado"]])
    vg_share = pd.Series(vg_src["share"].to_numpy(dtype=np.float64), index=vg_idx)

    return {
        "g_share": g_share,
        "vg_share": vg_share,
    }


def _get_share_grade_priors() -> dict[str, object]:
    global _SHARE_GRADE_PRIORS_CACHE
    if _SHARE_GRADE_PRIORS_CACHE is None:
        _SHARE_GRADE_PRIORS_CACHE = _build_share_grade_priors()
    return _SHARE_GRADE_PRIORS_CACHE


def _regularize_share_by_hist_prior(hg: pd.DataFrame, share_norm: pd.Series, day_key: list[str]) -> pd.Series:
    priors = _get_share_grade_priors()
    if not priors or "grado" not in hg.columns:
        return share_norm

    grade = pd.to_numeric(hg.get("grado"), errors="coerce").round().astype("Int64")
    g_share = priors.get("g_share")
    vg_share = priors.get("vg_share")
    prior = pd.Series(np.nan, index=hg.index, dtype="float64")

    if "variedad_canon" in hg.columns and isinstance(vg_share, pd.Series) and len(vg_share) > 0:
        variedad = _canon_str(hg["variedad_canon"]).astype("string")
        idx = pd.MultiIndex.from_arrays([variedad, grade], names=["variedad_canon", "grado"])
        prior = pd.Series(idx.map(vg_share), index=hg.index, dtype="float64")

    if isinstance(g_share, pd.Series) and len(g_share) > 0:
        prior = prior.where(prior.notna(), grade.map(g_share))

    # If prior is unknown, keep current normalized share.
    prior = prior.where(prior.notna(), share_norm)
    prior = prior.fillna(0.0).clip(lower=0.0)

    # Normalize prior per day (simplex per (cycle, day, block, variety)).
    grp_vals = [hg[c] for c in day_key]
    pden = prior.groupby(grp_vals, dropna=False).transform("sum")
    pn = prior.groupby(grp_vals, dropna=False).transform("size").clip(lower=1)
    prior_norm = pd.Series(np.where(pden > 0.0, prior / pden, 1.0 / pn.astype(float)), index=hg.index, dtype="float64")
    prior_norm = prior_norm.fillna(0.0).clip(lower=0.0)

    pred0 = pd.to_numeric(share_norm, errors="coerce").fillna(0.0).clip(lower=0.0)

    # Data-driven shrinkage: estimate lambda from observed real rows (no fixed multipliers).
    # y_blend = (1-lambda)*pred + lambda*prior
    # lambda chosen by minimizing weighted MAE on rows with real share labels.
    lam = 0.0
    lam_by_grade: dict[int, float] = {}
    if {"target_share_grado", "mask_target_share_grado"} <= set(hg.columns):
        y_true = pd.to_numeric(hg.get("target_share_grado"), errors="coerce")
        w = pd.to_numeric(hg.get("mask_target_share_grado"), errors="coerce").fillna(0.0)
        m_obs = y_true.notna() & (w > 0.0)
        if bool(m_obs.any()):
            cand = np.linspace(0.0, 0.90, 19, dtype=np.float64)
            best = np.inf
            best_lam = 0.0
            yt = y_true.loc[m_obs].to_numpy(dtype=np.float64)
            ww = w.loc[m_obs].to_numpy(dtype=np.float64)
            pp = pred0.loc[m_obs].to_numpy(dtype=np.float64)
            pr = prior_norm.loc[m_obs].to_numpy(dtype=np.float64)
            wsum = float(np.sum(ww))
            if wsum > 0.0:
                for l in cand:
                    yy = (1.0 - l) * pp + l * pr
                    mae = float(np.sum(np.abs(yy - yt) * ww) / wsum)
                    if mae + 1e-12 < best:
                        best = mae
                        best_lam = float(l)
                lam = best_lam

            # Grade-specific calibration (data-driven) where support is sufficient.
            if "grado" in hg.columns:
                gnum = pd.to_numeric(hg["grado"], errors="coerce").round().astype("Int64")
                obs = hg.loc[m_obs].copy()
                obs["__g"] = gnum.loc[m_obs]
                obs["__y"] = y_true.loc[m_obs]
                obs["__w"] = w.loc[m_obs]
                obs["__p"] = pred0.loc[m_obs]
                obs["__r"] = prior_norm.loc[m_obs]
                for gg, sg in obs.groupby("__g", dropna=False, sort=False):
                    if pd.isna(gg) or len(sg) < 120:
                        continue
                    ytg = pd.to_numeric(sg["__y"], errors="coerce").to_numpy(dtype=np.float64)
                    wwg = pd.to_numeric(sg["__w"], errors="coerce").to_numpy(dtype=np.float64)
                    ppg = pd.to_numeric(sg["__p"], errors="coerce").to_numpy(dtype=np.float64)
                    prg = pd.to_numeric(sg["__r"], errors="coerce").to_numpy(dtype=np.float64)
                    wsumg = float(np.sum(wwg))
                    if wsumg <= 0.0:
                        continue
                    bestg = np.inf
                    best_lg = lam
                    for l in cand:
                        yy = (1.0 - l) * ppg + l * prg
                        mae = float(np.sum(np.abs(yy - ytg) * wwg) / wsumg)
                        if mae + 1e-12 < bestg:
                            bestg = mae
                            best_lg = float(l)
                    lam_by_grade[int(gg)] = best_lg
    if "grado" in hg.columns and lam_by_grade:
        gnum = pd.to_numeric(hg["grado"], errors="coerce").round().astype("Int64")
        lam_row = gnum.map(lam_by_grade).astype("float64")
        lam_row = lam_row.where(lam_row.notna(), lam).clip(lower=0.0, upper=0.95)
        out = ((1.0 - lam_row) * pred0 + lam_row * prior_norm).clip(lower=0.0)
    else:
        out = ((1.0 - lam) * pred0 + lam * prior_norm).clip(lower=0.0)
    den = out.groupby(grp_vals, dropna=False).transform("sum")
    nn = out.groupby(grp_vals, dropna=False).transform("size").clip(lower=1)
    out = pd.Series(np.where(den > 0.0, out / den, 1.0 / nn.astype(float)), index=hg.index, dtype="float64")
    return out.fillna(0.0).clip(lower=0.0)


def _apply_pred_factor_peso_caps(df: pd.DataFrame) -> None:
    if "pred_factor_peso_tallo" not in df.columns or "peso_tallo_baseline_g" not in df.columns:
        return

    caps = _get_stem_weight_caps()
    if not caps:
        return

    base = pd.to_numeric(df["peso_tallo_baseline_g"], errors="coerce")
    fac = pd.to_numeric(df["pred_factor_peso_tallo"], errors="coerce")
    m = base.notna() & (base > 0.0) & fac.notna()
    if not bool(m.any()):
        return

    grade = pd.to_numeric(df.get("grado"), errors="coerce").round().astype("Int64")
    if "variedad_canon" in df.columns:
        variedad = _canon_str(df["variedad_canon"]).astype("string")
    else:
        variedad = pd.Series(["UNKNOWN"] * len(df), index=df.index, dtype="string")

    pred_w = base * fac

    lo = pd.Series(np.nan, index=df.index, dtype="float64")
    hi = pd.Series(np.nan, index=df.index, dtype="float64")

    vg_lo = caps.get("vg_lo")
    vg_hi = caps.get("vg_hi")
    if isinstance(vg_lo, pd.Series) and isinstance(vg_hi, pd.Series) and len(vg_lo) > 0 and len(vg_hi) > 0:
        idx = pd.MultiIndex.from_arrays([variedad, grade], names=["variedad", "grado"])
        lo = pd.Series(idx.map(vg_lo), index=df.index, dtype="float64")
        hi = pd.Series(idx.map(vg_hi), index=df.index, dtype="float64")

    g_lo = caps.get("g_lo")
    g_hi = caps.get("g_hi")
    if isinstance(g_lo, pd.Series) and isinstance(g_hi, pd.Series) and len(g_lo) > 0 and len(g_hi) > 0:
        lo = lo.where(lo.notna(), grade.map(g_lo))
        hi = hi.where(hi.notna(), grade.map(g_hi))

    lo = lo.fillna(float(caps.get("global_lo", 5.0)))
    hi = hi.fillna(float(caps.get("global_hi", 70.0)))
    hi = hi.where(hi >= lo, lo)

    pred_w_clip = pred_w.clip(lower=lo, upper=hi)
    fac_new = _safe_div(pred_w_clip, base, default=np.nan).clip(lower=0.60, upper=1.60)
    m_set = m & fac_new.notna()
    if bool(m_set.any()):
        vals = pd.to_numeric(fac_new.loc[m_set], errors="coerce").astype(np.float32).to_numpy()
        df.loc[m_set, "pred_factor_peso_tallo"] = vals
        if "factor_peso_tallo_ML1" in df.columns:
            df.loc[m_set, "factor_peso_tallo_ML1"] = vals


def _assign_predictions(
    df: pd.DataFrame,
    yhat: np.ndarray,
    targets: list[str],
    target_clips: dict,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    pred_cols: list[str] = []
    pred_alias_map: dict[str, str] = {}
    for i, t in enumerate(targets):
        col = "pred_" + t.replace("target_", "", 1)
        ycol = yhat[:, i].astype(np.float32)
        if t in target_clips and isinstance(target_clips[t], (list, tuple)) and len(target_clips[t]) == 2:
            lo, hi = float(target_clips[t][0]), float(target_clips[t][1])
            ycol = np.clip(ycol, lo, hi)
        out[col] = ycol
        pred_cols.append(col)
        alias = t.replace("target_", "", 1) + "_ML1"
        pred_alias_map[alias] = col

    for alias, src in pred_alias_map.items():
        if alias not in out.columns and src in out.columns:
            out[alias] = out[src]

    # Keep stem weight factors within historical agronomic ranges by grade/variety.
    _apply_pred_factor_peso_caps(out)
    return out, pred_cols


def _expand_ml1_horizon(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "stage" not in df.columns or "ciclo_id" not in df.columns:
        return df, 0

    work = df.copy()
    work["stage"] = _canon_str(work["stage"])
    work["ciclo_id"] = work["ciclo_id"].astype("string").fillna("UNKNOWN")
    work["fecha_evento"] = _to_date(work["fecha_evento"]) if "fecha_evento" in work.columns else pd.NaT
    if "fecha_post" in work.columns:
        work["fecha_post"] = _to_date(work["fecha_post"])

    if "pred_d_start" not in work.columns or "pred_n_harvest_days" not in work.columns:
        return work, 0

    veg = work.loc[work["stage"].eq("VEG"), ["ciclo_id", "fecha_evento", "pred_d_start", "pred_n_harvest_days"]].copy()
    if veg.empty:
        return work, 0
    veg = veg.sort_values("fecha_evento", kind="mergesort").drop_duplicates(subset=["ciclo_id"], keep="last")
    veg["d_start_i"] = np.rint(pd.to_numeric(veg["pred_d_start"], errors="coerce")).astype("Int64")
    veg["n_days_i"] = np.rint(pd.to_numeric(veg["pred_n_harvest_days"], errors="coerce")).astype("Int64")
    veg["n_days_i"] = veg["n_days_i"].clip(lower=1)
    veg["harvest_start_i"] = _to_date(veg["fecha_evento"]) + pd.to_timedelta(veg["d_start_i"].fillna(0).astype(int), unit="D")
    veg = veg.dropna(subset=["harvest_start_i", "n_days_i"])
    if veg.empty:
        return work, 0

    target_cols = [c for c in work.columns if c.startswith("target_")]
    mask_cols = [c for c in work.columns if c.startswith("mask_target_")]

    max_row_id = pd.to_numeric(work.get("row_id"), errors="coerce").max()
    next_row_id = int(max_row_id) + 1 if np.isfinite(max_row_id) else 1

    new_rows: list[pd.DataFrame] = []
    for r in veg.itertuples(index=False):
        cid = str(r.ciclo_id)
        n_days = int(r.n_days_i)

        cyc = work.loc[work["ciclo_id"].eq(cid)].copy()
        hg = cyc.loc[cyc["stage"].eq("HARVEST_GRADE")].copy()
        if hg.empty:
            continue

        fe_hg = pd.to_datetime(hg.get("fecha_evento"), errors="coerce").dt.normalize()
        if fe_hg.notna().any():
            day1_mask = pd.to_numeric(hg.get("day_in_harvest"), errors="coerce").round().eq(1.0)
            if bool(day1_mask.any()):
                h_start = pd.to_datetime(hg.loc[day1_mask, "fecha_evento"], errors="coerce").dt.normalize().min()
            else:
                h_start = fe_hg.min()
            h_start = pd.Timestamp(h_start).normalize()
        else:
            h_start = pd.Timestamp(r.harvest_start_i).normalize()

        fe_hg = pd.to_datetime(hg.get("fecha_evento"), errors="coerce").dt.normalize()
        existing_dates = set(fe_hg.dropna().tolist())
        if not existing_dates:
            continue
        missing_dates = [
            (h_start + pd.Timedelta(days=d - 1)).normalize()
            for d in range(1, n_days + 1)
            if (h_start + pd.Timedelta(days=d - 1)).normalize() not in existing_dates
        ]
        if not missing_dates:
            continue

        last_date = max(existing_dates)
        hg_tpl = hg.loc[fe_hg.eq(last_date)].copy()
        if hg_tpl.empty:
            hg_tpl = hg.tail(1).copy()
            if hg_tpl.empty:
                continue

        post = cyc.loc[cyc["stage"].eq("POST")].copy()
        post_tpl = pd.DataFrame()
        if not post.empty:
            fe_post = pd.to_datetime(post.get("fecha_evento"), errors="coerce").dt.normalize()
            post_tpl = post.loc[fe_post.eq(last_date)].copy()
            if post_tpl.empty:
                post_tpl = post.tail(min(len(post), len(hg_tpl))).copy()

        for fev in missing_dates:
            d = int((pd.Timestamp(fev) - h_start).days) + 1
            rel = float(d) / float(max(n_days, 1))

            add_hg = hg_tpl.copy()
            add_hg["fecha_evento"] = fev
            add_hg["day_in_harvest"] = float(d)
            add_hg["n_harvest_days"] = float(n_days)
            add_hg["rel_pos"] = rel
            if "dow" in add_hg.columns:
                add_hg["dow"] = float(pd.Timestamp(fev).dayofweek)
            if "month" in add_hg.columns:
                add_hg["month"] = float(pd.Timestamp(fev).month)
            if "weekofyear" in add_hg.columns:
                add_hg["weekofyear"] = float(pd.Timestamp(fev).isocalendar().week)
            if "fecha_post" in add_hg.columns:
                add_hg["fecha_post"] = pd.NaT
            for c in target_cols:
                add_hg[c] = np.nan
            for c in mask_cols:
                add_hg[c] = 0.0
            if "has_any_target" in add_hg.columns:
                add_hg["has_any_target"] = 0
            if "row_id" in add_hg.columns:
                add_hg["row_id"] = np.arange(next_row_id, next_row_id + len(add_hg), dtype=np.int64)
                next_row_id += len(add_hg)
            new_rows.append(add_hg)

            if not post_tpl.empty:
                add_p = post_tpl.copy()
                add_p["fecha_evento"] = fev
                add_p["day_in_harvest"] = float(d)
                add_p["n_harvest_days"] = float(n_days)
                add_p["rel_pos"] = rel
                if "pred_dh_dias" in add_p.columns:
                    dh = np.rint(pd.to_numeric(add_p["pred_dh_dias"], errors="coerce")).astype("Int64")
                    add_p["fecha_post"] = _to_date(pd.to_datetime(fev) + pd.to_timedelta(dh.fillna(0).astype(int), unit="D"))
                elif "fecha_post" in add_p.columns:
                    add_p["fecha_post"] = pd.NaT
                if "dow" in add_p.columns:
                    add_p["dow"] = float(pd.Timestamp(fev).dayofweek)
                if "month" in add_p.columns:
                    add_p["month"] = float(pd.Timestamp(fev).month)
                if "weekofyear" in add_p.columns:
                    add_p["weekofyear"] = float(pd.Timestamp(fev).isocalendar().week)
                if "dow_post" in add_p.columns:
                    add_p["dow_post"] = pd.to_datetime(add_p["fecha_post"], errors="coerce").dt.dayofweek
                if "month_post" in add_p.columns:
                    add_p["month_post"] = pd.to_datetime(add_p["fecha_post"], errors="coerce").dt.month
                if "weekofyear_post" in add_p.columns:
                    add_p["weekofyear_post"] = pd.to_datetime(add_p["fecha_post"], errors="coerce").dt.isocalendar().week.astype("Int64")
                for c in target_cols:
                    add_p[c] = np.nan
                for c in mask_cols:
                    add_p[c] = 0.0
                if "has_any_target" in add_p.columns:
                    add_p["has_any_target"] = 0
                if "row_id" in add_p.columns:
                    add_p["row_id"] = np.arange(next_row_id, next_row_id + len(add_p), dtype=np.int64)
                    next_row_id += len(add_p)
                new_rows.append(add_p)

    if not new_rows:
        return work, 0

    ext = pd.concat([work] + new_rows, ignore_index=True)
    ext = ext.sort_values(["fecha_evento", "stage", "row_source", "ciclo_id", "bloque_base", "variedad_canon", "grado", "destino"], kind="mergesort").reset_index(drop=True)
    return ext, int(len(ext) - len(work))


def _latest_meta() -> Path:
    files = sorted(MODELS_DIR.glob("ml1_multitask_nn_*_meta.json"))
    if not files:
        raise FileNotFoundError(f"No model metadata found in {MODELS_DIR}")
    return files[-1]


def _meta_for_run(run_id: str) -> Path:
    path = MODELS_DIR / f"ml1_multitask_nn_{run_id}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata for run_id={run_id} not found: {path}")
    return path


def _load_artifacts(run_id: str | None) -> tuple[str, dict, dict[str, np.ndarray]]:
    meta_path = _latest_meta() if run_id is None else _meta_for_run(run_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rid = str(meta["run_id"])

    model_path = Path(meta["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    arr = np.load(model_path)
    params = {
        "w1": arr["w1"],
        "b1": arr["b1"],
        "w2": arr["w2"],
        "b2": arr["b2"],
        "w3": arr["w3"],
        "b3": arr["b3"],
        "x_fill": arr["x_fill"] if "x_fill" in arr.files else arr["x_mean"],
        "x_clip_lo": arr["x_clip_lo"] if "x_clip_lo" in arr.files else None,
        "x_clip_hi": arr["x_clip_hi"] if "x_clip_hi" in arr.files else None,
        "x_mean": arr["x_mean"],
        "x_std": arr["x_std"],
        "y_mean": arr["y_mean"],
        "y_std": arr["y_std"],
    }
    return rid, meta, params


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _predict(x: np.ndarray, p: dict[str, np.ndarray]) -> np.ndarray:
    a1 = _relu(x @ p["w1"] + p["b1"])
    a2 = _relu(a1 @ p["w2"] + p["b2"])
    y_norm = a2 @ p["w3"] + p["b3"]
    y = y_norm * p["y_std"].reshape(1, -1) + p["y_mean"].reshape(1, -1)
    return y


def _build_features(df: pd.DataFrame, meta: dict, p: dict[str, np.ndarray]) -> np.ndarray:
    cat_cols = meta["features"]["cat_cols"]
    num_cols = meta["features"]["num_cols"]
    cat_dummy_cols = meta["features"]["cat_dummy_cols"]

    for c in cat_cols:
        if c not in df.columns:
            df[c] = "UNKNOWN"
        df[c] = _canon_str(df[c])

    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    x_num = df[num_cols].to_numpy(dtype=np.float32)
    x_fill = p.get("x_fill", p["x_mean"]).astype(np.float32)
    x_clip_lo = p.get("x_clip_lo")
    x_clip_hi = p.get("x_clip_hi")
    x_mean = p["x_mean"].astype(np.float32)
    x_std = p["x_std"].astype(np.float32)

    if x_num.shape[1] != len(x_mean):
        raise ValueError(
            f"Numeric feature mismatch. got={x_num.shape[1]} expected={len(x_mean)}"
        )

    for j in range(x_num.shape[1]):
        col = x_num[:, j]
        m = ~np.isfinite(col)
        if m.any():
            col[m] = x_fill[j]
            x_num[:, j] = col
    if isinstance(x_clip_lo, np.ndarray) and isinstance(x_clip_hi, np.ndarray):
        if len(x_clip_lo) == x_num.shape[1] and len(x_clip_hi) == x_num.shape[1]:
            x_num = np.clip(x_num, x_clip_lo.reshape(1, -1), x_clip_hi.reshape(1, -1))
    x_num = (x_num - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)

    x_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, prefix_sep="=", dtype=np.float32)
    x_cat = x_cat.reindex(columns=cat_dummy_cols, fill_value=0.0)
    x_cat_arr = x_cat.to_numpy(dtype=np.float32)

    x = np.concatenate([x_num, x_cat_arr], axis=1).astype(np.float32)
    return x


def _eval_if_possible(df: pd.DataFrame, yhat: np.ndarray, targets: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    maes: list[float] = []
    for i, t in enumerate(targets):
        mcol = f"mask_{t}"
        if t not in df.columns or mcol not in df.columns:
            out[f"mae_{t}"] = np.nan
            continue
        y = pd.to_numeric(df[t], errors="coerce").to_numpy(dtype=np.float32)
        m = pd.to_numeric(df[mcol], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0.0
        m = m & np.isfinite(y)
        if m.sum() == 0:
            out[f"mae_{t}"] = np.nan
            continue
        mae = float(np.mean(np.abs(yhat[m, i] - y[m])))
        out[f"mae_{t}"] = mae
        maes.append(mae)
    out["mae_avg"] = float(np.mean(maes)) if maes else np.nan
    return out


def _rebuild_ml1_harvest_chain(df: pd.DataFrame) -> pd.DataFrame:
    if "stage" not in df.columns:
        return df

    out = df.copy()
    st = _canon_str(out["stage"])
    is_hg = st.eq("HARVEST_GRADE")
    is_post = st.eq("POST")
    is_chain = is_hg | is_post
    if not bool(is_chain.any()):
        return out

    if "fecha_evento" in out.columns:
        out["fecha_evento"] = _to_date(out["fecha_evento"])

    day_key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon"] if c in out.columns]
    grade_key = day_key + ["grado"] if ("grado" in out.columns and day_key) else []
    if not day_key:
        return out

    # 1) Rebuild day-level harvest curve from baseline x predicted factor.
    hg = out.loc[is_hg].copy()
    fac = pd.to_numeric(hg.get("pred_factor_tallos_dia"), errors="coerce")
    base_day = pd.to_numeric(hg.get("tallos_pred_baseline_dia"), errors="coerce")
    ml1_day_ref = pd.to_numeric(hg.get("tallos_pred_ml1_dia"), errors="coerce")
    day_ref = base_day.where(base_day > 0.0, ml1_day_ref)

    day_tbl = hg[day_key].copy()
    day_tbl["fac"] = fac
    day_tbl["day_ref"] = day_ref
    day_tbl["day_ml1_ref"] = ml1_day_ref
    day_agg = (
        day_tbl.groupby(day_key, dropna=False, as_index=False)
        .agg(
            fac=("fac", "median"),
            day_ref=("day_ref", "median"),
            day_ml1_ref=("day_ml1_ref", "median"),
        )
    )
    day_agg["tallos_day_new"] = (day_agg["day_ref"] * day_agg["fac"]).where(
        day_agg["day_ref"].notna() & day_agg["fac"].notna(),
        day_agg["day_ml1_ref"],
    )
    day_agg["tallos_day_new"] = pd.to_numeric(day_agg["tallos_day_new"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Empirical cap profile for factor_tallos_dia by harvest day index.
    # This avoids implausible start-day spikes while preserving learned dynamics.
    cap_days = np.array([], dtype=np.int64)
    cap_vals = np.array([], dtype=np.float64)
    if {"mask_target_factor_tallos_dia", "target_factor_tallos_dia", "day_in_harvest"} <= set(out.columns):
        hist = out.loc[is_hg, ["mask_target_factor_tallos_dia", "target_factor_tallos_dia", "day_in_harvest"]].copy()
        hist["w"] = pd.to_numeric(hist["mask_target_factor_tallos_dia"], errors="coerce").fillna(0.0)
        hist["f"] = pd.to_numeric(hist["target_factor_tallos_dia"], errors="coerce")
        hist["d"] = pd.to_numeric(hist["day_in_harvest"], errors="coerce").round().astype("Int64")
        hist = hist.loc[hist["w"] > 0.0]
        hist = hist.loc[hist["f"].notna() & hist["d"].notna()]
        hist = hist.loc[(hist["d"] >= 1) & (hist["d"] <= 90)]
        if not hist.empty:
            cap_tbl = (
                hist.groupby("d", dropna=False, as_index=False)
                .agg(
                    n=("f", "size"),
                    q80=("f", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.80))),
                )
                .sort_values("d", kind="mergesort")
            )
            cap_tbl = cap_tbl.loc[cap_tbl["n"] >= 10]
            if not cap_tbl.empty:
                cap_days = cap_tbl["d"].astype(int).to_numpy(dtype=np.int64)
                cap_vals = cap_tbl["q80"].to_numpy(dtype=np.float64)

    # Blend with baseline day-shape to avoid pathological flat/zero-like segments.
    # This preserves predicted signal while keeping agronomic curve structure.
    date_col = "fecha_evento" if "fecha_evento" in day_key else day_key[1]
    grp_cols = [c for c in day_key if c != date_col]
    if grp_cols:
        smooth_parts: list[pd.DataFrame] = []
        for _, g in day_agg.groupby(grp_cols, dropna=False, sort=False):
            gg = g.sort_values(date_col, kind="mergesort").copy()
            model = pd.to_numeric(gg["tallos_day_new"], errors="coerce").fillna(0.0)
            base = pd.to_numeric(gg["day_ref"], errors="coerce").fillna(0.0)
            s_model = float(model.sum())
            s_base = float(base.sum())
            if s_base > 0.0 and s_model > 0.0:
                base_scaled = base * (s_model / s_base)
            else:
                base_scaled = model

            # Adaptive smoothing:
            # - If baseline curve is near-flat, trust model shape more.
            # - Otherwise, blend with baseline and apply light temporal smoothing.
            mean_model = float(model.mean()) if len(model) else 0.0
            mean_base = float(base_scaled.mean()) if len(base_scaled) else 0.0
            cv_model = float(model.std(ddof=0) / mean_model) if mean_model > 1e-9 else 0.0
            cv_base = float(base_scaled.std(ddof=0) / mean_base) if mean_base > 1e-9 else 0.0

            if cv_base < 0.08:
                blend = model
            else:
                # Keep model dominant while preserving agronomic profile.
                alpha_model = 0.80 if cv_model >= cv_base else 0.70
                blend = alpha_model * model + (1.0 - alpha_model) * base_scaled

            roll = blend.rolling(3, center=True, min_periods=1).mean()
            smooth_w = 0.10 if cv_model >= 0.25 else 0.20
            vals = ((1.0 - smooth_w) * blend + smooth_w * roll).clip(lower=0.0).to_numpy(dtype=np.float64)
            n = len(vals)
            if n > 0:
                day_ord = np.arange(1, n + 1, dtype=np.int64)

                # 1) Empirical day-index cap (factor space -> tallos space).
                if cap_days.size > 0 and cap_vals.size > 0:
                    cap_fac = np.interp(day_ord, cap_days, cap_vals, left=cap_vals[0], right=cap_vals[-1])
                    base_arr = pd.to_numeric(gg["day_ref"], errors="coerce").to_numpy(dtype=np.float64)
                    base_med = float(np.nanmedian(base_arr[np.isfinite(base_arr) & (base_arr > 0.0)])) if np.isfinite(base_arr).any() else 0.0
                    base_eff = np.where(np.isfinite(base_arr) & (base_arr > 0.0), base_arr, base_med)
                    cap_abs = np.clip(cap_fac * np.where(base_eff > 0.0, base_eff, 1.0), 0.0, np.inf)
                else:
                    cap_abs = np.full(n, np.inf, dtype=np.float64)

                # 2) Local spike cap to avoid one-day explosions (e.g. day1 >> day2/day3).
                med3 = pd.Series(vals).rolling(3, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
                spike_cap = np.where(np.isfinite(med3) & (med3 > 0.0), med3 * 2.0, np.inf)
                cap_all = np.minimum(cap_abs, spike_cap)
                vals_cap = np.minimum(vals, cap_all)

                # 2.b) Stronger early-day guard.
                # Harvest day 1/2 can be noisy in model outputs; enforce consistency
                # against neighboring days so first-day spikes do not dominate.
                if n >= 3:
                    nxt = vals_cap[1 : min(n, 5)]
                    nxt = nxt[np.isfinite(nxt) & (nxt >= 0.0)]
                    if nxt.size > 0:
                        cap0 = float(np.nanmedian(nxt) * 2.25)
                        if np.isfinite(cap0) and cap0 > 0.0:
                            vals_cap[0] = min(vals_cap[0], cap0)
                if n >= 4:
                    nxt2 = vals_cap[2 : min(n, 6)]
                    nxt2 = nxt2[np.isfinite(nxt2) & (nxt2 >= 0.0)]
                    if nxt2.size > 0:
                        cap1 = float(np.nanmedian(nxt2) * 2.25)
                        if np.isfinite(cap1) and cap1 > 0.0:
                            vals_cap[1] = min(vals_cap[1], cap1)

                # 3) If generated tail is flat, apply a gentle decay in the last days.
                tail_n = int(min(6, n))
                if tail_n >= 4:
                    tail = vals_cap[-tail_n:]
                    t_mean = float(np.nanmean(tail))
                    t_std = float(np.nanstd(tail))
                    if t_mean > 1e-6 and (t_std / t_mean) < 0.01:
                        decay = np.linspace(1.00, 0.92, tail_n, dtype=np.float64)
                        vals_cap[-tail_n:] = tail * decay

                # 4) Redistribute clipped mass to central non-capped days (preserve cycle total).
                removed = float(np.sum(vals) - np.sum(vals_cap))
                if removed > 1e-9:
                    was_capped = vals_cap < (vals - 1e-9)
                    rel = day_ord.astype(np.float64) / float(max(n, 1))
                    eligible = (~was_capped) & (rel >= 0.25) & (rel <= 0.80)
                    if not bool(np.any(eligible)):
                        eligible = ~was_capped
                    if bool(np.any(eligible)):
                        w = vals_cap[eligible].copy()
                        if not np.isfinite(w).any() or float(np.nansum(w)) <= 0.0:
                            w = np.ones(int(np.sum(eligible)), dtype=np.float64)
                        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
                        wsum = float(np.sum(w))
                        if wsum > 0.0:
                            vals_cap[eligible] = vals_cap[eligible] + removed * (w / wsum)
                vals = np.clip(vals_cap, 0.0, np.inf)

            gg["tallos_day_new"] = vals
            smooth_parts.append(gg)
        day_agg = pd.concat(smooth_parts, ignore_index=True)

    day_idx = pd.MultiIndex.from_frame(day_agg[day_key])
    day_map = pd.Series(day_agg["tallos_day_new"].to_numpy(dtype=np.float64), index=day_idx)
    row_day_idx = pd.MultiIndex.from_frame(out.loc[is_chain, day_key])
    out.loc[is_chain, "tallos_pred_ml1_dia"] = row_day_idx.map(day_map).to_numpy(dtype=np.float64)

    # 2) Rebuild grade distribution from predicted share (normalized by day).
    if bool(is_hg.any()) and grade_key:
        hg = out.loc[is_hg].copy()
        hg["tallos_day_new"] = pd.to_numeric(hg["tallos_pred_ml1_dia"], errors="coerce").fillna(0.0)

        share_pred = pd.to_numeric(hg.get("pred_share_grado"), errors="coerce")
        share_base = pd.to_numeric(hg.get("share_grado_baseline"), errors="coerce")
        share_old = _safe_div(
            pd.to_numeric(hg.get("tallos_pred_ml1_grado_dia"), errors="coerce"),
            pd.to_numeric(hg.get("tallos_pred_ml1_dia"), errors="coerce"),
            default=np.nan,
        )
        share_use = share_pred.where(share_pred.notna(), share_base)
        share_use = share_use.where(share_use.notna(), share_old).fillna(0.0).clip(lower=0.0)

        grp_vals = [hg[c] for c in day_key]
        den = share_use.groupby(grp_vals, dropna=False).transform("sum")
        ngrp = share_use.groupby(grp_vals, dropna=False).transform("size").clip(lower=1)
        share_norm = pd.Series(np.where(den > 0.0, share_use / den, 1.0 / ngrp.astype(float)), index=hg.index).fillna(0.0)
        share_norm = _regularize_share_by_hist_prior(hg=hg, share_norm=share_norm, day_key=day_key)
        tallos_grade_new = (hg["tallos_day_new"] * share_norm).clip(lower=0.0)
        out.loc[hg.index, "tallos_pred_ml1_grado_dia"] = tallos_grade_new.to_numpy(dtype=np.float64)
        out.loc[hg.index, "pred_share_grado"] = share_norm.to_numpy(dtype=np.float32)

        # Keep cycle mass aligned to tallos_proy when target is available.
        if "tallos_proy" in out.columns and "ciclo_id" in out.columns:
            cyc = (
                out.loc[is_hg, ["ciclo_id", "tallos_pred_ml1_grado_dia", "tallos_proy"]]
                .groupby("ciclo_id", dropna=False, as_index=False)
                .agg(
                    tallos_ml1_total=("tallos_pred_ml1_grado_dia", "sum"),
                    tallos_target=("tallos_proy", "max"),
                )
            )
            cyc["scale"] = np.where(
                pd.to_numeric(cyc["tallos_target"], errors="coerce").fillna(0.0) > 0.0,
                _safe_div(cyc["tallos_target"], cyc["tallos_ml1_total"], default=1.0),
                1.0,
            )
            scale_map = pd.Series(cyc["scale"].to_numpy(dtype=np.float64), index=cyc["ciclo_id"].astype("string"))
            srow = out["ciclo_id"].astype("string").map(scale_map).fillna(1.0)
            out.loc[is_hg, "tallos_pred_ml1_grado_dia"] = (
                pd.to_numeric(out.loc[is_hg, "tallos_pred_ml1_grado_dia"], errors="coerce").fillna(0.0) * srow.loc[is_hg]
            )
            out.loc[is_chain, "tallos_pred_ml1_dia"] = (
                pd.to_numeric(out.loc[is_chain, "tallos_pred_ml1_dia"], errors="coerce").fillna(0.0) * srow.loc[is_chain]
            )

    # 3) Propagate HG grade/day values to POST rows of same day/grade.
    if bool(is_post.any()) and grade_key:
        hg_map = (
            out.loc[is_hg, grade_key + ["tallos_pred_ml1_grado_dia"]]
            .groupby(grade_key, dropna=False, as_index=False)
            .agg(tallos_pred_ml1_grado_dia=("tallos_pred_ml1_grado_dia", "mean"))
        )
        hg_idx = pd.MultiIndex.from_frame(hg_map[grade_key])
        hg_val_map = pd.Series(pd.to_numeric(hg_map["tallos_pred_ml1_grado_dia"], errors="coerce").to_numpy(dtype=np.float64), index=hg_idx)
        post_rows = out.loc[is_post, grade_key].copy()
        post_idx = pd.MultiIndex.from_frame(post_rows)
        post_grade_new = pd.Series(post_idx.map(hg_val_map), index=post_rows.index, dtype="float64")
        post_old = pd.to_numeric(out.loc[is_post, "tallos_pred_ml1_grado_dia"], errors="coerce")
        out.loc[is_post, "tallos_pred_ml1_grado_dia"] = post_grade_new.where(post_grade_new.notna(), post_old)

    # 4) Rebuild tallos_post_proy with previous row-level split ratio.
    if "tallos_post_proy" in out.columns and "tallos_pred_ml1_grado_dia" in out.columns:
        old_tallos_post = pd.to_numeric(df.get("tallos_post_proy"), errors="coerce")
        old_tallos_grade = pd.to_numeric(df.get("tallos_pred_ml1_grado_dia"), errors="coerce")
        split_ratio = _safe_div(old_tallos_post, old_tallos_grade, default=1.0).clip(lower=0.0)
        out.loc[is_post, "tallos_post_proy"] = (
            pd.to_numeric(out.loc[is_post, "tallos_pred_ml1_grado_dia"], errors="coerce").fillna(0.0)
            * split_ratio.loc[is_post].fillna(1.0)
        )

    # 5) Rebuild kg/gramos and caja aliases from updated tallos.
    # For POST rows, green mass must come from projected post stems and
    # estimated stem weight (baseline * predicted factor).
    if "kg_verde_ref" in out.columns and "tallos_pred_ml1_grado_dia" in out.columns:
        old_kg = pd.to_numeric(df.get("kg_verde_ref"), errors="coerce")
        old_tallos_grade = pd.to_numeric(df.get("tallos_pred_ml1_grado_dia"), errors="coerce")
        kg_per_tallo = _safe_div(old_kg, old_tallos_grade, default=np.nan)
        kg_new = pd.to_numeric(out.get("tallos_pred_ml1_grado_dia"), errors="coerce") * kg_per_tallo

        if bool(is_post.any()) and "tallos_post_proy" in out.columns:
            tallos_post_new = pd.to_numeric(out.get("tallos_post_proy"), errors="coerce")
            peso_base_g = pd.to_numeric(out.get("peso_tallo_baseline_g"), errors="coerce")
            fac_peso = pd.to_numeric(out.get("pred_factor_peso_tallo"), errors="coerce")
            peso_est_g = (peso_base_g * fac_peso).where(peso_base_g.notna() & fac_peso.notna(), np.nan)
            # Fallback to previous implicit weight only when estimated weight is unavailable.
            peso_est_g = peso_est_g.where(peso_est_g > 0.0, kg_per_tallo * 1000.0)
            kg_post_green_new = tallos_post_new * (peso_est_g / 1000.0)
            kg_new.loc[is_post] = kg_post_green_new.loc[is_post].where(
                kg_post_green_new.loc[is_post].notna(),
                kg_new.loc[is_post],
            )

        out["kg_verde_ref"] = kg_new.where(kg_new.notna(), pd.to_numeric(out.get("kg_verde_ref"), errors="coerce"))
        if "gramos_verde_ref" in out.columns:
            out["gramos_verde_ref"] = pd.to_numeric(out["kg_verde_ref"], errors="coerce") * 1000.0

        for c in ["cajas_split_grado_dia", "cajas_ml1_grado_dia", "cajas_post_seed"]:
            if c in out.columns:
                old_c = pd.to_numeric(df.get(c), errors="coerce")
                c_per_kg = _safe_div(old_c, old_kg, default=np.nan)
                c_new = pd.to_numeric(out["kg_verde_ref"], errors="coerce") * c_per_kg
                if c == "cajas_split_grado_dia" and bool(is_post.any()):
                    cajas_post_green = pd.to_numeric(out["kg_verde_ref"], errors="coerce") / 10.0
                    c_new.loc[is_post] = cajas_post_green.loc[is_post].where(
                        cajas_post_green.loc[is_post].notna(),
                        c_new.loc[is_post],
                    )
                out[c] = c_new.where(c_new.notna(), pd.to_numeric(out.get(c), errors="coerce"))

    return out


def _compute_post_outputs(df: pd.DataFrame, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage = df["stage"] if "stage" in df.columns else pd.Series(["UNKNOWN"] * len(df), index=df.index)
    post = df[_canon_str(stage).eq("POST")].copy()
    if post.empty:
        return post, pd.DataFrame(), pd.DataFrame()

    post["fecha_evento"] = pd.to_datetime(post["fecha_evento"], errors="coerce").dt.normalize()
    post["fecha_post"] = pd.to_datetime(post["fecha_post"], errors="coerce").dt.normalize()
    post["destino"] = _canon_str(post["destino"])

    pred_h = pd.to_numeric(post.get("pred_factor_hidr"), errors="coerce").fillna(1.0).clip(0.80, 3.00)
    pred_d = pd.to_numeric(post.get("pred_factor_desp"), errors="coerce").fillna(1.0).clip(0.05, 1.00)
    pred_a = pd.to_numeric(post.get("pred_factor_ajuste"), errors="coerce").fillna(1.0).clip(0.50, 2.00)

    post["pred_factor_hidr"] = pred_h
    post["pred_factor_desp"] = pred_d
    post["pred_factor_ajuste"] = pred_a

    kg_verde = pd.to_numeric(post.get("kg_verde_ref"), errors="coerce")
    base_cajas = pd.to_numeric(post.get("cajas_split_grado_dia"), errors="coerce")

    post["pred_kg_post_ml1_nn"] = kg_verde * pred_h * pred_d * pred_a
    post["pred_gramos_post_ml1_nn"] = post["pred_kg_post_ml1_nn"] * 1000.0

    kg_por_caja_ref = np.where(
        (base_cajas.fillna(0.0) > 0.0) & np.isfinite(kg_verde.fillna(np.nan)),
        kg_verde / base_cajas,
        np.nan,
    )
    post["kg_por_caja_ref"] = kg_por_caja_ref

    post["pred_cajas_post_ml1_nn"] = np.where(
        base_cajas.notna(),
        base_cajas * pred_h * pred_d * pred_a,
        np.where(
            np.isfinite(post["kg_por_caja_ref"]) & (post["kg_por_caja_ref"] > 0),
            post["pred_kg_post_ml1_nn"] / post["kg_por_caja_ref"],
            np.nan,
        ),
    )
    post["aprovechamiento"] = np.where(
        pd.to_numeric(post["kg_verde_ref"], errors="coerce").fillna(0.0) > 0.0,
        post["pred_kg_post_ml1_nn"] / post["kg_verde_ref"],
        np.nan,
    )

    # Explicit naming conventions
    post["tallos_post_real"] = pd.to_numeric(post.get("tallos_post"), errors="coerce")
    post["tallos_post_ML1"] = pd.to_numeric(post.get("tallos_post_proy"), errors="coerce")
    post["tallos_proy_baseline"] = pd.to_numeric(post.get("tallos_proy"), errors="coerce")
    post["tallos_dia_ML1"] = pd.to_numeric(post.get("tallos_pred_ml1_dia"), errors="coerce")
    post["tallos_grado_dia_ML1"] = pd.to_numeric(post.get("tallos_pred_ml1_grado_dia"), errors="coerce")
    post["tallos_grado_dia_baseline"] = pd.to_numeric(post.get("tallos_pred_baseline_grado_dia"), errors="coerce")

    post["gramos_verde_ML1"] = pd.to_numeric(post.get("gramos_verde_ref"), errors="coerce")
    post["gramos_post_ML1"] = pd.to_numeric(post.get("pred_gramos_post_ml1_nn"), errors="coerce")
    post["gramos_post_real"] = pd.to_numeric(post.get("gramos_post_real_ref"), errors="coerce")
    post["kg_verde_ML1"] = pd.to_numeric(post.get("kg_verde_ref"), errors="coerce")
    post["kg_post_ML1"] = pd.to_numeric(post.get("pred_kg_post_ml1_nn"), errors="coerce")
    post["kg_post_real"] = pd.to_numeric(post.get("kg_post_real_ref"), errors="coerce")
    post["cajas_verde_ML1"] = pd.to_numeric(post.get("cajas_split_grado_dia"), errors="coerce")
    post["cajas_post_ML1"] = pd.to_numeric(post.get("pred_cajas_post_ml1_nn"), errors="coerce")
    post["cajas_post_seed_baseline"] = pd.to_numeric(post.get("cajas_post_seed"), errors="coerce")
    post["aprovechamiento_ML1"] = pd.to_numeric(post.get("aprovechamiento"), errors="coerce")
    post["ml1_multitask_nn_run_id"] = run_id

    # Put final green/final-processed weight columns at the end for quick visual comparison.
    final_tail = [
        "gramos_verde_ML1",
        "gramos_post_ML1",
        "gramos_post_real",
        "kg_verde_ML1",
        "kg_post_ML1",
        "kg_post_real",
        "cajas_verde_ML1",
        "cajas_post_ML1",
        "cajas_post_seed_baseline",
        "aprovechamiento_ML1",
    ]
    keep = [c for c in post.columns if c not in final_tail] + [c for c in final_tail if c in post.columns]
    post = post[keep]

    by_dest = (
        post.groupby(["fecha_evento", "fecha_post", "destino"], dropna=False, as_index=False)
        .agg(
            kg_verde_ML1=("kg_verde_ML1", "sum"),
            kg_post_ML1=("kg_post_ML1", "sum"),
            cajas_verde_ML1=("cajas_verde_ML1", "sum"),
            cajas_post_ML1=("cajas_post_ML1", "sum"),
        )
    )
    by_dest["aprovechamiento"] = np.where(
        pd.to_numeric(by_dest["kg_verde_ML1"], errors="coerce").fillna(0.0) > 0.0,
        by_dest["kg_post_ML1"] / by_dest["kg_verde_ML1"],
        np.nan,
    )
    by_dest["aprovechamiento_ML1"] = by_dest["aprovechamiento"]
    by_dest["ml1_multitask_nn_run_id"] = run_id

    total = (
        by_dest.groupby(["fecha_evento", "fecha_post"], dropna=False, as_index=False)
        .agg(
            kg_verde_ML1=("kg_verde_ML1", "sum"),
            kg_post_ML1=("kg_post_ML1", "sum"),
            cajas_verde_ML1=("cajas_verde_ML1", "sum"),
            cajas_post_ML1=("cajas_post_ML1", "sum"),
        )
    )
    total["aprovechamiento"] = np.where(
        pd.to_numeric(total["kg_verde_ML1"], errors="coerce").fillna(0.0) > 0.0,
        total["kg_post_ML1"] / total["kg_verde_ML1"],
        np.nan,
    )
    total["aprovechamiento_ML1"] = total["aprovechamiento"]
    total["ml1_multitask_nn_run_id"] = run_id
    return post, by_dest, total


def main() -> None:
    args = _parse_args()
    run_id, meta, params = _load_artifacts(args.run_id)

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {in_path}")
    df = read_parquet(in_path).copy()

    targets = list(meta["targets"])
    target_clips = meta.get("target_clips", {})

    # Pass 1: predict on original dataset to obtain cycle-level horizon estimates.
    x1 = _build_features(df, meta=meta, p=params)
    yhat1 = _predict(x1, params)
    df1, pred_cols = _assign_predictions(df=df, yhat=yhat1, targets=targets, target_clips=target_clips)

    # Expand missing days up to predicted ML1 horizon (HG/POST), then re-predict.
    df2, n_added = _expand_ml1_horizon(df1)
    if n_added > 0:
        x2 = _build_features(df2, meta=meta, p=params)
        yhat2 = _predict(x2, params)
        df_final, pred_cols = _assign_predictions(df=df2, yhat=yhat2, targets=targets, target_clips=target_clips)
    else:
        df_final = df1
        yhat2 = yhat1

    # Keep harvest curve, grade split and downstream mass flow consistent with ML1 predictions.
    df_final = _rebuild_ml1_harvest_chain(df_final)

    df_final["ml1_multitask_nn_run_id"] = run_id

    eval_metrics = _eval_if_possible(df_final, yhat=yhat2, targets=targets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (OUT_DIR / f"pred_ml1_multitask_nn_{run_id}.parquet")
    write_parquet(df_final, out_path)

    post_detail, post_dd, post_dt = _compute_post_outputs(df_final, run_id=run_id)
    post_detail_path = OUT_DIR / f"pred_ml1_multitask_nn_post_final_{run_id}.parquet"
    post_dd_path = OUT_DIR / f"pred_ml1_multitask_nn_post_dia_destino_{run_id}.parquet"
    post_dt_path = OUT_DIR / f"pred_ml1_multitask_nn_post_dia_total_{run_id}.parquet"
    if not post_detail.empty:
        write_parquet(post_detail, post_detail_path)
    if not post_dd.empty:
        write_parquet(post_dd, post_dd_path)
    if not post_dt.empty:
        write_parquet(post_dt, post_dt_path)

    print(f"[OK] Predictions written: {out_path}")
    print(f"     rows={len(df_final):,} cols_added={len(pred_cols)} run_id={run_id} rows_expanded={n_added:,}")
    if np.isfinite(eval_metrics.get("mae_avg", np.nan)):
        print(f"     masked_mae_avg={eval_metrics['mae_avg']:.6f}")
    if not post_detail.empty:
        print(f"[OK] Post final detail : {post_detail_path}")
        print(f"[OK] Post dia destino : {post_dd_path}")
        print(f"[OK] Post dia total   : {post_dt_path}")


if __name__ == "__main__":
    main()
