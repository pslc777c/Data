from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD_ML2 = DATA / "gold" / "ml2_nn"
SILVER = DATA / "silver"

IN_FACT_PESO = SILVER / "fact_peso_tallo_real_grado_dia.parquet"
IN_FACT_POST = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"
IN_DIM_VAR = SILVER / "dim_variedad_canon.parquet"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("build_views_ml1_ml2_real_compare")
    ap.add_argument("--pred-ml2", default=None, help="Legacy ML2 path (treated as operativo input).")
    ap.add_argument("--pred-ml2-puro", default=None, help="Path pred_ml2_multitask_nn_puro_*.parquet")
    ap.add_argument("--pred-ml2-operativo", default=None, help="Path pred_ml2_multitask_nn_operativo_*.parquet")
    ap.add_argument("--output-dir", default=str(GOLD_ML2))
    return ap.parse_args()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round().astype("Int64")


def _num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _safe_div(a: pd.Series, b: pd.Series, default: float = np.nan) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    out = aa / bb.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def _latest_pred_ml2() -> Path:
    files = sorted(GOLD_ML2.glob("pred_ml2_multitask_nn_operativo_*.parquet"))
    if files:
        return files[-1]
    files = sorted(GOLD_ML2.glob("pred_ml2_multitask_nn_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pred_ml2 file found in {GOLD_ML2}")
    return files[-1]


def _run_id_from_path(path: Path) -> str:
    m = re.search(r"pred_ml2_multitask_nn_(?:puro_|operativo_)?(.+)\.parquet$", path.name)
    return m.group(1) if m else "unknown"


def _base_from_pred(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if "row_id" not in out.columns:
        out["row_id"] = np.arange(len(out), dtype=np.int64)

    for c in ["fecha_evento", "fecha_post", "fecha_fin_cosecha_ciclo", "fecha_post_ml2"]:
        if c in out.columns:
            out[c] = _to_date(out[c])

    out["stage"] = _canon_str(out["stage"]) if "stage" in out.columns else "UNKNOWN"
    if "ciclo_id" in out.columns:
        out["ciclo_id"] = out["ciclo_id"].astype("string").fillna("UNKNOWN")
    for c in ["row_source", "variedad_canon", "area", "tipo_sp", "destino", "estado_ciclo"]:
        if c in out.columns:
            out[c] = _canon_str(out[c])
    for c in ["bloque_base", "grado"]:
        if c in out.columns:
            out[c] = _to_int(out[c]).astype("string")

    if "is_active_cycle" in out.columns:
        out["is_active_cycle"] = out["is_active_cycle"].fillna(False).astype(bool)
    else:
        out["is_active_cycle"] = False
    if "is_closed_cycle" in out.columns:
        out["is_closed_cycle"] = out["is_closed_cycle"].fillna(False).astype(bool)
    else:
        out["is_closed_cycle"] = False
    return out


def _common_view_base(base: pd.DataFrame, source: str, run_id: str) -> pd.DataFrame:
    cols = [
        "row_id",
        "stage",
        "row_source",
        "fecha_evento",
        "fecha_post",
        "ciclo_id",
        "bloque_base",
        "variedad_canon",
        "area",
        "tipo_sp",
        "grado",
        "destino",
        "estado_ciclo",
        "is_active_cycle",
        "is_closed_cycle",
        "fecha_fin_cosecha_ciclo",
        "tallos_proy",
    ]
    cols = [c for c in cols if c in base.columns]
    out = base[cols].copy()
    out["source_model"] = source
    out["ml2_multitask_nn_run_id"] = run_id

    # Unified metric columns (same schema for ML1/ML2/REAL)
    metric_cols = [
        "fecha_sp",
        "harvest_start",
        "harvest_end",
        "d_start",
        "n_harvest_days",
        "day_in_harvest",
        "rel_pos",
        "factor_tallos_dia",
        "share_grado",
        "factor_peso_tallo",
        "tallos_dia",
        "tallos_grado_dia",
        "dh_dias",
        "factor_hidr",
        "factor_desp",
        "factor_ajuste",
        "tallos_post",
        "kg_verde",
        "gramos_verde",
        "kg_post",
        "cajas_verde",
        "cajas_post",
        "aprovechamiento",
        "tallos_total_ciclo",
        "kg_verde_total_ciclo",
        "tallos_post_total_ciclo",
        "kg_post_total_ciclo",
        "cajas_post_total_ciclo",
    ]
    for c in metric_cols:
        out[c] = np.nan
    return out


def _veg_calendar_map(
    base: pd.DataFrame,
    d_col: str,
    n_col: str,
    start_col: str | None = None,
    end_col: str | None = None,
) -> pd.DataFrame:
    veg = base.loc[base["stage"].eq("VEG"), ["ciclo_id", "fecha_evento", d_col, n_col] + [c for c in [start_col, end_col] if c and c in base.columns]].copy()
    if veg.empty:
        return pd.DataFrame(columns=["ciclo_id", "fecha_sp", "d_start", "n_harvest_days", "harvest_start", "harvest_end"])
    veg["fecha_sp"] = _to_date(veg["fecha_evento"])
    veg["d_start"] = _num(veg.get(d_col), default=np.nan)
    veg["n_harvest_days"] = _num(veg.get(n_col), default=np.nan)
    veg = veg.sort_values("fecha_sp", kind="mergesort").drop_duplicates(subset=["ciclo_id"], keep="last")

    d_days = np.rint(veg["d_start"]).astype("Int64")
    n_days = np.rint(veg["n_harvest_days"]).astype("Int64").clip(lower=1)
    start = veg["fecha_sp"] + pd.to_timedelta(d_days.fillna(0).astype(int), unit="D")
    end = start + pd.to_timedelta(n_days.fillna(1).astype(int) - 1, unit="D")

    if start_col and start_col in veg.columns:
        start = _to_date(veg[start_col]).fillna(start)
    if end_col and end_col in veg.columns:
        end = _to_date(veg[end_col]).fillna(end)

    out = veg[["ciclo_id", "fecha_sp", "d_start", "n_harvest_days"]].copy()
    out["harvest_start"] = _to_date(start)
    out["harvest_end"] = _to_date(end)
    return out


def _load_real_harvest_join(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Returns: row-level HG real, cycle-level real window, day-level real totals
    hg = base.loc[base["stage"].eq("HARVEST_GRADE"), ["row_id", "ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado", "tallos_pred_baseline_dia", "peso_tallo_baseline_g"]].copy()
    if hg.empty or not IN_FACT_PESO.exists():
        return (
            pd.DataFrame(columns=["row_id"]),
            pd.DataFrame(columns=["ciclo_id", "harvest_start", "harvest_end", "n_harvest_days"]),
            pd.DataFrame(columns=["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "tallos_dia_real"]),
        )

    hg["fecha_evento"] = _to_date(hg["fecha_evento"])
    hg["bloque_base_int"] = _to_int(hg["bloque_base"])
    hg["grado_int"] = _to_int(hg["grado"])
    hg["variedad_canon"] = _canon_str(hg["variedad_canon"])
    # Deduplicate upstream repeated HG rows by business key to avoid real double-counting.
    hg_key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"] if c in hg.columns]
    if hg_key:
        if "row_id" in hg.columns:
            hg = hg.sort_values("row_id", kind="mergesort")
        hg = hg.drop_duplicates(subset=hg_key, keep="first")

    fr = read_parquet(IN_FACT_PESO).copy()
    fr.columns = [str(c).strip() for c in fr.columns]
    fr["fecha_evento"] = _to_date(fr["fecha"])
    if "bloque_base" not in fr.columns:
        if "bloque_padre" in fr.columns:
            fr["bloque_base"] = fr["bloque_padre"]
        elif "bloque" in fr.columns:
            fr["bloque_base"] = fr["bloque"]
        else:
            fr["bloque_base"] = pd.NA
    fr["bloque_base_int"] = _to_int(fr["bloque_base"])
    fr["grado_int"] = _to_int(fr["grado"])

    if "variedad_canon" not in fr.columns:
        if "variedad" in fr.columns and IN_DIM_VAR.exists():
            dimv = read_parquet(IN_DIM_VAR).copy()
            dimv.columns = [str(c).strip() for c in dimv.columns]
            dimv["variedad_raw_norm"] = _canon_str(dimv["variedad_raw"])
            dimv["variedad_canon"] = _canon_str(dimv["variedad_canon"])
            fr["variedad_raw_norm"] = _canon_str(fr["variedad"])
            fr = fr.merge(
                dimv[["variedad_raw_norm", "variedad_canon"]].drop_duplicates(),
                on="variedad_raw_norm",
                how="left",
            )
            fr["variedad_canon"] = fr["variedad_canon"].fillna(fr["variedad_raw_norm"])
        else:
            fr["variedad_canon"] = "UNKNOWN"
    fr["variedad_canon"] = _canon_str(fr["variedad_canon"])

    fr["tallos_real"] = _num(fr.get("tallos_real"), default=0.0)
    fr["peso_real_g"] = _num(fr.get("peso_real_g"), default=np.nan)
    fr["peso_tallo_real_g"] = _num(fr.get("peso_tallo_real_g"), default=np.nan)
    fr["kg_verde_real"] = fr["peso_real_g"] / 1000.0
    fr["kg_verde_real"] = fr["kg_verde_real"].fillna(fr["tallos_real"] * fr["peso_tallo_real_g"] / 1000.0)
    fr["peso_tallo_real_g"] = fr["peso_tallo_real_g"].fillna(_safe_div(fr["peso_real_g"], fr["tallos_real"], default=np.nan))

    key = ["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"]
    frg = (
        fr[key + ["tallos_real", "kg_verde_real", "peso_tallo_real_g"]]
        .dropna(subset=key)
        .groupby(key, dropna=False, as_index=False)
        .agg(
            tallos_grado_real=("tallos_real", "sum"),
            kg_verde_grado_real=("kg_verde_real", "sum"),
            peso_tallo_real_g=("peso_tallo_real_g", "mean"),
        )
    )

    hgr = hg.merge(
        frg,
        on=["fecha_evento", "bloque_base_int", "variedad_canon", "grado_int"],
        how="left",
    )

    day_key = ["ciclo_id", "fecha_evento", "bloque_base_int", "variedad_canon"]
    day_real = (
        hgr.groupby(day_key, dropna=False, as_index=False)
        .agg(tallos_dia_real=("tallos_grado_real", "sum"))
    )
    hgr = hgr.merge(day_real, on=day_key, how="left")
    hgr["share_grado_real"] = _safe_div(hgr["tallos_grado_real"], hgr["tallos_dia_real"], default=np.nan)
    hgr["factor_tallos_dia_real"] = _safe_div(hgr["tallos_dia_real"], _num(hgr["tallos_pred_baseline_dia"], default=np.nan), default=np.nan)
    hgr["factor_peso_tallo_real"] = _safe_div(hgr["peso_tallo_real_g"], _num(hgr["peso_tallo_baseline_g"], default=np.nan), default=np.nan)

    cyc = (
        hgr.loc[_num(hgr["tallos_grado_real"], default=0.0) > 0.0]
        .groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            harvest_start=("fecha_evento", "min"),
            harvest_end=("fecha_evento", "max"),
        )
    )
    if not cyc.empty:
        cyc["n_harvest_days"] = (pd.to_datetime(cyc["harvest_end"]) - pd.to_datetime(cyc["harvest_start"])).dt.days + 1
    else:
        cyc["n_harvest_days"] = np.nan

    day_real = day_real.rename(columns={"bloque_base_int": "bloque_base_int_day"})
    return hgr, cyc, day_real


def _load_real_post_alloc(base: pd.DataFrame) -> pd.DataFrame:
    post = base.loc[base["stage"].eq("POST"), ["row_id", "fecha_evento", "fecha_post", "grado", "destino", "share_block_post"]].copy()
    if post.empty or not IN_FACT_POST.exists():
        return pd.DataFrame(columns=["row_id"])

    post["fecha_evento"] = _to_date(post["fecha_evento"])
    post["fecha_post"] = _to_date(post["fecha_post"])
    post["grado_int"] = _to_int(post["grado"])
    post["destino"] = _canon_str(post["destino"])

    fr = read_parquet(IN_FACT_POST).copy()
    fr.columns = [str(c).strip() for c in fr.columns]
    fr["fecha_evento"] = _to_date(fr["fecha_cosecha"])
    fr["fecha_post"] = _to_date(fr["fecha_post"])
    fr["grado_int"] = _to_int(fr["grado"])
    fr["destino"] = _canon_str(fr["destino"])
    fr["dh_dias"] = _num(fr.get("dh_dias"), default=np.nan)
    fr["hidr_pct"] = _num(fr.get("hidr_pct"), default=np.nan)
    fr["tallos_total"] = _num(fr.get("tallos"), default=0.0)
    fr["kg_verde_total"] = _num(fr.get("peso_base_g"), default=0.0) / 1000.0
    fr["kg_post_total"] = _num(fr.get("peso_post_g"), default=0.0) / 1000.0

    key = ["fecha_evento", "fecha_post", "grado_int", "destino"]
    agg = (
        fr[key + ["dh_dias", "hidr_pct", "tallos_total", "kg_verde_total", "kg_post_total"]]
        .dropna(subset=key)
        .groupby(key, dropna=False, as_index=False)
        .agg(
            dh_dias_real=("dh_dias", "median"),
            hidr_pct_real=("hidr_pct", "median"),
            tallos_post_real_total=("tallos_total", "sum"),
            kg_verde_real_total=("kg_verde_total", "sum"),
            kg_post_real_total=("kg_post_total", "sum"),
        )
    )

    m = post.merge(agg, on=key, how="left")
    share = _num(m.get("share_block_post"), default=np.nan)
    den = share.groupby([m["fecha_evento"], m["fecha_post"], m["grado_int"], m["destino"]], dropna=False).transform("sum")
    ngrp = share.groupby([m["fecha_evento"], m["fecha_post"], m["grado_int"], m["destino"]], dropna=False).transform("size").clip(lower=1)
    share_norm = np.where(den > 0.0, share / den, 1.0 / ngrp.astype(float))
    m["share_alloc"] = pd.to_numeric(pd.Series(share_norm, index=m.index), errors="coerce").fillna(0.0)

    m["tallos_post_real"] = _num(m["tallos_post_real_total"], default=np.nan) * m["share_alloc"]
    m["kg_verde_real"] = _num(m["kg_verde_real_total"], default=np.nan) * m["share_alloc"]
    m["kg_post_real"] = _num(m["kg_post_real_total"], default=np.nan) * m["share_alloc"]
    m["factor_hidr_real"] = 1.0 + _num(m["hidr_pct_real"], default=np.nan)
    m["aprovechamiento_real"] = _safe_div(m["kg_post_real"], m["kg_verde_real"], default=np.nan)
    return m


def _fill_cycle_totals(v: pd.DataFrame) -> None:
    hg_mask = v["stage"].eq("HARVEST_GRADE")
    post_mask = v["stage"].eq("POST")

    cyc_hg = (
        v.loc[hg_mask]
        .groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            tallos_total_ciclo=("tallos_grado_dia", "sum"),
            kg_verde_total_ciclo=("kg_verde", "sum"),
        )
    )
    cyc_post = (
        v.loc[post_mask]
        .groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            tallos_post_total_ciclo=("tallos_post", "sum"),
            kg_post_total_ciclo=("kg_post", "sum"),
            cajas_post_total_ciclo=("cajas_post", "sum"),
        )
    )
    cyc = cyc_hg.merge(cyc_post, on="ciclo_id", how="outer")
    v2 = v.merge(cyc, on="ciclo_id", how="left", suffixes=("", "_map"))
    for c in ["tallos_total_ciclo", "kg_verde_total_ciclo", "tallos_post_total_ciclo", "kg_post_total_ciclo", "cajas_post_total_ciclo"]:
        map_col = f"{c}_map"
        if map_col in v2.columns:
            v2[c] = v2[map_col]
            v2 = v2.drop(columns=[map_col])
    v.drop(v.index, inplace=True)
    for c in v2.columns:
        v[c] = v2[c]


def _expand_missing_horizon_rows(v: pd.DataFrame) -> None:
    if v.empty:
        return

    if "ciclo_id" not in v.columns or "stage" not in v.columns:
        return

    work = v.copy()
    work["stage"] = _canon_str(work["stage"])
    work["fecha_evento"] = _to_date(work["fecha_evento"])
    if "harvest_start" in work.columns:
        work["harvest_start"] = _to_date(work["harvest_start"])

    max_row_id = pd.to_numeric(work.get("row_id"), errors="coerce").max()
    next_row_id = int(max_row_id) + 1 if np.isfinite(max_row_id) else 1

    new_rows: list[pd.DataFrame] = []

    for cid, cdf in work.groupby("ciclo_id", dropna=False):
        hg = cdf.loc[cdf["stage"].eq("HARVEST_GRADE")].copy()
        if hg.empty:
            continue

        fe_hg = pd.to_datetime(hg.get("fecha_evento"), errors="coerce").dropna()
        if not fe_hg.empty:
            harvest_start = fe_hg.min().normalize()
        else:
            hs = pd.to_datetime(cdf.get("harvest_start"), errors="coerce").dropna()
            if hs.empty:
                continue
            harvest_start = hs.iloc[0].normalize()

        n_ser = pd.to_numeric(cdf.get("n_harvest_days"), errors="coerce").dropna()
        if n_ser.empty:
            continue
        n_days = int(max(1, round(float(n_ser.max()))))

        day_ser = pd.to_numeric(hg.get("day_in_harvest"), errors="coerce")
        if day_ser.notna().any():
            day_obs = day_ser.round().astype("Int64")
        else:
            day_obs = ((pd.to_datetime(hg["fecha_evento"], errors="coerce") - harvest_start).dt.days + 1).round().astype("Int64")

        m_day_valid = day_obs.notna() & (day_obs >= 1)
        if not bool(m_day_valid.any()):
            continue
        day_obs_valid = day_obs.loc[m_day_valid]

        existing_days = set(day_obs_valid.astype(int).tolist())
        missing_days = [d for d in range(1, n_days + 1) if d not in existing_days]
        if not missing_days:
            continue

        last_day = int(day_obs_valid.max())
        hg_valid = hg.loc[m_day_valid].copy()
        hg_valid["__day"] = day_obs_valid.astype(int).to_numpy()
        hg_tpl = hg_valid.loc[hg_valid["__day"].eq(last_day)].drop(columns=["__day"])
        if hg_tpl.empty:
            continue

        post = cdf.loc[cdf["stage"].eq("POST")].copy()
        post_tpl = pd.DataFrame()
        if not post.empty:
            day_post = pd.to_numeric(post.get("day_in_harvest"), errors="coerce").round().astype("Int64")
            post["__day"] = day_post
            post_tpl = post.loc[post["__day"].eq(last_day)].drop(columns=["__day"])
            if post_tpl.empty:
                post_tpl = post.tail(min(len(hg_tpl), len(post))).copy()

        for d in missing_days:
            new_date = (harvest_start + pd.Timedelta(days=d - 1)).normalize()
            rel = float(d) / float(n_days)

            add_hg = hg_tpl.copy()
            add_hg["fecha_evento"] = new_date
            add_hg["day_in_harvest"] = float(d)
            add_hg["n_harvest_days"] = float(n_days)
            add_hg["rel_pos"] = rel
            for c in ["factor_tallos_dia", "share_grado", "factor_peso_tallo", "tallos_dia", "tallos_grado_dia", "kg_verde", "gramos_verde"]:
                if c in add_hg.columns:
                    add_hg[c] = np.nan
            if "fecha_post" in add_hg.columns:
                add_hg["fecha_post"] = pd.NaT
            if "row_id" in add_hg.columns:
                add_hg["row_id"] = np.arange(next_row_id, next_row_id + len(add_hg), dtype=np.int64)
                next_row_id += len(add_hg)
            new_rows.append(add_hg)

            if not post_tpl.empty:
                add_p = post_tpl.copy()
                add_p["fecha_evento"] = new_date
                add_p["day_in_harvest"] = float(d)
                add_p["n_harvest_days"] = float(n_days)
                add_p["rel_pos"] = rel
                if "dh_dias" in add_p.columns:
                    dh = pd.to_numeric(add_p["dh_dias"], errors="coerce")
                    dh_int = np.rint(dh).astype("Int64")
                    add_p["fecha_post"] = pd.to_datetime(new_date) + pd.to_timedelta(dh_int.fillna(0).astype(int), unit="D")
                    add_p["fecha_post"] = _to_date(add_p["fecha_post"])
                for c in ["factor_hidr", "factor_desp", "factor_ajuste", "tallos_post", "kg_verde", "gramos_verde", "kg_post", "cajas_verde", "cajas_post", "aprovechamiento"]:
                    if c in add_p.columns:
                        add_p[c] = np.nan
                if "row_id" in add_p.columns:
                    add_p["row_id"] = np.arange(next_row_id, next_row_id + len(add_p), dtype=np.int64)
                    next_row_id += len(add_p)
                new_rows.append(add_p)

    if new_rows:
        ext = pd.concat([work] + new_rows, ignore_index=True)
        ext = ext.sort_values(["ciclo_id", "fecha_evento", "stage", "bloque_base", "variedad_canon", "grado", "destino"], kind="mergesort").reset_index(drop=True)
        v.drop(v.index, inplace=True)
        for c in ext.columns:
            v[c] = ext[c]


def _realign_temporal_axis(v: pd.DataFrame) -> None:
    m_hg_post = v["stage"].isin(["HARVEST_GRADE", "POST"])
    day = pd.to_numeric(v.get("day_in_harvest"), errors="coerce")
    n_days = pd.to_numeric(v.get("n_harvest_days"), errors="coerce")

    # Keep only valid day positions inside horizon.
    m_day_valid = day.notna() & (day >= 1)
    m_n_valid = n_days.notna() & (n_days >= 1)
    if m_n_valid.any():
        m_day_valid = m_day_valid & ((~m_n_valid) | (day <= n_days))

    hs = pd.to_datetime(v.get("harvest_start"), errors="coerce").dt.normalize()
    m_retime = m_hg_post & hs.notna() & m_day_valid
    if m_retime.any():
        day_int = np.rint(day.loc[m_retime]).astype(int)
        v.loc[m_retime, "fecha_evento"] = (hs.loc[m_retime] + pd.to_timedelta(day_int - 1, unit="D")).dt.normalize()

    # Recalculate rel_pos after temporal alignment.
    v["rel_pos"] = _safe_div(pd.to_numeric(v.get("day_in_harvest"), errors="coerce"), pd.to_numeric(v.get("n_harvest_days"), errors="coerce"), default=np.nan)

    # Rebuild fecha_post from dh_dias when available.
    m_post = v["stage"].eq("POST")
    dh = pd.to_numeric(v.get("dh_dias"), errors="coerce")
    fev = pd.to_datetime(v.get("fecha_evento"), errors="coerce").dt.normalize()
    m_fp = m_post & fev.notna() & dh.notna() & (dh >= 0)
    if m_fp.any():
        dh_int = np.rint(dh.loc[m_fp]).astype(int)
        v.loc[m_fp, "fecha_post"] = (fev.loc[m_fp] + pd.to_timedelta(dh_int, unit="D")).dt.normalize()


def _balance_ml2_to_tallos_proy(v: pd.DataFrame) -> None:
    m_hg = v["stage"].eq("HARVEST_GRADE")
    m_post = v["stage"].eq("POST")
    if not m_hg.any():
        return

    cyc = (
        v.loc[m_hg]
        .groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            tallos_pred_total=("tallos_grado_dia", "sum"),
            tallos_target=("tallos_proy", "max"),
        )
    )
    cyc["tallos_target"] = pd.to_numeric(cyc["tallos_target"], errors="coerce")
    cyc["tallos_pred_total"] = pd.to_numeric(cyc["tallos_pred_total"], errors="coerce")
    cyc["scale_hg"] = np.where(
        cyc["tallos_target"].notna() & (cyc["tallos_target"] > 0) & (cyc["tallos_pred_total"] > 0),
        cyc["tallos_target"] / cyc["tallos_pred_total"],
        1.0,
    )
    scale_map = pd.Series(cyc["scale_hg"].to_numpy(dtype="float64"), index=cyc["ciclo_id"].astype("string"))
    sc = v["ciclo_id"].astype("string").map(scale_map).fillna(1.0)

    # HG: enforce mass balance on stems and derived green weight.
    v.loc[m_hg, "tallos_grado_dia"] = pd.to_numeric(v.loc[m_hg, "tallos_grado_dia"], errors="coerce") * sc.loc[m_hg]
    v.loc[m_hg, "tallos_dia"] = pd.to_numeric(v.loc[m_hg, "tallos_dia"], errors="coerce") * sc.loc[m_hg]

    # Recompute kg verde from tallos * peso for internal consistency.
    peso_base = pd.to_numeric(v.get("factor_peso_tallo"), errors="coerce").fillna(1.0)
    # factor_peso_tallo multiplies baseline stem weight available in source base;
    # if base weight isn't materialized here, keep proportional scaling on kg.
    if "kg_verde" in v.columns:
        v.loc[m_hg, "kg_verde"] = pd.to_numeric(v.loc[m_hg, "kg_verde"], errors="coerce") * sc.loc[m_hg]
        v.loc[m_hg, "gramos_verde"] = pd.to_numeric(v.loc[m_hg, "gramos_verde"], errors="coerce") * sc.loc[m_hg]

    # POST: keep consistency with upstream stems/green split.
    if m_post.any():
        v.loc[m_post, "tallos_post"] = pd.to_numeric(v.loc[m_post, "tallos_post"], errors="coerce") * sc.loc[m_post]
        v.loc[m_post, "kg_verde"] = pd.to_numeric(v.loc[m_post, "kg_verde"], errors="coerce") * sc.loc[m_post]
        v.loc[m_post, "gramos_verde"] = pd.to_numeric(v.loc[m_post, "gramos_verde"], errors="coerce") * sc.loc[m_post]
        v.loc[m_post, "cajas_verde"] = pd.to_numeric(v.loc[m_post, "cajas_verde"], errors="coerce") * sc.loc[m_post]

        prod = (
            pd.to_numeric(v.get("factor_hidr"), errors="coerce")
            * pd.to_numeric(v.get("factor_desp"), errors="coerce")
            * pd.to_numeric(v.get("factor_ajuste"), errors="coerce")
        )
        v.loc[m_post, "kg_post"] = pd.to_numeric(v.loc[m_post, "kg_verde"], errors="coerce") * prod.loc[m_post]
        v.loc[m_post, "cajas_post"] = pd.to_numeric(v.loc[m_post, "cajas_verde"], errors="coerce") * prod.loc[m_post]
        v.loc[m_post, "aprovechamiento"] = _safe_div(v.loc[m_post, "kg_post"], v.loc[m_post, "kg_verde"], default=np.nan)


def _apply_stage_masks(v: pd.DataFrame) -> None:
    m_veg = v["stage"].eq("VEG")
    m_hg = v["stage"].eq("HARVEST_GRADE")
    m_post = v["stage"].eq("POST")

    veg_na = [
        "day_in_harvest",
        "rel_pos",
        "factor_tallos_dia",
        "share_grado",
        "factor_peso_tallo",
        "tallos_dia",
        "tallos_grado_dia",
        "dh_dias",
        "factor_hidr",
        "factor_desp",
        "factor_ajuste",
        "tallos_post",
        "kg_verde",
        "gramos_verde",
        "kg_post",
        "cajas_verde",
        "cajas_post",
        "aprovechamiento",
    ]
    for c in veg_na:
        if c in v.columns:
            v.loc[m_veg, c] = np.nan

    hg_na = ["dh_dias", "factor_hidr", "factor_desp", "factor_ajuste", "tallos_post", "kg_post", "cajas_post", "aprovechamiento"]
    for c in hg_na:
        if c in v.columns:
            v.loc[m_hg, c] = np.nan

    post_na = ["factor_tallos_dia", "share_grado", "factor_peso_tallo"]
    for c in post_na:
        if c in v.columns:
            v.loc[m_post, c] = np.nan


def _build_view_ml1(base: pd.DataFrame, run_id: str) -> pd.DataFrame:
    v = _common_view_base(base, source="ML1", run_id=run_id)
    cyc = _veg_calendar_map(base, d_col="pred_d_start", n_col="pred_n_harvest_days")
    v = v.merge(cyc, on="ciclo_id", how="left", suffixes=("", "_cyc"))
    for c in ["fecha_sp", "harvest_start", "harvest_end", "d_start", "n_harvest_days"]:
        if f"{c}_cyc" in v.columns:
            v[c] = v[f"{c}_cyc"]
            v = v.drop(columns=[f"{c}_cyc"])

    # Recompute ML1 day index from its own harvest_start to avoid contamination from ML2-expanded rows.
    fev = _to_date(v["fecha_evento"])
    hs = _to_date(v["harvest_start"])
    d_calc = (pd.to_datetime(fev, errors="coerce") - pd.to_datetime(hs, errors="coerce")).dt.days + 1
    m_chain = v["stage"].isin(["HARVEST_GRADE", "POST"]) & hs.notna() & fev.notna()
    day_base = _num(base.get("day_in_harvest"), default=np.nan)
    v["day_in_harvest"] = day_base
    v.loc[m_chain, "day_in_harvest"] = pd.to_numeric(d_calc.loc[m_chain], errors="coerce")
    v["rel_pos"] = _safe_div(v["day_in_harvest"], v["n_harvest_days"], default=np.nan)
    v["factor_tallos_dia"] = _num(base.get("pred_factor_tallos_dia"), default=np.nan)
    v["share_grado"] = _num(base.get("pred_share_grado"), default=np.nan)
    v["factor_peso_tallo"] = _num(base.get("pred_factor_peso_tallo"), default=np.nan)
    v["tallos_dia"] = _num(base.get("tallos_pred_ml1_dia"), default=np.nan)
    v["tallos_grado_dia"] = _num(base.get("tallos_pred_ml1_grado_dia"), default=np.nan)

    v["dh_dias"] = _num(base.get("pred_dh_dias"), default=np.nan)
    v["factor_hidr"] = _num(base.get("pred_factor_hidr"), default=np.nan)
    v["factor_desp"] = _num(base.get("pred_factor_desp"), default=np.nan)
    v["factor_ajuste"] = _num(base.get("pred_factor_ajuste"), default=np.nan)
    v["tallos_post"] = _num(base.get("tallos_post_proy"), default=np.nan)

    v["kg_verde"] = _num(base.get("kg_verde_ref"), default=np.nan)
    v["gramos_verde"] = _num(base.get("gramos_verde_ref"), default=np.nan)
    v["gramos_verde"] = v["gramos_verde"].fillna(v["kg_verde"] * 1000.0)
    prod = v["factor_hidr"] * v["factor_desp"] * v["factor_ajuste"]
    v["kg_post"] = v["kg_verde"] * prod
    v["cajas_verde"] = _num(base.get("cajas_split_grado_dia"), default=np.nan)
    v["cajas_post"] = v["cajas_verde"] * prod
    v["aprovechamiento"] = _safe_div(v["kg_post"], v["kg_verde"], default=np.nan)

    # Expand missing HG/POST days to the predicted horizon.
    _expand_missing_horizon_rows(v)
    # ML1 is kept as pure model timeline (no temporal replacement).
    _apply_stage_masks(v)
    _fill_cycle_totals(v)
    return v


def _build_view_ml2(base: pd.DataFrame, run_id: str, source: str) -> pd.DataFrame:
    v = _common_view_base(base, source=source, run_id=run_id)
    cyc = _veg_calendar_map(
        base,
        d_col="pred_ml2_d_start",
        n_col="pred_ml2_n_harvest_days",
        start_col="pred_harvest_start_ml2",
        end_col="pred_harvest_end_ml2",
    )
    v = v.merge(cyc, on="ciclo_id", how="left", suffixes=("", "_cyc"))
    for c in ["fecha_sp", "harvest_start", "harvest_end", "d_start", "n_harvest_days"]:
        if f"{c}_cyc" in v.columns:
            v[c] = v[f"{c}_cyc"]
            v = v.drop(columns=[f"{c}_cyc"])

    # If real start exists, ML2 planning timeline anchors from real start.
    veg_anchor = base.loc[base["stage"].eq("VEG"), ["ciclo_id", "fecha_evento", "target_d_start"]].copy()
    if not veg_anchor.empty:
        veg_anchor = veg_anchor.sort_values("fecha_evento").drop_duplicates(subset=["ciclo_id"], keep="last")
        veg_anchor["fecha_sp"] = _to_date(veg_anchor["fecha_evento"])
        veg_anchor["d_start_real"] = _num(veg_anchor.get("target_d_start"), default=np.nan)
        m_real = veg_anchor["d_start_real"].notna()
        if m_real.any():
            veg_anchor.loc[m_real, "harvest_start_real"] = veg_anchor.loc[m_real, "fecha_sp"] + pd.to_timedelta(
                np.rint(veg_anchor.loc[m_real, "d_start_real"]).astype(int), unit="D"
            )
            anchor_map = pd.Series(
                pd.to_datetime(veg_anchor["harvest_start_real"], errors="coerce").dt.normalize().to_numpy(),
                index=veg_anchor["ciclo_id"].astype("string"),
            )
            d_map = pd.Series(veg_anchor["d_start_real"].to_numpy(dtype="float64"), index=veg_anchor["ciclo_id"].astype("string"))
            cid = v["ciclo_id"].astype("string")
            hs_anchor = pd.to_datetime(cid.map(anchor_map), errors="coerce").dt.normalize()
            m_has = hs_anchor.notna()
            v.loc[m_has, "harvest_start"] = hs_anchor.loc[m_has]
            v.loc[m_has, "d_start"] = pd.to_numeric(cid.loc[m_has].map(d_map), errors="coerce")
            nd = pd.to_numeric(v.loc[m_has, "n_harvest_days"], errors="coerce").fillna(1.0).clip(lower=1.0)
            v.loc[m_has, "harvest_end"] = (
                pd.to_datetime(v.loc[m_has, "harvest_start"], errors="coerce")
                + pd.to_timedelta(np.rint(nd).astype(int) - 1, unit="D")
            ).dt.normalize()

    v["fecha_post"] = _to_date(base.get("fecha_post_ml2")).fillna(_to_date(base["fecha_post"]))
    day_ml2 = _num(base.get("day_in_harvest_ml2"), default=np.nan)
    day_ml1 = _num(base.get("day_in_harvest"), default=np.nan)
    day_ml2 = day_ml2.where(day_ml2 >= 1, np.nan)
    v["day_in_harvest"] = day_ml2.fillna(day_ml1)
    v["rel_pos"] = _num(base.get("rel_pos_ml2"), default=np.nan).fillna(_num(base.get("rel_pos"), default=np.nan))

    v["factor_tallos_dia"] = _num(base.get("pred_ml2_factor_tallos_dia"), default=np.nan).fillna(_num(base.get("pred_factor_tallos_dia"), default=np.nan))
    v["share_grado"] = _num(base.get("share_grado_ml2_norm"), default=np.nan).fillna(_num(base.get("pred_ml2_share_grado"), default=np.nan)).fillna(_num(base.get("pred_share_grado"), default=np.nan))
    v["factor_peso_tallo"] = _num(base.get("pred_ml2_factor_peso_tallo"), default=np.nan).fillna(_num(base.get("pred_factor_peso_tallo"), default=np.nan))
    v["tallos_dia"] = _num(base.get("tallos_pred_ml2_dia"), default=np.nan).fillna(_num(base.get("tallos_pred_ml1_dia"), default=np.nan))
    v["tallos_grado_dia"] = _num(base.get("tallos_pred_ml2_grado_dia"), default=np.nan).fillna(_num(base.get("tallos_pred_ml1_grado_dia"), default=np.nan))

    v["dh_dias"] = _num(base.get("pred_ml2_dh_dias"), default=np.nan).fillna(_num(base.get("pred_dh_dias"), default=np.nan))
    v["factor_hidr"] = _num(base.get("pred_ml2_factor_hidr"), default=np.nan).fillna(_num(base.get("pred_factor_hidr"), default=np.nan))
    v["factor_desp"] = _num(base.get("pred_ml2_factor_desp"), default=np.nan).fillna(_num(base.get("pred_factor_desp"), default=np.nan))
    v["factor_ajuste"] = _num(base.get("pred_ml2_factor_ajuste"), default=np.nan).fillna(_num(base.get("pred_factor_ajuste"), default=np.nan))
    v["tallos_post"] = _num(base.get("tallos_post_ml2_proy"), default=np.nan).fillna(_num(base.get("tallos_post_proy"), default=np.nan))

    v["kg_verde"] = _num(base.get("kg_verde_ml2"), default=np.nan).fillna(_num(base.get("kg_verde_ref"), default=np.nan))
    v["gramos_verde"] = _num(base.get("gramos_verde_ml2"), default=np.nan).fillna(_num(base.get("gramos_verde_ref"), default=np.nan))
    v["gramos_verde"] = v["gramos_verde"].fillna(v["kg_verde"] * 1000.0)
    prod = v["factor_hidr"] * v["factor_desp"] * v["factor_ajuste"]
    v["kg_post"] = _num(base.get("kg_post_ml2"), default=np.nan).fillna(v["kg_verde"] * prod)
    v["cajas_verde"] = _num(base.get("cajas_split_grado_dia_ml2"), default=np.nan).fillna(_num(base.get("cajas_split_grado_dia"), default=np.nan))
    v["cajas_post"] = _num(base.get("cajas_post_ml2"), default=np.nan).fillna(v["cajas_verde"] * prod)
    v["aprovechamiento"] = _safe_div(v["kg_post"], v["kg_verde"], default=np.nan)

    _expand_missing_horizon_rows(v)
    _realign_temporal_axis(v)
    _balance_ml2_to_tallos_proy(v)
    _apply_stage_masks(v)
    _fill_cycle_totals(v)
    return v


def _build_view_real(base: pd.DataFrame, run_id: str) -> pd.DataFrame:
    v = _common_view_base(base, source="REAL", run_id=run_id)

    hgr, cyc_real, day_real = _load_real_harvest_join(base)
    post_real = _load_real_post_alloc(base)

    # Cycle-level real calendar from observed harvest.
    veg = base.loc[base["stage"].eq("VEG"), ["ciclo_id", "fecha_evento", "target_d_start", "target_n_harvest_days"]].copy()
    veg["fecha_sp"] = _to_date(veg["fecha_evento"])
    veg = veg.sort_values("fecha_sp", kind="mergesort").drop_duplicates(subset=["ciclo_id"], keep="last")

    cyc = veg[["ciclo_id", "fecha_sp"]].merge(cyc_real, on="ciclo_id", how="left")
    cyc["d_start"] = _safe_div((pd.to_datetime(cyc["harvest_start"]) - pd.to_datetime(cyc["fecha_sp"])).dt.days, pd.Series(1.0, index=cyc.index), default=np.nan)
    cyc["n_harvest_days"] = _num(cyc.get("n_harvest_days"), default=np.nan)
    cyc = cyc.merge(
        veg[["ciclo_id", "target_d_start", "target_n_harvest_days"]].rename(
            columns={"target_d_start": "d_start_target", "target_n_harvest_days": "n_harvest_days_target"}
        ),
        on="ciclo_id",
        how="left",
    )
    cyc["d_start"] = _num(cyc["d_start_target"], default=np.nan).fillna(cyc["d_start"])
    cyc["n_harvest_days"] = _num(cyc["n_harvest_days_target"], default=np.nan).fillna(cyc["n_harvest_days"])
    cyc = cyc.drop(columns=[c for c in ["d_start_target", "n_harvest_days_target"] if c in cyc.columns])

    v = v.merge(cyc, on="ciclo_id", how="left", suffixes=("", "_cyc"))
    for c in ["fecha_sp", "harvest_start", "harvest_end", "d_start", "n_harvest_days"]:
        if f"{c}_cyc" in v.columns:
            v[c] = v[f"{c}_cyc"]
            v = v.drop(columns=[f"{c}_cyc"])

    # Harvest-grade real rows.
    if not hgr.empty:
        hr = hgr[["row_id", "tallos_dia_real", "tallos_grado_real", "kg_verde_grado_real", "factor_tallos_dia_real", "share_grado_real", "factor_peso_tallo_real"]].copy()
        v = v.merge(hr, on="row_id", how="left")
        v["tallos_dia"] = _num(v.get("tallos_dia_real"), default=np.nan)
        v["tallos_grado_dia"] = _num(v.get("tallos_grado_real"), default=np.nan)
        v["kg_verde"] = _num(v.get("kg_verde_grado_real"), default=np.nan)
        v["gramos_verde"] = v["kg_verde"] * 1000.0
        v["factor_tallos_dia"] = _num(v.get("factor_tallos_dia_real"), default=np.nan)
        v["share_grado"] = _num(v.get("share_grado_real"), default=np.nan)
        v["factor_peso_tallo"] = _num(v.get("factor_peso_tallo_real"), default=np.nan)

    # day_in_harvest / rel_pos in real.
    dhr = _num(base.get("dia_rel_cosecha_real"), default=np.nan)
    dhr = np.where(pd.to_numeric(dhr, errors="coerce").notna(), pd.to_numeric(dhr, errors="coerce") + 1.0, np.nan)
    dhr = pd.Series(dhr, index=v.index, dtype="float64")
    dhr = dhr.where(dhr >= 1, np.nan)
    v["day_in_harvest"] = dhr
    m_chain = v["stage"].isin(["HARVEST_GRADE", "POST"])
    d_calc = (pd.to_datetime(v["fecha_evento"]) - pd.to_datetime(v["harvest_start"])).dt.days + 1
    m_calc = m_chain & pd.to_datetime(v["harvest_start"], errors="coerce").notna() & pd.to_datetime(v["fecha_evento"], errors="coerce").notna()
    # REAL view must follow observed dates; derive day index from real fecha_evento when available.
    v.loc[m_calc, "day_in_harvest"] = pd.to_numeric(d_calc[m_calc], errors="coerce")
    v["rel_pos"] = _safe_div(v["day_in_harvest"], v["n_harvest_days"], default=np.nan)

    # Post real rows (allocated by share_block_post)
    if not post_real.empty:
        pr = post_real[
            [
                "row_id",
                "dh_dias_real",
                "factor_hidr_real",
                "tallos_post_real",
                "kg_verde_real",
                "kg_post_real",
                "aprovechamiento_real",
            ]
        ].copy()
        v = v.merge(pr, on="row_id", how="left")
        m_post = v["stage"].eq("POST")
        v.loc[m_post, "dh_dias"] = _num(v.loc[m_post, "dh_dias_real"], default=np.nan).to_numpy(dtype="float64")
        v.loc[m_post, "factor_hidr"] = _num(v.loc[m_post, "factor_hidr_real"], default=np.nan).to_numpy(dtype="float64")
        v.loc[m_post, "factor_desp"] = np.nan
        v.loc[m_post, "factor_ajuste"] = np.nan
        v.loc[m_post, "tallos_post"] = _num(v.loc[m_post, "tallos_post_real"], default=np.nan).to_numpy(dtype="float64")
        v.loc[m_post, "kg_verde"] = _num(v.loc[m_post, "kg_verde_real"], default=np.nan).to_numpy(dtype="float64")
        v.loc[m_post, "gramos_verde"] = v.loc[m_post, "kg_verde"] * 1000.0
        v.loc[m_post, "kg_post"] = _num(v.loc[m_post, "kg_post_real"], default=np.nan).to_numpy(dtype="float64")
        v.loc[m_post, "cajas_verde"] = np.nan
        v.loc[m_post, "cajas_post"] = np.nan
        v.loc[m_post, "aprovechamiento"] = _num(v.loc[m_post, "aprovechamiento_real"], default=np.nan).to_numpy(dtype="float64")

    # Real view should keep only observed rows in HG/POST.
    m_veg = v["stage"].eq("VEG")
    m_hg_real = v["stage"].eq("HARVEST_GRADE") & pd.to_numeric(v.get("tallos_grado_dia"), errors="coerce").fillna(0.0).gt(0.0)
    m_post_real = v["stage"].eq("POST") & (
        pd.to_numeric(v.get("tallos_post"), errors="coerce").fillna(0.0).gt(0.0)
        | pd.to_numeric(v.get("kg_post"), errors="coerce").fillna(0.0).gt(0.0)
    )
    v = v.loc[m_veg | m_hg_real | m_post_real].copy()

    # Some upstream ML rows may be duplicated for the same harvest-grade business key.
    # In REAL view we must keep one row per observed key to avoid double-counting fact values.
    m_hg = v["stage"].eq("HARVEST_GRADE")
    hg_key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado"] if c in v.columns]
    if bool(m_hg.any()) and hg_key:
        hg = v.loc[m_hg].copy()
        if "row_id" in hg.columns:
            hg = hg.sort_values("row_id", kind="mergesort")
        hg = hg.drop_duplicates(subset=hg_key, keep="first")
        v = pd.concat([v.loc[~m_hg], hg], ignore_index=True)

    _realign_temporal_axis(v)

    # Enforce real operational window where available.
    m_hg_post = v["stage"].isin(["HARVEST_GRADE", "POST"])
    hs = pd.to_datetime(v.get("harvest_start"), errors="coerce").dt.normalize()
    he = pd.to_datetime(v.get("harvest_end"), errors="coerce").dt.normalize()
    fev = pd.to_datetime(v.get("fecha_evento"), errors="coerce").dt.normalize()
    m_keep = (~m_hg_post) | ((~hs.notna() | (fev >= hs)) & (~he.notna() | (fev <= he)))
    v = v.loc[m_keep].copy()

    _apply_stage_masks(v)
    _fill_cycle_totals(v)
    return v


def _write_named_and_latest(df: pd.DataFrame, out_dir: Path, stem: str, run_id: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_run = out_dir / f"{stem}_{run_id}.parquet"
    latest = out_dir / f"{stem}.parquet"
    write_parquet(df, by_run)
    write_parquet(df, latest)
    return by_run, latest


def main() -> None:
    args = _parse_args()
    pred_oper_path = (
        Path(args.pred_ml2_operativo)
        if args.pred_ml2_operativo
        else (Path(args.pred_ml2) if args.pred_ml2 else _latest_pred_ml2())
    )
    if not pred_oper_path.exists():
        raise FileNotFoundError(f"ML2 operativo prediction file not found: {pred_oper_path}")

    run_id = _run_id_from_path(pred_oper_path)
    pred_pure_path = Path(args.pred_ml2_puro) if args.pred_ml2_puro else (
        pred_oper_path.parent / f"pred_ml2_multitask_nn_puro_{run_id}.parquet"
    )
    if not pred_pure_path.exists():
        pred_pure_path = pred_oper_path

    pred_oper = read_parquet(pred_oper_path).copy()
    pred_pure = read_parquet(pred_pure_path).copy()
    base_oper = _base_from_pred(pred_oper)
    base_pure = _base_from_pred(pred_pure)

    # ML1 should come from its own pure prediction file (if trace exists), not from ML2-expanded rows.
    base_ml1 = base_oper
    if "ml1_input_file" in pred_oper.columns:
        ml1_files = pred_oper["ml1_input_file"].dropna().astype("string").unique().tolist()
        if ml1_files:
            ml1_name = str(ml1_files[0])
            ml1_path = (DATA / "gold" / "ml1_nn" / ml1_name)
            if ml1_path.exists():
                pred_ml1 = read_parquet(ml1_path).copy()
                base_ml1 = _base_from_pred(pred_ml1)

    view_ml1 = _build_view_ml1(base=base_ml1, run_id=run_id)
    view_ml2_puro = _build_view_ml2(base=base_pure, run_id=run_id, source="ML2_PURO")
    view_ml2_oper = _build_view_ml2(base=base_oper, run_id=run_id, source="ML2_OPERATIVO")
    # REAL view must be anchored on ML1 base (closest to raw operational timeline),
    # not on ML2-generated rows.
    base_real = base_ml1.copy()
    if "ml2_row_generated" in base_real.columns:
        base_real = base_real.loc[pd.to_numeric(base_real["ml2_row_generated"], errors="coerce").fillna(0).eq(0)].copy()
    view_real = _build_view_real(base=base_real, run_id=run_id)

    out_dir = Path(args.output_dir)
    p1_run, p1_latest = _write_named_and_latest(view_ml1, out_dir, "view_pipeline_ml1_global", run_id)
    p2p_run, p2p_latest = _write_named_and_latest(view_ml2_puro, out_dir, "view_pipeline_ml2_puro_global", run_id)
    p2o_run, p2o_latest = _write_named_and_latest(view_ml2_oper, out_dir, "view_pipeline_ml2_operativo_global", run_id)
    # Legacy ML2 alias keeps operativo semantics.
    p2_run, p2_latest = _write_named_and_latest(view_ml2_oper, out_dir, "view_pipeline_ml2_global", run_id)
    p3_run, p3_latest = _write_named_and_latest(view_real, out_dir, "view_pipeline_real_global", run_id)

    print("[OK] Wrote separated global views (VEG/HARVEST_GRADE/POST):")
    print(f"     ML1            : {p1_run}")
    print(f"     ML2_puro       : {p2p_run}")
    print(f"     ML2_operativo  : {p2o_run}")
    print(f"     REAL           : {p3_run}")
    print(f"     ML2 legacy     : {p2_run}")
    print("[OK] Latest aliases:")
    print(f"     {p1_latest}")
    print(f"     {p2p_latest}")
    print(f"     {p2o_latest}")
    print(f"     {p2_latest}")
    print(f"     {p3_latest}")
    print(
        "     rows="
        f"ml1:{len(view_ml1):,} "
        f"ml2_puro:{len(view_ml2_puro):,} "
        f"ml2_operativo:{len(view_ml2_oper):,} "
        f"real:{len(view_real):,}"
    )


if __name__ == "__main__":
    main()
