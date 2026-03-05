from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold" / "ml2_nn"
EVAL = DATA / "eval" / "ml2_nn"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("audit_ml_pipeline_views")
    ap.add_argument("--ml1", default=str(GOLD / "view_pipeline_ml1_global.parquet"))
    ap.add_argument("--ml2-global", default=str(GOLD / "view_pipeline_ml2_global.parquet"))
    ap.add_argument("--ml2-puro", default=str(GOLD / "view_pipeline_ml2_puro_global.parquet"))
    ap.add_argument("--ml2-operativo", default=str(GOLD / "view_pipeline_ml2_operativo_global.parquet"))
    ap.add_argument("--real", default=str(GOLD / "view_pipeline_real_global.parquet"))
    ap.add_argument("--output-dir", default=str(EVAL))
    ap.add_argument("--tag", default="latest")
    return ap.parse_args()


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _num(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input view: {path}")
    df = read_parquet(path).copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "stage" in df.columns:
        df["stage"] = df["stage"].astype("string").str.upper().str.strip()
    if "ciclo_id" in df.columns:
        df["ciclo_id"] = df["ciclo_id"].astype("string")
    if "fecha_evento" in df.columns:
        df["fecha_evento"] = _to_date(df["fecha_evento"])
    if "harvest_end" in df.columns:
        df["harvest_end"] = _to_date(df["harvest_end"])
    return df


def _check_hg_day_consistency(df: pd.DataFrame, source: str) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df.columns:
        return out
    hg = df.loc[df["stage"].eq("HARVEST_GRADE")].copy()
    if hg.empty:
        return out
    day = _num(hg.get("day_in_harvest"), default=np.nan).round()
    fev = _to_date(hg.get("fecha_evento"))
    hg["__day_key"] = day.where(day.notna(), fev)
    key = [c for c in ["ciclo_id", "__day_key", "bloque_base", "variedad_canon"] if c in hg.columns]
    if len(key) < 2:
        return out
    by_day = (
        hg.groupby(key, dropna=False, as_index=False)
        .agg(tallos_dia_first=("tallos_dia", "first"))
    )
    day_sum = (
        by_day.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(sum_day=("tallos_dia_first", "sum"))
    )
    grade_sum = (
        hg.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(sum_grade=("tallos_grado_dia", "sum"))
    )
    cyc = day_sum.merge(grade_sum, on="ciclo_id", how="outer")
    cyc["sum_day"] = _num(cyc["sum_day"], default=0.0)
    cyc["sum_grade"] = _num(cyc["sum_grade"], default=0.0)
    cyc["diff"] = (cyc["sum_day"] - cyc["sum_grade"]).abs()
    cyc["tol"] = np.maximum(1e-3, cyc["sum_grade"].abs() * 1e-5)
    for r in cyc.itertuples(index=False):
        ok = bool(r.diff <= r.tol)
        out.append(
            {
                "check_id": "hg_day_consistency",
                "source_model": source,
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.diff),
                "expected": 0.0,
                "tolerance": float(r.tol),
                "details": f"sum_day={float(r.sum_day):.6f} sum_grade={float(r.sum_grade):.6f}",
            }
        )
    return out


def _check_post_duplicates(df: pd.DataFrame, source: str) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df.columns:
        return out
    post = df.loc[df["stage"].eq("POST")].copy()
    if post.empty:
        return out
    key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado", "destino"] if c in post.columns]
    if len(key) < 2:
        return out
    dups = post.groupby(key, dropna=False, as_index=False).size()
    dups = dups.loc[_num(dups["size"], default=0.0) > 1.0].copy()
    if dups.empty:
        out.append(
            {
                "check_id": "post_duplicates",
                "source_model": source,
                "ciclo_id": "__ALL__",
                "status": "PASS",
                "value": 0.0,
                "expected": 0.0,
                "tolerance": 0.0,
                "details": "no duplicated POST business keys",
            }
        )
        return out
    by_cyc = dups.groupby("ciclo_id", dropna=False, as_index=False).agg(dup_keys=("size", "count"))
    for r in by_cyc.itertuples(index=False):
        out.append(
            {
                "check_id": "post_duplicates",
                "source_model": source,
                "ciclo_id": str(r.ciclo_id),
                "status": "FAIL",
                "value": float(r.dup_keys),
                "expected": 0.0,
                "tolerance": 0.0,
                "details": f"duplicated_post_keys={int(r.dup_keys)}",
            }
        )
    return out


def _check_total_vs_tallos_proy(df: pd.DataFrame, source: str) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df.columns or "tallos_proy" not in df.columns:
        return out
    hg = df.loc[df["stage"].eq("HARVEST_GRADE")].copy()
    if hg.empty:
        return out
    cyc = (
        hg.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(sum_grade=("tallos_grado_dia", "sum"), tallos_proy=("tallos_proy", "max"))
    )
    cyc["sum_grade"] = _num(cyc["sum_grade"], default=0.0)
    cyc["tallos_proy"] = _num(cyc["tallos_proy"], default=0.0)
    cyc["diff"] = (cyc["sum_grade"] - cyc["tallos_proy"]).abs()
    cyc["tol"] = np.maximum(1e-3, cyc["tallos_proy"].abs() * 1e-5)
    for r in cyc.itertuples(index=False):
        ok = bool(r.diff <= r.tol)
        out.append(
            {
                "check_id": "total_vs_tallos_proy",
                "source_model": source,
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.diff),
                "expected": 0.0,
                "tolerance": float(r.tol),
                "details": f"sum_grade={float(r.sum_grade):.6f} tallos_proy={float(r.tallos_proy):.6f}",
            }
        )
    return out


def _check_ml2_start_vs_real(df_ml2: pd.DataFrame, source: str, real_start: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    hg = df_ml2.loc[df_ml2["stage"].eq("HARVEST_GRADE"), ["ciclo_id", "fecha_evento"]].copy()
    if hg.empty or real_start.empty:
        return out
    ml2_start = hg.groupby("ciclo_id", dropna=False, as_index=False).agg(start_ml2=("fecha_evento", "min"))
    z = ml2_start.merge(real_start, on="ciclo_id", how="inner")
    if z.empty:
        return out
    z["delta_days"] = (pd.to_datetime(z["start_ml2"]) - pd.to_datetime(z["start_real"])).dt.days
    # Allow small negative drift for operational backfill/normalization edge cases.
    min_delta_days = -3
    for r in z.itertuples(index=False):
        ok = bool(r.delta_days >= min_delta_days)
        out.append(
            {
                "check_id": "ml2_start_vs_real",
                "source_model": source,
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.delta_days),
                "expected": float(min_delta_days),
                "tolerance": 0.0,
                "details": f"start_ml2={str(r.start_ml2)} start_real={str(r.start_real)}",
            }
        )
    return out


def _check_oper_forward_projection(df_oper: pd.DataFrame, real_start: pd.DataFrame, real_end: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    hg = df_oper.loc[df_oper["stage"].eq("HARVEST_GRADE")].copy()
    if hg.empty:
        return out
    z = hg.merge(real_end, on="ciclo_id", how="left").merge(real_start, on="ciclo_id", how="left")
    m_fwd = z["last_real"].notna() & (pd.to_datetime(z["fecha_evento"], errors="coerce") > pd.to_datetime(z["last_real"], errors="coerce"))
    zf = z.loc[m_fwd].copy()
    if zf.empty:
        out.append(
            {
                "check_id": "oper_forward_non_null",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": "__ALL__",
                "status": "PASS",
                "value": 0.0,
                "expected": 0.0,
                "tolerance": 0.0,
                "details": "no forward rows after last_real",
            }
        )
        return out
    grp = (
        zf.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_rows=("tallos_grado_dia", "size"),
            n_null=("tallos_grado_dia", lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
        )
    )
    for r in grp.itertuples(index=False):
        ok = int(r.n_null) == 0
        out.append(
            {
                "check_id": "oper_forward_non_null",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.n_null),
                "expected": 0.0,
                "tolerance": 0.0,
                "details": f"forward_rows={int(r.n_rows)}",
            }
        )
    return out


def _check_oper_forward_post_non_null(df_oper: pd.DataFrame, real_end: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if post.empty:
        return out
    z = post.merge(real_end, on="ciclo_id", how="left")
    m_fwd = z["last_real"].notna() & (pd.to_datetime(z["fecha_evento"], errors="coerce") > pd.to_datetime(z["last_real"], errors="coerce"))
    zf = z.loc[m_fwd].copy()
    if zf.empty:
        return out
    grp = (
        zf.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_rows=("tallos_post", "size"),
            n_null=("tallos_post", lambda s: int(pd.to_numeric(s, errors="coerce").isna().sum())),
        )
    )
    for r in grp.itertuples(index=False):
        ok = int(r.n_null) == 0
        out.append(
            {
                "check_id": "oper_forward_post_non_null",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.n_null),
                "expected": 0.0,
                "tolerance": 0.0,
                "details": f"forward_post_rows={int(r.n_rows)}",
            }
        )
    return out


def _check_oper_post_mass_balance(df_oper: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df_oper.columns:
        return out
    key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado"] if c in df_oper.columns]
    if len(key) < 2:
        return out

    hg = df_oper.loc[df_oper["stage"].eq("HARVEST_GRADE"), key + ["tallos_grado_dia"]].copy()
    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if hg.empty or post.empty:
        return out

    post_mass_col = "tallos_post_real" if "tallos_post_real" in post.columns else "tallos_post"
    post_agg = (
        post.groupby(key, dropna=False, as_index=False)
        .agg(
            tallos_post_real_sum=(post_mass_col, "sum"),
            tallos_post_sum=("tallos_post", "sum"),
            kg_verde_sum=("kg_verde", "sum"),
            kg_post_sum=("kg_post", "sum"),
        )
    )
    z = hg.merge(post_agg, on=key, how="left")
    z["tallos_hg"] = _num(z["tallos_grado_dia"], default=0.0)
    z["tallos_post_real_sum"] = _num(z["tallos_post_real_sum"], default=0.0)
    z["diff"] = (z["tallos_post_real_sum"] - z["tallos_hg"]).abs()
    z["tol"] = np.maximum(1e-3, z["tallos_hg"].abs() * 1e-5)
    z["fail"] = z["diff"] > z["tol"]

    cyc = (
        z.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_keys=("fail", "size"),
            n_fail=("fail", "sum"),
            max_diff=("diff", "max"),
        )
    )
    for r in cyc.itertuples(index=False):
        ok = int(r.n_fail) == 0
        out.append(
            {
                "check_id": "oper_post_mass_balance",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.max_diff),
                "expected": 0.0,
                "tolerance": 0.0,
                "details": f"failed_keys={int(r.n_fail)}/{int(r.n_keys)}",
            }
        )
    return out


def _check_oper_zero_hg_zero_post(df_oper: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df_oper.columns:
        return out
    key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado"] if c in df_oper.columns]
    if len(key) < 2:
        return out

    hg = df_oper.loc[df_oper["stage"].eq("HARVEST_GRADE"), key + ["tallos_grado_dia"] + [c for c in ["ml2_anchor_real"] if c in df_oper.columns]].copy()
    if "ml2_anchor_real" in hg.columns:
        hg = hg.loc[pd.to_numeric(hg["ml2_anchor_real"], errors="coerce").fillna(0.0).gt(0.0)].copy()
    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if hg.empty or post.empty:
        return out

    post_agg = (
        post.groupby(key, dropna=False, as_index=False)
        .agg(
            n_post=("tallos_post", "size"),
            tallos_post_real_sum=("tallos_post_real", "sum") if "tallos_post_real" in post.columns else ("tallos_post", "sum"),
            tallos_post_sum=("tallos_post", "sum"),
            kg_verde_sum=("kg_verde", "sum"),
            kg_post_sum=("kg_post", "sum"),
        )
    )
    z = hg.merge(post_agg, on=key, how="left")
    z = z.loc[_num(z.get("n_post"), default=0.0) > 0.0].copy()
    if z.empty:
        out.append(
            {
                "check_id": "oper_zero_hg_zero_post",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": "__ALL__",
                "status": "PASS",
                "value": 0.0,
                "expected": 0.0,
                "tolerance": 0.0,
                "details": "no anchored HG keys with POST rows",
            }
        )
        return out
    z["tallos_hg"] = _num(z["tallos_grado_dia"], default=0.0)
    z["tallos_post_real_sum"] = _num(z["tallos_post_real_sum"], default=0.0)
    z["tallos_post_sum"] = _num(z["tallos_post_sum"], default=0.0)
    z["kg_verde_sum"] = _num(z["kg_verde_sum"], default=0.0)
    z["kg_post_sum"] = _num(z["kg_post_sum"], default=0.0)

    tol_zero = 1e-6
    z0 = z.loc[z["tallos_hg"].abs() <= tol_zero].copy()
    if z0.empty:
        out.append(
            {
                "check_id": "oper_zero_hg_zero_post",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": "__ALL__",
                "status": "PASS",
                "value": 0.0,
                "expected": 0.0,
                "tolerance": 0.0,
                "details": "no hg=0 keys with post rows",
            }
        )
        return out

    z0["fail"] = (
        (z0["tallos_post_real_sum"].abs() > tol_zero)
        | (z0["tallos_post_sum"].abs() > tol_zero)
        | (z0["kg_verde_sum"].abs() > tol_zero)
        | (z0["kg_post_sum"].abs() > tol_zero)
    )
    z0["max_abs"] = z0[["tallos_post_real_sum", "tallos_post_sum", "kg_verde_sum", "kg_post_sum"]].abs().max(axis=1)
    cyc = (
        z0.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_keys=("fail", "size"),
            n_fail=("fail", "sum"),
            max_abs=("max_abs", "max"),
        )
    )
    for r in cyc.itertuples(index=False):
        ok = int(r.n_fail) == 0
        out.append(
            {
                "check_id": "oper_zero_hg_zero_post",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.max_abs),
                "expected": 0.0,
                "tolerance": float(tol_zero),
                "details": f"failed_keys={int(r.n_fail)}/{int(r.n_keys)}",
            }
        )
    return out


def _check_oper_post_factor_chain(df_oper: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df_oper.columns:
        return out
    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if post.empty:
        return out

    has_real_cols = {"factor_hidr_real", "factor_desp_real", "factor_ajuste_real"} <= set(post.columns)
    if not has_real_cols:
        return out

    # Validate factor chain only where real post mermas/ajuste is available.
    m_real = (
        pd.to_numeric(post.get("factor_desp_real"), errors="coerce").notna()
        | pd.to_numeric(post.get("factor_ajuste_real"), errors="coerce").notna()
    )
    p = post.loc[m_real].copy()
    if p.empty:
        return out

    kg_verde = _num(p.get("kg_verde"), default=np.nan)
    kg_post = _num(p.get("kg_post"), default=np.nan)
    f_h = _num(p.get("factor_hidr"), default=np.nan)
    f_d = _num(p.get("factor_desp"), default=np.nan)
    f_a = _num(p.get("factor_ajuste"), default=np.nan)

    m_ok_base = kg_verde.notna() & kg_post.notna() & f_h.notna() & f_d.notna() & f_a.notna()
    if not bool(m_ok_base.any()):
        return out

    p = p.loc[m_ok_base].copy()
    calc = (
        _num(p.get("kg_verde"), default=np.nan)
        * _num(p.get("factor_hidr"), default=np.nan)
        * _num(p.get("factor_desp"), default=np.nan)
        * _num(p.get("factor_ajuste"), default=np.nan)
    )
    actual = _num(p.get("kg_post"), default=np.nan)
    diff = (actual - calc).abs()
    tol = np.maximum(1e-4, actual.abs() * 1e-4)
    p["__fail"] = diff > tol
    p["__diff"] = diff

    grp = (
        p.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_rows=("__fail", "size"),
            n_fail=("__fail", "sum"),
            max_diff=("__diff", "max"),
        )
    )
    for r in grp.itertuples(index=False):
        ok = int(r.n_fail) == 0
        out.append(
            {
                "check_id": "oper_post_factor_chain",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.max_diff),
                "expected": 0.0,
                "tolerance": 1e-4,
                "details": f"failed_rows={int(r.n_fail)}/{int(r.n_rows)}",
            }
        )
    return out


def _check_oper_post_boxes_from_kg(df_oper: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df_oper.columns:
        return out
    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if post.empty:
        return out

    kg_verde = _num(post.get("kg_verde"), default=np.nan)
    kg_post = _num(post.get("kg_post"), default=np.nan)
    cajas_verde = _num(post.get("cajas_verde"), default=np.nan)
    cajas_post = _num(post.get("cajas_post"), default=np.nan)

    # Validate only rows where masses and boxes are present.
    m_cv = kg_verde.notna() & cajas_verde.notna()
    m_cp = kg_post.notna() & cajas_post.notna()
    if not bool((m_cv | m_cp).any()):
        return out

    chk = post.loc[m_cv | m_cp, ["ciclo_id"]].copy()
    err_cv = (cajas_verde - (kg_verde / 10.0)).abs().where(m_cv, 0.0)
    err_cp = (cajas_post - (kg_post / 10.0)).abs().where(m_cp, 0.0)
    chk["err_cv"] = _num(err_cv.loc[chk.index], default=0.0)
    chk["err_cp"] = _num(err_cp.loc[chk.index], default=0.0)
    chk["err_max"] = chk[["err_cv", "err_cp"]].max(axis=1)
    tol = 1e-6
    chk["fail"] = chk["err_max"] > tol

    grp = (
        chk.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_rows=("fail", "size"),
            n_fail=("fail", "sum"),
            max_err=("err_max", "max"),
        )
    )
    for r in grp.itertuples(index=False):
        ok = int(r.n_fail) == 0
        out.append(
            {
                "check_id": "oper_post_boxes_from_kg",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.max_err),
                "expected": 0.0,
                "tolerance": float(tol),
                "details": f"failed_rows={int(r.n_fail)}/{int(r.n_rows)}",
            }
        )
    return out


def _check_oper_post_destino_share_b2(df_oper: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "stage" not in df_oper.columns:
        return out
    need = {"destino", "tallos_post", "share_dest_real_b2"}
    if not need.issubset(set(df_oper.columns)):
        return out

    post = df_oper.loc[df_oper["stage"].eq("POST")].copy()
    if post.empty:
        return out

    post["destino"] = post["destino"].astype("string").str.upper().str.strip()
    m_bt = post["destino"].isin(["BLANCO", "TINTURADO", "ARCOIRIS"])
    p = post.loc[m_bt].copy()
    if p.empty:
        return out

    key = [c for c in ["ciclo_id", "fecha_evento", "bloque_base", "variedad_canon", "grado"] if c in p.columns]
    if len(key) < 2:
        return out

    tallos = _num(p.get("tallos_post"), default=np.nan)
    w = _num(p.get("share_dest_real_b2"), default=np.nan)
    has_w = w.notna()
    if not bool(has_w.any()):
        return out

    den_w = w.where(has_w, 0.0).groupby([p[c] for c in key], dropna=False).transform("sum")
    den_t = tallos.where(has_w, np.nan).groupby([p[c] for c in key], dropna=False).transform("sum")
    m_valid = has_w & (den_w > 0.0) & den_t.notna() & (den_t > 0.0)
    if not bool(m_valid.any()):
        return out

    p = p.loc[m_valid].copy()
    w = _num(p.get("share_dest_real_b2"), default=np.nan)
    tallos = _num(p.get("tallos_post"), default=np.nan)
    den_w = w.groupby([p[c] for c in key], dropna=False).transform("sum")
    den_t = tallos.groupby([p[c] for c in key], dropna=False).transform("sum")
    p["share_w"] = np.where(den_w > 0.0, w / den_w, np.nan)
    p["share_t"] = np.where(den_t > 0.0, tallos / den_t, np.nan)
    p["diff"] = (p["share_t"] - p["share_w"]).abs()

    tol = 5e-3
    p["fail"] = p["diff"] > tol
    grp = (
        p.groupby("ciclo_id", dropna=False, as_index=False)
        .agg(
            n_rows=("fail", "size"),
            n_fail=("fail", "sum"),
            max_diff=("diff", "max"),
        )
    )
    for r in grp.itertuples(index=False):
        ok = int(r.n_fail) == 0
        out.append(
            {
                "check_id": "oper_post_destino_share_b2",
                "source_model": "ML2_OPERATIVO",
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.max_diff),
                "expected": 0.0,
                "tolerance": float(tol),
                "details": f"failed_rows={int(r.n_fail)}/{int(r.n_rows)}",
            }
        )
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_ml1 = _load(Path(args.ml1))
    df_g = _load(Path(args.ml2_global))
    df_p = _load(Path(args.ml2_puro))
    df_o = _load(Path(args.ml2_operativo))
    df_r = _load(Path(args.real))

    real_hg = df_r.loc[df_r["stage"].eq("HARVEST_GRADE"), ["ciclo_id", "fecha_evento", "tallos_grado_dia"]].copy()
    real_hg = real_hg.loc[_num(real_hg["tallos_grado_dia"], default=0.0) > 0.0]
    real_start = real_hg.groupby("ciclo_id", dropna=False, as_index=False).agg(start_real=("fecha_evento", "min"))
    real_end = real_hg.groupby("ciclo_id", dropna=False, as_index=False).agg(last_real=("fecha_evento", "max"))

    rows: list[dict] = []
    rows += _check_hg_day_consistency(df_ml1, "ML1")
    rows += _check_hg_day_consistency(df_g, "ML2_GLOBAL")
    rows += _check_hg_day_consistency(df_p, "ML2_PURO")
    rows += _check_hg_day_consistency(df_o, "ML2_OPERATIVO")
    rows += _check_hg_day_consistency(df_r, "REAL")

    rows += _check_total_vs_tallos_proy(df_ml1, "ML1")
    rows += _check_total_vs_tallos_proy(df_g, "ML2_GLOBAL")
    rows += _check_total_vs_tallos_proy(df_p, "ML2_PURO")

    rows += _check_post_duplicates(df_ml1, "ML1")
    rows += _check_post_duplicates(df_g, "ML2_GLOBAL")
    rows += _check_post_duplicates(df_p, "ML2_PURO")
    rows += _check_post_duplicates(df_o, "ML2_OPERATIVO")
    rows += _check_post_duplicates(df_r, "REAL")

    # Start-vs-real applies only to operational layer.
    rows += _check_ml2_start_vs_real(df_o, "ML2_OPERATIVO", real_start)

    rows += _check_oper_forward_projection(df_o, real_start, real_end)
    rows += _check_oper_forward_post_non_null(df_o, real_end)
    rows += _check_oper_zero_hg_zero_post(df_o)
    rows += _check_oper_post_factor_chain(df_o)
    rows += _check_oper_post_boxes_from_kg(df_o)
    rows += _check_oper_post_destino_share_b2(df_o)

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("Audit produced no rows.")

    audit["ok"] = audit["status"].eq("PASS")
    summary = (
        audit.groupby(["check_id", "source_model", "status"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    totals = (
        audit.groupby(["check_id", "source_model"], dropna=False, as_index=False)
        .agg(total=("ok", "size"), n_pass=("ok", "sum"))
    )
    totals["n_fail"] = totals["total"] - totals["n_pass"]

    p_audit = out_dir / f"ml_pipeline_audit_{args.tag}_{stamp}.parquet"
    p_audit_latest = out_dir / f"ml_pipeline_audit_{args.tag}.parquet"
    p_sum = out_dir / f"ml_pipeline_audit_summary_{args.tag}_{stamp}.parquet"
    p_sum_latest = out_dir / f"ml_pipeline_audit_summary_{args.tag}.parquet"
    p_json = out_dir / f"ml_pipeline_audit_summary_{args.tag}_{stamp}.json"
    p_json_latest = out_dir / f"ml_pipeline_audit_summary_{args.tag}.json"

    write_parquet(audit, p_audit)
    write_parquet(audit, p_audit_latest)
    write_parquet(summary, p_sum)
    write_parquet(summary, p_sum_latest)

    payload = {
        "tag": args.tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "totals": totals.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "n_rows": int(len(audit)),
        "n_fail": int((~audit["ok"]).sum()),
    }
    p_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    p_json_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] audit rows={len(audit):,} fails={(~audit['ok']).sum():,}")
    print(f"[OK] {p_audit}")
    print(f"[OK] {p_sum}")
    print(f"[OK] {p_json}")


if __name__ == "__main__":
    main()
