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
    for r in z.itertuples(index=False):
        ok = bool(r.delta_days >= 0)
        out.append(
            {
                "check_id": "ml2_start_vs_real",
                "source_model": source,
                "ciclo_id": str(r.ciclo_id),
                "status": _status(ok),
                "value": float(r.delta_days),
                "expected": 0.0,
                "tolerance": np.nan,
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

    rows += _check_ml2_start_vs_real(df_g, "ML2_GLOBAL", real_start)
    rows += _check_ml2_start_vs_real(df_p, "ML2_PURO", real_start)
    rows += _check_ml2_start_vs_real(df_o, "ML2_OPERATIVO", real_start)

    rows += _check_oper_forward_projection(df_o, real_start, real_end)
    rows += _check_oper_forward_post_non_null(df_o, real_end)

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

