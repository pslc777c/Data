from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.io import read_parquet, write_parquet


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"

IN_B2_RAW = BRONZE / "balanza_2_raw.parquet"
IN_DIM_VAR = SILVER / "dim_variedad_canon.parquet"
OUT_DIAG = SILVER / "dim_b2_share_real_fecha_post_variedad_grado_destino.parquet"
OUT_SUM = SILVER / "dim_b2_share_real_fecha_post_variedad_grado_summary.parquet"

DESTINOS = ["BLANCO", "TINTURADO", "ARCOIRIS"]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.strip().fillna("UNKNOWN")


def _to_num(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _parse_int_from_mixed(s: pd.Series) -> pd.Series:
    x = _canon_str(s)
    n = x.str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(n, errors="coerce").round().astype("Int64")


def _build_var_map() -> dict[str, str]:
    d: dict[str, str] = {
        "GYPXLE": "XL",
        "XLENCE": "XL",
        "XL": "XL",
        "GYPCLO": "CLO",
        "CLOUD": "CLO",
        "CLO": "CLO",
    }
    if IN_DIM_VAR.exists():
        v = read_parquet(IN_DIM_VAR).copy()
        v.columns = [str(c).strip() for c in v.columns]
        need = {"variedad_raw", "variedad_canon"}
        if need.issubset(set(v.columns)):
            vr = _canon_str(v["variedad_raw"])
            vc = _canon_str(v["variedad_canon"])
            for rr, cc in zip(vr, vc):
                if rr and cc and rr != "UNKNOWN" and cc != "UNKNOWN":
                    d[str(rr)] = str(cc)
    return d


def _complete_dest_grid(keys: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    k = keys[key_cols].drop_duplicates().copy()
    k["_k"] = 1
    d = pd.DataFrame({"destino": DESTINOS})
    d["_k"] = 1
    out = k.merge(d, on="_k", how="inner").drop(columns=["_k"])
    return out


def main() -> None:
    if not IN_B2_RAW.exists():
        raise FileNotFoundError(f"Missing input: {IN_B2_RAW}")

    b2 = read_parquet(IN_B2_RAW).copy()
    b2.columns = [str(c).strip() for c in b2.columns]
    need = {"fecha_entrega", "Grado", "Destino", "peso_neto", "variedad"}
    miss = need - set(b2.columns)
    if miss:
        raise ValueError(f"balanza_2_raw missing columns: {sorted(miss)}")

    var_map = _build_var_map()

    b2["fecha_post"] = _to_date(b2["fecha_entrega"])
    b2["grado_int"] = _parse_int_from_mixed(b2["Grado"])
    b2["destino"] = _canon_str(b2["Destino"]).replace({"CLASIFICACION": "BLANCO", "GUIRNALDA": "BLANCO"})
    b2["variedad_raw"] = _canon_str(b2["variedad"])
    b2["variedad_canon"] = b2["variedad_raw"].map(var_map).fillna(b2["variedad_raw"])
    b2["peso_neto_kg"] = _to_num(b2["peso_neto"], default=np.nan)
    b2["tallos"] = _to_num(b2.get("Tallos"), default=np.nan)

    b2 = b2.loc[
        b2["fecha_post"].notna()
        & b2["grado_int"].notna()
        & b2["destino"].isin(DESTINOS)
        & b2["variedad_canon"].notna()
        & (b2["peso_neto_kg"] > 0.0)
    ].copy()

    key_gv = ["fecha_post", "variedad_canon", "grado_int", "destino"]
    g_gv_raw = (
        b2.groupby(key_gv, dropna=False, as_index=False)
        .agg(
            b2_peso_dest_grado_var_kg=("peso_neto_kg", "sum"),
            b2_tallos_dest_grado_var=("tallos", "sum"),
            b2_rows_dest_grado_var=("peso_neto_kg", "size"),
        )
    )
    gv_keys = g_gv_raw[["fecha_post", "variedad_canon", "grado_int"]].drop_duplicates()
    gv_grid = _complete_dest_grid(gv_keys, ["fecha_post", "variedad_canon", "grado_int"])
    g_gv = gv_grid.merge(g_gv_raw, on=key_gv, how="left")
    g_gv["b2_peso_dest_grado_var_kg"] = _to_num(g_gv["b2_peso_dest_grado_var_kg"], default=0.0)
    g_gv["b2_tallos_dest_grado_var"] = _to_num(g_gv["b2_tallos_dest_grado_var"], default=0.0)
    g_gv["b2_rows_dest_grado_var"] = _to_num(g_gv["b2_rows_dest_grado_var"], default=0.0)

    den_gv = g_gv.groupby(["fecha_post", "variedad_canon", "grado_int"], dropna=False)["b2_peso_dest_grado_var_kg"].transform("sum")
    g_gv["b2_peso_total_grado_var_kg"] = den_gv
    g_gv["share_dest_real_b2_gv"] = np.where(den_gv > 0.0, g_gv["b2_peso_dest_grado_var_kg"] / den_gv, np.nan)

    key_v = ["fecha_post", "variedad_canon", "destino"]
    g_v_raw = (
        b2.groupby(key_v, dropna=False, as_index=False)
        .agg(b2_peso_dest_var_kg=("peso_neto_kg", "sum"))
    )
    v_keys = g_v_raw[["fecha_post", "variedad_canon"]].drop_duplicates()
    v_grid = _complete_dest_grid(v_keys, ["fecha_post", "variedad_canon"])
    g_v = v_grid.merge(g_v_raw, on=key_v, how="left")
    g_v["b2_peso_dest_var_kg"] = _to_num(g_v["b2_peso_dest_var_kg"], default=0.0)
    den_v = g_v.groupby(["fecha_post", "variedad_canon"], dropna=False)["b2_peso_dest_var_kg"].transform("sum")
    g_v["b2_peso_total_var_kg"] = den_v
    g_v["share_dest_real_b2_v"] = np.where(den_v > 0.0, g_v["b2_peso_dest_var_kg"] / den_v, np.nan)

    key_d = ["fecha_post", "destino"]
    g_d_raw = (
        b2.groupby(key_d, dropna=False, as_index=False)
        .agg(b2_peso_dest_d_kg=("peso_neto_kg", "sum"))
    )
    d_keys = g_d_raw[["fecha_post"]].drop_duplicates()
    d_grid = _complete_dest_grid(d_keys, ["fecha_post"])
    g_d = d_grid.merge(g_d_raw, on=key_d, how="left")
    g_d["b2_peso_dest_d_kg"] = _to_num(g_d["b2_peso_dest_d_kg"], default=0.0)
    den_d = g_d.groupby(["fecha_post"], dropna=False)["b2_peso_dest_d_kg"].transform("sum")
    g_d["b2_peso_total_d_kg"] = den_d
    g_d["share_dest_real_b2_d"] = np.where(den_d > 0.0, g_d["b2_peso_dest_d_kg"] / den_d, np.nan)

    out = g_gv.merge(
        g_v[["fecha_post", "variedad_canon", "destino", "b2_peso_dest_var_kg", "b2_peso_total_var_kg", "share_dest_real_b2_v"]],
        on=["fecha_post", "variedad_canon", "destino"],
        how="left",
    ).merge(
        g_d[["fecha_post", "destino", "b2_peso_dest_d_kg", "b2_peso_total_d_kg", "share_dest_real_b2_d"]],
        on=["fecha_post", "destino"],
        how="left",
    )

    out["share_dest_real_b2"] = _to_num(out["share_dest_real_b2_gv"], default=np.nan)
    out["share_dest_real_b2"] = out["share_dest_real_b2"].where(out["share_dest_real_b2"].notna(), _to_num(out["share_dest_real_b2_v"], default=np.nan))
    out["share_dest_real_b2"] = out["share_dest_real_b2"].where(out["share_dest_real_b2"].notna(), _to_num(out["share_dest_real_b2_d"], default=np.nan))

    out["fallback_level_b2"] = np.select(
        [
            _to_num(out["share_dest_real_b2_gv"], default=np.nan).notna(),
            _to_num(out["share_dest_real_b2_v"], default=np.nan).notna(),
            _to_num(out["share_dest_real_b2_d"], default=np.nan).notna(),
        ],
        ["variedad_grado", "variedad", "fecha"],
        default="none",
    )
    out["b2_peso_dest_real_kg"] = out["b2_peso_dest_grado_var_kg"]
    out["created_at"] = pd.Timestamp.now("UTC")

    out = out.sort_values(["fecha_post", "variedad_canon", "grado_int", "destino"], kind="mergesort").reset_index(drop=True)
    write_parquet(out, OUT_DIAG)

    s = (
        out.groupby(["fecha_post", "variedad_canon", "grado_int"], dropna=False, as_index=False)
        .agg(
            b2_peso_total_grado_var_kg=("b2_peso_total_grado_var_kg", "max"),
            share_sum=("share_dest_real_b2", "sum"),
            n_dest_weight_gt0=("b2_peso_dest_grado_var_kg", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            n_dest=("destino", "size"),
        )
    )
    s["share_gap_abs"] = (s["share_sum"] - 1.0).abs()
    s["created_at"] = pd.Timestamp.now("UTC")
    write_parquet(s, OUT_SUM)

    print(f"[OK] {OUT_DIAG}")
    print(f"     rows={len(out):,} days={out['fecha_post'].nunique():,} variedades={out['variedad_canon'].nunique():,}")
    print(f"[OK] {OUT_SUM}")
    print(f"     rows={len(s):,} max_share_gap={float(pd.to_numeric(s['share_gap_abs'], errors='coerce').max()):.6f}")


if __name__ == "__main__":
    main()
