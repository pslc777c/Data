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
SILVER = DATA / "silver"

IN_DIM_HIDR = SILVER / "dim_hidratacion_fecha_post_grado_destino.parquet"
OUT_DIR = SILVER / "balanzas"
OUT_B2_SHARE = OUT_DIR / "silver_b2_share_grado_dia_destino.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cols:
            return cols[k.lower()]
    raise KeyError(f"No encuentro columnas. Busqué: {candidates}. Disponibles: {list(df.columns)[:80]}")


def main() -> None:
    if not IN_DIM_HIDR.exists():
        raise FileNotFoundError(f"No existe input: {IN_DIM_HIDR}")

    df = read_parquet(IN_DIM_HIDR).copy()
    df.columns = [str(c).strip() for c in df.columns]

    fecha_col = _resolve_col(df, ["fecha_post", "FECHA_POST", "fecha", "Fecha"])
    destino_col = _resolve_col(df, ["destino", "DESTINO", "codigo_actividad", "CODIGO_ACTIVIDAD"])
    grado_col = _resolve_col(df, ["grado", "GRADO"])
    # peso para shares: tallos o peso_post o paso_post_g
    w_col = None
    for cands in [
        ["tallos", "TALLOS", "tallos_w"],
        ["paso_post_g", "PASO_POST_G", "peso_post_g", "PESO_POST_G"],
        ["peso", "PESO", "kg", "KG"],
    ]:
        try:
            w_col = _resolve_col(df, cands)
            break
        except Exception:
            pass
    if w_col is None:
        raise KeyError(
            "No pude inferir columna de peso para shares en dim_hidratacion_fecha_post_grado_destino. "
            "Esperaba algo como tallos / paso_post_g."
        )

    out = df[[fecha_col, destino_col, grado_col, w_col]].copy()
    out["fecha_post"] = _to_date(out[fecha_col])
    out["destino"] = _canon_str(out[destino_col])
    out["w"] = _safe_float(out[w_col]).fillna(0.0)

    # grado a num (si viene como "40", "45.0", o "40GR")
    g = out[grado_col]
    if pd.api.types.is_numeric_dtype(g):
        out["grado_num"] = _safe_float(g)
    else:
        gg = _canon_str(g).str.replace("GR", "", regex=False)
        out["grado_num"] = _safe_float(gg)

    out = out.loc[
        out["fecha_post"].notna()
        & out["destino"].notna()
        & np.isfinite(out["grado_num"])
        & (out["w"] > 0)
    ].copy()

    # agrega por (fecha_post, destino, grado_num)
    g1 = (
        out.groupby(["fecha_post", "destino", "grado_num"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "w_grado"})
    )
    gtot = (
        g1.groupby(["fecha_post", "destino"], as_index=False)["w_grado"]
        .sum()
        .rename(columns={"w_grado": "w_total"})
    )
    g1 = g1.merge(gtot, on=["fecha_post", "destino"], how="left")
    g1["share"] = np.where(g1["w_total"] > 0, g1["w_grado"] / g1["w_total"], np.nan)

    # wide: share_b2__{grado}
    def _colname(x: float) -> str:
        # normaliza 45.0 -> 45, 45.5 -> 45.5
        if np.isfinite(x) and float(x).is_integer():
            return f"share_b2__{int(x)}"
        return f"share_b2__{x:.1f}"

    g1["col"] = g1["grado_num"].map(_colname)
    wide = (
        g1.pivot_table(
            index=["fecha_post", "destino"],
            columns="col",
            values="share",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None

    wide = wide.merge(gtot, on=["fecha_post", "destino"], how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(wide, OUT_B2_SHARE)

    print(f"[OK] silver B2 share: {OUT_B2_SHARE}")
    print(f"     rows={len(wide):,}  days={wide['fecha_post'].nunique():,}  destinos={wide['destino'].nunique():,}")


if __name__ == "__main__":
    main()
