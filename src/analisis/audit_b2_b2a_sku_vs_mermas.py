from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

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
SILVER_BAL = SILVER / "balanzas"
EVAL = DATA / "eval" / "ml2"

IN_B2A = SILVER_BAL / "silver_b2a_sku_gradoideal_dia_destino.parquet"
IN_DIM = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"

OUT_DIR = EVAL / "audit_b2a_sku_vs_mermas"
OUT_PDF = OUT_DIR / "audit_b2a_sku_vs_mermas.pdf"

# outputs silver (shares)
OUT_SHARE_SKU = SILVER_BAL / "silver_b2a_share_sku_dia_destino.parquet"
OUT_SHARE_GI = SILVER_BAL / "silver_b2a_share_gradoideal_bins_dia_destino.parquet"

# (debug) long bases to audit
OUT_LONG_SKU = SILVER_BAL / "silver_b2a_share_sku_long_dia_destino.parquet"
OUT_LONG_GI = SILVER_BAL / "silver_b2a_share_gi_long_dia_destino.parquet"


# -------------------------
# Helpers
# -------------------------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _col_map(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().upper(): str(c).strip() for c in df.columns}


def _pick_first(df: pd.DataFrame, candidates_upper: list[str]) -> str:
    m = _col_map(df)
    for c in candidates_upper:
        if c in m:
            return m[c]
    raise KeyError(f"No encuentro ninguna de columnas: {candidates_upper}. Tengo: {list(df.columns)[:120]}")


def _pick_optional(df: pd.DataFrame, candidates_upper: list[str]) -> str | None:
    m = _col_map(df)
    for c in candidates_upper:
        if c in m:
            return m[c]
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _map_destino(raw: pd.Series) -> pd.Series:
    """
    Normaliza codigo_actividad / destino a:
    BLANCO, TINTURADO, ARCOIRIS, GUIRNALDA, (fallback) mismo codigo en upper.
    IMPORTANTE: también deja pasar si ya viene BLANCO/TINTURADO/...
    """
    s = _canon_str(raw)

    # ya normalizado
    already = s.isin(["BLANCO", "TINTURADO", "ARCOIRIS", "GUIRNALDA"])
    out = s.copy()

    # reglas (tus reglas)
    out = np.select(
        [
            already,
            s.eq("05CBMB") | s.eq("CBMC") | s.eq("CBM") | s.eq("CBX"),
            s.eq("CXLTA1") | s.eq("05CTS"),
            s.eq("0504GUFDGU"),
            s.eq("CXLTARH"),
        ],
        [
            s,             # ya OK
            "BLANCO",
            "TINTURADO",
            "GUIRNALDA",
            "ARCOIRIS",
        ],
        default=s,
    ).astype(str)

    return pd.Series(out, index=raw.index, dtype="string")


def _deciles(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    d = df[[xcol, ycol]].copy()
    d = d.loc[np.isfinite(d[xcol]) & np.isfinite(d[ycol])].copy()
    if len(d) < 30:
        return pd.DataFrame(columns=["decile", "n", "x_mean", "y_mean"])
    d["decile"] = pd.qcut(d[xcol], 10, duplicates="drop")
    out = (
        d.groupby("decile", as_index=False)
        .agg(n=(ycol, "size"), x_mean=(xcol, "mean"), y_mean=(ycol, "mean"))
    )
    return out


def _plot_deciles(dec: pd.DataFrame, title: str, ylab: str) -> bytes:
    buf = io.BytesIO()
    plt.figure()
    if len(dec) > 0:
        plt.plot(range(len(dec)), dec["y_mean"].to_numpy(), marker="o")
        plt.xticks(range(len(dec)), [str(x) for x in dec["decile"]], rotation=45, ha="right")
        plt.ylabel(ylab)
        plt.title(title)
    else:
        plt.title(f"{title} (sin datos suficientes)")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=220)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def _plot_scatter(df: pd.DataFrame, x: str, y: str, title: str) -> bytes:
    buf = io.BytesIO()
    xx = _safe_float(df[x])
    yy = _safe_float(df[y])
    m = np.isfinite(xx) & np.isfinite(yy)
    plt.figure()
    if m.sum() > 0:
        plt.scatter(xx[m], yy[m], s=10, alpha=0.6)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(title)
    else:
        plt.title(f"{title} (sin datos)")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=220)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def _df_to_table_data(df: pd.DataFrame, max_rows: int = 15) -> list[list[str]]:
    df2 = df.head(max_rows).copy()
    cols = list(df2.columns)
    data = [cols]
    for _, r in df2.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.4f}")
            else:
                row.append("" if pd.isna(v) else str(v))
        data.append(row)
    return data


def _ols_robust(df: pd.DataFrame, ycol: str, xcols: list[str]) -> pd.DataFrame:
    d = df.copy()
    y = _safe_float(d[ycol])

    X = pd.DataFrame({"const": np.ones(len(d), dtype=float)}, index=d.index)
    for c in xcols:
        X[c] = _safe_float(d[c]).fillna(0.0)

    m = np.isfinite(y) & np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
    if m.sum() < 80:
        return pd.DataFrame([{"error": "Insuficientes filas para OLS", "n": int(m.sum())}])

    res = sm.OLS(y[m].to_numpy(dtype=float), X.loc[m].to_numpy(dtype=float)).fit(cov_type="HC3")

    rows = []
    for name, coef, se, t, p in zip(X.columns, res.params, res.bse, res.tvalues, res.pvalues):
        rows.append({
            "term": str(name),
            "coef": float(coef),
            "se_hc3": float(se),
            "t": float(t),
            "p": float(p),
            "n": int(m.sum()),
            "r2": float(res.rsquared),
            "r2_adj": float(res.rsquared_adj),
        })
    return pd.DataFrame(rows).sort_values("p")


def _make_gradoideal_bins(b2a: pd.DataFrame, step: float = 1.0) -> np.ndarray:
    x = b2a["grado_ideal"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.arange(0, 101, step)
    lo = float(np.nanquantile(x, 0.01))
    hi = float(np.nanquantile(x, 0.99))
    lo2 = np.floor(lo / step) * step
    hi2 = np.ceil(hi / step) * step
    if hi2 <= lo2:
        hi2 = lo2 + step
    return np.arange(lo2, hi2 + step, step)


def main() -> None:
    _ensure_dir(OUT_DIR)
    _ensure_dir(SILVER_BAL)

    for p in [IN_B2A, IN_DIM]:
        if not p.exists():
            raise FileNotFoundError(f"No existe input: {p}")

    # -------------------------
    # 1) Load B2A
    # -------------------------
    b2a = read_parquet(IN_B2A).copy()
    b2a.columns = [str(c).strip() for c in b2a.columns]

    c_fecha = _pick_first(b2a, ["FECHA", "FECHA_POST"])
    c_dest = _pick_first(b2a, ["DESTINO", "CODIGO_ACTIVIDAD"])
    c_sku = _pick_first(b2a, ["SKU"])
    c_gi = _pick_first(b2a, ["GRADO_IDEAL"])
    c_w = _pick_first(b2a, ["TALLOS_TOTALES", "TALLOS", "W"])
    c_var = _pick_optional(b2a, ["VARIEDAD"])

    b2a["fecha_post"] = _to_date(b2a[c_fecha])
    b2a["destino"] = _map_destino(b2a[c_dest])
    b2a["sku"] = _safe_float(b2a[c_sku])
    b2a["grado_ideal"] = _safe_float(b2a[c_gi])
    b2a["w"] = _safe_float(b2a[c_w]).fillna(0.0)
    if c_var:
        b2a["variedad"] = _canon_str(b2a[c_var])
    else:
        b2a["variedad"] = "ALL"

    # filtro mínimo
    b2a = b2a.loc[
        b2a["fecha_post"].notna()
        & b2a["destino"].notna()
        & np.isfinite(b2a["sku"])
        & np.isfinite(b2a["grado_ideal"])
        & (b2a["w"] > 0)
    ].copy()

    # diagnostico destinos
    print("[DBG] destinos B2A (mapeados):", sorted(b2a["destino"].dropna().unique().tolist())[:50])


    # -------------------------
    # 2) Shares SKU (wide)
    # -------------------------
    gday = b2a.groupby(["fecha_post", "destino"], as_index=False)["w"].sum().rename(columns={"w": "w_day"})
    b2a = b2a.merge(gday, on=["fecha_post", "destino"], how="left")
    b2a["share_sku"] = np.where(b2a["w_day"] > 0, b2a["w"] / b2a["w_day"], np.nan)

    # SKU como entero (evita 625.0 vs 625)
    b2a["sku_int"] = pd.to_numeric(b2a["sku"], errors="coerce").round(0).astype("Int64")
    b2a = b2a.loc[b2a["sku_int"].notna()].copy()

    wide_sku = (
        b2a.pivot_table(
            index=["fecha_post", "destino"],
            columns="sku_int",
            values="share_sku",
            aggfunc="sum",
            fill_value=0.0,      # ausencias -> 0 (no NaN)
        )
        .reset_index()
    )
    wide_sku.columns.name = None

    # Renombra columnas SKU a texto estable: sku__625, sku__250, ...
    rename_map = {}
    for c in wide_sku.columns:
        if c in ("fecha_post", "destino"):
            continue
        try:
            rename_map[c] = f"sku__{int(c)}"
        except Exception:
            pass
    wide_sku = wide_sku.rename(columns=rename_map)

    # Seguridad: NaN residuales -> 0
    sku_cols = [c for c in wide_sku.columns if str(c).startswith("sku__")]
    if sku_cols:
        wide_sku[sku_cols] = wide_sku[sku_cols].fillna(0.0)

    write_parquet(wide_sku, OUT_SHARE_SKU)


    # -------------------------
    # 3) Shares por bins de grado_ideal (long + wide)
    # -------------------------
    bins = _make_gradoideal_bins(b2a, step=1.0)
    mids = (bins[:-1] + bins[1:]) / 2.0

    gi = b2a["grado_ideal"].to_numpy(dtype=float)
    bin_idx = np.digitize(gi, bins) - 1
    ok = (bin_idx >= 0) & (bin_idx < len(mids)) & np.isfinite(b2a["w"].to_numpy(dtype=float))

    tmp = b2a.loc[ok, ["fecha_post", "destino", "w", "w_day"]].copy()
    tmp["bin"] = bin_idx[ok].astype(int)

    gb = tmp.groupby(["fecha_post", "destino", "bin"], as_index=False)["w"].sum()
    gb = gb.merge(tmp.groupby(["fecha_post", "destino"], as_index=False)["w_day"].first(), on=["fecha_post", "destino"], how="left")
    gb["share"] = np.where(gb["w_day"] > 0, gb["w"] / gb["w_day"], np.nan)
    gb["col"] = gb["bin"].map(lambda i: f"share_gi__{mids[i]:.1f}")
    write_parquet(gb, OUT_LONG_GI)

    wide_gi = gb.pivot_table(index=["fecha_post", "destino"], columns="col", values="share", aggfunc="first").reset_index()
    wide_gi.columns.name = None
    write_parquet(wide_gi, OUT_SHARE_GI)

    # GI features
    gi_cols = [c for c in wide_gi.columns if c.startswith("share_gi__")]
    if gi_cols:
        mat = np.nan_to_num(wide_gi[gi_cols].to_numpy(dtype=float), nan=0.0)
        gi_mid_vals = np.array([float(c.split("__")[1]) for c in gi_cols], dtype=float)
        wide_gi["gi_avg"] = (mat * gi_mid_vals.reshape(1, -1)).sum(axis=1)

        def ent(row):
            r = row[row > 0]
            return float(-(r * np.log(r)).sum()) if r.size else float("nan")

        wide_gi["gi_entropy"] = np.array([ent(r) for r in mat], dtype=float)
    else:
        wide_gi["gi_avg"] = np.nan
        wide_gi["gi_entropy"] = np.nan

    # -------------------------
    # 4) Load DIM
    # -------------------------
    dim = read_parquet(IN_DIM).copy()
    dim.columns = [str(c).strip() for c in dim.columns]

    c_dim_fecha = _pick_first(dim, ["FECHA_POST", "FECHA", "FECHA_POST_PRED_USED"])
    c_dim_dest = _pick_first(dim, ["DESTINO", "CODIGO_ACTIVIDAD"])

    dim["fecha_post"] = _to_date(dim[c_dim_fecha])
    dim["destino"] = _map_destino(dim[c_dim_dest])

    # targets
    c_factor = _pick_first(dim, ["FACTOR_DESP", "FACTOR_DESP_REAL", "FACTOR_DESPERDICIO"])
    c_ajuste = _pick_optional(dim, ["AJUSTE", "AJUSTE_REAL", "FACTOR_AJUSTE_REAL", "AJUSTE_PESO"])

    dim["factor_desp"] = _safe_float(dim[c_factor])
    dim["ajuste"] = _safe_float(dim[c_ajuste]) if c_ajuste else np.nan

    print("[DBG] destinos DIM (mapeados):", sorted(dim["destino"].dropna().unique().tolist())[:50])

    # -------------------------
    # 5) Diagnóstico match keys (esto es lo que te estaba matando)
    # -------------------------
    keys_dim = dim[["fecha_post", "destino"]].dropna().drop_duplicates()
    keys_b2a = wide_sku[["fecha_post", "destino"]].dropna().drop_duplicates()
    keys_inner = keys_dim.merge(keys_b2a, on=["fecha_post", "destino"], how="inner")

    if len(keys_dim) > 0:
        match_rate = len(keys_inner) / len(keys_dim)
    else:
        match_rate = 0.0

    print(f"[DBG] keys_dim={len(keys_dim):,} keys_b2a={len(keys_b2a):,} keys_inner={len(keys_inner):,} match_rate={match_rate:.2%}")
    if match_rate < 0.50:
        print("[WARN] Match rate bajo. Esto implica que el merge dejará muchos NaN en shares (y parecerá 'todo 0'). Revisa mapeo destino/fechas.")

    # -------------------------
    # 6) Merge analítico
    # -------------------------
    df = dim.merge(wide_sku, on=["fecha_post", "destino"], how="left")
    df = df.merge(wide_gi[["fecha_post", "destino", "gi_avg", "gi_entropy"]], on=["fecha_post", "destino"], how="left")

    # calendario
    df["dow"] = df["fecha_post"].dt.dayofweek.astype("Int64")
    df["month"] = df["fecha_post"].dt.month.astype("Int64")
    df["weekofyear"] = df["fecha_post"].dt.isocalendar().week.astype("Int64")

    # share_625 (si existe como columna)
    col_625 = None
    if 625.0 in df.columns:
        col_625 = 625.0
    elif 625 in df.columns:
        col_625 = 625

    if col_625 is not None:
        df["share_625"] = _safe_float(df[col_625]).fillna(0.0)
    else:
        df["share_625"] = np.nan  # preferible a 0 para no engañarnos

    # filtro
    df = df.loc[df["fecha_post"].notna() & df["destino"].notna() & df["factor_desp"].notna()].copy()

    # diagnósticos shares
    share_cols = [c for c in df.columns if isinstance(c, (int, float))]  # columnas SKU numéricas
    if share_cols:
        nonzero = (df[share_cols].fillna(0.0).sum(axis=1) > 0).mean()
        print(f"[DBG] % filas con suma_shares_SKU>0: {nonzero:.2%}")
    else:
        print("[WARN] No hay columnas de SKU en el dataframe final (wide_sku quedó vacío o el merge no trajo columnas).")

    # -------------------------
    # 7) Deciles + OLS (demo con 625)
    # -------------------------
    # Nota: aquí solo queda como ejemplo, el análisis macro lo hacemos en el siguiente audit (vector completo).
    dec_desp = _deciles(df.dropna(subset=["share_625"]).copy(), "share_625", "factor_desp")
    dec_adj = _deciles(df.loc[df["ajuste"].notna() & df["share_625"].notna()].copy(), "share_625", "ajuste") if df["ajuste"].notna().any() else pd.DataFrame()

    xcols = ["share_625", "gi_avg", "gi_entropy", "dow", "month", "weekofyear"]
    ols_desp = _ols_robust(df.dropna(subset=["share_625"]).copy(), "factor_desp", xcols=xcols) if df["share_625"].notna().any() else pd.DataFrame([{"error": "No existe share_625 (columna SKU no presente)"}])
    ols_adj = _ols_robust(df.loc[df["ajuste"].notna() & df["share_625"].notna()].copy(), "ajuste", xcols=xcols) if df["ajuste"].notna().any() and df["share_625"].notna().any() else pd.DataFrame([{"error": "No hay ajuste o no existe share_625"}])

    # -------------------------
    # 8) PDF
    # -------------------------
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.grey)

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Audit B2A SKU vs mermas",
    )

    story = []
    story.append(Paragraph("Audit estadístico: efecto SKU macro (B2A) sobre mermas y ajuste", title))
    story.append(Paragraph(f"Inputs: {IN_B2A.name} + {IN_DIM.name}", small))
    story.append(Paragraph(f"Filas analíticas: {len(df):,} (fecha_post x destino)", small))
    story.append(Paragraph(f"Match rate (keys dim vs B2A): {match_rate:.2%}", small))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1) Efectos marginales (deciles) para share_625 (solo demo)", h2))
    if len(dec_desp) > 0:
        story.append(Image(io.BytesIO(_plot_deciles(dec_desp, "Deciles share_625 vs factor_desp", "factor_desp")), width=16*cm, height=7*cm))
    else:
        story.append(Paragraph("No hay datos suficientes para deciles de share_625 (o no existe la columna).", styles["BodyText"]))

    if len(dec_adj) > 0:
        story.append(Spacer(1, 6))
        story.append(Image(io.BytesIO(_plot_deciles(dec_adj, "Deciles share_625 vs ajuste", "ajuste")), width=16*cm, height=7*cm))

    story.append(PageBreak())
    story.append(Paragraph("2) OLS robusto (HC3) (solo demo con 625)", h2))

    story.append(Paragraph("2.1) Target: factor_desp", styles["Heading3"]))
    t1 = Table(_df_to_table_data(ols_desp, max_rows=18), hAlign="LEFT")
    t1.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.2),
    ]))
    story.append(t1)

    story.append(Spacer(1, 10))
    story.append(Paragraph("2.2) Target: ajuste", styles["Heading3"]))
    t2 = Table(_df_to_table_data(ols_adj, max_rows=18), hAlign="LEFT")
    t2.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.2),
    ]))
    story.append(t2)

    story.append(PageBreak())
    story.append(Paragraph("3) Scatter rápido (solo demo con 625)", h2))
    if df["share_625"].notna().any():
        story.append(Image(io.BytesIO(_plot_scatter(df.dropna(subset=["share_625"]), "share_625", "factor_desp", "share_625 vs factor_desp")), width=16*cm, height=7*cm))
        if df["ajuste"].notna().any():
            story.append(Spacer(1, 6))
            story.append(Image(io.BytesIO(_plot_scatter(df.loc[df["ajuste"].notna() & df["share_625"].notna()], "share_625", "ajuste", "share_625 vs ajuste")), width=16*cm, height=7*cm))
    else:
        story.append(Paragraph("No existe share_625 en el merge final (revisar share parquet).", styles["BodyText"]))

    doc.build(story)

    print(f"[OK] PDF: {OUT_PDF}")
    print(f"[OK] silver share sku (wide): {OUT_SHARE_SKU}")
    print(f"[OK] silver share gi bins (wide): {OUT_SHARE_GI}")
    print(f"[OK] silver share sku (long): {OUT_LONG_SKU}")
    print(f"[OK] silver share gi (long): {OUT_LONG_GI}")
    print(f"     rows={len(df):,}  days={df['fecha_post'].nunique():,}")


if __name__ == "__main__":
    main()
