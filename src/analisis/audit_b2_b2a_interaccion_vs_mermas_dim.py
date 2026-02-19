from __future__ import annotations

from pathlib import Path
import io
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

import statsmodels.api as sm

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
EVAL = DATA / "eval" / "ml2"

OUT_DIR = EVAL / "audit_b2_b2a_vs_mermas_dim"
OUT_PDF = OUT_DIR / "audit_b2_b2a_vs_mermas_dim_report.pdf"

# ===== INPUTS =====
IN_DIM = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"
IN_B2 = SILVER / "fact_hidratacion_real_post_grado_destino.parquet"
IN_B2A = SILVER / "silver_balanza2a_grado_ideal_dia_variedad_actividad.parquet"

# ===== OUTPUTS (NEW) =====
OUT_SILVER_DIR = SILVER / "balanzas"
OUT_B2_SHARE = OUT_SILVER_DIR / "silver_b2_share_bins_fecha_post_destino.parquet"
OUT_B2A_SHARE = OUT_SILVER_DIR / "silver_b2a_share_bins_fecha_post_destino.parquet"


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if p.size == 0:
        return float("nan")
    return float(-(p * np.log(p)).sum())


def _mae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(m):
        return float("nan")
    err = np.abs(y_true[m] - y_pred[m])
    if w is None:
        return float(np.mean(err))
    ww = np.asarray(w, dtype=float)[m]
    denom = float(np.sum(ww))
    if denom <= 0:
        return float(np.mean(err))
    return float(np.sum(err * ww) / denom)


def _make_bins(x_all: np.ndarray, step: float = 1.0) -> np.ndarray:
    x = x_all[np.isfinite(x_all)]
    if x.size == 0:
        return np.arange(0, 101, step)
    lo = float(np.nanquantile(x, 0.01))
    hi = float(np.nanquantile(x, 0.99))
    lo2 = np.floor(lo / step) * step
    hi2 = np.ceil(hi / step) * step
    if hi2 <= lo2:
        hi2 = lo2 + step
    return np.arange(lo2, hi2 + step, step)


def _hist_share_wide(df_long: pd.DataFrame, val_col: str, w_col: str, bins: np.ndarray, prefix: str) -> pd.DataFrame:
    """
    Crea shares por (fecha_post, destino) y por bin del valor val_col.
    Retorna wide con columnas: share_{prefix}__{mid}
    """
    v = df_long[val_col].to_numpy(dtype=float)
    w = df_long[w_col].to_numpy(dtype=float)
    fecha = df_long["fecha_post"].to_numpy()
    dest = df_long["destino"].to_numpy()

    bin_idx = np.digitize(v, bins) - 1
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0) & (bin_idx >= 0) & (bin_idx < len(bins) - 1)

    tmp = pd.DataFrame(
        {
            "fecha_post": pd.to_datetime(fecha[valid]),
            "destino": dest[valid],
            "bin": bin_idx[valid].astype(int),
            "w": w[valid].astype(float),
        }
    )

    g = tmp.groupby(["fecha_post", "destino", "bin"], as_index=False)["w"].sum()
    tot = g.groupby(["fecha_post", "destino"], as_index=False)["w"].sum().rename(columns={"w": "w_day"})
    g = g.merge(tot, on=["fecha_post", "destino"], how="left")
    g["share"] = np.where(g["w_day"] > 0, g["w"] / g["w_day"], np.nan)

    mids = (bins[:-1] + bins[1:]) / 2.0
    g["col"] = g["bin"].map(lambda i: f"share_{prefix}__{mids[i]:.1f}")

    wide = g.pivot_table(index=["fecha_post", "destino"], columns="col", values="share", aggfunc="first").reset_index()
    wide.columns.name = None

    # conservar totales para debug
    wide = wide.merge(tot, on=["fecha_post", "destino"], how="left")

    return wide


def _distribution_feats(wide: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in wide.columns if c.startswith(f"share_{prefix}__")]
    out = wide[["fecha_post", "destino"]].copy()
    if not cols:
        out[f"{prefix}_entropy"] = np.nan
        out[f"{prefix}_top1_share"] = np.nan
        return out

    mat = wide[cols].to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0)
    sort_sh = np.sort(mat, axis=1)[:, ::-1]
    out[f"{prefix}_top1_share"] = sort_sh[:, 0]
    out[f"{prefix}_entropy"] = np.array([_entropy(row) for row in mat], dtype=float)
    return out


def _distance_feats(wide_b2: pd.DataFrame, wide_b2a: pd.DataFrame) -> pd.DataFrame:
    df = wide_b2.merge(wide_b2a, on=["fecha_post", "destino"], how="inner", suffixes=("", "_b2a"))
    b2_cols = [c for c in df.columns if c.startswith("share_b2__")]
    out = df[["fecha_post", "destino"]].copy()
    if not b2_cols:
        out["dist_L1"] = np.nan
        out["dist_L2"] = np.nan
        out["cosine_sim"] = np.nan
        out["corr_share"] = np.nan
        return out

    b2a_cols = [c.replace("share_b2__", "share_b2a__") for c in b2_cols]
    for c in b2a_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = np.nan_to_num(df[b2_cols].to_numpy(dtype=float), nan=0.0)
    Y = np.nan_to_num(df[b2a_cols].to_numpy(dtype=float), nan=0.0)

    out["dist_L1"] = np.sum(np.abs(X - Y), axis=1)
    out["dist_L2"] = np.sqrt(np.sum((X - Y) ** 2, axis=1))

    nx = np.sqrt(np.sum(X * X, axis=1))
    ny = np.sqrt(np.sum(Y * Y, axis=1))
    denom = nx * ny
    out["cosine_sim"] = np.where(denom > 0, np.sum(X * Y, axis=1) / denom, np.nan)

    corr = []
    for i in range(X.shape[0]):
        x = X[i]
        y = Y[i]
        if np.allclose(x, 0) or np.allclose(y, 0):
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(x, y)[0, 1]))
    out["corr_share"] = corr
    return out


_mid_re = re.compile(r"__([0-9]+(?:\.[0-9]+)?)$")


def _parse_mids_from_cols(cols: list[str]) -> np.ndarray:
    mids = []
    for c in cols:
        m = _mid_re.search(c)
        if not m:
            mids.append(np.nan)
        else:
            mids.append(float(m.group(1)))
    mids = np.asarray(mids, dtype=float)
    return mids


def _directional_shift_feats(wide_b2: pd.DataFrame, wide_b2a: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas direccionales usando únicamente las columnas share existentes,
    alineando mids por nombre de columna (evita broadcast mismatch).
    """
    df = wide_b2.merge(wide_b2a, on=["fecha_post", "destino"], how="inner", suffixes=("", "_b2a"))
    b2_cols = [c for c in df.columns if c.startswith("share_b2__")]
    out = df[["fecha_post", "destino"]].copy()

    if not b2_cols:
        out["avg_b2"] = np.nan
        out["avg_b2a"] = np.nan
        out["delta_avg"] = np.nan
        out["emd"] = np.nan
        out["mass_up"] = np.nan
        out["mass_down"] = np.nan
        return out

    b2a_cols = [c.replace("share_b2__", "share_b2a__") for c in b2_cols]
    for c in b2a_cols:
        if c not in df.columns:
            df[c] = 0.0

    P = np.nan_to_num(df[b2_cols].to_numpy(float), nan=0.0)
    Q = np.nan_to_num(df[b2a_cols].to_numpy(float), nan=0.0)

    mids = _parse_mids_from_cols(b2_cols)
    if not np.isfinite(mids).any():
        # si no se puede parsear, no calculamos promedios/emd direccionales
        out["avg_b2"] = np.nan
        out["avg_b2a"] = np.nan
        out["delta_avg"] = np.nan
        out["emd"] = np.nan
        out["mass_up"] = np.nan
        out["mass_down"] = np.nan
        return out

    # reemplazo NaN por mediana para estabilidad (solo si hubo algún NaN aislado)
    mids = np.where(np.isfinite(mids), mids, np.nanmedian(mids))

    avg_p = (P * mids.reshape(1, -1)).sum(axis=1)
    avg_q = (Q * mids.reshape(1, -1)).sum(axis=1)

    out["avg_b2"] = avg_p
    out["avg_b2a"] = avg_q
    out["delta_avg"] = avg_q - avg_p

    out["mass_up"] = (np.maximum(Q - P, 0.0) * mids.reshape(1, -1)).sum(axis=1)
    out["mass_down"] = (np.maximum(P - Q, 0.0) * mids.reshape(1, -1)).sum(axis=1)

    # EMD aproximado por CDFs; el step se infiere de los mids observados
    mids_sorted = np.sort(np.unique(mids))
    step = float(np.nanmedian(np.diff(mids_sorted))) if mids_sorted.size > 1 else 1.0
    cdf_p = np.cumsum(P, axis=1)
    cdf_q = np.cumsum(Q, axis=1)
    out["emd"] = np.sum(np.abs(cdf_p - cdf_q), axis=1) * step

    return out


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


def _plot_hist(df: pd.DataFrame, col: str, title: str) -> bytes:
    buf = io.BytesIO()
    x = _safe_float(df[col])
    x = x[np.isfinite(x)]
    plt.figure()
    if x.size > 0:
        plt.hist(x, bins=30)
        plt.xlabel(col)
        plt.ylabel("frecuencia")
        plt.title(title)
    else:
        plt.title(f"{title} (sin datos)")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=220)
    plt.close()
    buf.seek(0)
    return buf.getvalue()


def _df_to_table_data(df: pd.DataFrame, max_rows: int = 20) -> list[list[str]]:
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


def _fit_models(df: pd.DataFrame, target: str, weight_col: str) -> pd.DataFrame:
    d = df.sort_values(["fecha_post", "destino"]).copy()
    max_date = d["fecha_post"].max()
    cut = max_date - pd.Timedelta(days=30)

    tr = d.loc[d["fecha_post"] < cut].copy()
    va = d.loc[d["fecha_post"] >= cut].copy()

    y_tr = _safe_float(tr[target]).to_numpy()
    y_va = _safe_float(va[target]).to_numpy()

    w_tr = np.clip(_safe_float(tr[weight_col]).to_numpy(), 0, None)
    w_va = np.clip(_safe_float(va[weight_col]).to_numpy(), 0, None)

    base_cols = [
        "dow", "month", "weekofyear",
        "w2_kg", "w2a_kg", "share_w2", "share_w2a", "ratio_out_in",
        "avg_b2", "avg_b2a", "delta_avg", "emd", "mass_up", "mass_down",
        "b2_entropy", "b2_top1_share", "b2a_entropy", "b2a_top1_share",
        "dist_L1", "dist_L2", "cosine_sim", "corr_share",
    ]

    cols = [c for c in base_cols if c in df.columns]

    def run(cols_use: list[str], tag: str) -> dict:
        Xtr = tr[cols_use].copy()
        Xva = va[cols_use].copy()
        for c in cols_use:
            Xtr[c] = _safe_float(Xtr[c]).fillna(0.0)
            Xva[c] = _safe_float(Xva[c]).fillna(0.0)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=42,
        )
        model.fit(Xtr.to_numpy(), y_tr, sample_weight=w_tr)
        p_tr = model.predict(Xtr.to_numpy())
        p_va = model.predict(Xva.to_numpy())

        return {
            "model": tag,
            "cut_date": str(cut.date()),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "mae_train": _mae(y_tr, p_tr, w_tr),
            "mae_val": _mae(y_va, p_va, w_va),
        }

    rows = [run(cols, "calendar+volumes+mix+dist")]

    out = pd.DataFrame(rows)
    out["delta_mae_val_vs_base"] = 0.0
    return out


def _ols_robust(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    d = df.copy()
    y = _safe_float(d[ycol])

    X = pd.DataFrame({
        "const": 1.0,
        "dow": _safe_float(d["dow"]).fillna(0),
        "month": _safe_float(d["month"]).fillna(0),
        "weekofyear": _safe_float(d["weekofyear"]).fillna(0),
        "w2_kg": _safe_float(d["w2_kg"]).fillna(0),
        "w2a_kg": _safe_float(d["w2a_kg"]).fillna(0),
        "ratio_out_in": _safe_float(d["ratio_out_in"]).fillna(0),

        # mismatch / interacción
        "avg_b2": _safe_float(d["avg_b2"]),
        "avg_b2a": _safe_float(d["avg_b2a"]),
        "delta_avg": _safe_float(d["delta_avg"]),
        "emd": _safe_float(d["emd"]),
        "mass_up": _safe_float(d["mass_up"]),
        "mass_down": _safe_float(d["mass_down"]),
        "dist_L1": _safe_float(d["dist_L1"]),
        "cosine_sim": _safe_float(d["cosine_sim"]),
        "corr_share": _safe_float(d["corr_share"]),
    })

    dest = _canon_str(d["destino"]).fillna("UNKNOWN")
    X = pd.concat([X, pd.get_dummies(dest, prefix="dest", drop_first=True)], axis=1)

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


def _build_pdf(targets: list[tuple[str, str]], df: pd.DataFrame, results: dict) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=12)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=12)
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8.5, leading=11, textColor=colors.grey)

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 1.5 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Audit B2 vs B2A vs dim mermas/ajustes",
        author="Data-LakeHouse",
    )

    story = []
    story.append(Paragraph("Audit estadístico: interacción B2 (mix físico) vs B2A (Grado_Ideal) sobre mermas y ajuste", title))
    story.append(Paragraph(f"Filas: {len(df):,} (fecha_post x destino)", small))
    story.append(Spacer(1, 8))

    # sección por target
    for tcol, tname in targets:
        story.append(Paragraph(f"Target: <b>{tcol}</b> — {tname}", h2))

        story.append(Image(io.BytesIO(_plot_hist(df, tcol, f"Distribución target: {tcol}")), width=16 * cm, height=6.5 * cm))
        story.append(Spacer(1, 6))

        # scatter con métricas que sí son interpretables
        for x in ["delta_avg", "emd", "mass_down", "dist_L1"]:
            if x in df.columns:
                story.append(Image(io.BytesIO(_plot_scatter(df, x, tcol, f"{x} vs {tcol}")), width=16 * cm, height=6.5 * cm))
                story.append(Spacer(1, 6))

        # OLS robusta
        ols = results.get(tcol, {}).get("ols")
        story.append(Paragraph("OLS robusta (HC3) con controles (calendario + volúmenes + destino)", body))
        if ols is None or ("term" not in ols.columns):
            story.append(Paragraph("OLS no disponible o insuficiente.", small))
        else:
            show = ols[["term", "coef", "se_hc3", "t", "p", "r2", "r2_adj"]].head(18)
            tbl = Table(_df_to_table_data(show, max_rows=18), hAlign="LEFT")
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(tbl)

        story.append(PageBreak())

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SILVER_DIR.mkdir(parents=True, exist_ok=True)

    for p in [IN_DIM, IN_B2, IN_B2A]:
        if not p.exists():
            raise FileNotFoundError(f"No existe input: {p}")

    # ===== DIM =====
    dim = read_parquet(IN_DIM).copy()
    dim.columns = [str(c).strip() for c in dim.columns]

    need = {"FECHA_POST", "DESTINO", "W2_KG", "W2A_KG"}
    cols = {c.upper(): c for c in dim.columns}
    miss = [k for k in need if k not in cols]
    if miss:
        raise ValueError(f"dim_mermas_ajuste... no tiene columnas: {miss}. Columnas: {list(dim.columns)[:80]}")

    fecha_post_col = cols["FECHA_POST"]
    destino_col = cols["DESTINO"]
    w2_col = cols["W2_KG"]
    w2a_col = cols["W2A_KG"]

    dim["fecha_post"] = _to_date(dim[fecha_post_col])
    dim["destino"] = _canon_str(dim[destino_col])

    dim["w2_kg"] = _safe_float(dim[w2_col]).fillna(0.0)
    dim["w2a_kg"] = _safe_float(dim[w2a_col]).fillna(0.0)

    dim["dow"] = dim["fecha_post"].dt.dayofweek.astype("Int64")
    dim["month"] = dim["fecha_post"].dt.month.astype("Int64")
    dim["weekofyear"] = dim["fecha_post"].dt.isocalendar().week.astype("Int64")

    gday = dim.groupby("fecha_post", as_index=False).agg(w2_day=("w2_kg", "sum"), w2a_day=("w2a_kg", "sum"))
    dim = dim.merge(gday, on="fecha_post", how="left")
    dim["share_w2"] = np.where(dim["w2_day"] > 0, dim["w2_kg"] / dim["w2_day"], np.nan)
    dim["share_w2a"] = np.where(dim["w2a_day"] > 0, dim["w2a_kg"] / dim["w2a_day"], np.nan)
    dim["ratio_out_in"] = np.where(dim["w2_kg"] > 0, dim["w2a_kg"] / dim["w2_kg"], np.nan)

    TARGETS = [
        ("factor_desp", "Factor desperdicio real (dim)"),
        ("ajuste", "Ajuste peso real (dim)"),
    ]
    for tcol, _ in TARGETS:
        if tcol not in dim.columns:
            raise ValueError(f"dim no tiene columna target requerida: {tcol}. cols={list(dim.columns)[:120]}")

    # ===== B2 =====
    b2 = read_parquet(IN_B2).copy()
    b2.columns = [str(c).strip() for c in b2.columns]

    b2_fecha = "Fecha" if "Fecha" in b2.columns else ("fecha_post" if "fecha_post" in b2.columns else "fecha")
    b2["fecha_post"] = _to_date(b2[b2_fecha])

    if "destino" in [c.lower() for c in b2.columns]:
        b2_dest = [c for c in b2.columns if c.lower() == "destino"][0]
        b2["destino"] = _canon_str(b2[b2_dest])
    else:
        raise ValueError("B2 debe tener 'destino' para cruzar con dim (BLANCO/ARCOIRIS/TINTURADO/GUIRNALDA).")

    grado_col = "Grado" if "Grado" in b2.columns else ("grado" if "grado" in b2.columns else None)
    if grado_col is None:
        raise ValueError("B2: no encuentro columna Grado/grado.")
    g = b2[grado_col]
    if pd.api.types.is_numeric_dtype(g):
        b2["grado_num"] = pd.to_numeric(g, errors="coerce").astype(float)
    else:
        gg = _canon_str(g).str.replace("GR", "", regex=False)
        b2["grado_num"] = pd.to_numeric(gg, errors="coerce").astype(float)

    tallos_col = None
    for c in ["tallos", "Tallos", "tallos_total", "TALLOS", "TALLOS_TOTALES", "tallos_w", "tallos_pred", "tallos_real"]:
        if c in b2.columns:
            tallos_col = c
            break
    if tallos_col is None:
        raise ValueError("B2: no encuentro columna de tallos/peso para ponderar.")
    b2["w"] = _safe_float(b2[tallos_col]).fillna(0.0)

    b2 = b2.loc[b2["fecha_post"].notna() & b2["destino"].notna() & np.isfinite(b2["grado_num"]) & (b2["w"] > 0)].copy()

    # ===== B2A =====
    b2a = read_parquet(IN_B2A).copy()
    b2a.columns = [str(c).strip() for c in b2a.columns]

    if "Fecha" not in b2a.columns:
        raise ValueError("B2A: no encuentro columna 'Fecha'.")
    if "codigo_actividad" not in b2a.columns:
        raise ValueError("B2A: no encuentro columna 'codigo_actividad'.")
    if "Grado_Ideal" not in b2a.columns:
        raise ValueError("B2A: no encuentro columna 'Grado_Ideal'.")
    if "Tallos_Totales" not in b2a.columns:
        raise ValueError("B2A: no encuentro columna 'Tallos_Totales'.")

    b2a["fecha_post"] = _to_date(b2a["Fecha"])
    b2a["destino"] = _canon_str(b2a["codigo_actividad"])
    b2a["grado_ideal"] = _safe_float(b2a["Grado_Ideal"])
    b2a["w"] = _safe_float(b2a["Tallos_Totales"]).fillna(0.0)

    b2a = b2a.loc[b2a["fecha_post"].notna() & b2a["destino"].notna() & np.isfinite(b2a["grado_ideal"]) & (b2a["w"] > 0)].copy()

    # ===== BINS + SHARES (WIDE) =====
    bins = _make_bins(np.concatenate([b2["grado_num"].to_numpy(), b2a["grado_ideal"].to_numpy()]), step=1.0)

    wide_b2 = _hist_share_wide(b2, "grado_num", "w", bins, prefix="b2")
    wide_b2a = _hist_share_wide(b2a, "grado_ideal", "w", bins, prefix="b2a")

    # --- Guardar 2 parquets SILVER solicitados ---
    # (esto es la base exacta de shares por bin que usa el audit)
    write_parquet(wide_b2, OUT_B2_SHARE)
    write_parquet(wide_b2a, OUT_B2A_SHARE)

    # ===== FEATURES =====
    feats_b2 = _distribution_feats(wide_b2, prefix="b2")
    feats_b2a = _distribution_feats(wide_b2a, prefix="b2a")
    dist = _distance_feats(wide_b2, wide_b2a)
    shift = _directional_shift_feats(wide_b2, wide_b2a)

    df = dim.merge(feats_b2, on=["fecha_post", "destino"], how="left")
    df = df.merge(feats_b2a, on=["fecha_post", "destino"], how="left")
    df = df.merge(dist, on=["fecha_post", "destino"], how="left")
    df = df.merge(shift, on=["fecha_post", "destino"], how="left")

    # filtro base mínimo (fecha/destino)
    df = df.loc[df["fecha_post"].notna() & df["destino"].notna()].copy()

    # ===== Resultados estadísticos por target =====
    results = {}
    for tcol, tname in TARGETS:
        df_t = df.loc[df[tcol].notna()].copy()
        results[tcol] = {
            "name": tname,
            "ols": _ols_robust(df_t, ycol=tcol),
        }

    # ===== (Opcional) comparar modelo solo para factor_desp como sanity =====
    # (no lo usamos como decisión principal, pero ayuda a validar pipeline)
    cmp = _fit_models(df.loc[df["factor_desp"].notna()].copy(), target="factor_desp", weight_col="w2_kg")

    # ===== PDF =====
    _build_pdf(TARGETS, df=df, results=results)

    best = cmp.sort_values("mae_val").iloc[0]
    print(f"[OK] PDF report: {OUT_PDF}")
    print(f"[OK] Silver shares B2 : {OUT_B2_SHARE}")
    print(f"[OK] Silver shares B2A: {OUT_B2A_SHARE}")
    print(f"     model_sanity={best['model']} mae_val={best['mae_val']:.4f}")
    print(f"     rows_used={len(df):,}  days={df['fecha_post'].nunique():,}")


if __name__ == "__main__":
    main()
