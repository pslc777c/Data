from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import json
import io

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

import statsmodels.api as sm

import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

from common.io import read_parquet


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
GOLD = DATA / "gold"
EVAL = DATA / "eval" / "ml2"

IN_DS = GOLD / "ml2_datasets" / "ds_desp_poscosecha_ml2_v1.parquet"

OUT_DIR = EVAL / "audit_mix_grados_desp"
OUT_PDF = OUT_DIR / "audit_mix_grados_vs_desp_report.pdf"

TARGET = "log_ratio_desp_clipped"
WEIGHT = "tallos_w"
DATE_COL = "fecha_post_pred_used"
DESTINO_COL = "destino"
GRADO_COL = "grado"

BASE_NUM = ["dow", "month", "weekofyear"]
BASE_CAT = [DESTINO_COL, GRADO_COL]


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _canon_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _entropy(p: np.ndarray) -> float:
    p = p[np.isfinite(p)]
    p = p[p > 0]
    if len(p) == 0:
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


def _make_daily_mix_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d[DATE_COL] = _to_date(d[DATE_COL])
    d[DESTINO_COL] = _canon_str(d[DESTINO_COL])
    d[GRADO_COL] = _canon_str(d[GRADO_COL])

    d[WEIGHT] = _safe_float(d[WEIGHT]).fillna(0.0)
    d[TARGET] = _safe_float(d[TARGET])

    g_tot = (
        d.groupby([DATE_COL, DESTINO_COL], dropna=False, as_index=False)[WEIGHT]
        .sum()
        .rename(columns={WEIGHT: "tallos_total"})
    )

    g = (
        d.groupby([DATE_COL, DESTINO_COL, GRADO_COL], dropna=False, as_index=False)[WEIGHT]
        .sum()
        .rename(columns={WEIGHT: "tallos_grado"})
    )

    piv = g.pivot_table(
        index=[DATE_COL, DESTINO_COL],
        columns=GRADO_COL,
        values="tallos_grado",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()

    out = piv.merge(g_tot, on=[DATE_COL, DESTINO_COL], how="left")

    grado_cols = [c for c in out.columns if c not in {DATE_COL, DESTINO_COL, "tallos_total"}]
    for c in grado_cols:
        out[f"share_grado__{c}"] = np.where(
            out["tallos_total"] > 0,
            out[c] / out["tallos_total"],
            np.nan,
        )

    share_cols = [f"share_grado__{c}" for c in grado_cols]
    shares_mat = out[share_cols].to_numpy(dtype=float)

    sort_sh = np.sort(shares_mat, axis=1)[:, ::-1]
    out["mix_top1_share"] = sort_sh[:, 0]
    out["mix_top2_share"] = np.where(sort_sh.shape[1] > 1, sort_sh[:, 1], np.nan)
    out["mix_entropy"] = [_entropy(row[np.isfinite(row)]) for row in shares_mat]

    grado_vals = []
    for c in grado_cols:
        try:
            grado_vals.append(float(str(c).strip()))
        except Exception:
            grado_vals.append(np.nan)
    grado_vals = np.asarray(grado_vals, dtype=float)

    if np.isfinite(grado_vals).any():
        num_mask = np.isfinite(grado_vals)
        w_sh = shares_mat[:, num_mask]
        v = grado_vals[num_mask]
        out["mix_grado_avg"] = np.sum(w_sh * v.reshape(1, -1), axis=1)
    else:
        out["mix_grado_avg"] = np.nan

    def _wavg(x: pd.DataFrame) -> float:
        w = np.clip(x[WEIGHT].to_numpy(dtype=float), 0, None)
        y = x[TARGET].to_numpy(dtype=float)
        if np.sum(w) <= 0:
            return float("nan")
        return float(np.average(y, weights=w))

    agg_target = (
        d.groupby([DATE_COL, DESTINO_COL], as_index=False)
        .apply(lambda x: pd.Series({"target_wavg": _wavg(x), "target_avg": float(np.nanmean(x[TARGET].to_numpy(dtype=float)))}))
        .reset_index(drop=True)
    )
    out = out.merge(agg_target, on=[DATE_COL, DESTINO_COL], how="left")

    return out


def _corr_table(df: pd.DataFrame, ycol: str, xcols: list[str]) -> pd.DataFrame:
    rows = []
    y = _safe_float(df[ycol])
    for x in xcols:
        xx = _safe_float(df[x])
        m = np.isfinite(y) & np.isfinite(xx)
        if m.sum() < 20:
            rows.append({"x": x, "n": int(m.sum()), "pearson": np.nan, "spearman": np.nan})
            continue
        pear = np.corrcoef(xx[m], y[m])[0, 1]
        spear = pd.Series(xx[m]).corr(pd.Series(y[m]), method="spearman")
        rows.append({"x": x, "n": int(m.sum()), "pearson": float(pear), "spearman": float(spear)})
    out = pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(), ascending=False)
    return out


def _fit_ols_robust(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["dow"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.dayofweek.astype("Int64")
    df["month"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.month.astype("Int64")
    df["weekofyear"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.isocalendar().week.astype("Int64")

    y = _safe_float(df["target_wavg"])
    X_num = df[["dow", "month", "weekofyear", "mix_top1_share", "mix_entropy", "mix_grado_avg", "tallos_total"]].copy()
    for c in X_num.columns:
        X_num[c] = _safe_float(X_num[c])

    destino = _canon_str(df[DESTINO_COL]).fillna("UNKNOWN")
    X_cat = pd.get_dummies(destino, prefix="destino", dummy_na=False)

    X = pd.concat([X_num, X_cat], axis=1)
    X = sm.add_constant(X, has_constant="add")

    m = np.isfinite(y) & np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
    if m.sum() < 50:
        return pd.DataFrame([{"error": "Insuficientes filas para OLS", "n": int(m.sum())}])

    model = sm.OLS(y[m].to_numpy(dtype=float), X.loc[m].to_numpy(dtype=float))
    res = model.fit(cov_type="HC3")

    rows = []
    for name, coef, se, tval, pval in zip(X.columns, res.params, res.bse, res.tvalues, res.pvalues):
        rows.append(
            {
                "term": str(name),
                "coef": float(coef),
                "se_hc3": float(se),
                "t": float(tval),
                "p": float(pval),
                "n": int(m.sum()),
                "r2": float(res.rsquared),
                "r2_adj": float(res.rsquared_adj),
            }
        )
    out = pd.DataFrame(rows).sort_values("p")
    return out


def _compare_models(d: pd.DataFrame) -> pd.DataFrame:
    df = d.copy()
    df[DATE_COL] = _to_date(df[DATE_COL])
    df[DESTINO_COL] = _canon_str(df[DESTINO_COL])
    df[GRADO_COL] = _canon_str(df[GRADO_COL])

    df["dow"] = df[DATE_COL].dt.dayofweek.astype("Int64")
    df["month"] = df[DATE_COL].dt.month.astype("Int64")
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype("Int64")

    df[TARGET] = _safe_float(df[TARGET])
    df[WEIGHT] = _safe_float(df[WEIGHT]).fillna(0.0)

    daily = _make_daily_mix_features(df)
    keep = [DATE_COL, DESTINO_COL, "mix_top1_share", "mix_entropy", "mix_grado_avg", "tallos_total"]
    df = df.merge(daily[keep], on=[DATE_COL, DESTINO_COL], how="left")

    max_date = df[DATE_COL].max()
    cut = max_date - pd.Timedelta(days=30)

    tr = df.loc[df[DATE_COL] < cut].copy()
    va = df.loc[df[DATE_COL] >= cut].copy()

    def fit_and_eval(cols_num: list[str], cols_cat: list[str], tag: str) -> dict:
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat),
                ("num", "passthrough", cols_num),
            ],
            remainder="drop",
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=6,
            max_iter=400,
            random_state=42,
        )
        pipe = Pipeline([("prep", pre), ("model", model)])

        X_tr = tr[cols_num + cols_cat].copy()
        y_tr = tr[TARGET].to_numpy(dtype=float)
        w_tr = np.clip(tr[WEIGHT].to_numpy(dtype=float), 0, None)

        X_va = va[cols_num + cols_cat].copy()
        y_va = va[TARGET].to_numpy(dtype=float)
        w_va = np.clip(va[WEIGHT].to_numpy(dtype=float), 0, None)

        pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
        p_tr = pipe.predict(X_tr)
        p_va = pipe.predict(X_va)

        return {
            "model": tag,
            "cut_date": str(cut.date()),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "mae_train": _mae(y_tr, p_tr, w_tr),
            "mae_val": _mae(y_va, p_va, w_va),
        }

    base = fit_and_eval(
        cols_num=BASE_NUM,
        cols_cat=[DESTINO_COL, GRADO_COL],
        tag="baseline_calendar+destino+grado",
    )

    mix = fit_and_eval(
        cols_num=BASE_NUM + ["mix_top1_share", "mix_entropy", "mix_grado_avg", "tallos_total"],
        cols_cat=[DESTINO_COL, GRADO_COL],
        tag="baseline+mix_daily(top1,entropy,avg,vol)",
    )

    out = pd.DataFrame([base, mix])
    out["delta_mae_val"] = out["mae_val"] - float(out.loc[out["model"] == "baseline_calendar+destino+grado", "mae_val"].iloc[0])
    return out


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


def _make_plot_images(daily: pd.DataFrame) -> tuple[bytes, bytes]:
    """
    Returns (scatter_png_bytes, hist_png_bytes)
    """
    dd = daily.copy()
    x = _safe_float(dd["mix_grado_avg"])
    y = _safe_float(dd["target_wavg"])
    m = np.isfinite(x) & np.isfinite(y)

    # Scatter
    buf1 = io.BytesIO()
    plt.figure()
    if m.sum() > 0:
        plt.scatter(x[m], y[m], s=10, alpha=0.6)
        plt.xlabel("mix_grado_avg")
        plt.ylabel("target_wavg (log_ratio_desp_clipped)")
        plt.title("Mix grado promedio vs residual de desperdicio (agregado dia/destino)")
    else:
        plt.title("Sin datos suficientes para scatter")
    plt.tight_layout()
    plt.savefig(buf1, format="png", dpi=200)
    plt.close()
    buf1.seek(0)

    # Histogram
    buf2 = io.BytesIO()
    plt.figure()
    if np.isfinite(y).sum() > 0:
        plt.hist(y[np.isfinite(y)], bins=30)
        plt.xlabel("target_wavg (log_ratio_desp_clipped)")
        plt.ylabel("frecuencia")
        plt.title("Distribucion del residual de desperdicio (target_wavg)")
    else:
        plt.title("Sin datos suficientes para histograma")
    plt.tight_layout()
    plt.savefig(buf2, format="png", dpi=200)
    plt.close()
    buf2.seek(0)

    return buf1.getvalue(), buf2.getvalue()


def _build_pdf(
    pdf_path: Path,
    meta: dict,
    corr: pd.DataFrame,
    reg: pd.DataFrame,
    cmp: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    h2 = ParagraphStyle(
        "h2",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
    )
    small = ParagraphStyle(
        "small",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.grey,
    )

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0] - 1.5 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Audit mix de grados vs desperdicio",
        author="Data-LakeHouse",
    )

    story = []
    story.append(Paragraph("Audit estadistico: efecto de mix de grados sobre desperdicio (ML2)", title))
    story.append(Paragraph(f"Dataset: {meta['dataset']}", small))
    story.append(Paragraph(f"Filas con real: {meta['n_rows_has_real']:,} | Agregados dia/destino: {meta['n_days_destino']:,}", small))
    story.append(Spacer(1, 8))

    # KPI table
    story.append(Paragraph("1) KPI principal (comparacion de modelos)", h2))
    kpi_df = cmp.copy()
    # format
    kpi_data = _df_to_table_data(kpi_df[["model", "cut_date", "n_train", "n_val", "mae_train", "mae_val", "delta_mae_val"]], max_rows=10)
    t = Table(kpi_data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)

    # concise conclusion line
    try:
        base_mae = float(cmp.loc[cmp["model"].str.startswith("baseline_calendar"), "mae_val"].iloc[0])
        mix_mae = float(cmp.loc[cmp["model"].str.contains("baseline\\+mix"), "mae_val"].iloc[0])
        delta = mix_mae - base_mae
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Resultado: MAE val baseline={base_mae:.4f} vs with_mix={mix_mae:.4f} (delta={delta:+.4f}).", body))
    except Exception:
        pass

    # Plots
    story.append(Paragraph("2) Graficos exploratorios (agregado por dia/destino)", h2))
    scatter_png, hist_png = _make_plot_images(daily)

    img1 = Image(io.BytesIO(scatter_png))
    img1.drawHeight = 7.0 * cm
    img1.drawWidth = 16.0 * cm
    story.append(img1)
    story.append(Spacer(1, 6))

    img2 = Image(io.BytesIO(hist_png))
    img2.drawHeight = 7.0 * cm
    img2.drawWidth = 16.0 * cm
    story.append(img2)

    story.append(PageBreak())

    # Correlations
    story.append(Paragraph("3) Correlaciones (target_wavg vs features de mix)", h2))
    corr_show = corr.copy()
    # keep most relevant
    corr_show = corr_show[["x", "n", "pearson", "spearman"]].head(20)
    corr_data = _df_to_table_data(corr_show, max_rows=20)
    tc = Table(corr_data, hAlign="LEFT", colWidths=[8.0 * cm, 1.5 * cm, 2.5 * cm, 2.5 * cm])
    tc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(tc)
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Nota: Spearman captura relaciones monotónicas (no necesariamente lineales). "
            "Si hay señal real, deberia aparecer en top1_share, entropy, grado_avg o shares especificos.",
            small,
        )
    )

    # OLS
    story.append(Paragraph("4) Regresion OLS robusta (HC3) con controles de calendario y destino", h2))
    if "term" not in reg.columns:
        story.append(Paragraph("OLS no se pudo estimar (insuficientes filas o datos).", body))
    else:
        reg_show = reg[["term", "coef", "se_hc3", "t", "p", "r2", "r2_adj"]].head(20)
        reg_data = _df_to_table_data(reg_show, max_rows=20)
        tr = Table(reg_data, hAlign="LEFT", colWidths=[6.5 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm])
        tr.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(tr)
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Interpretacion: coeficientes positivos implican mayor target_wavg (residual de desperdicio) "
                "a mayor valor del feature, controlando por calendario y destino. "
                "Usar p-val como guia, no como verdad absoluta (posible confounding).",
                small,
            )
        )

    # Cierre
    story.append(Spacer(1, 10))
    story.append(Paragraph("5) Conclusiones operativas", h2))
    concl = []
    # conclusion heuristic based on delta_mae
    try:
        delta = float(cmp.loc[cmp["model"].str.contains("baseline\\+mix"), "delta_mae_val"].iloc[0])
        if np.isfinite(delta) and delta < -0.003:
            concl.append("- Hay evidencia cuantitativa de que el mix diario (grado/volumen) aporta señal al residual de desperdicio.")
            concl.append("- Siguiente paso recomendado: incorporar mix de ventas (SKU_macro) y, cuando exista, B2A para refinar causalidad.")
        else:
            concl.append("- La mejora por mix diario es baja; conviene esperar a tener vector de SKU por dia (ventas) y/o B2A antes de integrar al ML2.")
    except Exception:
        concl.append("- No se pudo computar delta de MAE de manera confiable; revisar salidas de comparacion de modelos.")

    concl.append("- Importante: para forecast futuro, features tipo B2/B2A no existen; deben tener fallback o entrenar un modelo con features disponibles a futuro.")
    story.append(Paragraph("<br/>".join(concl), body))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Archivo generado automaticamente por audit_mix_grados_vs_desp.py", small))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IN_DS.exists():
        raise FileNotFoundError(f"No encuentro IN_DS={IN_DS}  (ROOT={ROOT})")

    df = read_parquet(IN_DS).copy()
    df.columns = [str(c).strip() for c in df.columns]

    need = {TARGET, WEIGHT, "has_real", DATE_COL, DESTINO_COL, GRADO_COL}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Dataset sin columnas necesarias: {sorted(miss)}")

    d = df.loc[df["has_real"].astype(bool)].copy()
    d[DATE_COL] = _to_date(d[DATE_COL])
    d = d.loc[d[DATE_COL].notna()].copy()

    daily = _make_daily_mix_features(d)
    mix_cols = [c for c in daily.columns if c.startswith("share_grado__")] + [
        "mix_top1_share",
        "mix_top2_share",
        "mix_entropy",
        "mix_grado_avg",
        "tallos_total",
    ]
    corr = _corr_table(daily, ycol="target_wavg", xcols=mix_cols)
    reg = _fit_ols_robust(daily)
    cmp = _compare_models(d)

    meta = {
        "dataset": str(IN_DS),
        "n_rows_has_real": int(len(d)),
        "n_days_destino": int(len(daily)),
    }

    _build_pdf(OUT_PDF, meta=meta, corr=corr, reg=reg, cmp=cmp, daily=daily)

    # Mensaje final
    try:
        base_mae = float(cmp.loc[cmp["model"].str.startswith("baseline_calendar"), "mae_val"].iloc[0])
        mix_mae = float(cmp.loc[cmp["model"].str.contains("baseline\\+mix"), "mae_val"].iloc[0])
        delta = mix_mae - base_mae
        print(f"[OK] PDF report: {OUT_PDF}")
        print(f"     MAE val baseline={base_mae:.4f}  with_mix={mix_mae:.4f}  delta={delta:+.4f}")
    except Exception:
        print(f"[OK] PDF report: {OUT_PDF}")


if __name__ == "__main__":
    main()
