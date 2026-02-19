from __future__ import annotations

from pathlib import Path
import io
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import statsmodels.api as sm

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

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

OUT_DIR = EVAL / "audit_b2_vs_sku_interaccion"
OUT_PDF = OUT_DIR / "audit_b2_vs_sku_interaccion_vs_mermas.pdf"

# Inputs
IN_DIM = SILVER / "dim_mermas_ajuste_fecha_post_destino.parquet"
IN_B2_SHARE = SILVER / "balanzas" / "silver_b2_share_grado_dia_destino.parquet"
IN_B2A_SKU_SHARE = SILVER / "balanzas" / "silver_b2a_share_sku_dia_destino.parquet"

# Outputs trazables
OUT_MODEL_DF = OUT_DIR / "model_df.parquet"
OUT_COEFS = OUT_DIR / "ols_coefs.parquet"
OUT_MARGINALS = OUT_DIR / "marginal_effects.parquet"

# Top-K/Top-J para evitar explosión de interacciones
TOP_K_SKU = 20
TOP_J_B2 = 8


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
    raise KeyError(f"No encuentro columnas. Busqué: {candidates}. Disponibles: {list(df.columns)[:120]}")


def _df_to_table_data(df: pd.DataFrame, max_rows: int = 25) -> list[list[str]]:
    df2 = df.head(max_rows).copy()
    cols = list(df2.columns)
    data = [cols]
    for _, r in df2.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(f"{v:.6f}")
            else:
                row.append("" if pd.isna(v) else str(v))
        data.append(row)
    return data


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


def _pick_top_share_cols(df: pd.DataFrame, prefix: str, top_n: int, weight_col: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return []
    w = np.clip(_safe_float(df[weight_col]).fillna(0.0).to_numpy(), 0, None)
    if np.sum(w) <= 0:
        # fallback simple mean
        masses = {c: float(np.nanmean(_safe_float(df[c]).to_numpy())) for c in cols}
    else:
        masses = {c: float(np.average(np.nan_to_num(_safe_float(df[c]).to_numpy(), nan=0.0), weights=w)) for c in cols}
    # ordena por masa desc
    top = sorted(cols, key=lambda c: masses.get(c, 0.0), reverse=True)[:top_n]
    return top


def _collapse_other(df: pd.DataFrame, prefix: str, keep_cols: list[str], other_name: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(prefix)]
    drop_cols = [c for c in cols if c not in keep_cols]
    out = df.copy()
    if drop_cols:
        out[other_name] = np.nan_to_num(out[drop_cols], nan=0.0).sum(axis=1)
        out = out.drop(columns=drop_cols)
    else:
        if other_name not in out.columns:
            out[other_name] = 0.0
    return out


def _ols_hc3(df: pd.DataFrame, y: str, xcols: list[str], destino_col: str) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    yv = _safe_float(d[y])

    X = pd.DataFrame(index=d.index)
    X["const"] = 1.0
    for c in xcols:
        X[c] = _safe_float(d[c]).fillna(0.0)

    dest = _canon_str(d[destino_col]).fillna("UNKNOWN")
    X = pd.concat([X, pd.get_dummies(dest, prefix="dest", drop_first=True)], axis=1)

    m = np.isfinite(yv) & np.all(np.isfinite(X.to_numpy(dtype=float)), axis=1)
    n = int(m.sum())
    if n < 80:
        return pd.DataFrame([{"error": "Insuficientes filas para OLS", "n": n}]), {"n": n}

    res = sm.OLS(yv[m].to_numpy(dtype=float), X.loc[m].to_numpy(dtype=float)).fit(cov_type="HC3")

    rows = []
    for name, coef, se, t, p in zip(X.columns, res.params, res.bse, res.tvalues, res.pvalues):
        rows.append(
            {
                "term": str(name),
                "coef": float(coef),
                "se_hc3": float(se),
                "t": float(t),
                "p": float(p),
                "n": n,
                "r2": float(res.rsquared),
                "r2_adj": float(res.rsquared_adj),
            }
        )

    meta = {
        "n": n,
        "r2": float(res.rsquared),
        "r2_adj": float(res.rsquared_adj),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }
    return pd.DataFrame(rows).sort_values("p"), meta


def _interaction_cols(sku_cols: list[str], b2_cols: list[str]) -> list[tuple[str, str, str]]:
    # (new_col, sku_col, b2_col)
    out = []
    for s in sku_cols:
        for g in b2_cols:
            out.append((f"int__{s}__x__{g}", s, g))
    return out


def _compute_marginals(
    coefs: pd.DataFrame,
    sku_cols: list[str],
    b2_cols: list[str],
    b2_mix_avg: dict[str, float],
    scenarios: dict[str, dict[str, float]],
) -> pd.DataFrame:
    # coefs table contains term->coef
    c = {r["term"]: float(r["coef"]) for _, r in coefs.iterrows() if "term" in coefs.columns}

    rows = []
    for sku in sku_cols:
        beta = c.get(sku, 0.0)

        # avg marginal: beta + sum_j gamma_{sku,j} * avg_b2_j
        marg_avg = beta
        for b2 in b2_cols:
            gamma = c.get(f"int__{sku}__x__{b2}", 0.0)
            marg_avg += gamma * float(b2_mix_avg.get(b2, 0.0))

        row = {"sku": sku, "marginal_avg": float(marg_avg)}
        # scenarios
        for sc_name, mix in scenarios.items():
            marg_sc = beta
            for b2 in b2_cols:
                gamma = c.get(f"int__{sku}__x__{b2}", 0.0)
                marg_sc += gamma * float(mix.get(b2, 0.0))
            row[f"marginal__{sc_name}"] = float(marg_sc)

        rows.append(row)

    return pd.DataFrame(rows)


def _build_pdf(
    pdf_path: Path,
    meta: dict,
    examples: pd.DataFrame,
    results: dict,
    df: pd.DataFrame,
) -> None:
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
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Audit B2 (grado) vs B2A (SKU) interaccion vs mermas",
        author="Data-LakeHouse",
    )

    story = []
    story.append(Paragraph("Audit macro: B2 (mix por grado) + B2A (mix por SKU) + interacción vs mermas", title))
    story.append(Paragraph(f"Filas model_df: {meta['rows']:,} | días: {meta['days']:,} | destinos: {meta['destinos']:,}", small))
    story.append(Paragraph(f"Top SKUs: {meta['top_k_sku']} | Top B2 grados: {meta['top_j_b2']}", small))
    story.append(Spacer(1, 8))

    # Ejemplos
    story.append(Paragraph("0) Ejemplos (para validar que shares y mismatches están bien)", h2))
    ex = examples.copy()
    t = Table(_df_to_table_data(ex, max_rows=10), hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("Nota: los shares deben sumar ~1 por (fecha_post,destino) dentro de cada vector.", small))

    story.append(PageBreak())

    # Para cada target
    for target_name, pack in results.items():
        story.append(Paragraph(f"Target: {target_name}", h2))
        story.append(Paragraph(pack["desc"], body))
        story.append(Spacer(1, 6))

        # KPIs
        story.append(Paragraph("1) KPIs de modelos OLS (R2_adj / AIC / n)", h2))
        kpi = pack["kpi"].copy()
        tt = Table(_df_to_table_data(kpi, max_rows=10), hAlign="LEFT")
        tt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(tt)
        story.append(Spacer(1, 8))

        # Coefs (top 25 por p)
        story.append(Paragraph("2) Coeficientes (OLS HC3) — Top términos por p-value", h2))
        coef = pack["coefs"].copy()
        show = coef if "term" in coef.columns else coef
        show = show.head(25)
        tc = Table(_df_to_table_data(show, max_rows=25), hAlign="LEFT")
        tc.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(tc)
        story.append(Spacer(1, 8))

        # Marginal effects
        story.append(Paragraph("3) Efectos marginales por SKU (condicionados al mix B2)", h2))
        marg = pack["marginals"].copy().head(20)
        tm = Table(_df_to_table_data(marg, max_rows=20), hAlign="LEFT")
        tm.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(tm)
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "Interpretación: marginal_avg es ∂target/∂share_sku. "
                "Si share_sku sube +0.01 (1pp), el target cambia ~ marginal_avg * 0.01 (puntos absolutos).",
                small,
            )
        )
        story.append(Spacer(1, 10))

        # Plots simples
        story.append(Paragraph("4) Gráficos rápidos", h2))
        story.append(Image(io.BytesIO(_plot_hist(df, target_name, f"Distribución: {target_name}")), width=16 * cm, height=7 * cm))
        story.append(Spacer(1, 6))
        story.append(Image(io.BytesIO(_plot_scatter(df, "w2_kg", target_name, "w2_kg vs target")), width=16 * cm, height=7 * cm))

        story.append(PageBreak())

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in [IN_DIM, IN_B2_SHARE, IN_B2A_SKU_SHARE]:
        if not p.exists():
            raise FileNotFoundError(f"No existe input: {p}")

    # ----------------------------
    # 1) DIM (targets y controles)
    # ----------------------------
    dim = read_parquet(IN_DIM).copy()
    dim.columns = [str(c).strip() for c in dim.columns]

    fecha_col = _resolve_col(dim, ["FECHA_POST", "fecha_post", "Fecha_Post", "fecha"])
    dest_col = _resolve_col(dim, ["DESTINO", "destino", "codigo_actividad", "CODIGO_ACTIVIDAD"])

    dim["fecha_post"] = _to_date(dim[fecha_col])
    dim["destino"] = _canon_str(dim[dest_col])

    # pesos/volúmenes base (ya los usabas)
    w2_col = _resolve_col(dim, ["W2_KG", "w2_kg"])
    w2a_col = _resolve_col(dim, ["W2A_KG", "w2a_kg"])
    dim["w2_kg"] = _safe_float(dim[w2_col]).fillna(0.0)
    dim["w2a_kg"] = _safe_float(dim[w2a_col]).fillna(0.0)

    # targets requeridos
    if "factor_desp" not in dim.columns:
        raise KeyError(f"dim no tiene 'factor_desp'. cols={list(dim.columns)[:120]}")
    if "ajuste" not in dim.columns:
        raise KeyError(f"dim no tiene 'ajuste'. cols={list(dim.columns)[:120]}")
    dim["factor_desp"] = _safe_float(dim["factor_desp"])
    dim["ajuste"] = _safe_float(dim["ajuste"])

    # calendario
    dim["dow"] = dim["fecha_post"].dt.dayofweek.astype("Int64")
    dim["month"] = dim["fecha_post"].dt.month.astype("Int64")
    dim["weekofyear"] = dim["fecha_post"].dt.isocalendar().week.astype("Int64")

    # shares por día entre destinos (macro)
    gday = dim.groupby("fecha_post", as_index=False).agg(w2_day=("w2_kg", "sum"), w2a_day=("w2a_kg", "sum"))
    dim = dim.merge(gday, on="fecha_post", how="left")
    dim["share_w2"] = np.where(dim["w2_day"] > 0, dim["w2_kg"] / dim["w2_day"], np.nan)
    dim["share_w2a"] = np.where(dim["w2a_day"] > 0, dim["w2a_kg"] / dim["w2a_day"], np.nan)
    dim["ratio_out_in"] = np.where(dim["w2_kg"] > 0, dim["w2a_kg"] / dim["w2_kg"], np.nan)

    dim = dim.loc[dim["fecha_post"].notna() & dim["destino"].notna()].copy()

    # ----------------------------
    # 2) B2 share (grado)
    # ----------------------------
    b2 = read_parquet(IN_B2_SHARE).copy()
    b2.columns = [str(c).strip() for c in b2.columns]
    if "fecha_post" not in b2.columns:
        b2_fecha = _resolve_col(b2, ["fecha_post", "FECHA_POST", "Fecha"])
        b2["fecha_post"] = _to_date(b2[b2_fecha])
    else:
        b2["fecha_post"] = _to_date(b2["fecha_post"])
    if "destino" not in [c.lower() for c in b2.columns]:
        b2_dest = _resolve_col(b2, ["destino", "DESTINO", "codigo_actividad"])
        b2["destino"] = _canon_str(b2[b2_dest])
    else:
        b2_dest = [c for c in b2.columns if c.lower() == "destino"][0]
        b2["destino"] = _canon_str(b2[b2_dest])

    # ----------------------------
    # 3) B2A share (SKU)
    # ----------------------------
    sku = read_parquet(IN_B2A_SKU_SHARE).copy()
    sku.columns = [str(c).strip() for c in sku.columns]
    if "fecha_post" not in sku.columns:
        s_fecha = _resolve_col(sku, ["fecha_post", "FECHA_POST", "Fecha"])
        sku["fecha_post"] = _to_date(sku[s_fecha])
    else:
        sku["fecha_post"] = _to_date(sku["fecha_post"])
    if "destino" not in [c.lower() for c in sku.columns]:
        s_dest = _resolve_col(sku, ["destino", "DESTINO", "codigo_actividad"])
        sku["destino"] = _canon_str(sku[s_dest])
    else:
        s_dest = [c for c in sku.columns if c.lower() == "destino"][0]
        sku["destino"] = _canon_str(sku[s_dest])

    # ----------------------------
    # 4) Merge master dataset
    # ----------------------------
    df = dim.merge(b2, on=["fecha_post", "destino"], how="left", suffixes=("", "_b2"))
    df = df.merge(sku, on=["fecha_post", "destino"], how="left", suffixes=("", "_sku"))

    # fill NA shares -> 0 (si faltan bins/sku ese día)
    for c in df.columns:
        if c.startswith("share_b2__") or c.startswith("share_sku__"):
            df[c] = _safe_float(df[c]).fillna(0.0)

    # filtra targets válidos
    df = df.loc[df["factor_desp"].notna() | df["ajuste"].notna()].copy()
    df = df.loc[df["fecha_post"].notna() & df["destino"].notna()].copy()

    # ----------------------------
    # 5) Selección Top-K/Top-J y OTHER
    # ----------------------------
    base_controls = ["dow", "month", "weekofyear", "w2_kg", "w2a_kg", "share_w2", "share_w2a", "ratio_out_in"]

    sku_top = _pick_top_share_cols(df, prefix="share_sku__", top_n=TOP_K_SKU, weight_col="w2_kg")
    b2_top = _pick_top_share_cols(df, prefix="share_b2__", top_n=TOP_J_B2, weight_col="w2_kg")

    df = _collapse_other(df, prefix="share_sku__", keep_cols=sku_top, other_name="share_sku__OTHER")
    df = _collapse_other(df, prefix="share_b2__", keep_cols=b2_top, other_name="share_b2__OTHER")

    sku_cols = sku_top + ["share_sku__OTHER"]
    b2_cols = b2_top + ["share_b2__OTHER"]

    # Interacciones
    inter = _interaction_cols(sku_cols, b2_cols)
    for newc, s, g in inter:
        df[newc] = _safe_float(df[s]).fillna(0.0) * _safe_float(df[g]).fillna(0.0)

    inter_cols = [x[0] for x in inter]

    # ----------------------------
    # 6) Escenarios B2 para marginales
    # ----------------------------
    # mix promedio ponderado por w2_kg
    w = np.clip(_safe_float(df["w2_kg"]).fillna(0.0).to_numpy(), 0, None)
    if np.sum(w) <= 0:
        b2_mix_avg = {c: float(np.nanmean(df[c].to_numpy(dtype=float))) for c in b2_cols}
    else:
        b2_mix_avg = {c: float(np.average(df[c].to_numpy(dtype=float), weights=w)) for c in b2_cols}

    # escenarios: cada grado top como "dominante"
    scenarios = {}
    for c in b2_top[: min(4, len(b2_top))]:  # escenarios con top 4 grados
        mix = {k: 0.0 for k in b2_cols}
        mix[c] = 0.8
        mix["share_b2__OTHER"] = 0.2
        scenarios[f"b2_dom_{c.replace('share_b2__','')}"] = mix

    # ----------------------------
    # 7) OLS por target (y por bloque)
    # ----------------------------
    results = {}
    all_coef_rows = []
    all_marg_rows = []

    # bloques de features
    x_base = base_controls
    x_b2 = base_controls + b2_cols
    x_sku = base_controls + sku_cols
    x_both = base_controls + b2_cols + sku_cols
    x_full = base_controls + b2_cols + sku_cols + inter_cols

    models = [
        ("base_controls", x_base),
        ("+B2", x_b2),
        ("+SKU", x_sku),
        ("+B2+SKU", x_both),
        ("+B2+SKU+interactions", x_full),
    ]

    for target, desc in [
        ("factor_desp", "Factor desperdicio real (dim)"),
        ("ajuste", "Ajuste real de peso (dim)"),
    ]:
        dft = df.loc[df[target].notna()].copy()

        kpi_rows = []
        best_full = None
        coef_full = None

        for tag, xcols in models:
            coef, meta = _ols_hc3(dft, y=target, xcols=xcols, destino_col="destino")
            row = {
                "target": target,
                "model": tag,
                "n": meta.get("n", np.nan),
                "r2": meta.get("r2", np.nan),
                "r2_adj": meta.get("r2_adj", np.nan),
                "aic": meta.get("aic", np.nan),
                "bic": meta.get("bic", np.nan),
            }
            kpi_rows.append(row)

            if tag == "+B2+SKU+interactions":
                best_full = row
                coef_full = coef.copy()
                if "term" in coef_full.columns:
                    tmp = coef_full.copy()
                    tmp["target"] = target
                    tmp["model"] = tag
                    all_coef_rows.append(tmp)

        kpi = pd.DataFrame(kpi_rows)

        # Marginal effects se calculan sobre el modelo FULL (si estimó bien)
        if coef_full is not None and "term" in coef_full.columns:
            marg = _compute_marginals(
                coefs=coef_full,
                sku_cols=sku_cols,
                b2_cols=b2_cols,
                b2_mix_avg=b2_mix_avg,
                scenarios=scenarios,
            ).sort_values("marginal_avg", key=lambda s: s.abs(), ascending=False)

            marg["target"] = target
            all_marg_rows.append(marg)
        else:
            marg = pd.DataFrame([{"error": "No se pudo estimar OLS full (insuficientes filas o datos)."}])

        results[target] = {
            "desc": desc,
            "kpi": kpi,
            "coefs": coef_full if coef_full is not None else pd.DataFrame([{"error": "sin coefs"}]),
            "marginals": marg,
        }

    # ----------------------------
    # 8) Ejemplos de validación (top mismatch / top shares)
    # ----------------------------
    # arma un resumen para inspección rápida
    ex_cols = ["fecha_post", "destino", "w2_kg", "w2a_kg", "factor_desp", "ajuste"]
    # agrega 2 SKUs + 2 B2 grados más masivos
    ex_cols += sku_top[: min(2, len(sku_top))]
    ex_cols += b2_top[: min(2, len(b2_top))]
    ex_cols = [c for c in ex_cols if c in df.columns]

    # orden: días con mucho share del SKU #1 y donde haya targets
    if sku_top:
        examples = df.loc[(df["factor_desp"].notna() | df["ajuste"].notna())].copy()
        examples = examples.sort_values(sku_top[0], ascending=False)[ex_cols].head(10)
    else:
        examples = df.loc[(df["factor_desp"].notna() | df["ajuste"].notna()), ex_cols].head(10)

    # ----------------------------
    # 9) Persist outputs
    # ----------------------------
    write_parquet(df, OUT_MODEL_DF)

    if all_coef_rows:
        write_parquet(pd.concat(all_coef_rows, ignore_index=True), OUT_COEFS)
    else:
        write_parquet(pd.DataFrame([{"error": "no coef rows"}]), OUT_COEFS)

    if all_marg_rows:
        write_parquet(pd.concat(all_marg_rows, ignore_index=True), OUT_MARGINALS)
    else:
        write_parquet(pd.DataFrame([{"error": "no marginal rows"}]), OUT_MARGINALS)

    meta = {
        "rows": int(len(df)),
        "days": int(df["fecha_post"].nunique()),
        "destinos": int(df["destino"].nunique()),
        "top_k_sku": int(TOP_K_SKU),
        "top_j_b2": int(TOP_J_B2),
    }

    _build_pdf(OUT_PDF, meta=meta, examples=examples, results=results, df=df)

    print(f"[OK] PDF: {OUT_PDF}")
    print(f"[OK] model_df: {OUT_MODEL_DF}")
    print(f"[OK] coefs   : {OUT_COEFS}")
    print(f"[OK] marginals: {OUT_MARGINALS}")
    print(f"     rows={meta['rows']:,}  days={meta['days']:,}")


if __name__ == "__main__":
    main()
