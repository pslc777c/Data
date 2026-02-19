from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# =====================================================
# ROOT DINÁMICO
# =====================================================
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
OUTDIR = DATA / "gold"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = OUTDIR / "dataset_analisis_desperdicio_dia_destino.parquet"
OUTPUT_PDF = OUTDIR / "EJECUTIVO_Sistema_Ajuste_Desperdicio.pdf"

# Plots
PLOT_SCATTER = OUTDIR / "scatter_ajuste_vs_desp.png"
PLOT_M0_REALPRED = OUTDIR / "m0_mix_gb_realpred_test.png"
PLOT_M2_REALPRED = OUTDIR / "m2_mix_ajuste_gb_realpred_test.png"
PLOT_M2_IMPORT = OUTDIR / "m2_perm_import_test.png"

# Config
TEST_FRAC = 0.20
TOP_K = 12
ALPHAS = np.logspace(-4, 4, 41)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _plot_real_vs_pred(y_true, y_pred, title: str, path: Path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_bar(series: pd.Series, title: str, ylabel: str, path: Path):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _make_metrics_table(rows: list[list[str]]) -> Table:
    t = Table([["Métrica", "Valor"]] + rows, colWidths=[3.2 * inch, 3.8 * inch], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def main():
    df = pd.read_parquet(DATASET_PATH).copy()

    required = {"Fecha", "Destino", "desp_pct", "ajuste"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Faltan columnas requeridas: {miss}")

    # Normalización básica
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
    df["Destino"] = df["Destino"].astype(str).str.upper().str.strip()

    df["desp_pct"] = pd.to_numeric(df["desp_pct"], errors="coerce")
    df["ajuste"] = pd.to_numeric(df["ajuste"], errors="coerce")

    # Features mix = todo excepto claves y targets
    exclude_cols = ["Fecha", "Destino", "desp_pct", "ajuste"]
    X_mix_df = df.drop(columns=[c for c in exclude_cols if c in df.columns]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Filtrar filas válidas
    mask = df["desp_pct"].notna() & df["ajuste"].notna()
    df = df.loc[mask].copy()
    X_mix_df = X_mix_df.loc[mask].copy()

    # Split temporal: últimas fechas como test
    df_sorted = df.sort_values(["Fecha", "Destino"]).reset_index(drop=True)
    X_mix_sorted = X_mix_df.loc[df_sorted.index].reset_index(drop=True)

    n = len(df_sorted)
    cut = int(np.floor((1.0 - TEST_FRAC) * n))

    train = df_sorted.iloc[:cut].copy()
    test = df_sorted.iloc[cut:].copy()

    Xmix_train = X_mix_sorted.iloc[:cut].values
    Xmix_test = X_mix_sorted.iloc[cut:].values

    y_train = train["desp_pct"].values
    y_test = test["desp_pct"].values

    aj_train = train["ajuste"].values.reshape(-1, 1)
    aj_test = test["ajuste"].values.reshape(-1, 1)

    # =====================================================
    # 1) Diagnóstico directo: Ajuste vs Desperdicio
    # =====================================================
    corr_all = float(np.corrcoef(df_sorted["ajuste"].values, df_sorted["desp_pct"].values)[0, 1])

    plt.figure()
    plt.scatter(df_sorted["ajuste"].values, df_sorted["desp_pct"].values)
    plt.xlabel("Ajuste (%)")
    plt.ylabel("Desperdicio (%)")
    plt.title(f"Ajuste vs Desperdicio (corr={corr_all:.3f})")
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER)
    plt.close()

    # =====================================================
    # 2) M1: Desp ~ Ajuste (lineal interpretable)
    # =====================================================
    lin = LinearRegression()
    lin.fit(aj_train, y_train)
    pred1_train = lin.predict(aj_train)
    pred1_test = lin.predict(aj_test)

    m1_r2_train = float(r2_score(y_train, pred1_train))
    m1_r2_test = float(r2_score(y_test, pred1_test))
    m1_rmse_train = _rmse(y_train, pred1_train)
    m1_rmse_test = _rmse(y_test, pred1_test)

    beta_aj = float(lin.coef_[0])  # Δdesp (pp) por +1.0 (100pp) de ajuste
    impact_1pp = beta_aj * 0.01    # Δdesp (pp) por +1pp de ajuste

    # =====================================================
    # 3) M0: Desp ~ Mix (GB)  (baseline de mezcla)
    # =====================================================
    m0 = GradientBoostingRegressor(
        random_state=42, n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8
    )
    m0.fit(Xmix_train, y_train)
    pred0_test = m0.predict(Xmix_test)

    m0_r2_test = float(r2_score(y_test, pred0_test))
    m0_rmse_test = _rmse(y_test, pred0_test)

    _plot_real_vs_pred(y_test, pred0_test, "M0 (TEST): Desp ~ Mix (GB)", PLOT_M0_REALPRED)

    # =====================================================
    # 4) M2: Desp ~ Mix + Ajuste (GB)  (hipótesis: ajuste media/explica)
    # =====================================================
    Xmixaj_train = np.hstack([Xmix_train, aj_train])
    Xmixaj_test = np.hstack([Xmix_test, aj_test])

    m2 = GradientBoostingRegressor(
        random_state=42, n_estimators=500, learning_rate=0.03, max_depth=3, subsample=0.8
    )
    m2.fit(Xmixaj_train, y_train)
    pred2_test = m2.predict(Xmixaj_test)

    m2_r2_test = float(r2_score(y_test, pred2_test))
    m2_rmse_test = _rmse(y_test, pred2_test)

    _plot_real_vs_pred(y_test, pred2_test, "M2 (TEST): Desp ~ Mix + Ajuste (GB)", PLOT_M2_REALPRED)

    # Permutation importance (TEST) para M2
    feature_names_mix = X_mix_sorted.columns.to_list()
    feat_names_m2 = feature_names_mix + ["AJUSTE_PCT"]

    perm = permutation_importance(m2, Xmixaj_test, y_test, n_repeats=10, random_state=42, scoring="r2")
    imp = pd.Series(perm.importances_mean, index=feat_names_m2).sort_values(ascending=False).head(TOP_K)

    _plot_bar(imp, f"M2 (TEST): Top {TOP_K} Permutation Importance (ΔR²)", "Importancia (ΔR²)", PLOT_M2_IMPORT)

    # =====================================================
    # 5) PDF Ejecutivo
    # =====================================================
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PDF))
    elements = []

    elements.append(Paragraph("DOCUMENTO EJECUTIVO — SISTEMA AJUSTE ↔ DESPERDICIO", styles["Heading1"]))
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph(
        "Objetivo: evaluar si el desperdicio (%) está principalmente explicado por el ajuste (%) "
        "(hipótesis operacional: forzar ajuste genera desperdicio), y cómo se compara contra "
        "modelos basados solo en mezcla (B2 + B2A). Se reporta desempeño out-of-sample usando split temporal.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Métricas clave
    metrics_tbl = _make_metrics_table([
        ["Split", f"Temporal: {int((1-TEST_FRAC)*100)}% train / {int(TEST_FRAC*100)}% test (último bloque como test)"],
        ["Corr(Ajuste, Desp)", f"{corr_all:.4f}"],
        ["M0: Desp ~ Mix (GB) — R² test / RMSE test", f"{m0_r2_test:.4f} / {m0_rmse_test:.4f}"],
        ["M1: Desp ~ Ajuste (Lineal) — R² test / RMSE test", f"{m1_r2_test:.4f} / {m1_rmse_test:.4f}"],
        ["M2: Desp ~ Mix + Ajuste (GB) — R² test / RMSE test", f"{m2_r2_test:.4f} / {m2_rmse_test:.4f}"],
        ["Efecto marginal (lineal): +1pp Ajuste ⇒ Δ Desp", f"{impact_1pp:.4f} pp"],
    ])
    elements.append(metrics_tbl)
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("1) Evidencia directa: Ajuste vs Desperdicio", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_SCATTER), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("2) Comparación de modelos (TEST)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_M0_REALPRED), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(str(PLOT_M2_REALPRED), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("3) Drivers del modelo conjunto (Mix + Ajuste)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_M2_IMPORT), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.25 * inch))

    # Conclusiones automáticas (reglas simples)
    concl = []
    if m2_r2_test > max(m0_r2_test, m1_r2_test) + 0.02:
        concl.append("• El modelo conjunto (Mix + Ajuste) mejora de forma material la predicción de desperdicio: evidencia a favor de una relación sistémica.")
    if m1_r2_test > m0_r2_test + 0.02:
        concl.append("• Ajuste explica más desperdicio que la mezcla por sí sola: consistente con hipótesis 'forzar ajuste genera desperdicio'.")
    if "AJUSTE_PCT" in imp.index[:5].to_list():
        concl.append("• Ajuste aparece como driver top en importancia del modelo conjunto: el ajuste aporta señal incremental sobre la mezcla.")
    if not concl:
        concl.append("• No se observa mejora contundente: el desperdicio sigue dominado por ruido/variables omitidas; ajuste puede ser parte pero no explica la mayor variabilidad.")

    elements.append(Paragraph("Conclusiones ejecutivas", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("<br/>".join(concl), styles["BodyText"]))

    doc.build(elements)

    print(f"[OK] PDF generado en: {OUTPUT_PDF}")
    print("[INFO] R2 test: M0 mix=", m0_r2_test, " M1 ajuste=", m1_r2_test, " M2 mix+aj=", m2_r2_test)
    print("[INFO] corr(ajuste, desp)=", corr_all, " beta_aj=", beta_aj, " impacto(+1pp)=", impact_1pp)
    print("[INFO] Top importance M2:\n", imp)


if __name__ == "__main__":
    main()
