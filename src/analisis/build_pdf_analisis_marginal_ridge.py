from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
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

DATASET_PATH = DATA / "gold" / "dataset_analisis_desperdicio_dia_destino.parquet"
OUTDIR = DATA / "gold"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = OUTDIR / "Analisis_Marginal_Ridge_Desperdicio_y_Ajuste.pdf"

PLOT_REALPRED_DESP = OUTDIR / "ridge_real_vs_pred_desp.png"
PLOT_TOP_DESP = OUTDIR / "ridge_top_effects_desp.png"
PLOT_REALPRED_AJUSTE = OUTDIR / "ridge_real_vs_pred_ajuste.png"
PLOT_TOP_AJUSTE = OUTDIR / "ridge_top_effects_ajuste.png"

# Config
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_K = 12

# RidgeCV: grilla razonable (logspace)
ALPHAS = np.logspace(-4, 4, 41)  # 1e-4 ... 1e4


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _fit_ridge(X_train, y_train) -> RidgeCV:
    # RidgeCV con validación interna (default: Leave-One-Out si cv=None).
    # Para estabilidad en datasets medianos, usamos cv=5.
    model = RidgeCV(alphas=ALPHAS, cv=5, fit_intercept=True)
    model.fit(X_train, y_train)
    return model


def _top_effects(model: RidgeCV, feature_names: list[str], k: int = 10) -> pd.Series:
    coefs = pd.Series(model.coef_, index=feature_names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(k)


def _plot_real_vs_pred(y_true, y_pred, title: str, path: Path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("Real")
    plt.ylabel("Predicho")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_top_effects(top: pd.Series, title: str, path: Path):
    plt.figure()
    top.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Coeficiente (Ridge)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _pp_effect_from_coef(beta: float, delta_pp: float = 1.0) -> float:
    """
    beta está en unidades de: cambio_y por +1.0 de share (100pp).
    delta_pp=1.0 => +0.01 en share
    """
    return float(beta * (delta_pp / 100.0))


def _make_metrics_table(metrics: dict) -> Table:
    data = [
        ["Métrica", "Desperdicio (desp_pct)", "Ajuste"],
        ["R² Train", f"{metrics['desp_r2_train']:.4f}", f"{metrics['aj_r2_train']:.4f}"],
        ["R² Test", f"{metrics['desp_r2_test']:.4f}", f"{metrics['aj_r2_test']:.4f}"],
        ["RMSE Train", f"{metrics['desp_rmse_train']:.4f}", f"{metrics['aj_rmse_train']:.4f}"],
        ["RMSE Test", f"{metrics['desp_rmse_test']:.4f}", f"{metrics['aj_rmse_test']:.4f}"],
        ["Alpha (Ridge)", f"{metrics['desp_alpha']:.6g}", f"{metrics['aj_alpha']:.6g}"],
        ["N (filas)", str(metrics["n_rows"]), str(metrics["n_rows"])],
        ["P (features)", str(metrics["n_features"]), str(metrics["n_features"])],
    ]
    t = Table(data, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def main():
    # =====================================================
    # 1) Cargar dataset
    # =====================================================
    df = pd.read_parquet(DATASET_PATH).copy()

    # Targets
    if "desp_pct" not in df.columns:
        raise ValueError("No existe columna 'desp_pct' en el dataset.")
    if "ajuste" not in df.columns:
        raise ValueError("No existe columna 'ajuste' en el dataset.")

    # Features
    exclude_cols = ["Fecha", "Destino", "desp_pct", "ajuste"]
    X_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    # Limpieza defensiva
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_desp = pd.to_numeric(df["desp_pct"], errors="coerce").values
    y_aj = pd.to_numeric(df["ajuste"], errors="coerce").values

    # Quitamos filas con y NaN
    mask = np.isfinite(y_desp) & np.isfinite(y_aj)
    X_df = X_df.loc[mask].copy()
    y_desp = y_desp[mask]
    y_aj = y_aj[mask]

    feature_names = X_df.columns.tolist()
    X = X_df.values

    n_rows, n_features = X.shape
    if n_rows < 30:
        raise ValueError(f"Dataset demasiado pequeño (n={n_rows}). No es estable para entrenar.")

    # =====================================================
    # 2) Split train/test (misma partición para ambos targets)
    # =====================================================
    X_train, X_test, y_desp_train, y_desp_test, y_aj_train, y_aj_test = train_test_split(
        X, y_desp, y_aj, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # =====================================================
    # 3) RidgeCV para desperdicio
    # =====================================================
    m_desp = _fit_ridge(X_train, y_desp_train)
    pred_desp_train = m_desp.predict(X_train)
    pred_desp_test = m_desp.predict(X_test)

    # =====================================================
    # 4) RidgeCV para ajuste
    # =====================================================
    m_aj = _fit_ridge(X_train, y_aj_train)
    pred_aj_train = m_aj.predict(X_train)
    pred_aj_test = m_aj.predict(X_test)

    # Métricas
    metrics = {
        "n_rows": n_rows,
        "n_features": n_features,
        "desp_alpha": float(m_desp.alpha_),
        "aj_alpha": float(m_aj.alpha_),
        "desp_r2_train": float(r2_score(y_desp_train, pred_desp_train)),
        "desp_r2_test": float(r2_score(y_desp_test, pred_desp_test)),
        "aj_r2_train": float(r2_score(y_aj_train, pred_aj_train)),
        "aj_r2_test": float(r2_score(y_aj_test, pred_aj_test)),
        "desp_rmse_train": _rmse(y_desp_train, pred_desp_train),
        "desp_rmse_test": _rmse(y_desp_test, pred_desp_test),
        "aj_rmse_train": _rmse(y_aj_train, pred_aj_train),
        "aj_rmse_test": _rmse(y_aj_test, pred_aj_test),
    }

    # Top drivers
    top_desp = _top_effects(m_desp, feature_names, k=TOP_K)
    top_aj = _top_effects(m_aj, feature_names, k=TOP_K)

    # =====================================================
    # 5) Simulación marginal (1pp)
    # =====================================================
    # Nota: esto es "marginal local" del modelo lineal regularizado.
    # Para una simulación composicional estricta (+1pp y renormalizar el resto)
    # eso lo haremos en el siguiente paso con un bloque dedicado.
    top_feat_desp = top_desp.index[0]
    top_feat_aj = top_aj.index[0]

    impact_1pp_desp = _pp_effect_from_coef(top_desp.iloc[0], delta_pp=1.0)
    impact_1pp_aj = _pp_effect_from_coef(top_aj.iloc[0], delta_pp=1.0)

    # =====================================================
    # 6) Gráficos
    # =====================================================
    _plot_real_vs_pred(y_desp_test, pred_desp_test, "Ridge (TEST): Desperdicio Real vs Predicho", PLOT_REALPRED_DESP)
    _plot_top_effects(top_desp, f"Top {TOP_K} Coeficientes Ridge: Desperdicio", PLOT_TOP_DESP)

    _plot_real_vs_pred(y_aj_test, pred_aj_test, "Ridge (TEST): Ajuste Real vs Predicho", PLOT_REALPRED_AJUSTE)
    _plot_top_effects(top_aj, f"Top {TOP_K} Coeficientes Ridge: Ajuste", PLOT_TOP_AJUSTE)

    # =====================================================
    # 7) PDF Ejecutivo
    # =====================================================
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PDF))
    elements = []

    elements.append(Paragraph("ANÁLISIS MARGINAL (RIDGE) — DESPERDICIO Y AJUSTE", styles["Heading1"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(
        "Objetivo: estimar drivers de desperdicio (desp_pct) y ajuste a partir de la distribución "
        "de Grado en B2 y de mix (SKU + Grado_Ideal bin) en B2A, a nivel Día × Destino. "
        "Se utiliza Ridge Regression para estabilizar coeficientes bajo colinealidad composicional.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Métricas (Train/Test)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(_make_metrics_table(metrics))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Simulación marginal (aprox. lineal, +1pp en share)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(
        f"Desperdicio: si '{top_feat_desp}' sube +1pp, el modelo estima Δdesp_pct ≈ {impact_1pp_desp:.4f} puntos.",
        styles["BodyText"]
    ))
    elements.append(Paragraph(
        f"Ajuste: si '{top_feat_aj}' sube +1pp, el modelo estima Δajuste ≈ {impact_1pp_aj:.4f} unidades.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 0.25 * inch))

    elements.append(Paragraph("Desperdicio — Ajuste del modelo (TEST)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_REALPRED_DESP), width=6.5 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(str(PLOT_TOP_DESP), width=6.5 * inch, height=4.2 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Ajuste — Ajuste del modelo (TEST)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_REALPRED_AJUSTE), width=6.5 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(str(PLOT_TOP_AJUSTE), width=6.5 * inch, height=4.2 * inch))

    doc.build(elements)

    print(f"[OK] PDF generado en: {OUTPUT_PDF}")
    print("[INFO] Métricas:", metrics)
    print(f"[INFO] Top desperdicio: {top_feat_desp} (impacto +1pp: {impact_1pp_desp:.4f})")
    print(f"[INFO] Top ajuste: {top_feat_aj} (impacto +1pp: {impact_1pp_aj:.4f})")


if __name__ == "__main__":
    main()
