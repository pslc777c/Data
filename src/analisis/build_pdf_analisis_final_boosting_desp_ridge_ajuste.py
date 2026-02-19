from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error

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
OUTPUT_PDF = OUTDIR / "Analisis_FINAL_Boosting_Desperdicio_Ridge_Ajuste.pdf"

# Plots
PLOT_D_REALPRED = OUTDIR / "gb_real_vs_pred_desp_test.png"
PLOT_D_IMPORT = OUTDIR / "gb_perm_import_desp_test.png"
PLOT_A_REALPRED = OUTDIR / "ridge_real_vs_pred_ajuste_test.png"
PLOT_A_TOP = OUTDIR / "ridge_top_effects_ajuste.png"

# Config
TEST_SIZE = 0.20
RANDOM_STATE = 42
TOP_K = 15

# RidgeCV alphas
ALPHAS = np.logspace(-4, 4, 41)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_table(title: str, rows: list[list[str]]) -> Table:
    data = [[title, ""]] + rows
    t = Table(data, colWidths=[3.0 * inch, 4.0 * inch], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("SPAN", (0, 0), (-1, 0)),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


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


def _safe_renormalize_add_1pp(x: np.ndarray, idx_target: int, group_idx: np.ndarray, delta_pp: float = 1.0) -> np.ndarray:
    """
    Simulación composicional:
    - Incrementa feature idx_target en +delta_pp (porcentaje puntos => 0.01)
    - Resta proporcionalmente al resto del grupo para mantener suma del grupo ~ constante.
    - Trunca a 0 si cae negativo y renormaliza en segunda pasada.
    """
    x2 = x.copy()
    delta = delta_pp / 100.0

    if idx_target not in set(group_idx.tolist()):
        # Si no está en grupo, solo suma (fallback)
        x2[idx_target] = x2[idx_target] + delta
        return x2

    # masa disponible para restar
    others = group_idx[group_idx != idx_target]
    sum_others = float(np.sum(x2[others]))

    x2[idx_target] = float(x2[idx_target]) + delta

    if sum_others <= 1e-12:
        # No hay de dónde restar: dejamos y listo
        return x2

    # restar proporcional
    x2[others] = x2[others] - delta * (x2[others] / sum_others)

    # truncar negativos y re-ajustar si truncamos
    neg_mask = x2[others] < 0
    if np.any(neg_mask):
        x2[others[neg_mask]] = 0.0
        # renormalizar para que el grupo conserve (aprox) suma original + delta - delta = original
        # conservamos la suma total del grupo como: sum_group_old (antes) + delta - delta = sum_group_old
        # como truncamos, distribuimos el faltante sobre los positivos
        sum_group_old = float(np.sum(x[group_idx]))
        sum_group_new = float(np.sum(x2[group_idx]))
        gap = sum_group_old - sum_group_new
        if gap > 1e-12:
            pos = group_idx[x2[group_idx] > 0]
            sum_pos = float(np.sum(x2[pos]))
            if sum_pos > 1e-12:
                x2[pos] = x2[pos] + gap * (x2[pos] / sum_pos)

    return x2


def main():
    # =====================================================
    # 1) Carga y preparación
    # =====================================================
    df = pd.read_parquet(DATASET_PATH).copy()

    required = {"desp_pct", "ajuste", "Fecha", "Destino"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Faltan columnas requeridas: {miss}")

    exclude_cols = ["Fecha", "Destino", "desp_pct", "ajuste"]
    X_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_desp = pd.to_numeric(df["desp_pct"], errors="coerce").values
    y_aj = pd.to_numeric(df["ajuste"], errors="coerce").values

    mask = np.isfinite(y_desp) & np.isfinite(y_aj)
    X_df = X_df.loc[mask].copy()
    y_desp = y_desp[mask]
    y_aj = y_aj[mask]

    feature_names = X_df.columns.to_list()
    X = X_df.values

    n_rows, n_features = X.shape
    if n_rows < 50:
        raise ValueError(f"Dataset demasiado pequeño (n={n_rows}).")

    # Bloques: B2A vs B2
    # Regla: B2A tiene patrón "_GI_" en el nombre
    idx_b2a = np.array([i for i, c in enumerate(feature_names) if "_GI_" in str(c)], dtype=int)
    idx_b2 = np.array([i for i, c in enumerate(feature_names) if "_GI_" not in str(c)], dtype=int)

    # =====================================================
    # 2) Split train/test (misma partición para ambos targets)
    # =====================================================
    X_train, X_test, y_desp_train, y_desp_test, y_aj_train, y_aj_test = train_test_split(
        X, y_desp, y_aj, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # =====================================================
    # 3) DESPERDICIO — Gradient Boosting
    # =====================================================
    gb = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        loss="squared_error",
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8
    )
    gb.fit(X_train, y_desp_train)

    pred_desp_train = gb.predict(X_train)
    pred_desp_test = gb.predict(X_test)

    desp_r2_train = float(r2_score(y_desp_train, pred_desp_train))
    desp_r2_test = float(r2_score(y_desp_test, pred_desp_test))
    desp_rmse_train = _rmse(y_desp_train, pred_desp_train)
    desp_rmse_test = _rmse(y_desp_test, pred_desp_test)

    # Permutation importance en TEST (más interpretable que feature_importances_ en este contexto)
    perm = permutation_importance(
        gb, X_test, y_desp_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="r2"
    )
    imp = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
    top_imp = imp.head(TOP_K)

    # Importancia acumulada por bloque
    total_imp = float(np.sum(np.maximum(imp.values, 0)))
    imp_b2a = float(np.sum(np.maximum(imp.iloc[[i for i, c in enumerate(imp.index) if "_GI_" in str(c)]].values, 0)))
    imp_b2 = float(np.sum(np.maximum(imp.iloc[[i for i, c in enumerate(imp.index) if "_GI_" not in str(c)]].values, 0)))
    pct_b2a = (imp_b2a / total_imp) if total_imp > 1e-12 else 0.0
    pct_b2 = (imp_b2 / total_imp) if total_imp > 1e-12 else 0.0

    # Simulación marginal composicional (+1pp) para top 3 features
    sim_rows = []
    top3 = top_imp.index[:3].to_list()
    feat_to_idx = {c: i for i, c in enumerate(feature_names)}

    for feat in top3:
        j = feat_to_idx[feat]
        group_idx = idx_b2a if "_GI_" in feat else idx_b2

        deltas = []
        for k in range(X_test.shape[0]):
            x0 = X_test[k]
            y0 = gb.predict(x0.reshape(1, -1))[0]
            x1 = _safe_renormalize_add_1pp(x0, j, group_idx, delta_pp=1.0)
            y1 = gb.predict(x1.reshape(1, -1))[0]
            deltas.append(float(y1 - y0))

        sim_rows.append([feat, f"{np.mean(deltas):.5f}", f"{np.std(deltas):.5f}", "B2A" if "_GI_" in feat else "B2"])

    # Plots desperdicio
    _plot_real_vs_pred(
        y_desp_test, pred_desp_test,
        "GB (TEST): Desperdicio Real vs Predicho",
        PLOT_D_REALPRED
    )
    _plot_bar(
        top_imp,
        f"GB (TEST): Top {TOP_K} Permutation Importance (R² drop)",
        "Importancia (ΔR²)",
        PLOT_D_IMPORT
    )

    # =====================================================
    # 4) AJUSTE — RidgeCV (estable y explicable)
    # =====================================================
    ridge = RidgeCV(alphas=ALPHAS, cv=5, fit_intercept=True)
    ridge.fit(X_train, y_aj_train)

    pred_aj_train = ridge.predict(X_train)
    pred_aj_test = ridge.predict(X_test)

    aj_r2_train = float(r2_score(y_aj_train, pred_aj_train))
    aj_r2_test = float(r2_score(y_aj_test, pred_aj_test))
    aj_rmse_train = _rmse(y_aj_train, pred_aj_train)
    aj_rmse_test = _rmse(y_aj_test, pred_aj_test)

    coefs = pd.Series(ridge.coef_, index=feature_names)
    top_coef_aj = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(TOP_K)

    # Plots ajuste
    _plot_real_vs_pred(
        y_aj_test, pred_aj_test,
        "Ridge (TEST): Ajuste Real vs Predicho",
        PLOT_A_REALPRED
    )
    _plot_bar(
        top_coef_aj,
        f"Ridge: Top {TOP_K} Coeficientes (Ajuste)",
        "Coeficiente",
        PLOT_A_TOP
    )

    # =====================================================
    # 5) PDF Ejecutivo único
    # =====================================================
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(OUTPUT_PDF))
    elements = []

    elements.append(Paragraph("ANÁLISIS FINAL — DESPERDICIO (GB) + AJUSTE (RIDGE)", styles["Heading1"]))
    elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph(
        "Base: Día × Destino. Features incluyen (i) distribución de Grado en B2 (entrada) "
        "y (ii) distribución de mix SKU + Grado_Ideal (bin) en B2A (salida). "
        "Para desperdicio se usa Gradient Boosting (no lineal). Para ajuste, RidgeCV (regularizado) por estabilidad.",
        styles["BodyText"]
    ))
    elements.append(Spacer(1, 0.2 * inch))

    # Summary table
    tbl = _make_table("Métricas (Train/Test)", [
        ["N filas / P features", f"{n_rows} / {n_features}"],
        ["Desperdicio — Considerado", "GradientBoostingRegressor + Permutation Importance"],
        ["Desperdicio R² (Train / Test)", f"{desp_r2_train:.4f} / {desp_r2_test:.4f}"],
        ["Desperdicio RMSE (Train / Test)", f"{desp_rmse_train:.4f} / {desp_rmse_test:.4f}"],
        ["Ajuste — Considerado", f"RidgeCV (alpha={ridge.alpha_:.6g})"],
        ["Ajuste R² (Train / Test)", f"{aj_r2_train:.4f} / {aj_r2_test:.4f}"],
        ["Ajuste RMSE (Train / Test)", f"{aj_rmse_train:.4f} / {aj_rmse_test:.4f}"],
        ["Desperdicio — % importancia B2A vs B2", f"{pct_b2a*100:.1f}% (B2A) / {pct_b2*100:.1f}% (B2)"],
    ])
    elements.append(tbl)
    elements.append(Spacer(1, 0.25 * inch))

    # Desperdicio section
    elements.append(Paragraph("1) DESPERDICIO — Modelo no lineal (Gradient Boosting)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_D_REALPRED), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Image(str(PLOT_D_IMPORT), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.2 * inch))

    # Simulation table
    sim_table_data = [["Feature", "Δpred mean (+1pp)", "Δpred std", "Bloque"]] + sim_rows
    sim_table = Table(sim_table_data, hAlign="LEFT")
    sim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(Paragraph("Simulación marginal composicional (+1pp con renormalización dentro del bloque)", styles["Heading3"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(sim_table)
    elements.append(Spacer(1, 0.25 * inch))

    # Ajuste section
    elements.append(Paragraph("2) AJUSTE — Modelo regularizado interpretable (RidgeCV)", styles["Heading2"]))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Image(str(PLOT_A_REALPRED), width=6.8 * inch, height=4.2 * inch))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Image(str(PLOT_A_TOP), width=6.8 * inch, height=4.2 * inch))

    doc.build(elements)

    print(f"[OK] PDF generado en: {OUTPUT_PDF}")

    print("[INFO] Desperdicio GB R2 train/test:", desp_r2_train, desp_r2_test)
    print("[INFO] Ajuste Ridge R2 train/test:", aj_r2_train, aj_r2_test)
    print("[INFO] % importancia desperdicio por bloque (B2A/B2):", pct_b2a, pct_b2)
    print("[INFO] Top 10 importance desperdicio:\n", top_imp.head(10))
    print("[INFO] Top 10 coef ajuste:\n", top_coef_aj.head(10))


if __name__ == "__main__":
    main()
