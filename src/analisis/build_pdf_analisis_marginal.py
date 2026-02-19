from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# ROOT DINAMICO
# =====================================================
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
DATA_PDF = ROOT / "data" / "eval" 

DATASET_PATH = DATA / "gold" / "dataset_analisis_desperdicio_dia_destino.parquet"
OUTPUT_PDF = DATA_PDF / "Analisis_Marginal_Desperdicio.pdf"

PLOT1 = DATA_PDF / "plot_real_vs_pred.png"
PLOT2 = DATA_PDF / "plot_top_effects.png"


# =====================================================
# 1️⃣ CARGA
# =====================================================
df = pd.read_parquet(DATASET_PATH)

y = df["desp_pct"].values

exclude_cols = ["Fecha", "Destino", "desp_pct", "ajuste"]
X = df.drop(columns=exclude_cols)

feature_names = X.columns.tolist()
X = X.values


# =====================================================
# 2️⃣ MODELO LINEAL
# =====================================================
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

coefs = pd.Series(model.coef_, index=feature_names)
coefs_sorted = coefs.reindex(coefs.abs().sort_values(ascending=False).index)
top_effects = coefs_sorted.head(10)


# =====================================================
# 3️⃣ SIMULACION MARGINAL
# =====================================================
# Elegimos el efecto más fuerte
top_feature = top_effects.index[0]
beta_top = top_effects.iloc[0]

impacto_1pp = beta_top * 0.01  # incremento 1% share


# =====================================================
# 4️⃣ GRAFICOS
# =====================================================

# Real vs Predicho
plt.figure()
plt.scatter(y, y_pred)
plt.xlabel("Desperdicio Real (%)")
plt.ylabel("Desperdicio Predicho (%)")
plt.title("Modelo Lineal: Real vs Predicho")
plt.tight_layout()
plt.savefig(PLOT1)
plt.close()


# Top 10 coeficientes
plt.figure()
top_effects.plot(kind="bar")
plt.title("Top 10 Efectos Marginales")
plt.ylabel("Impacto en desp_pct por +1 unidad de share")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT2)
plt.close()


# =====================================================
# 5️⃣ PDF EJECUTIVO
# =====================================================
doc = SimpleDocTemplate(str(OUTPUT_PDF))
elements = []
styles = getSampleStyleSheet()

elements.append(Paragraph("ANALISIS MARGINAL DE DESPERDICIO", styles["Heading1"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph(f"R² del modelo: {round(r2,4)}", styles["BodyText"]))
elements.append(Paragraph(f"RMSE: {round(rmse,4)}", styles["BodyText"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Paragraph("Simulacion Marginal", styles["Heading2"]))
elements.append(Spacer(1, 0.2 * inch))
elements.append(Paragraph(
    f"Si el mix '{top_feature}' aumenta 1 punto porcentual, "
    f"el desperdicio cambia aproximadamente {round(impacto_1pp,4)} puntos porcentuales.",
    styles["BodyText"]
))
elements.append(Spacer(1, 0.4 * inch))

elements.append(Paragraph("1. Ajuste del Modelo", styles["Heading2"]))
elements.append(Spacer(1, 0.2 * inch))
elements.append(Image(str(PLOT1), width=6*inch, height=4*inch))
elements.append(Spacer(1, 0.4 * inch))

elements.append(Paragraph("2. Principales Efectos Marginales", styles["Heading2"]))
elements.append(Spacer(1, 0.2 * inch))
elements.append(Image(str(PLOT2), width=6*inch, height=4*inch))

doc.build(elements)

print(f"[OK] PDF generado en: {OUTPUT_PDF}")
