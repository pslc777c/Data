from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# =============================
# ROOT
# =============================
def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path(__file__).resolve().parents[3]


ROOT = _project_root()
DATA = ROOT / "data"
OUTDIR = DATA / "gold"

DATASET_PATH = OUTDIR / "dataset_analisis_desperdicio_dia_destino.parquet"
OUTPUT_PDF = OUTDIR / "Analisis_Extremos_Desperdicio.pdf"

PLOT_IMPORT = OUTDIR / "extremos_importance.png"


# =============================
# LOAD
# =============================
df = pd.read_parquet(DATASET_PATH).copy()

df["desp_pct"] = pd.to_numeric(df["desp_pct"], errors="coerce")

# Definir extremo como > percentil 90
threshold = df["desp_pct"].quantile(0.90)
df["desp_extremo"] = (df["desp_pct"] > threshold).astype(int)

# Features
exclude_cols = ["Fecha", "Destino", "desp_pct", "ajuste", "desp_extremo"]
X_df = df.drop(columns=[c for c in exclude_cols if c in df.columns]).fillna(0)

X = X_df.values
y = df["desp_extremo"].values

# =============================
# Split
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# =============================
# Clasificador
# =============================
clf = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    random_state=42
)

clf.fit(X_train, y_train)

probs = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)

# Permutation importance
perm = permutation_importance(
    clf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc"
)

importances = pd.Series(
    perm.importances_mean,
    index=X_df.columns
).sort_values(ascending=False).head(15)

# Plot
plt.figure()
importances.plot(kind="bar")
plt.title("Drivers de días con Desperdicio Extremo (ROC AUC)")
plt.ylabel("Importancia (ΔAUC)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_IMPORT)
plt.close()

# =============================
# Comparación media normal vs extremo
# =============================
normal = df[df["desp_extremo"] == 0]
extremo = df[df["desp_extremo"] == 1]

comparacion = []

for col in importances.index[:10]:
    mean_normal = normal[col].mean()
    mean_ext = extremo[col].mean()
    lift = (mean_ext / mean_normal) if mean_normal != 0 else np.nan
    comparacion.append([col, round(mean_normal,4), round(mean_ext,4), round(lift,2)])

# =============================
# PDF
# =============================
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(OUTPUT_PDF))
elements = []

elements.append(Paragraph("ANALISIS DE DIAS EXTREMOS DE DESPERDICIO", styles["Heading1"]))
elements.append(Spacer(1, 0.2 * inch))

elements.append(Paragraph(
    f"Umbral extremo definido como > percentil 90 (>{round(threshold,4)}%).",
    styles["BodyText"]
))
elements.append(Spacer(1, 0.2 * inch))

elements.append(Paragraph(f"ROC AUC del clasificador: {round(auc,4)}", styles["BodyText"]))
elements.append(Spacer(1, 0.3 * inch))

elements.append(Image(str(PLOT_IMPORT), width=6.5*inch, height=4*inch))
elements.append(Spacer(1, 0.3 * inch))

# Tabla comparativa
data = [["Feature", "Media Normal", "Media Extremo", "Lift"]] + comparacion

table = Table(data)
table.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
]))

elements.append(table)

doc.build(elements)

print("[OK] PDF generado:", OUTPUT_PDF)
print("ROC AUC:", auc)
print("Top drivers extremos:\n", importances.head(10))
