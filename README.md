# Data Lakehouse – ML1 / ML2 Planning Pipeline

Arquitectura modular para construcción de datasets analíticos y modelos predictivos
orientados a planificación de cosecha y poscosecha.

## Arquitectura por Capas

### Bronze
Ingesta y normalización inicial de fuentes crudas.

### Silver
Construcción de dimensiones, facts y ventanas operativas.

### Features
Generación de variables derivadas por bloque, ciclo, grado y día.

### Models
#### ML1
Modelos base:
- Curvas de tallos
- Distribución de grado
- Peso de tallo
- Ventanas de cosecha
- Ajustes poscosecha baseline

#### ML2
Modelos de refinamiento:
- Ajuste poscosecha
- Desperdicio
- Hidratación
- Harvest horizon dinámico
- Share grado refinado

### Gold
Vistas finales listas para consumo operativo / BI.

### Preds
Construcción de predicciones agregadas:
- Oferta día
- Tallos día
- Peso final
- Cajas
- Plan horas

### Audit
Validaciones:
- Balance de masa
- Curva vs real
- Horizon consistency
- KPI poscosecha

### Ops
Orquestación y registro de ejecuciones.

---

## Flujo General

Bronze → Silver → Features → ML1 → ML2 → Gold → Preds → Audit

---

## Principios de Diseño

- Separación estricta por capas
- Reproducibilidad modular
- Versionado por tags
- Sin versionar datos ni artefactos
