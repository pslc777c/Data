# Data Lakehouse – Planificación Macro (ML / Vistas Gold)

Repositorio del pipeline para construir datasets y vistas finales (Gold) y modelos (ML1/ML2) orientados a planificación/poscosecha, con enfoque en trazabilidad de etapas y reproducibilidad.

## Objetivo
- Estandarizar la cadena de transformación desde fuentes/consultas hasta salidas analíticas.
- Generar vistas finales (Gold) para consumo BI/operación.
- Entrenar y evaluar modelos (ML1/ML2) como capa predictiva del proceso.

## Estructura del repositorio (alto nivel)
> Ajusta esta sección si tu estructura real difiere.

- `src/`  
  Código principal del proyecto.
  - `models/` : entrenamiento/inferencia de ML (ML1, ML2, etc.)
  - `gold/` : construcción de vistas finales (outputs listos para BI)
  - `silver/` / `bronze/` : (si aplica) capas intermedias de transformación

- `sql_scripts/`  
  SQL de apoyo para extracción/transformaciones.

- `config/`  
  Configuración del proyecto.
  > **No versionar credenciales reales.** Usar archivos ejemplo.

- `esquema/`  
  Definiciones/estructuras (si aplica).

- `init_duckdb.py`  
  Inicialización local (si aplica) para pruebas/ejecución con DuckDB.

## Requisitos
- Python 3.10+ (recomendado)
- Git
- (Opcional) DuckDB si se usa ejecución local

Instalación (ejemplo):
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
