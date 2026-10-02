# Customer Intelligence ML Platform

Plataforma de *Customer Churn Prediction* con integración de **MLOps / Monitoreo**.
Repositorio del **Curso 2 — Monitoreo, Data Drift y Mantenimiento de Modelos**, que
integra el monitoreo sobre la plataforma base del curso anterior.

> La idea central del curso: **"deploy no es el final del ciclo MLE"**. Tras desplegar
> el modelo hay que *observarlo*, detectar **data drift** y mantenerlo.

---

## ¿Qué incluye?

- **Dataset sintético de churn** (`data/raw/customer_churn.csv`), reproducible.
- **Pipeline de ML** (`src/`): preprocesamiento + `RandomForest`, con entrenamiento,
  evaluación y predicción.
- **CLI** (`main.py`): `make-data`, `train`, `predict`, `drift`, `serve`.
- **Servicio REST** (FastAPI) que sirve predicciones y **expone métricas Prometheus**.
- **Monitoreo**: métricas Prometheus + **detección de data drift (PSI)** + **alertas**.
- **Docker Compose**: levanta el modelo + Prometheus.
- **Pruebas automatizadas** (`pytest`), **GitHub Actions** (CI) y **configuración DVC**.

## Estructura

```text
customer-intelligence-ml-platform/
├── main.py                      # CLI
├── src/
│   ├── config.py                # rutas, columnas, constantes
│   ├── data.py                  # dataset sintético (normal y con drift)
│   ├── features.py              # preprocesamiento (ColumnTransformer)
│   ├── model.py                 # entrenamiento / evaluación / persistencia
│   ├── predict.py               # predicción por lotes
│   ├── monitoring.py            # métricas Prometheus + PSI (drift)
│   └── service.py               # FastAPI: /predict, /metrics, /health
├── scripts/drift_experiment.py  # experimento NORMAL vs DRIFT
├── tests/                       # pruebas (pytest)
├── notebooks/                   # 6 notebooks del entregable
├── prometheus/                  # prometheus.yml + alerts.yml
├── artifacts/                   # modelos y reportes (salida)
├── data/raw/                    # dataset
├── Dockerfile · docker-compose.yml
├── dvc.yaml · .github/workflows/ci.yml
└── requirements*.txt
```

## Instalación

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Uso rápido

```bash
# 1) Generar datos y entrenar
python main.py make-data
python main.py train

# 2) Predicción por lotes
python main.py predict --data data/raw/customer_churn.csv --output artifacts/reports/preds.csv

# 3) Experimento de data drift (NORMAL vs DRIFT)
python main.py drift

# 4) Servir el modelo con métricas Prometheus
python main.py serve --port 8000
#   POST http://localhost:8000/predict   GET http://localhost:8000/metrics
```

## Monitoreo con Docker + Prometheus

```bash
docker compose up --build
# Modelo:      http://localhost:8000/metrics
# Prometheus:  http://localhost:9090  (ver métricas y alertas)
```

Métricas expuestas: `churn_predictions_total`, `churn_positive_total`,
`churn_probability` (histograma), `prediction_latency_seconds`,
`feature_mean{feature=...}` y `feature_drift_psi{feature=...}`.

Alertas (`prometheus/alerts.yml`): `FeatureDriftDetected` (PSI > 0.25),
`HighChurnRate` y `HighAveragePaymentDelay`.

## Data drift (PSI)

El módulo `src/monitoring.py` calcula el **Population Stability Index**:

| PSI | Interpretación |
|-----|----------------|
| < 0.10 | Estable |
| 0.10 – 0.25 | Cambio moderado (vigilar) |
| > 0.25 | Drift fuerte (posible reentrenamiento) |

`python main.py drift` genera `artifacts/reports/drift_report.json` comparando un
lote "normal" contra uno con drift y reporta qué variables cambiaron.

## Pruebas

```bash
pytest -v
```

## Cómo adaptarlo a tu modelo real

1. Reemplaza `data/raw/customer_churn.csv` por tu dataset (o ajusta `src/data.py`).
2. Actualiza las columnas en `src/config.py` (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, `TARGET`).
3. Si cambias el algoritmo, edítalo en `src/model.py` (`build_pipeline`).
4. Vuelve a entrenar (`python main.py train`) y valida (`pytest`).

---

*Dataset sintético y reproducible con fines académicos.*
