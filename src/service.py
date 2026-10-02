"""Servicio REST (FastAPI) que sirve predicciones y expone métricas Prometheus.

Endpoints:
  GET  /health   -> estado del servicio y si el modelo está cargado.
  POST /predict  -> predice churn para un cliente y registra métricas.
  GET  /metrics  -> métricas en formato Prometheus (scrapeado por Prometheus).
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import config, monitoring
from .model import load_model

app = FastAPI(title="Customer Intelligence ML Platform", version="2.0.0")

_model = None


def get_model():
    """Carga perezosa del modelo (una sola vez)."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0)
    monthly_fee: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_calls: int = Field(..., ge=0)
    payment_delay: float = Field(..., ge=0)
    digital_usage: float = Field(..., ge=0, le=100)
    contract_type: str = "month-to-month"
    payment_method: str = "card"
    has_internet: str = "yes"


@app.get("/health")
def health():
    ok = True
    try:
        get_model()
    except Exception:
        ok = False
    return {"status": "ok" if ok else "model_not_loaded", "model_loaded": ok}


@app.post("/predict")
def predict(customer: Customer):
    import pandas as pd
    start = time.perf_counter()
    model = get_model()
    row = pd.DataFrame([customer.model_dump()])
    proba = float(model.predict_proba(row[config.FEATURES])[:, 1][0])
    monitoring.PREDICTION_LATENCY.observe(time.perf_counter() - start)
    monitoring.record_prediction(customer.model_dump(), proba)
    return {
        "churn_probability": round(proba, 4),
        "churn_prediction": int(proba >= 0.5),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
