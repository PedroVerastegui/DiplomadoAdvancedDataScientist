"""Prueba del servicio REST (FastAPI) y sus métricas Prometheus."""
import pandas as pd
from fastapi.testclient import TestClient

from src import config, service
from src.model import build_pipeline
from src.data import generate_churn_data


def _train_tmp_model(tmp_path):
    df = generate_churn_data(n=600)
    pipe = build_pipeline().fit(df[config.FEATURES], df[config.TARGET])
    import joblib
    p = tmp_path / "m.joblib"
    joblib.dump(pipe, p)
    return str(p)


def test_health_and_predict(tmp_path, monkeypatch):
    model_path = _train_tmp_model(tmp_path)
    # Fuerza al servicio a usar el modelo temporal.
    service._model = None
    monkeypatch.setattr(service, "load_model", lambda *a, **k: __import__("joblib").load(model_path))

    client = TestClient(service.app)
    assert client.get("/health").json()["model_loaded"] is True

    payload = {
        "tenure_months": 3, "monthly_fee": 95.0, "total_charges": 300.0,
        "support_calls": 6, "payment_delay": 40.0, "digital_usage": 20.0,
        "contract_type": "month-to-month", "payment_method": "cash", "has_internet": "yes",
    }
    r = client.post("/predict", json=payload).json()
    assert 0 <= r["churn_probability"] <= 1
    assert r["churn_prediction"] in (0, 1)

    # /metrics debe exponer las métricas Prometheus.
    m = client.get("/metrics").text
    assert "churn_predictions_total" in m
