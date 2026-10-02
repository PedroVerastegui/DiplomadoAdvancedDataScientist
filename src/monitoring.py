"""Monitoreo del modelo: métricas Prometheus y detección de data drift (PSI).

Este módulo es el corazón de la integración del Curso 2 (Monitoreo,
Data Drift y Mantenimiento). Expone:
  - Métricas Prometheus que el servicio publica en /metrics.
  - El cálculo del PSI (Population Stability Index) para medir drift.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from . import config

# --------------------------------------------------------------------------
# Métricas Prometheus
# --------------------------------------------------------------------------
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total", "Número total de predicciones servidas")
CHURN_POSITIVE_TOTAL = Counter(
    "churn_positive_total", "Predicciones con churn = 1")
CHURN_PROBABILITY = Histogram(
    "churn_probability", "Distribución de la probabilidad de churn predicha",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Latencia de la predicción en segundos")
FEATURE_MEAN = Gauge(
    "feature_mean", "Media observada de cada feature numérica", ["feature"])
DRIFT_PSI = Gauge(
    "feature_drift_psi", "PSI de drift por feature numérica", ["feature"])


def record_prediction(features: dict, probability: float) -> None:
    """Actualiza las métricas Prometheus con una predicción individual."""
    PREDICTIONS_TOTAL.inc()
    CHURN_PROBABILITY.observe(float(probability))
    if probability >= 0.5:
        CHURN_POSITIVE_TOTAL.inc()
    for feat in config.NUMERIC_FEATURES:
        if feat in features and features[feat] is not None:
            FEATURE_MEAN.labels(feature=feat).set(float(features[feat]))


# --------------------------------------------------------------------------
# Detección de drift (PSI)
# --------------------------------------------------------------------------
def population_stability_index(expected: np.ndarray, actual: np.ndarray,
                               bins: int = 10) -> float:
    """Calcula el PSI entre una distribución esperada y una observada.

    Interpretación habitual:
      PSI < 0.1  -> sin cambios significativos.
      0.1-0.25   -> cambio moderado (vigilar).
      > 0.25     -> cambio fuerte (drift; posible reentrenamiento).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 2:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf

    e_perc = np.histogram(expected, bins=cuts)[0] / len(expected)
    a_perc = np.histogram(actual, bins=cuts)[0] / len(actual)
    eps = 1e-6
    e_perc = np.clip(e_perc, eps, None)
    a_perc = np.clip(a_perc, eps, None)
    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))


def compute_drift(baseline: pd.DataFrame, current: pd.DataFrame,
                  features: list[str] | None = None) -> dict:
    """Calcula el PSI por feature numérica y lo clasifica.

    Devuelve {feature: {psi, status, pct_change}} y actualiza el Gauge DRIFT_PSI.
    """
    feats = features or config.NUMERIC_FEATURES
    result = {}
    for feat in feats:
        if feat not in baseline or feat not in current:
            continue
        psi = population_stability_index(baseline[feat].values, current[feat].values)
        if psi < 0.1:
            status = "estable"
        elif psi < 0.25:
            status = "moderado"
        else:
            status = "drift"
        base_mean = float(baseline[feat].mean())
        cur_mean = float(current[feat].mean())
        pct = ((cur_mean - base_mean) / base_mean * 100) if base_mean else 0.0
        result[feat] = {
            "psi": round(psi, 4),
            "status": status,
            "baseline_mean": round(base_mean, 2),
            "current_mean": round(cur_mean, 2),
            "pct_change": round(pct, 1),
        }
        DRIFT_PSI.labels(feature=feat).set(psi)
    return result


def load_baseline_stats() -> dict:
    """Carga las estadísticas baseline guardadas en el entrenamiento."""
    with open(config.BASELINE_PATH, encoding="utf-8") as f:
        return json.load(f)
