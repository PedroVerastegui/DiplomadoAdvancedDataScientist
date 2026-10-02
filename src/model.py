"""Entrenamiento, evaluación y persistencia del modelo de churn."""
from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import load_data
from .features import build_preprocessor


def build_pipeline() -> Pipeline:
    """Pipeline completo: preprocesamiento + clasificador."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def train(data_path: str | None = None, model_path: str | None = None) -> dict:
    """Entrena el modelo, lo evalúa en un hold-out y lo guarda en disco.

    Devuelve un diccionario con las métricas de evaluación.
    """
    config.ensure_dirs()
    df = load_data(data_path)
    X = df[config.FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
    }

    mpath = model_path or str(config.MODEL_PATH)
    joblib.dump(pipe, mpath)
    with open(config.REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Guarda estadísticas de referencia (baseline) para la detección de drift.
    _save_baseline(X_train)
    return metrics


def _save_baseline(X: pd.DataFrame) -> None:
    """Guarda medias/percentiles de las numéricas como baseline de drift."""
    stats = {}
    for col in config.NUMERIC_FEATURES:
        stats[col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std()),
            "q": [float(v) for v in X[col].quantile([0, .25, .5, .75, 1]).tolist()],
        }
    with open(config.BASELINE_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_model(model_path: str | None = None) -> Pipeline:
    """Carga el pipeline entrenado desde disco."""
    return joblib.load(model_path or str(config.MODEL_PATH))
