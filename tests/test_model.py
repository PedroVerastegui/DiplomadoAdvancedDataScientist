"""Pruebas de entrenamiento y predicción."""
import pandas as pd

from src import config
from src.data import generate_churn_data
from src.model import build_pipeline


def test_pipeline_trains_and_predicts():
    df = generate_churn_data(n=800)
    X, y = df[config.FEATURES], df[config.TARGET]
    pipe = build_pipeline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape[0] == len(df)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_beats_baseline_auc():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    df = generate_churn_data(n=2000)
    X, y = df[config.FEATURES], df[config.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    pipe = build_pipeline().fit(Xtr, ytr)
    auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
    assert auc > 0.7  # el modelo debe aprender señal real


def test_handles_unknown_category():
    df = generate_churn_data(n=500)
    pipe = build_pipeline().fit(df[config.FEATURES], df[config.TARGET])
    row = df[config.FEATURES].iloc[[0]].copy()
    row["payment_method"] = "crypto"  # categoría no vista
    proba = pipe.predict_proba(row)[:, 1]
    assert 0 <= proba[0] <= 1
