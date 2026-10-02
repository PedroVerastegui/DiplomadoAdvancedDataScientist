"""Pruebas de datos y features."""
import pandas as pd

from src import config
from src.data import generate_churn_data
from src.features import build_preprocessor


def test_generate_schema():
    df = generate_churn_data(n=200)
    for col in config.FEATURES + [config.TARGET, config.ID_COL]:
        assert col in df.columns
    assert len(df) == 200
    assert set(df[config.TARGET].unique()).issubset({0, 1})


def test_generate_is_deterministic():
    a = generate_churn_data(n=100, seed=7)
    b = generate_churn_data(n=100, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_drift_changes_distribution():
    normal = generate_churn_data(n=1000, drift=False)
    drift = generate_churn_data(n=1000, drift=True)
    # La morosidad media debe subir con drift.
    assert drift["payment_delay"].mean() > normal["payment_delay"].mean()


def test_preprocessor_fits():
    df = generate_churn_data(n=200)
    pre = build_preprocessor()
    X = pre.fit_transform(df[config.FEATURES])
    assert X.shape[0] == 200
