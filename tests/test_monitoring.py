"""Pruebas de monitoreo y detección de drift."""
import numpy as np

from src.data import generate_churn_data
from src.monitoring import compute_drift, population_stability_index


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    psi = population_stability_index(x, x.copy())
    assert psi < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)
    assert population_stability_index(base, shifted) > 0.25


def test_compute_drift_flags_drifted_features():
    normal = generate_churn_data(n=2000, drift=False)
    drift = generate_churn_data(n=2000, seed=99, drift=True)
    report = compute_drift(normal, drift)
    # payment_delay debe marcar drift fuerte.
    assert report["payment_delay"]["status"] == "drift"
    assert report["payment_delay"]["pct_change"] > 0
