"""Generación y carga del dataset sintético de churn.

El dataset imita un caso de 'Customer Churn Prediction'. Es sintético y
determinista (semilla fija) para que el repositorio sea reproducible sin
depender de datos externos. Reemplázalo por tu dataset real cuando lo tengas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def generate_churn_data(n: int = 4000, seed: int = config.RANDOM_STATE,
                        drift: bool = False) -> pd.DataFrame:
    """Genera un DataFrame de churn.

    Parameters
    ----------
    n : número de clientes.
    seed : semilla para reproducibilidad.
    drift : si es True, desplaza algunas distribuciones para simular
            'data drift' (se usa en el experimento de monitoreo).
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, size=n)
    monthly_fee = rng.normal(70, 25, size=n).clip(15, 200)
    support_calls = rng.poisson(2, size=n)
    payment_delay = rng.exponential(5, size=n).clip(0, 90)
    digital_usage = rng.normal(55, 20, size=n).clip(0, 100)

    if drift:
        # Simula un cambio real en la operación del negocio:
        # suben las tarifas, las llamadas a soporte y la morosidad; baja el uso digital.
        monthly_fee = (monthly_fee * 1.69).clip(15, 400)      # +69%
        support_calls = support_calls + rng.poisson(6, size=n)  # ~ +296%
        payment_delay = (payment_delay * 10.27).clip(0, 365)   # +927%
        digital_usage = (digital_usage * 0.39).clip(0, 100)    # -61%

    contract_type = rng.choice(
        ["month-to-month", "one_year", "two_year"], size=n, p=[0.55, 0.28, 0.17])
    payment_method = rng.choice(
        ["card", "bank_transfer", "digital_wallet", "cash"], size=n,
        p=[0.4, 0.25, 0.25, 0.1])
    has_internet = rng.choice(["yes", "no"], size=n, p=[0.8, 0.2])

    total_charges = (monthly_fee * tenure * rng.uniform(0.85, 1.15, size=n)).round(2)

    # Probabilidad latente de churn en función de las variables.
    logit = (
        -3.1
        + 0.055 * payment_delay
        + 0.40 * support_calls
        - 0.050 * tenure
        + 0.018 * (monthly_fee - 70)
        - 0.022 * (digital_usage - 55)
        + np.where(contract_type == "month-to-month", 1.3, 0.0)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        config.ID_COL: [f"C{i:06d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_fee": monthly_fee.round(2),
        "total_charges": total_charges,
        "support_calls": support_calls,
        "payment_delay": payment_delay.round(2),
        "digital_usage": digital_usage.round(2),
        "contract_type": contract_type,
        "payment_method": payment_method,
        "has_internet": has_internet,
        config.TARGET: churn,
    })
    return df


def write_raw_dataset(n: int = 4000) -> str:
    """Genera y guarda el dataset base en data/raw/."""
    config.ensure_dirs()
    df = generate_churn_data(n=n)
    df.to_csv(config.RAW_DATA, index=False)
    return str(config.RAW_DATA)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Carga el dataset desde CSV; si no existe, lo genera."""
    import os
    p = path or str(config.RAW_DATA)
    if not os.path.exists(p):
        write_raw_dataset()
    return pd.read_csv(p)
