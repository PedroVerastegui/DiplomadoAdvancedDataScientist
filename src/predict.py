"""Predicción por lotes (batch)."""
from __future__ import annotations

import pandas as pd

from . import config
from .model import load_model


def predict_batch(data_path: str, model_path: str | None = None,
                  output_path: str | None = None) -> pd.DataFrame:
    """Aplica el modelo a un CSV y devuelve el DataFrame con predicciones."""
    model = load_model(model_path)
    df = pd.read_csv(data_path)
    proba = model.predict_proba(df[config.FEATURES])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["churn_prediction"] = (proba >= 0.5).astype(int)

    if output_path:
        out.to_csv(output_path, index=False)
    return out
