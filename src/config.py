"""Rutas y constantes centrales del proyecto."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "customer_churn.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
BASELINE_PATH = REPORTS_DIR / "baseline_stats.json"

TARGET = "churn"
ID_COL = "customer_id"

# Columnas por tipo (se usan en el preprocesamiento y el monitoreo).
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_fee",
    "total_charges",
    "support_calls",
    "payment_delay",
    "digital_usage",
]
CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "has_internet",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (DATA_DIR / "raw", MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
