"""Load the trained model and predict default risk for one applicant."""

import os
import joblib
import mlflow
import mlflow.sklearn

from src import data_preprocessing as dp
from src.train import TRACKING_URI, REGISTERED_MODEL_NAME, MODEL_PATH

DEFAULT_THRESHOLD = 0.5


def load_model():
    """Try the MLflow registry first, then the local joblib file.

    Returns (model, where_it_came_from).
    """
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")
        return model, "MLflow Model Registry (LoanDefaultRiskModel@champion)"
    except Exception:
        pass

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), "local file models/best_model.joblib"

    raise FileNotFoundError("No trained model found. Run:  python -m src.train")


def risk_band(probability):
    """Turn a probability into an easy-to-read label."""
    if probability < 0.2:
        return "Low"
    if probability < 0.4:
        return "Medium"
    return "High"


def predict_one(model, applicant, threshold=DEFAULT_THRESHOLD):
    """Predict for one applicant (a dict of feature values).

    Raises ValueError if the input is invalid.
    """
    row = dp.validate_input(applicant)         # same checks as the training data
    probability = float(model.predict_proba(row)[0, 1])  # pipeline does the preprocessing
    return {
        "probability": probability,
        "prediction": "Default" if probability >= threshold else "No default",
        "risk_band": risk_band(probability),
        "threshold": threshold,
    }
