"""Inference utilities for a trained model + scaler."""
from typing import List
import joblib
import pandas as pd
from pathlib import Path
from src.config import SENSOR_COLUMNS


def load_artifacts(model_path: Path, scaler_path: Path):
    """Load a saved model and scaler from disk."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_single(model, scaler, sensor_values: List[float]) -> str:
    """Predict activity from a list of 17 sensor readings (in SENSOR_COLUMNS order)."""
    if len(sensor_values) != len(SENSOR_COLUMNS):
        raise ValueError(
            f"Expected {len(SENSOR_COLUMNS)} values, got {len(sensor_values)}"
        )
    df = pd.DataFrame([sensor_values], columns=SENSOR_COLUMNS)
    scaled = scaler.transform(df)
    return str(model.predict(scaled)[0])
