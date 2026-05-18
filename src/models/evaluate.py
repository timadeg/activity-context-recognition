"""Model evaluation: accuracy, precision, recall, F1, confusion matrix."""
from typing import Dict
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)


def evaluate_model(model, X_test, y_test, name: str = "") -> Dict[str, float]:
    """Compute headline metrics for a single fitted model."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    if name:
        print(f"  {name}: " + " | ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def evaluate_all(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every trained model and return a comparison DataFrame."""
    print("Evaluating models...")
    rows = []
    for name, (model, fit_time) in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, name=name)
        metrics["fit_time_seconds"] = fit_time
        metrics["model"] = name
        rows.append(metrics)
    df = pd.DataFrame(rows).set_index("model")
    return df
