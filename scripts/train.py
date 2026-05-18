"""End-to-end training pipeline.

Run from the project root:
    python -m scripts.train
"""
import joblib
from src.config import MODELS_DIR
from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.features.balance import balance_classes
from src.features.build import split_data, scale_features
from src.models.train import train_all
from src.models.evaluate import evaluate_all


def main():
    # 1. Load
    df = load_raw_data()

    # 2. Clean
    df = clean_data(df)

    # 3. Balance
    df = balance_classes(df)

    # 4. Split & scale
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # 5. Train all models
    trained = train_all(X_train_scaled, y_train)

    # 6. Evaluate
    results = evaluate_all(trained, X_test_scaled, y_test)
    print("\n=== Results ===")
    print(results.round(3).to_string())

    # 7. Persist best model + scaler
    MODELS_DIR.mkdir(exist_ok=True)
    best_model_name = results["f1"].idxmax()
    best_model = trained[best_model_name][0]

    model_path = MODELS_DIR / f"{best_model_name}_v1.pkl"
    scaler_path = MODELS_DIR / "scaler_v1.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nSaved best model ({best_model_name}) to {model_path}")
    print(f"Saved scaler to {scaler_path}")


if __name__ == "__main__":
    main()
