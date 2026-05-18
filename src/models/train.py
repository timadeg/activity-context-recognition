"""Model training. Each function returns the fitted estimator and fit time."""
import time
from typing import Tuple, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.config import RANDOM_STATE


def _timed_fit(model: Any, X_train, y_train) -> Tuple[Any, float]:
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Fitted {model.__class__.__name__} in {elapsed:.2f}s")
    return model, elapsed


def train_decision_tree(X_train, y_train) -> Tuple[DecisionTreeClassifier, float]:
    return _timed_fit(DecisionTreeClassifier(random_state=RANDOM_STATE), X_train, y_train)


def train_svm(X_train, y_train) -> Tuple[SVC, float]:
    return _timed_fit(SVC(random_state=RANDOM_STATE), X_train, y_train)


def train_random_forest(X_train, y_train) -> Tuple[RandomForestClassifier, float]:
    return _timed_fit(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), X_train, y_train
    )


def train_mlp(X_train, y_train) -> Tuple[MLPClassifier, float]:
    return _timed_fit(
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            random_state=RANDOM_STATE,
            max_iter=500,
        ),
        X_train, y_train,
    )


def train_all(X_train, y_train) -> dict:
    """Train all four classifiers. Returns dict keyed by model name."""
    print("Training models...")
    results = {}
    results["decision_tree"] = train_decision_tree(X_train, y_train)
    results["svm"] = train_svm(X_train, y_train)
    results["random_forest"] = train_random_forest(X_train, y_train)
    results["mlp"] = train_mlp(X_train, y_train)
    return results
