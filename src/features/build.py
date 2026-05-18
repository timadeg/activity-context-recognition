"""Feature preparation: train/test split and scaling."""
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COLUMN


def split_data(
    df: pd.DataFrame, target: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, apply to both. Return scaler for reuse."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
