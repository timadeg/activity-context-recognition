"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_FILE


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw activity context tracking CSV.

    Args:
        path: Path to the CSV file. Defaults to RAW_DATA_FILE from config.

    Returns:
        DataFrame containing the raw sensor data.
    """
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df
