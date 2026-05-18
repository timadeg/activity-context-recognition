"""Data cleaning: drop ID, remove duplicates, consolidate labels."""
import pandas as pd


# Mapping to consolidate duplicate activity labels found in the dataset
ACTIVITY_LABEL_MAP = {
    "ClimbingUpStairs": "AscendingStairs",
    "ClimbingDownStairs": "DescendingStairs",
    "MountainAscending": "AscendingStairs",
    "MountainDescending": "DescendingStairs",
}


def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the _id column so duplicates can be detected."""
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    print(f"Removed {before - after:,} duplicate rows ({before:,} -> {after:,})")
    return df


def consolidate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Merge synonymous activity labels into a single class each."""
    df = df.copy()
    df["activity"] = df["activity"].replace(ACTIVITY_LABEL_MAP)
    print(f"Consolidated labels. Classes: {sorted(df['activity'].unique())}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""
    df = drop_id_column(df)
    df = remove_duplicates(df)
    df = consolidate_labels(df)
    return df
