"""Class balancing via over- and under-sampling."""
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from src.config import RANDOM_STATE, TARGET_COLUMN, DESIRED_SAMPLES_PER_CLASS


def balance_classes(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    desired_samples: int = DESIRED_SAMPLES_PER_CLASS,
) -> pd.DataFrame:
    """Balance class distribution to roughly `desired_samples` per class.

    Classes below the target are oversampled; classes above are undersampled.
    """
    counts = df[target].value_counts()

    under_strategy = {
        cls: desired_samples for cls, n in counts.items() if n > desired_samples
    }
    over_strategy = {
        cls: desired_samples for cls, n in counts.items() if n < desired_samples
    }

    X = df.drop(columns=[target])
    y = df[target]

    if under_strategy:
        under = RandomUnderSampler(
            sampling_strategy=under_strategy, random_state=RANDOM_STATE
        )
        X, y = under.fit_resample(X, y)

    if over_strategy:
        over = RandomOverSampler(
            sampling_strategy=over_strategy, random_state=RANDOM_STATE
        )
        X, y = over.fit_resample(X, y)

    balanced = pd.concat([X, y], axis=1)
    print(f"Balanced dataset: {len(balanced):,} rows")
    print(balanced[target].value_counts().to_string())
    return balanced
