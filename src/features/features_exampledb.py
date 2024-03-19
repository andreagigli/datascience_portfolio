import pandas as pd

from typing import Tuple


def extract_features(X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Accepts features (X) and target (Y) data as separate DataFrames and returns them without modification.

    Args:
        X: Features data as a pandas DataFrame.
        Y: Target data as a pandas DataFrame.

    Returns:
        A tuple of (X, Y), the features and target data as provided.
    """
    return X, Y

