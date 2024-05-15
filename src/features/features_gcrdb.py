import warnings
from typing import Tuple

import numpy as np
import pandas as pd


def extract_features(data: pd.DataFrame, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract features and target variable from the given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing the features and target variable.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the features (X) and the target variable (Y).
            - X (pd.DataFrame): Design matrix (n_samples, n_features).
            - Y (pd.DataFrame): Series containing the target variable 'Good risk'.
    """
    Y = data["good_risk"]
    X = data.drop(columns=["good_risk"])

    return X, Y
