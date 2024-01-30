from typing import Tuple

from pandas import DataFrame


def extract_features(X: DataFrame, Y: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Takes dataframes X and Y and returns two dataframes X and Y."""
    return X, Y
