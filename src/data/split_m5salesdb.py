from pandas import DataFrame, Series
from typing import Union, Tuple, Optional


def split_data(X: DataFrame, Y: Union[DataFrame, Series]) -> Tuple[DataFrame,
Union[DataFrame, Series],
DataFrame,
Union[DataFrame, Series],
DataFrame,
Union[DataFrame, Series],
Optional[None]
]:
    """
    Splits the dataset into training, validation, and test sets based on the sample's day.
    The training set includes data from day 1 until 139, the validation set data from

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (Series or DataFrame): DataFrame containing the target variable.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (Series or DataFrame): Training set target variable.
        X_val (DataFrame): Validation set features.
        Y_val (Series or DataFrame): Validation set target variable.
        X_test (DataFrame): Test set features.
        Y_test (Series or DataFrame): Test set target variable.
        cv_indices (None): Placeholder for cross-validation indices, indicating no cross-validation indices are provided in this function.
    """
    idx_train = X["d"] < 1912
    idx_val = (X["d"] >= 1912) & (X["d"] < 1941)
    idx_test = X["d"] >= 1941

    X_train = X.loc[idx_train].reset_index(drop=True)
    Y_train = Y.loc[idx_train].reset_index(drop=True)
    X_val = X.loc[idx_val].reset_index(drop=True)
    Y_val = Y.loc[idx_val].reset_index(drop=True)
    X_test = X.loc[idx_test].reset_index(drop=True)
    Y_test = Y.loc[idx_test].reset_index(drop=True)

    cv_indices = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices
