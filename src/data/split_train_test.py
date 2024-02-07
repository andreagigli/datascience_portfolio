from typing import Optional, Tuple, List

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def split_data(
    X: DataFrame,
    Y: DataFrame,
    random_seed: Optional[int] = None,
    train_prc: int = 80,
    test_prc: int = 20,
    n_folds: Optional[int] = None,
    stratified: bool = False
) -> Tuple[DataFrame, DataFrame, Optional[DataFrame], Optional[DataFrame], DataFrame, DataFrame, Optional[List[Tuple[np.ndarray, np.ndarray]]]]:
    """
    Splits the dataset into training, validation (if applicable), and test sets, optionally providing cross-validation indices.

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (DataFrame): DataFrame containing the target variable.
        random_seed (Optional[int]): Seed for random number generator to ensure reproducibility.
        train_prc (int): Percentage of the dataset to include in the training set.
        test_prc (int): Percentage of the dataset to include in the test set.
        n_folds (Optional[int]): Number of folds for k-fold cross-validation, if applicable.
        stratified (bool): Whether to perform stratified k-fold cross-validation, relevant for classification tasks.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (DataFrame): Training set target variable.
        X_val (Optional[DataFrame]): Validation set features, if `n_folds` is None (indicative of a hold-out validation set).
        Y_val (Optional[DataFrame]): Validation set target variable, if `n_folds` is None.
        X_test (DataFrame): Test set features.
        Y_test (DataFrame): Test set target variable.
        cv_indices (Optional[List[Tuple[np.ndarray, np.ndarray]]]): List of tuples containing train and validation indices for each fold, if `n_folds` is not None. Useful for cross-validation.
    """
    assert train_prc + test_prc == 100, "Sum of ratios must be 100"

    # Splitting the dataset
    train_size = train_prc / 100
    test_size = test_prc / 100

    # Splitting the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=random_seed, shuffle=True)

    # Creating cross-validation indices
    if n_folds is not None:
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        cv_indices = [(train_idx, val_idx) for train_idx, val_idx in kf.split(X_train, Y_train)]
    else:
        cv_indices = None

    X_val = None
    Y_val = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices
