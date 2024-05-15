import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from typing import Optional, Tuple, List, Dict


def split_data_m5salesdb(X: pd.DataFrame,
                         Y: pd.DataFrame,
                         # look_back_days: int = 365
                         **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame,
                                            pd.DataFrame, pd.DataFrame,
                                            Optional[Dict[str, any]],
                                            Optional[Dict[str, any]]]:
    """
    Splits the dataset into training, validation, and an extended test set to accommodate sequential
    prediction needs. The training data covers days 1 - 1911, the validation set 1912 - 1940, the test set 1941 - 1969.
    The first day of the test set (1941) includes the number of items sold that day (in "sold"), whereas the following
    days will contain a zero by default, as that value must be predicted.

    The training set is used for model training, the validation set for model tuning and validation, and
    the extended test set for conducting sequential predictions where each day's forecast can be informed
    by the actual or predicted sales data from preceding days.

    Because the multi-day prediction problem is to be solved in a sequential way, the test set is extended backward to
    include a specified number of look-back days to enable feature computation for days 1942 - 1969, that depend on the
    model prediction of "sold" and on historical values of "sold" (such as lagged or rolling window features).

    Args:
        X (pd.DataFrame): DataFrame containing the features, with 'd' indicating the day.
        Y (pd.DataFrame): DataFrame containing the target variable, aligned with X.
        **kwargs:
            look_back_days (int): Number of days prior to the test period to include in the extended test set for feature computation.

    Returns:
        X_train (pd.DataFrame): Features for the training set.
        Y_train (pd.DataFrame): Target variable for the training set.
        X_val (pd.DataFrame): Features for the validation set.
        Y_val (pd.DataFrame): Target variable for the validation set.
        X_test_extended (pd.DataFrame): Features for the extended test set, including the look-back period for feature computation and the actual prediction period.
        Y_test (pd.DataFrame): Target variable for the actual test period, not including the look-back days.
        aux_split_params (Optional[Dict[str, any]]): Additional parameters like 'start_day_for_prediction' that may be useful for prediction.
    """
    # Checking required arguments
    look_back_days = kwargs.get('look_back_days_sequential_prediction', None)
    if look_back_days is None or look_back_days < 0:
        raise ValueError("look_back_days_sequential_prediction must be provided (>=0).")

    # Note: 'Y' is assumed to have a multi-index of ('id', 'd') and a column 'sold_next_day'. Resetting the index is not necessary
    X_train = X.loc[(slice(None), slice(None, 1911)), :]
    Y_train = Y.loc[(slice(None), slice(None, 1911)), :]
    X_val = X.loc[(slice(None), slice(1912, 1940)), :]
    Y_val = Y.loc[(slice(None), slice(1912, 1940)), :]
    X_test_extended = X.loc[(slice(None), slice(1941 - look_back_days, None)),
                      :]  # Extend the test set back by `look_back_days`
    Y_test = Y.loc[(slice(None), slice(1941, None)), :]

    cv_indices = None

    # Prepare optional returns with additional information
    aux_split_params = {
        'start_day_for_prediction': 1941,  # Day from which "actual" predictions start, after the look-back period
        "X_val": X_val,
        "Y_val": Y_val,
    }

    return X_train, Y_train, X_val, Y_val, X_test_extended, Y_test, cv_indices, aux_split_params


def split_data_train_test(
        X: DataFrame,
        Y: DataFrame,
        **kwargs,
) -> Tuple[DataFrame, DataFrame,
Optional[DataFrame], Optional[DataFrame],
DataFrame, DataFrame,
Optional[List[Tuple[ndarray, ndarray]]],
Optional[Dict[str, any]]]:
    """
    Splits the dataset into training, validation (if applicable), and test sets, optionally providing cross-validation indices.

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (DataFrame): DataFrame containing the target variable.
        **kwargs: Additional keyword arguments including:
            - random_seed (Optional[int]): Seed for random number generator to ensure reproducibility.
            - train_prc (int): Percentage of the dataset to include in the training set. Defaults to 80.
            - test_prc (int): Percentage of the dataset to include in the test set. Defaults to 20.
            - n_folds (Optional[int]): Number of folds for k-fold cross-validation, if applicable. None by default.
            - stratified (bool): Whether to perform stratified k-fold cross-validation, relevant for classification tasks. Defaults to False.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (DataFrame): Training set target variable.
        X_val (Optional[DataFrame]): Validation set features, if `n_folds` is None (indicative of a hold-out validation set). None if not applicable.
        Y_val (Optional[DataFrame]): Validation set target variable, if `n_folds` is None. None if not applicable.
        X_test (DataFrame): Test set features.
        Y_test (DataFrame): Test set target variable.
        cv_indices (Optional[List[Tuple[np.ndarray, np.ndarray]]]): List of tuples containing train and validation indices for each fold, if `n_folds` is not None. Useful for cross-validation.
        aux_split_params (Optional[Dict[str, Any]]): An empty dictionary, included for conformity with other splitting functions that may return additional optional parameters.
    """
    # Retrieve arguments from kwargs or assign default values
    random_seed = kwargs.get('random_seed', None)
    train_prc = kwargs.get('train_prc', 80)
    test_prc = kwargs.get('test_prc', 20)
    n_folds = kwargs.get('n_folds', None)
    stratified = kwargs.get('stratified', False)

    # Verify the consistency of the given split percentages
    assert train_prc + test_prc == 100, "Sum of ratios must be 100"

    # Splitting the dataset
    train_size = train_prc / 100
    test_size = test_prc / 100

    # Splitting the dataset into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=random_seed,
                                                        shuffle=True)

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

    aux_split_params = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices, aux_split_params


def split_data_train_val_test(X: DataFrame,
                              Y: DataFrame,
                              **kwargs,
                              ) -> Tuple[DataFrame, DataFrame,
Optional[DataFrame], Optional[DataFrame],
DataFrame, DataFrame,
Optional[List[Tuple[ndarray, ndarray]]],
Optional[Dict[str, any]]]:
    """
    Splits the dataset into training, validation, and test sets based on specified percentages.

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (DataFrame): DataFrame containing the target variable.
        **kwargs:
            train_prc (int): Percentage of the dataset to include in the training set. Default is 70.
            val_prc (int): Percentage of the dataset to include in the validation set. Default is 15.
            test_prc (int): Percentage of the dataset to include in the test set. Default is 15.
            random_seed (Optional[int]): Seed for random number generator to ensure reproducibility. Default is None.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (DataFrame): Training set target variable.
        X_val (DataFrame): Validation set features.
        Y_val (DataFrame): Validation set target variable.
        X_test (DataFrame): Test set features.
        Y_test (DataFrame): Test set target variable.
        cv_indices (None): Placeholder for cross-validation indices, indicating no cross-validation indices are provided in this function.
        aux_split_params (Dict[str, any]): An empty dictionary, included for conformity with other splitting functions that may return additional optional parameters.
    """
    # Retrieve arguments from kwargs or assign default values
    random_seed = kwargs.get('random_seed', None)
    train_prc = kwargs.get('train_prc', 70)
    val_prc = kwargs.get('val_prc', 15)
    test_prc = kwargs.get('test_prc', 15)

    # Verify the consistency of the given split percentages
    assert train_prc + val_prc + test_prc == 100, "Sum of percentages must be 100"

    # Splitting the dataset
    train_size = train_prc / 100
    val_size = val_prc / 100
    test_size = test_prc / 100

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size, random_state=random_seed,
                                                        shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size / (test_size + val_size),
                                                    random_state=random_seed)

    cv_indices = None

    aux_split_params = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices, aux_split_params


def split_data_passthrough(*args, **kwargs):
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], {}
