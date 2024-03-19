from typing import Optional, Dict, Tuple, List

from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def split_data(X: DataFrame,
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

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size, random_state=random_seed, shuffle=True)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_size / (test_size + val_size), random_state=random_seed)

    cv_indices = None

    aux_split_params = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices, aux_split_params