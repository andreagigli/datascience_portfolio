
def split_data(X, Y):
    """
    Splits the dataset into training, validation, and test sets based on the sample's day.
    The training set includes data from day 1 until 139, the validation set data from

    Args:
        X (DataFrame): DataFrame containing the features.
        Y (DataFrame): DataFrame containing the target variable.

    Returns:
        X_train (DataFrame): Training set features.
        Y_train (DataFrame): Training set target variable.
        X_val (DataFrame): Validation set features.
        Y_val (DataFrame): Validation set target variable.
        X_test (DataFrame): Test set features.
        Y_test (DataFrame): Test set target variable.
        cv_indices (None): Placeholder for cross-validation indices, indicating no cross-validation indices are provided in this function.
    """
    idx_train = X["d"] < 1912
    idx_val = 1912 <= X["d"] < 1941
    idx_test = X["d"] >= 1941

    X_train = X.loc[idx_train, :]
    Y_train = Y.loc[idx_train, :]
    X_val = X.loc[idx_val, :]
    Y_val = Y.loc[idx_val, :]
    X_test = X.loc[idx_test, :]
    Y_test = Y.loc[idx_test, :]

    cv_indices = None

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices