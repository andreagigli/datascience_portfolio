def split_data(X, Y):
    """Takes dataframes X and Y and returns training and test dataframes, as well as cross-validation indices.
     The cross validation indices are expressed as a list of tuples of lists."""
    X_train, Y_train, X_test, Y_test, cv_indices = None
    return X_train, Y_train, X_test, Y_test, cv_indices