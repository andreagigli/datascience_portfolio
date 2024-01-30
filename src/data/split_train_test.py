from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def split_data(X, Y, random_seed=None, train_prc=80, test_prc=20, n_folds=None, stratified=False):
    """
    Splits the dataset into training and test sets and provides cross-validation indices for k-fold cross-validation.

    Parameters:
    X (pd.DataFrame): DataFrame containing the features.
    Y (pd.DataFrame): DataFrame containing the target variable.
    n_splits (int): Number of folds for k-fold cross-validation.
    random_seed (int): Random seed for reproducibility.
    stratified (bool): Whether to perform stratified k-fold (only for classification).

    Returns:
    X_train, Y_train (pd.DataFrame): Training data.
    X_test, Y_test (pd.DataFrame): Test data.
    X_val, Y_val (None): Validation data is not explicitly returned in k-fold cross-validation. Only added for consistency across split_data functions.
    cv_indices (list of tuples): Indices for k-fold cross-validation.
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