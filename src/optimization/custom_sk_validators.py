import numpy as np

from sklearn.model_selection import BaseCrossValidator


class PredefinedSplit(BaseCrossValidator):
    """
    Custom cross-validator that uses predefined training and validation sets.

    Parameters:
    test_fold : array-like
        An array that specifies the split of the data into training and 
        validation set. It should contain 0 for samples that are part of the
        training set, and 1 for samples that are part of the validation set.

    Example:
    val_fold = [0] * len(train_data) + [1] * len(validation_data)
    X_combined = np.concatenate([train_data, validation_data])
    y_combined = np.concatenate([train_labels, validation_labels])
    ps = PredefinedSplit(test_fold=val_fold)
    estimator = RandomForestClassifier()
    param_distrib = {'n_estimators': randint(10, 100), 'max_depth': randint(10, 20)}
    search = RandomizedSearchCV(estimator, param_distrib, n_iter=5, cv=ps)
    search.fit(X_combined, y_combined)
    """
    def __init__(self, test_fold):
        self.test_fold = test_fold

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        # Ensure test_fold is an array for safe indexing
        test_fold = np.array(self.test_fold)
        # Indices for training and validation
        train_indices = np.where(test_fold == 0)[0]
        test_indices = np.where(test_fold == 1)[0]
        yield train_indices, test_indices


if __name__ == "__main__":
    from scipy.stats import randint
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    # Load a sample dataset and split it into train, validation, and test set
    iris = load_iris()
    X, y = iris.data, iris.target
    split_index = int(len(X) * 0.7)
    train_data, validation_data = X[:split_index], X[split_index:]
    train_labels, validation_labels = y[:split_index], y[split_index:]

    # Create an array where -1 indicates training samples, and 0 indicates validation samples
    val_fold = [0] * len(train_data) + [1] * len(validation_data)

    # Merge your training and validation set
    X_combined = np.concatenate([train_data, validation_data])
    y_combined = np.concatenate([train_labels, validation_labels])

    # Create the PredefinedSplit
    ps = PredefinedSplit(test_fold=val_fold)

    # Define a simple model and parameter grid for demonstration
    estimator = RandomForestClassifier()
    param_distrib = {'n_estimators': randint(10, 100), 'max_depth': randint(10, 20)}

    # GridSearchCV using the custom PredefinedSplit
    search = RandomizedSearchCV(estimator, param_distrib, n_iter=5, cv=ps)

    # Fit the grid search
    search.fit(X_combined, y_combined)

    # Print best parameters
    print("Best parameters:", search.best_params_)

    # For debug, verify that the computed split is as expected
    check_splits = list(ps.split(X_combined))
    assert np.all(np.array(val_fold)[check_splits[0][0]] == 0), "Error in the train split"
    assert np.all(np.array(val_fold)[check_splits[0][1]] == 1), "Error in the val split"
