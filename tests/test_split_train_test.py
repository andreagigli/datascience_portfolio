import pytest
from src.data.split_train_test import split_data
import pandas as pd
import numpy as np


def test_split_data_proportions():
    # Create a dummy dataset
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
    Y = pd.DataFrame(np.random.rand(100, 1))  # 100 samples, 1 target

    # Split the data into train and test sets
    X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data(X, Y, train_prc=80, test_prc=20, stratified=False)

    # Check expected outputs
    assert X_val is None
    assert Y_val is None
    assert cv_indices is not None
    assert len(X_train) / len(X) == 0.8
    assert len(X_test) / len(X) == 0.2


    # Split the data into train and test sets
    X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data(X, Y, train_prc=80, test_prc=20, stratified=True)

