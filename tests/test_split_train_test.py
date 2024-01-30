import pytest
from src.data.split_train_test import split_data
import pandas as pd
import numpy as np

# Fixture for creating a dummy dataset
@pytest.fixture
def create_dummy_data():
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(100, 10))  # 100 samples, 10 features
    Y = pd.DataFrame(np.random.rand(100, 1))  # 100 samples, 1 target
    return X, Y

# Basic test using fixture
def test_split_data_basic(create_dummy_data):
    X, Y = create_dummy_data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, cv_indices = split_data(X, Y, train_prc=70, test_prc=30, stratified=False)

    assert len(X_train) == 70
    assert len(X_test) == 30
    assert X_val is None
    assert Y_val is None
    assert cv_indices is None

# Parametrized test
@pytest.mark.parametrize("train_prc, test_prc, expected_train_size, expected_test_size", [
    (80, 20, 80, 20),
    (75, 25, 75, 25),
    (90, 10, 90, 10),
])
def test_split_data_parametrized(create_dummy_data, train_prc, test_prc, expected_train_size, expected_test_size):
    X, Y = create_dummy_data
    (X_train, Y_train,
     X_val, Y_val,
     X_test, Y_test,
     cv_indices) = split_data(X, Y, train_prc=train_prc, test_prc=test_prc, stratified=False)

    assert len(X_train) == expected_train_size
    assert len(X_test) == expected_test_size
    assert X_val is None
    assert Y_val is None