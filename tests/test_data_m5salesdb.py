import numpy as np
import pandas as pd
import pytest
from src.data.data_m5salesdb import subsample_items


@pytest.fixture
def toy_dataset():
    """
    A pytest fixture that provides a toy dataset for testing.

    This fixture generates a small dataset consisting of feature data `X_toy`, target data `Y_toy`,
    and cross-validation indices `cv_indices_toy`. It is designed for use in testing data processing
    functions, especially those that involve subsampling or cross-validation.

    Returns:
        tuple: A tuple containing three elements:
            - X_toy (pd.DataFrame): Feature data with 'item_id', 'feature1', and 'feature2' columns.
            - Y_toy (pd.DataFrame): Target data with a single 'target' binary column.
            - cv_indices_toy (list of tuples): A list where each tuple contains two numpy arrays
              representing the indices for training and testing in a cross-validation fold.

    Examples:
        To use the `toy_dataset` fixture in a test, define your test function to accept a parameter
        named after the fixture. pytest will invoke the fixture and pass its return value to your test:

        ```python
        def test_example(toy_dataset):
            X_toy, Y_toy, cv_indices_toy = toy_dataset
            # Your test code here
        ```
    """
    X_toy = pd.DataFrame({
        'item_id': [7, 4, 8, 5, 7, 3, 7, 8, 5, 4],
        'feature1': np.random.random(10),
        'feature2': np.random.random(10)
    })
    Y_toy = pd.DataFrame({
        'target': np.random.randint(0, 2, 10)  # Binary target variable
    })
    cv_indices_toy = [(np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])),
                      (np.array([5, 6, 7, 8, 9]), np.array([0, 1, 2, 3, 4]))]
    return X_toy, Y_toy, cv_indices_toy


def test_subsample_items(toy_dataset):
    """
    Tests the subsample_items function to ensure it correctly subsamples the dataset and adjusts CV indices.

    This test checks whether the subsample_items function from the data processing module
    correctly applies a specified subsampling rate to both feature and target datasets
    and adjusts the provided cross-validation indices accordingly. It uses a fixed random seed
    for reproducibility of the subsampling process.

    Args:
        toy_dataset (tuple): A fixture provided by pytest that returns a toy dataset,
                             including feature data (`X_toy`), target data (`Y_toy`),
                             and cross-validation indices (`cv_indices_toy`).

    Asserts:
        - The length of the subsampled feature dataset is less than or equal to 60% of the original dataset.
        - The lengths of the subsampled feature and target datasets are equal.
        - Each pair of training and testing indices in the adjusted CV indices is non-empty.
    """
    # Unpack the dataset
    X_toy, Y_toy, cv_indices_toy = toy_dataset

    # Given random seed for reproducibility
    random_seed = 42

    # When
    X_toy_subsampled, Y_toy_subsampled, cv_indices_subsampled = subsample_items(X_toy, Y_toy, cv_indices=cv_indices_toy,
                                                                                subsampling_rate=0.6,
                                                                                random_seed=random_seed)

    # Then
    assert len(X_toy_subsampled) <= len(X_toy) * 0.6  # Check if subsampling rate is respected
    assert len(X_toy_subsampled) == len(Y_toy_subsampled)  # Check if X and Y lengths match
    # Check if subsampled CV indices are valid
    for train_idx, test_idx in cv_indices_subsampled:
        assert len(train_idx) > 0 and len(test_idx) > 0  # Ensures we have non-empty train and test indices
