import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class CustomModel(BaseEstimator, RegressorMixin):
    def __init__(self, custom_parameter: float = 1.0) -> None:
        """
        Initializes the CustomModel with a specific parameter.

        Args:
            custom_parameter (float): A custom parameter influencing the model's behavior.
        """
        self.custom_parameter = custom_parameter
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomModel":
        """
        Fits the custom model to the training data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data target values.

        Returns:
            self (CustomModel): The instance of this CustomModel, fitted to the data.
        """
        # Custom preprocessing or modifications can be done here
        modified_X = X * self.custom_parameter

        # Create an internal scikit-learn model (e.g., LinearRegression) and fit it
        self.model = LinearRegression()
        self.model.fit(modified_X, y)
        return self

    def predict(self, X: np.ndarray):
        """
        Predicts target values using the fitted model for given features.

        Args:
            X (np.ndarray): Data features for which predictions are to be made.

        Returns:
            predictions (np.ndarray): Predicted target values.
        """
        # Custom post-processing or modifications can be done here
        modified_X = X * self.custom_parameter

        # Use the internal scikit-learn model to make predictions
        predictions = self.model.predict(modified_X)
        return predictions
