import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class CustomModel(BaseEstimator, RegressorMixin):
    def __init__(self, custom_parameter: float = 1.0) -> None:
        """
        Initialize the CustomModel model.

        Args:
            custom_parameter: A custom parameter specific to our custom model.
        """
        self.custom_parameter = custom_parameter
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomModel":
        """
        Fit the custom linear regression model to the data.

        Args:
            X: Training data features.
            y: Target values.

        Returns:
            self: The fitted model.
        """
        # Custom preprocessing or modifications can be done here
        modified_X = X * self.custom_parameter

        # Create an internal scikit-learn model (e.g., LinearRegression) and fit it
        self.model = LinearRegression()
        self.model.fit(modified_X, y)
        return self

    def predict(self, X: np.ndarray):
        """
        Make predictions using the custom linear regression model.

        Args:
            X: Data for which predictions should be made.

        Returns:
            predictions: Predicted values.
        """
        # Custom post-processing or modifications can be done here
        modified_X = X * self.custom_parameter

        # Use the internal scikit-learn model to make predictions
        predictions = self.model.predict(modified_X)
        return predictions
