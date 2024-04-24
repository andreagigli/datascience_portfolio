from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame
from scipy.stats import iqr
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(
    Y: np.ndarray, 
    Y_pred: np.ndarray, 
    model: BaseEstimator, 
    target_name: List[str], 
    Y_train: Optional[np.ndarray] = None, 
    Y_train_pred: Optional[np.ndarray] = None
) -> Tuple[DataFrame, Dict[str, Figure]]:
    """
    Evaluates a regression model's performance, comparing actual vs. predicted values, and generates plots for visualization.

    Args:
        Y (np.ndarray): Actual target values for the test set.
        Y_pred (np.ndarray): Predicted target values for the test set.
        model (BaseEstimator): The regression model used for predictions.
        target_name (List[str]): List containing the name(s) of the target variable(s).
        Y_train (Optional[np.ndarray]): Actual target values for the training set, if available.
        Y_train_pred (Optional[np.ndarray]): Predicted target values for the training set, if available.

    Returns:
        scores (DataFrame): A DataFrame containing evaluation metrics such as MAE, MSE, RMSE, NRMSE, and R2,
                            for both test and optionally for the training dataset.
        figs (Dict[str, Figure]): A dictionary of matplotlib Figures, including scatter plots of actual vs.
                                  predicted values for the test set and, if provided, for the training set.
    """
    scores, figs = None

    return scores, figs