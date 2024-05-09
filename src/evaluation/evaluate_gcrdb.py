import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.base import BaseEstimator
from typing import Dict, List, Optional, Tuple, Union

from src.evaluation.evaluate_misc import plot_confusion_matrix, calculate_metrics_clf


def evaluate(
        Y: np.ndarray,
        Y_pred: np.ndarray,
        model: BaseEstimator,
        target_name: Optional[List[str]] = None,
        Y_train: Optional[np.ndarray] = None,
        Y_train_pred: Optional[np.ndarray] = None
) -> Tuple[DataFrame, Dict[str, Figure]]:
    """
    Evaluates a binary classification model's performance, comparing actual vs. predicted values, and generates
    plots for visualization.

    Args:
        Y (np.ndarray): Actual target values for the test set.
        Y_pred (np.ndarray): Predicted target values for the test set.
        model (BaseEstimator): The classification model used for predictions.
        target_name (Optional[List[str]]): List containing the name(s) of the target variable(s).
        Y_train (Optional[np.ndarray]): Actual target values for the training set, if available.
        Y_train_pred (Optional[np.ndarray]): Predicted target values for the training set, if available.

    Returns:
        scores (DataFrame): A DataFrame containing evaluation metrics such as Accuracy, Precision, Recall, F1 Score,
                            ROC AUC, for both test and optionally for the training dataset.
        figs (Dict[str, Figure]): A dictionary of matplotlib Figures, including confusion matrices for the test set
                                  and, if provided, for the training set.
    """
    Y = Y.squeeze()
    Y_pred = Y_pred.squeeze()

    if Y_train is not None and Y_train_pred is not None:
        Y_train = Y_train.squeeze()
        Y_train_pred = Y_train_pred.squeeze()

    # Compute metrics
    test_metrics = calculate_metrics_clf(Y, Y_pred)
    scores_dict = {f"{key}_test": [test_metrics[key]] for key in test_metrics}

    if Y_train is not None and Y_train_pred is not None:
        train_metrics = calculate_metrics_clf(Y_train, Y_train_pred)
        for key in train_metrics:
            scores_dict[f"{key}_train"] = [train_metrics[key]]

    # Convert scores_dict to DataFrame
    scores = pd.DataFrame(scores_dict)

    # Plot confusion matrices
    figs = {'Confusion Matrix - Test': plot_confusion_matrix(Y, Y_pred, "Test")}

    if Y_train is not None and Y_train_pred is not None:
        figs['Confusion Matrix - Train'] = plot_confusion_matrix(Y_train, Y_train_pred, "Train")

    return scores, figs
