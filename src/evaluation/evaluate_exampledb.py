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


def calculate_metrics(Y_actual: np.ndarray, Y_predicted: np.ndarray, *args, **kwargs) -> Dict[str, np.ndarray]:
    """
    Calculates various regression metrics to evaluate the performance of a model.

    Args:
        Y_actual (np.ndarray): Actual target values.
        Y_predicted (np.ndarray): Predicted target values by the model.

    Returns:
        metrics (Dict[str, np.ndarray]): A dictionary containing the calculated metrics, including Mean Absolute Error (MAE),
                                         Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Normalized RMSE (NRMSE),
                                         and R^2 Score (R2), each keyed by their respective names.
    """
    mae = mean_absolute_error(Y_actual, Y_predicted, multioutput="raw_values")
    mse = mean_squared_error(Y_actual, Y_predicted, multioutput="raw_values")
    rmse = np.sqrt(mse)
    nrmse = np.divide(rmse, iqr(Y_predicted, axis=0))
    r2 = r2_score(Y_actual, Y_predicted, multioutput="raw_values")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R2': r2}


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

    # Compute metrics
    test_metrics = calculate_metrics(Y, Y_pred)
    scores_dict = {key: test_metrics[key] for key in test_metrics}
    if Y_train is not None and Y_train_pred is not None:
        train_metrics = calculate_metrics(Y_train, Y_train_pred)
        scores_dict.update({key + '_train': train_metrics[key] for key in train_metrics})

    # Convert scores_dict to DataFrame
    columns = ['y0'] if Y.ndim == 1 else ['y' + str(i) for i in range(Y.shape[1])]
    scores = pd.DataFrame.from_dict(scores_dict, orient="index")
    scores["Aggregated"] = scores.mean(axis=1)

    # Generate scatter plot for test data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y, y=Y_pred, alpha=0.6)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', lw=2)  # Line for perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test: Actual vs Predicted {target_name[0]}')
    figs = {'scatter_plot': plt.gcf()}

    # Generate scatter plot for training data if provided
    if Y_train is not None and Y_train_pred is not None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=Y_train, y=Y_train_pred, alpha=0.6)
        plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], color='red', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Train: Actual vs Predicted {target_name[0]}')
        figs['scatter_plot_train'] = plt.gcf()

    return scores, figs