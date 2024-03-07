from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from numpy import ptp
from pandas import DataFrame
from scipy.stats import iqr
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def calculate_metrics(Y_actual: np.ndarray, Y_predicted: np.ndarray) -> Dict[str, np.ndarray]:
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
    Y_actual = Y_actual.squeeze()
    Y_predicted = Y_predicted.squeeze()

    mae = mean_absolute_error(Y_actual, Y_predicted, multioutput="raw_values")
    mse = mean_squared_error(Y_actual, Y_predicted, multioutput="raw_values")
    rmse = np.sqrt(mse)
    nrmse = np.divide(rmse, ptp(Y_actual, axis=0))
    r2 = r2_score(Y_actual, Y_predicted, multioutput="raw_values")

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R2': r2}


def evaluate(Y: Union[np.ndarray, pd.DataFrame],
             Y_pred: Union[np.ndarray, pd.DataFrame],
             model: BaseEstimator,
             target_name: Optional[List[str]] = None,
             Y_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
             Y_train_pred: Optional[Union[np.ndarray, pd.DataFrame]] = None,
             ) -> Tuple[DataFrame, Dict[str, Figure]]:
    """
    Evaluates a regression model's performance, comparing actual vs. predicted values, and generates plots for visualization.

    Args:
        Y (Union[np.ndarray, pd.DataFrame]): Actual target values for the test set.
        Y_pred (Union[np.ndarray, pd.DataFrame]): Predicted target values for the test set.
        model (BaseEstimator): The regression model used for predictions.
        target_name (Optional[List[str]]): List containing the name(s) of the target variable(s).
        Y_train (Optional[Union[np.ndarray, pd.DataFrame]]): Actual target values for the training set, if available.
        Y_train_pred (Optional[Union[np.ndarray, pd.DataFrame]]): Predicted target values for the training set, if available.

    Returns:
        scores (DataFrame): A DataFrame containing evaluation metrics such as MAE, MSE, RMSE, NRMSE, and R2,
                            for both test and optionally for the training dataset.
        figs (Dict[str, plt.Figure]): A dictionary of matplotlib Figures, including scatter plots of actual vs.
                                  predicted values for the test set and, if provided, for the training set.
    """
    # Remove potentially redundant dimensions from the data
    Y = Y.squeeze()
    Y_pred = Y_pred.squeeze()
    # If data is stored in Pandas DataFrames make sure the index is contiguous
    Y = Y.reset_index(drop=True) if isinstance(Y, (pd.DataFrame, pd.Series)) else Y
    Y_pred = Y_pred.reset_index(drop=True) if isinstance(Y_pred, (pd.DataFrame, pd.Series)) else Y_pred

    if Y_train is not None and Y_train_pred is not None:
        Y_train = Y_train.squeeze()
        Y_train_pred = Y_train_pred.squeeze()
        Y_train = Y_train.reset_index(drop=True) if isinstance(Y_train, (pd.DataFrame, pd.Series)) else Y_train
        Y_train_pred = Y_train_pred.reset_index(drop=True) if isinstance(Y_train_pred, (pd.DataFrame, pd.Series)) else Y_train_pred

    # Print model type and parameters
    model_info = format_sklearn_estimator_info(model)
    print(model_info)

    # Compute metrics
    test_metrics = calculate_metrics(Y, Y_pred)
    scores_dict = {key: test_metrics[key] for key in test_metrics}
    if Y_train is not None and Y_train_pred is not None:
        train_metrics = calculate_metrics(Y_train, Y_train_pred)
        scores_dict.update({key + '_train': train_metrics[key] for key in train_metrics})

    # Convert scores_dict to DataFrame
    columns = ['y0'] if Y.ndim == 1 else ['y' + str(i) for i in range(Y.shape[1])]
    scores = pd.DataFrame.from_dict(scores_dict, orient="index", columns=columns)
    scores["Aggregated"] = scores.mean(axis=1)
    print(scores)

    # Handle target_name being None
    if target_name is None:  # It's assumed that Y is a numpy array
        target_name = ["y" + str(i) for i in range(Y.shape[1])] if Y.ndim > 1 else ["y"]

    # Generate scatter plot for test data
    n_scatter_samples = 100

    if len(Y) > n_scatter_samples:  # Limit the number of plotted datapoints
        indices = np.random.choice(range(len(Y)), size=n_scatter_samples, replace=False)
        Y_sampled = Y[indices]  # Assuming Y is a pandas Series or DataFrame
        Y_pred_sampled = Y_pred[indices]
    else:
        Y_sampled = Y
        Y_pred_sampled = Y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_sampled, y=Y_pred_sampled, alpha=0.6)
    plt.plot([Y_sampled.min(), Y_sampled.max()], [Y_sampled.min(), Y_sampled.max()], color='red', lw=2)  # Line for perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test: Actual vs Predicted {target_name[0]}. R2={scores.loc["R2", "Aggregated"]:.2f}')
    figs = {'scatter_plot': plt.gcf()}

    # Generate scatter plot for training data if provided
    if Y_train is not None and Y_train_pred is not None:
        if len(Y_train) > n_scatter_samples:  # Limit the number of plotted datapoints
            indices = np.random.choice(range(len(Y_train)), size=n_scatter_samples, replace=False)
            Y_train_sampled = Y_train[indices]  # Assuming Y is a pandas Series or DataFrame
            Y_train_pred_sampled = Y_train_pred[indices]
        else:
            Y_train_sampled = Y_train
            Y_train_pred_sampled = Y_train_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=Y_train_sampled, y=Y_train_pred_sampled, alpha=0.6)
        plt.plot([Y_train_sampled.min(), Y_train_sampled.max()], [Y_train_sampled.min(), Y_train_sampled.max()], color='red', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Train: Actual vs Predicted {target_name[0]}. R2={scores.loc["R2_train", "Aggregated"]:.2f}')
        figs['scatter_plot_train'] = plt.gcf()

    return scores, figs


def format_sklearn_estimator_info(estimator: BaseEstimator) -> str:
    """
    Formats detailed information about an estimator or pipeline, including its components and parameters,
    for better readability. This function supports both individual estimators and pipelines consisting of
    multiple steps (transformers and a final estimator).

    Args:
        estimator (BaseEstimator): The estimator or pipeline to format information for.

    Returns:
        str: A formatted string containing detailed information about the estimator or pipeline, including
             type, components (if any), and parameters for each component.
    """
    info_lines = []

    # Check if the estimator is a pipeline
    if isinstance(estimator, Pipeline):
        info_lines.append("Estimator Type: Pipeline")
        info_lines.append("Steps:")
        # Iterate through each step in the pipeline
        for step_name, step_estimator in estimator.steps:
            step_info = format_step_info(step_name, step_estimator)
            info_lines.append(step_info)
    else:
        # For a single estimator (not a pipeline)
        info_lines.append(format_step_info(estimator.__class__.__name__, estimator))

    return "\n".join(info_lines)


def format_step_info(step_name: str, step_model: BaseEstimator) -> str:
    """
    Formats the information about a single step in the pipeline or a single model.

    Args:
        step_name (str): The name of the step or model.
        step_model (BaseEstimator): The model or transformer in the step.

    Returns:
        str: Formatted string containing the step or model name and parameters.
    """
    step_type = f"  Step: {step_name}, Model: {step_model.__class__.__name__}"
    param_lines = ["    Parameters:"]
    params = step_model.get_params(deep=True)

    # Format each parameter with indentation
    for param, value in params.items():
        if isinstance(value, BaseEstimator):
            value_str = f"{value.__class__.__name__}(...)"
        else:
            value_str = repr(value)
        param_lines.append(f"      {param}: {value_str}")

    return "\n".join([step_type] + param_lines)
