from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.base import BaseEstimator

from src.evaluation.evaluation_misc import plot_confusion_matrix, calculate_metrics_clf, calculate_metrics_reg, \
    format_sklearn_estimator_info, plot_predictions_reg
from src.utils.my_dataframe import pprint_db
from src.utils.my_misc import pprint_string


def evaluate_exampledb(
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
    test_metrics = calculate_metrics_reg(Y, Y_pred)
    scores_dict = {key: test_metrics[key] for key in test_metrics}
    if Y_train is not None and Y_train_pred is not None:
        train_metrics = calculate_metrics_reg(Y_train, Y_train_pred)
        scores_dict.update({key + '_train': train_metrics[key] for key in train_metrics})

    # Convert scores_dict to DataFrame
    columns = ['y0'] if Y.ndim == 1 else ['y' + str(i) for i in range(Y.shape[1])]
    scores = pd.DataFrame.from_dict(scores_dict, orient="index")
    scores["Aggregated"] = scores.mean(axis=1)

    # Print model type and parameters
    pprint_string(format_sklearn_estimator_info(model), title="MODEL INFO")

    # Print evaluation scores
    pprint_db(scores, "EVALUATION METRICS FOR EXAMPLE DB")

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


def evaluate_gcrdb(
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
    scores = pd.DataFrame(scores_dict).T

    # Print model type and parameters
    pprint_string(format_sklearn_estimator_info(model), title="MODEL INFO")

    # Print evaluation scores
    pprint_db(scores, "EVALUATION METRICS FOR GCR DB")

    # Plot confusion matrices
    figs = {'Confusion Matrix - Test': plot_confusion_matrix(Y, Y_pred, "Test")}

    if Y_train is not None and Y_train_pred is not None:
        figs['Confusion Matrix - Train'] = plot_confusion_matrix(Y_train, Y_train_pred, "Train")

    return scores, figs


def evaluate_m5salesdb(Y: Union[np.ndarray, pd.DataFrame],
                       Y_pred: Union[np.ndarray, pd.DataFrame],
                       model: BaseEstimator,
                       target_name: Optional[List[str]] = None,
                       Y_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                       Y_train_pred: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                       *args, **kwargs) -> Tuple[DataFrame, Dict[str, Figure]]:
    """
    Evaluates a regression model's performance by comparing actual vs. predicted values across test, training,
    and optionally validation datasets. Generates various metrics and scatter plots for visualization of predictions.

    Args:
        Y (Union[np.ndarray, pd.DataFrame]): Actual target values for the test set.
        Y_pred (Union[np.ndarray, pd.DataFrame]): Predicted target values for the test set.
        model (BaseEstimator): The regression model used for predictions.
        target_name (Optional[List[str]]): List containing the name(s) of the target variable(s), used for plot labeling.
        Y_train (Optional[Union[np.ndarray, pd.DataFrame]]): Actual target values for the training set, if available.
        Y_train_pred (Optional[Union[np.ndarray, pd.DataFrame]]): Predicted target values for the training set, if available.

    Optional Keyword Args:
        Y_val (Optional[Union[np.ndarray, pd.DataFrame]]): Actual target values for the validation set, if available.
        Y_val_pred (Optional[Union[np.ndarray, pd.DataFrame]]): Predicted target values for the validation set, if available.

    Returns:
        scores (pd.DataFrame): A DataFrame containing evaluation metrics such as MAE, MSE, RMSE, NRMSE, and R^2
                               for test, training, and optionally validation datasets.
        figs (Dict[str, plt.Figure]): A dictionary of matplotlib Figures, including scatter plots of actual vs.
                                      predicted values for test, training, and optionally validation datasets.
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
        Y_train_pred = Y_train_pred.reset_index(drop=True) if isinstance(Y_train_pred,
                                                                         (pd.DataFrame, pd.Series)) else Y_train_pred

    if kwargs.get("Y_val") is not None and kwargs.get("Y_val_pred") is not None:
        kwargs["Y_val"] = kwargs.get("Y_val").squeeze()
        kwargs["Y_val_pred"] = kwargs.get("Y_val_pred").squeeze()
        kwargs["Y_val"] = kwargs["Y_val"].reset_index(drop=True) if isinstance(kwargs["Y_val"],
                                                                               (pd.DataFrame, pd.Series)) else kwargs[
            "Y_val"]
        kwargs["Y_val_pred"] = kwargs["Y_val_pred"].reset_index(drop=True) if isinstance(kwargs["Y_val_pred"],
                                                                                         (pd.DataFrame, pd.Series)) else \
            kwargs["Y_val_pred"]

    # Compute metrics
    test_metrics = calculate_metrics_reg(Y, Y_pred)
    scores_dict = {f"{key}_test": test_metrics[key] for key in test_metrics}
    if Y_train is not None and Y_train_pred is not None:
        train_metrics = calculate_metrics_reg(Y_train, Y_train_pred)
        scores_dict.update({f"{key}_train": train_metrics[key] for key in train_metrics})
    if kwargs.get("Y_val_pred") is not None:
        val_metrics = calculate_metrics_reg(kwargs["Y_val"], kwargs["Y_val_pred"])
        scores_dict.update({f"{key}_val": val_metrics[key] for key in val_metrics})

    # Convert scores_dict to DataFrame
    columns = ['y0'] if Y.ndim == 1 else ['y' + str(i) for i in range(Y.shape[1])]
    scores = pd.DataFrame.from_dict(scores_dict, orient="index", columns=columns)
    scores["Aggregated"] = scores.mean(axis=1)

    # Print model type and parameters
    pprint_string(format_sklearn_estimator_info(model), title="MODEL INFO")

    # Print evaluation scores
    pprint_db(scores, "EVALUATION METRICS FOR M5 SALES DB")

    # Handle target_name being None
    if target_name is None:  # It's assumed that Y is a numpy array
        target_name = ["y" + str(i) for i in range(Y.shape[1])] if Y.ndim > 1 else ["y"]

    # Generate scatter plot for test data
    figs = {}
    n_scatter_samples = 1000
    figs["Y_pred"] = plot_predictions_reg(Y, Y_pred, n_scatter_samples, scores, "Test", target_name)
    if Y_train is not None and Y_train_pred is not None:
        figs["Y_train_pred"] = plot_predictions_reg(Y_train, Y_train_pred, n_scatter_samples, scores, "Train",
                                                    target_name)
    if kwargs.get("Y_val") is not None and kwargs.get("Y_val_pred") is not None:
        figs["Y_val_pred"] = plot_predictions_reg(kwargs.get("Y_val"), kwargs.get("Y_val_pred"), n_scatter_samples,
                                                  scores, "Val", target_name)

    return scores, figs


def evaluate_passthrough(*args, **kwargs):
    return pd.DataFrame(), {}
