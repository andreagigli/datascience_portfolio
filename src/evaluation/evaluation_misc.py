from typing import Dict

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import iqr
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

COLORMAPS = {
    'sequential': 'viridis',
    'diverging': 'coolwarm',
    'categorical': 'colorblind'
}


def calculate_metrics_clf(Y_actual: np.ndarray, Y_predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculates various classification metrics to evaluate the performance of a classifier. Includes the baseline ROC_AUC
    score using a DummyClassifier with stratified strategy.

    Args:
        Y_actual (np.ndarray): Actual target values.
        Y_predicted (np.ndarray): Predicted target values by the model.

    Returns:
        metrics (Dict[str, float]): A dictionary containing the calculated metrics, including Accuracy,
                                    Precision, Recall, F1 Score, ROC AUC Score.
    """
    # Calculate baseline ROC AUC using a DummyClassifier
    dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
    dummy_clf.fit(np.zeros_like(Y_actual).reshape(-1, 1), Y_actual)
    Y_dummy_pred = dummy_clf.predict(np.zeros_like(Y_actual).reshape(-1, 1))

    if len(np.unique(Y_actual)) > 2:
        average = 'macro'
        roc_auc = roc_auc_score(Y_actual, Y_predicted, multi_class='ovr')
        bsl_roc_auc = roc_auc_score(Y_actual, Y_dummy_pred, multi_class='ovr')
    else:
        average = 'binary'
        roc_auc = roc_auc_score(Y_actual, Y_predicted)
        bsl_roc_auc = roc_auc_score(Y_actual, Y_dummy_pred)

    metrics = {
        'Accuracy': accuracy_score(Y_actual, Y_predicted),
        'Precision': precision_score(Y_actual, Y_predicted, average=average),
        'Recall': recall_score(Y_actual, Y_predicted, average=average),
        'F1': f1_score(Y_actual, Y_predicted, average=average),
        'ROC_AUC': roc_auc,
        'BSL_ROC_AUC': bsl_roc_auc,
    }

    return metrics


def calculate_metrics_reg(Y_actual: np.ndarray, Y_predicted: np.ndarray, *args, **kwargs) -> Dict[str, np.ndarray]:
    """
    Calculates various regression metrics to evaluate the performance of a model. Includes the baseline R^2 score using
    a DummyRegressor with mean strategy.

    Args:
        Y_actual (np.ndarray): Actual target values.
        Y_predicted (np.ndarray): Predicted target values by the model.

    Returns:
        metrics (Dict[str, np.ndarray]): A dictionary containing the calculated metrics, including Mean Absolute Error (MAE),
                                         Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Normalized RMSE (NRMSE),
                                         and R^2 Score (R2), each keyed by their respective names.
    """
    # Calculate baseline R^2 using a DummyRegressor
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(np.zeros((len(Y_actual), 1)), Y_actual)  # Dummy regressor doesn't use input features
    Y_dummy_pred = dummy_reg.predict(np.zeros((len(Y_actual), 1)))

    mae = mean_absolute_error(Y_actual, Y_predicted, multioutput="raw_values")
    mse = mean_squared_error(Y_actual, Y_predicted, multioutput="raw_values")
    rmse = np.sqrt(mse)
    nrmse = np.divide(rmse, iqr(Y_predicted, axis=0))
    r2 = r2_score(Y_actual, Y_predicted, multioutput="raw_values")
    bsl_r2 = r2_score(Y_actual, Y_dummy_pred, multioutput="raw_values")

    metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'NRMSE': nrmse, 'R2': r2, 'BSL_R2': bsl_r2}

    return metrics


def format_pipeline_step_info(step_name: str, step_model: BaseEstimator) -> str:
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
    model_info = []

    # Check if the estimator is a pipeline
    if isinstance(estimator, Pipeline):
        model_info.append("Estimator Type: Pipeline")
        model_info.append("Steps:")
        # Iterate through each step in the pipeline
        for step_name, step_estimator in estimator.steps:
            step_info = format_pipeline_step_info(step_name, step_estimator)
            model_info.append(step_info)
    else:
        # For a single estimator (not a pipeline)
        model_info.append(format_pipeline_step_info(estimator.__class__.__name__, estimator))

    model_info = "\n".join(model_info)

    return model_info


def plot_confusion_matrix(Y_actual: np.ndarray, Y_predicted: np.ndarray, dataset_split_name: str) -> Figure:
    """
    Plots the confusion matrix for the given actual and predicted values.

    Args:
        Y_actual (np.ndarray): Actual target values.
        Y_predicted (np.ndarray): Predicted target values by the model.
        dataset_split_name (str): The name of the dataset split (e.g., 'Test', 'Train').

    Returns:
        Figure: Matplotlib figure object of the confusion matrix plot.
    """
    cm = confusion_matrix(Y_actual, Y_predicted, normalize="all")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=COLORMAPS["sequential"], cbar=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {dataset_split_name}')
    return plt.gcf()


def plot_predictions_reg(Y, Y_pred, n_scatter_samples, scores, dataset_split_name, target_name):
    if len(Y) > n_scatter_samples:  # Limit the number of plotted datapoints
        indices = np.random.choice(range(len(Y)), size=n_scatter_samples, replace=False)
        Y_sampled = Y[indices]  # Assuming Y is a pandas Series or DataFrame
        Y_pred_sampled = Y_pred[indices]
    else:
        Y_sampled = Y
        Y_pred_sampled = Y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_sampled, y=Y_pred_sampled, alpha=0.6)
    plt.plot([Y_sampled.min(), Y_sampled.max()], [Y_sampled.min(), Y_sampled.max()], color='red',
             lw=2)  # Line for perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(
        f'{dataset_split_name}: Actual vs Predicted {target_name[0]}. R2={scores.loc[f"R2_{dataset_split_name.lower()}", "Aggregated"]:.2f}')
    return plt.gcf()
