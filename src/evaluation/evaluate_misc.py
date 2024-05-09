import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict


COLORMAPS = {
    'sequential': 'viridis',
    'diverging': 'coolwarm',
    'categorical': 'colorblind'
}


def calculate_metrics_clf(Y_actual: np.ndarray, Y_predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculates various classification metrics to evaluate the performance of a classifier.

    Args:
        Y_actual (np.ndarray): Actual target values.
        Y_predicted (np.ndarray): Predicted target values by the model.

    Returns:
        metrics (Dict[str, float]): A dictionary containing the calculated metrics, including Accuracy,
                                    Precision, Recall, F1 Score, ROC AUC Score.
    """
    if len(np.unique(Y_actual)) > 2:
        average = 'macro'
        roc_auc = roc_auc_score(Y_actual, Y_predicted, multi_class='ovr')
    else:
        average = 'binary'
        roc_auc = roc_auc_score(Y_actual, Y_predicted)

    metrics = {
        'Accuracy': accuracy_score(Y_actual, Y_predicted),
        'Precision': precision_score(Y_actual, Y_predicted, average=average),
        'Recall': recall_score(Y_actual, Y_predicted, average=average),
        'F1 Score': f1_score(Y_actual, Y_predicted, average=average),
        'ROC AUC': roc_auc
    }
    return metrics


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
