import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(Y, Y_pred, model, target_name):
    """
    Evaluates the performance of a regression model and generates a plot of actual vs. predicted values.

    Parameters:
    Y (array-like): Actual target values.
    Y_pred (array-like): Predicted target values from the model.
    model: Trained model used for prediction.
    target_name (list): List containing the name of the target variable.

    Returns:
    scores (pd.DataFrame): DataFrame containing evaluation metrics.
    figs (dict): Dictionary containing generated figures.
    """
    # Compute evaluation metrics
    mae = mean_absolute_error(Y, Y_pred, multioutput="raw_values")
    mse = mean_squared_error(Y, Y_pred, multioutput="raw_values")
    rmse = np.sqrt(mse)
    nrmse = np.divide(rmse, iqr(Y_pred, axis=0))
    r2 = r2_score(Y, Y_pred, multioutput="raw_values")

    # Create a DataFrame
    scores = pd.DataFrame(
        data=[mae, mse, rmse, nrmse, r2],  # processed as rows
        index=['MAE', 'MSE', 'RMSE', 'NRMSE', 'R2'],
        columns=['y0'] if Y.ndim == 1 else ['y' + str(i) for i in range(Y.shape[1])],
    )

    # Calculate Aggregated mean for each metric
    # Ensure all entries are numeric for mean calculation, replace non-numeric with NaN
    scores['Aggregated'] = scores.apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1)

    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y, y=Y_pred, alpha=0.6)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color='red', lw=2)  # Line for perfect predictions
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    formatted_mae = ', '.join([f'{x:.2f}' for x in mae])
    formatted_rmse = ', '.join([f'{x:.2f}' for x in rmse])
    formatted_r2 = ', '.join([f'{x:.2f}' for x in r2])
    plt.title(f'Actual vs Predicted {target_name[0]} - MAE: {formatted_mae}, RMSE: {formatted_rmse}, R2: {formatted_r2}')

    figs = {'scatter_plot': plt.gcf()}

    return scores, figs