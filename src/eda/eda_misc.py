import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List


def plot_correlation_heatmap(X: pd.DataFrame,
                             Y: Optional[pd.DataFrame] = None,
                             sample_size: int = 1000,
                             method: str = 'pearson'
                             ) -> plt.Figure:
    """
    Plot a heatmap of the correlation matrix for dataframe X and, optionally, between X and Y.

    Args:
        X (pd.DataFrame): The design matrix.
        Y (Optional[pd.DataFrame]): The target matrix. If provided, includes correlation between X and Y. Default is None.
        sample_size (int): The number of samples to take from X and (optionally) Y for correlation calculation. Default is 1000.
        method (str): The method for correlation calculation ('pearson' or 'spearman'). Default is 'pearson'.

    Returns:
        plt.Figure: A matplotlib figure object containing the correlation heatmap.
    """
    # Sample data if the sample size is less than the number of rows
    sampled_X = X.sample(n=min(sample_size, X.shape[0]), random_state=1)
    if Y is not None:
        sampled_Y = Y.sample(n=min(sample_size, Y.shape[0]), random_state=1)
        sampled_data = pd.concat([sampled_X, sampled_Y], axis=1)
    else:
        sampled_data = sampled_X

    # Calculate the correlation matrix
    corr = sampled_data.corr(method=method)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Title adjustment
    correlation_type = "Pearson" if method == "pearson" else "Spearman"
    title = f"{correlation_type} Correlation Heatmap"
    if Y is not None:
        title += " including X and Y"
    plt.suptitle(title)

    return fig


def plot_pairwise_scatterplots(X: pd.DataFrame,
                               Y: Optional[pd.DataFrame] = None,
                               columns_to_plot: Optional[List[str]] = None,
                               sample_size: int = 100,
                               ) -> plt.Figure:
    """
    Plot pairwise scatter plots for dataframe X and, optionally, between X and Y.

    Args:
        X (pd.DataFrame): The design matrix.
        Y (Optional[pd.DataFrame]): The target matrix. If provided, includes scatter plots between X and Y. Default is None.
        columns_to_plot (Optional[List[str]]): List of column names to consider for pairwise comparisons. If provided,
                                               only these columns will be used for plotting. Invalid column names are
                                               ignored, and if all provided names are invalid or the list is empty,
                                               the default behavior is to plot all comparisons. Default is None.
        sample_size (int): The number of samples to take from X and (optionally) Y to plot in each scatterplot. Default is 100.

    Returns:
        plt.Figure: A matplotlib figure object containing the pairwise scatterplots for the desired columns.
    """
    # Concatenate X and Y if Y is provided
    if Y is not None:
        data = pd.concat([X, Y], axis=1)
    else:
        data = X

    # Filter data for columns to plot if provided
    if columns_to_plot is not None:
        columns_to_plot = [col for col in columns_to_plot if col in data.columns]
        if not columns_to_plot:
            print("All provided column names are invalid. Plotting all comparisons instead.")
        else:
            data = data[columns_to_plot]

    # Sample data if necessary
    if sample_size < len(data):
        data = data.sample(n=sample_size, random_state=0)

    # Generate pairwise scatter plots
    fig = sns.pairplot(data)
    plt.suptitle("Pairwise Scatter Plots")  # Adjust 'size' and 'y' as needed

    return fig