import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from scipy.stats import skew, kurtosis


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


def plot_data_distribution(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates a single figure with vertical violin plots for continuous variables, including skewness and kurtosis,
    and another figure with count plots for discrete variables in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str], optional): List of columns to include in the plots. Defaults to all columns.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing the handles to the figures for continuous and discrete variables respectively.
    """
    if columns is None:
        columns = df.columns.tolist()

    # Determine which columns are continuous and which are discrete
    continuous_vars = [col for col in columns if df[col].nunique() > 10]  # Arbitrary cutoff for example
    discrete_vars = [col for col in columns if df[col].nunique() <= 10]

    # Create figure for continuous variables
    fig_cont, axs_cont = plt.subplots(nrows=len(continuous_vars), figsize=(10, 5 * len(continuous_vars)))
    plt.suptitle("Distribution of the Continuous Variables")
    if len(discrete_vars) == 1:  # Handle case of single subplot by making it iterable
        axs_cont = [axs_cont]
    for i, var in enumerate(continuous_vars):
        axs_cont[i].set_title(f'{var} Distribution')
        sns.violinplot(data=df, x=var, ax=axs_cont[i], inner=None, orient="h", color='lightgray')  # Violin plot without inner annotations
        sns.boxplot(data=df, x=var, ax=axs_cont[i], width=0.1, fliersize=3, whis=1.5, orient="h", color='blue')  # Superimposed thin boxplot
        # Calculate and annotate skewness and kurtosis
        skw = skew(df[var].dropna())
        kurt = kurtosis(df[var].dropna())
        axs_cont[i].text(0.95, 0, f'Skew: {skw:.2f}\nKurt: {kurt:.2f}', ha='right', va='bottom', transform=axs_cont[i].transAxes)
        axs_cont[i].set_xlabel("")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the title

    # Create figure for discrete variables
    fig_disc, axs_disc = plt.subplots(nrows=len(discrete_vars), figsize=(10, 5 * len(discrete_vars)))
    plt.suptitle("Distribution of the Discrete Variables")
    if len(discrete_vars) == 1:  # Handle case of single subplot
        axs_disc = [axs_disc]
    for i, var in enumerate(discrete_vars):
        sns.countplot(x=df[var], ax=axs_disc[i] if len(discrete_vars) > 1 else axs_disc)
        axs_disc[i].set_title(f'{var} Count Plot')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig_cont, fig_disc


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