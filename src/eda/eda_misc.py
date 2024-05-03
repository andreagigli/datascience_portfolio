from functools import partial

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_regression, f_classif, r_regression, chi2
from sklearn.manifold import TSNE
from src.utils.my_dataframe import subsample_regular_interval

from sklearn.preprocessing import StandardScaler

from src.utils.my_statstest import SpearmanScorer


def compute_mutual_information(data: pd.DataFrame,
                               columns_of_interest: Optional[List[str]] = None,
                               target: Optional[str] = None,
                               discrete_features_mask: Optional[List[bool]] = None,
                               sample_size: Optional[int] = None,
                               plot_heatmap: bool = False) -> Tuple[
    pd.DataFrame, pd.DataFrame, Optional[Figure], Optional[Figure]]:
    """
    Computes mutual information between selected feature columns and one optional target in the dataset.

    If the target is not explicitly provided, the function calculates mutual information for each feature against every
    other feature. Since a different variant of mutual information is computed depending on the target type, continuous
    or discrete, two sets of mutual information scores are computed, one using the continuous features as targets, the
    other using the discrete features as targets. This separation ensures that the mutual information scores are
    comparable within the same data type groups.

    Args:
        data (pd.DataFrame): The dataframe containing the data.
        columns_of_interest (Optional[List[str]]): Columns of interest for MI calculation. Defaults to all columns if None.
        target (Optional[str]): The target column for MI calculation. If None, calculates MI for features against each other.
        discrete_features_mask (Optional[List[bool]]): Mask indicating whether each feature in columns_of_interest is discrete.
        sample_size (Optional[int]): If specified, uses a random subset of this size for calculation. Useful for large datasets.
        plot_heatmap (bool): If True, plots a heatmap of the mutual information results.

    Returns:
        Tuple containing:
            - pd.DataFrame: MI scores for features considered as continuous targets.
            - pd.DataFrame: MI scores for features considered as discrete targets.
            - Optional[plt.Figure]: Heatmap for continuous features if generated.
            - Optional[plt.Figure]: Heatmap for discrete features if generated.
    """
    # Use all columns if none are specifically provided
    if columns_of_interest is None:
        columns_of_interest = data.columns.tolist()
    else:
        # Filter columns
        columns_of_interest = list(columns_of_interest)  # Ensure they are a list
        columns_of_interest = [col for col in columns_of_interest if col in data.columns]
        if len(columns_of_interest) == 0:
            columns_of_interest = data.columns.tolist()

    # Set up the discrete features mask
    if discrete_features_mask is None:
        discrete_features_mask = [False] * len(columns_of_interest)

    # Validate the discrete_features_mask size
    if len(discrete_features_mask) != len(columns_of_interest):
        raise ValueError("discrete_features_mask must match the length of columns_of_interest.")

    # Sample data if necessary
    if sample_size and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    results_cont = pd.DataFrame()
    results_disc = pd.DataFrame()

    # Define score functions with partial to incorporate discrete_features_mask
    score_func_cont = partial(mutual_info_regression, discrete_features=discrete_features_mask)
    score_func_disc = partial(mutual_info_classif, discrete_features=discrete_features_mask)

    if target:
        # Calculate MI between the provided target column and all the other columns_of_interest, choosing the mutual information criterion based on the nature of such target column
        X = data[columns_of_interest]
        y = data[target]
        if discrete_features_mask[columns_of_interest.index(target)]:  # Discrete target
            selector = SelectKBest(score_func=score_func_disc, k="all").fit(X, y)  # Use sklearn selector to quickly compute all the pairwise scores between y and all the X
            results_disc = pd.DataFrame(
                {'feature_name': selector.feature_names_in_, "target_name": target, "MI": selector.scores_})
        else:  # Continuous target
            selector = SelectKBest(score_func=score_func_cont, k="all").fit(X, y)
            results_cont = pd.DataFrame(
                {'feature_name': selector.feature_names_in_, "target_name": target, "MI": selector.scores_})
    else:
        # Calculate MI between each column, chosen as a temporary target, and the rest, choosing the mutual information criterion based on the nature of such temporary target column
        for current_target_col in columns_of_interest:  # Discrete current_target_col
            X = data[columns_of_interest]
            y = data[current_target_col]
            if discrete_features_mask[columns_of_interest.index(current_target_col)]:
                selector = SelectKBest(score_func=score_func_disc, k="all").fit(X, y)
                results_disc = pd.concat([results_disc, pd.DataFrame(
                    {'feature_name': selector.feature_names_in_, "target_name": current_target_col,
                     "MI": selector.scores_})], ignore_index=True)
            else:  # Continuous current_target_col
                selector = SelectKBest(score_func=score_func_cont, k="all").fit(X, y)
                results_cont = pd.concat([results_cont, pd.DataFrame(
                    {'feature_name': selector.feature_names_in_, "target_name": current_target_col,
                     "MI": selector.scores_})], ignore_index=True)

    # Plotting heatmaps if required
    fig_heatmap_mi_cont, fig_heatmap_mi_disc = None, None
    if plot_heatmap:
        if not results_cont.empty:
            fig_heatmap_mi_cont, ax1 = plt.subplots(figsize=(10, 5))
            tmp_results = results_cont[results_cont['feature_name'] != results_cont['target_name']]  # Do not plot scores pertaining the relationship of a feature with itself because this can mess up the visual scale of the heatmap
            heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="MI")
            sns.heatmap(heatmap_data, annot=True, cmap='rocket', fmt=".2g", ax=ax1)
            ax1.set_title('Mutual Information for Continuous Features')
            plt.show()

        if not results_disc.empty:
            figheatmap_mi_disc, ax2 = plt.subplots(figsize=(10, 5))
            tmp_results = results_disc[results_disc['feature_name'] != results_disc['target_name']]  # Do not plot scores pertaining the relationship of a feature with itself because this can mess up the visual scale of the heatmap
            heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="MI")
            sns.heatmap(heatmap_data, annot=True, cmap='rocket', fmt=".2g", ax=ax2)
            ax2.set_title('Mutual Information for Discrete Features')
            plt.show()

    return results_cont, results_disc, fig_heatmap_mi_cont, fig_heatmap_mi_disc


def compute_relationship(data: pd.DataFrame,
                         score_func: str,
                         columns_of_interest: Optional[List[str]] = None,
                         target: Optional[str] = None,
                         sample_size: Optional[int] = None,
                         plot_heatmap: bool = False,
                         include_diagonal: bool = False) -> Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
    """
    Computes relationships between selected feature columns and an optional target in the dataset using a specified scoring function.

    IMPORTANT: This function does not discriminate between discrete and continuous features. Users must ensure that the scoring function is appropriate for the data types of the features used.

    Args:
        data (pd.DataFrame): The dataframe containing the data.
        score_func (str): An identifier of the scoring function to use, from [f_classif, f_regression, mutual_info_classif, mutual_info_regression, chi2, pearson, spearman].
        columns_of_interest (Optional[List[str]]): Columns of interest for the calculation. Defaults to all columns if None.
        target (Optional[str]): The target column for calculation. If None, calculates for features against each other.
        sample_size (Optional[int]): If specified, uses a random subset of this size for calculation. Useful for large datasets.
        plot_heatmap (bool): If True, plots a heatmap of the results.
        include_diagonal (bool): If False, excludes the diagonal in heatmap where feature and target are the same. Default is False.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
            - DataFrame with calculated scores.
            - Optional Figure object if a heatmap is generated.
    """
    # Use all columns if none are specifically provided
    if columns_of_interest is None:
        columns_of_interest = data.columns.tolist()
    else:
        # Filter columns
        columns_of_interest = list(columns_of_interest)
        columns_of_interest = [col for col in columns_of_interest if col in data.columns]
        if len(columns_of_interest) == 0:
            columns_of_interest = data.columns.tolist()

    # Handle numeric requirements for certain scoring methods
    if score_func in ["f_regression", "f_classif", "r_regression", "pearson", "spearman", "chi2"]:
        if not all(pd.api.types.is_numeric_dtype(data[col]) for col in columns_of_interest):
            raise ValueError(f"All selected columns must be numeric for the score function {score_func}.")
        if score_func == "chi2":
            if not all([pd.api.types.is_integer_dtype(data[col]) for col in columns_of_interest]):
                raise ValueError("All selected columns must be of integer type for chi2.")
            if data[columns_of_interest].min().min() < 0:
                raise ValueError("All selected columns must be non-negative for the score function chi2.")

    # Resolve the scoring function
    score_func = {
        'f_classif': f_classif,
        'f_regression': f_regression,
        'mutual_info_classif': mutual_info_classif,
        'mutual_info_regression': mutual_info_regression,
        'chi2': chi2,
        'pearson': r_regression,
        'spearman': SpearmanScorer,  # Using an sklearn-compatible custom Spearman scorer
    }.get(score_func, None)
    if score_func is None:
        raise ValueError("Scoring function not supported. Choose from 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression', 'chi2', 'pearson', 'spearman'.")

    # Sample the data if necessary
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=42)

    results = pd.DataFrame()

    if target:
        # Compute the relationship between the target and other columns
        X = data[columns_of_interest]
        y = data[target]
        selector = SelectKBest(score_func, k='all').fit(X, y)  # Use sklearn selector to quickly compute all the pairwise scores between y and all the X
        results = pd.DataFrame({
            'feature_name': columns_of_interest,
            'target_name': target,
            'score': selector.scores_,
            'p_value': getattr(selector, 'pvalues_', None)  # Capture p-values if available
        })

    else:
        # Compute the relationship among all columns
        for column in columns_of_interest:
            X = data[columns_of_interest]
            y = data[column]
            selector = SelectKBest(score_func, k='all').fit(X, y)
            results = pd.concat([results, pd.DataFrame({'feature_name': columns_of_interest,
                                                        'target_name': column,
                                                        'score': selector.scores_,
                                                        'p_value': getattr(selector, 'pvalues_', None)})])

    # Plotting heatmap if required and results are not empty
    fig_heatmap_relationship = None
    if plot_heatmap and not results.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        tmp_results = results if include_diagonal else results[results['feature_name'] != results['target_name']]  # Do not plot scores pertaining the relationship of a feature with itself because this can mess up the visual scale of the heatmap
        heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="score")
        if not target:
            # Only plot lower triangle if no target is provided
            mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=+1)
            sns.heatmap(heatmap_data, mask=mask, annot=True, cmap='rocket', ax=ax, fmt=".2g")
        else:
            sns.heatmap(heatmap_data, annot=True, cmap='rocket', ax=ax, fmt=".2g")
        plt.title(f'Heatmap of {score_func.__name__} Scores')
        plt.xticks(rotation=30, horizontalalignment="right")
        plt.tight_layout()

    return results, fig_heatmap_relationship


# def compute_relationship_scores(data: pd.DataFrame,
#                                 columns_of_interest: Optional[List[str]] = None,
#                                 target_columns: Optional[List[str]] = None,
#                                 score_func: str = 'mutual_info_classif',
#                                 sample_size: Optional[int] = None,
#                                 plot_heatmap: bool = True,
#                                 ) -> Tuple[pd.DataFrame, Optional[Figure]]:
#     """
#     Computes the relationship scores between specified columns and target columns in a dataframe using different scoring methods.
#
#     Args:
#         data (pd.DataFrame): The dataframe containing the data.
#         columns_of_interest (Optional[List[str]]): Columns of interest for which to compute the relationship scores.
#         target_columns (Optional[List[str]]): The target columns against which to compute the scores.
#         score_func (str): Scoring method to use from ['mutual_info_classif', 'mutual_info_regression', 'f_regression', 'f_classif', 'chi2'].
#         sample_size (Optional[int]): Number of samples to use; if None, use all data.
#         plot_heatmap (bool): If True, plots a heatmap of the relationship scores.
#
#     Returns:
#         pd.DataFrame: DataFrame containing the scores between each feature and target column, and p-values where applicable.
#         Figure (Optional): Heatmap of the computed relationship scores.
#
#     Scoring Methods Reference:
#         - 'mutual_info_classif' and 'mutual_info_regression': Scores range from 0 to 1 where 0 indicates no mutual information and 1 indicates maximum mutual information.
#         - 'f_regression' and 'f_regression': Provides F-statistic and p-values; higher F-statistic indicates a stronger relationship, typically significant if p-value < 0.05.
#         - 'chi2': Returns chi-squared stats between each non-negative feature and class; larger values indicate higher dependency.
#     """
#     # Validate input columns
#     feature_cols = data.columns if columns_of_interest is None else [col for col in data.columns if col in columns_of_interest]
#     target_cols = data.columns if target_columns is None else [col for col in data.columns if col in target_columns]
#
#     # Handle numeric requirements for certain scoring methods
#     if score_func in ['chi2', 'f_regression']:
#         # Check if all selected columns are numeric and non-negative (for chi2)
#         if not all(pd.api.types.is_numeric_dtype(data[col]) for col in feature_cols + target_cols):
#             raise ValueError("All selected columns for chi2 and f_regression must be numeric.")
#         if score_func == 'chi2' and data[feature_cols].min().min() < 0:
#             raise ValueError("All selected columns for chi2 must be non-negative.")
#
#     # Determine the appropriate score function
#     score_func = {
#         'mutual_info_classif': mutual_info_classif,
#         'mutual_info_regression': mutual_info_regression,
#         'f_regression': f_regression,
#         'f_classif': f_classif,
#         'chi2': chi2
#     }.get(score_func)
#     if score_func is None:
#         raise ValueError("Unsupported scoring method")
#
#     # Sample data if necessary
#     if sample_size and sample_size < len(data):
#         data = data.sample(n=sample_size, random_state=42)
#
#     # Compute scores
#     results = []
#     for target in target_cols:
#         features = [col for col in feature_cols if col != target]
#         selector = SelectKBest(score_func, k='all')
#         selector.fit(data[features], data[target])
#         scores = selector.scores_
#         pvalues = getattr(selector, 'pvalues_', None)  # Get p-values if they exist
#         results.extend([
#             {"feature_name": feat, "target_name": target, "score": score, "p_value": pval if pvalues is not None else None}
#             for feat, score, pval in zip(features, scores, pvalues if pvalues is not None else [None] * len(scores))
#         ])
#     results = pd.DataFrame(results)
#
#     # Plot heatmap if requested
#     fig_heatmap_relationships = None
#     if plot_heatmap and not results.empty:
#         heatmap_data = results.pivot(index="feature_name", columns="target_name", values="score")
#         fig_heatmap_relationships = plt.figure(figsize=(10, len(heatmap_data.index) * 0.5))
#         sns.heatmap(heatmap_data, annot=True, cmap='rocket', fmt=".2g", linewidths=.5)
#         plt.title(f"Heatmap of {score_func.__name__}")
#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     return results, fig_heatmap_relationships


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


def plot_data_distribution(df: pd.DataFrame,
                           columns_to_plot: Optional[List[str]] = None,
                           ) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates a single figure with vertical violin plots for continuous variables, including skewness and kurtosis,
    and another figure with count plots for discrete variables in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns_to_plot (List[str], optional): List of columns to include in the plots. Defaults to all columns.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing the handles to the figures for continuous and discrete variables respectively.
    """
    if columns_to_plot is None:
        columns_to_plot = df.columns.tolist()

    # Determine which columns are continuous and which are discrete
    continuous_vars = [col for col in columns_to_plot if df[col].nunique() > 10]  # Arbitrary cutoff for example
    discrete_vars = [col for col in columns_to_plot if df[col].nunique() <= 10]

    # Create figure for continuous variables
    fig_cont, axs_cont = plt.subplots(nrows=len(continuous_vars), figsize=(10, 5 * len(continuous_vars)))
    plt.suptitle("Distribution of the Continuous Variables")
    if len(discrete_vars) == 1:  # Handle case of single subplot by making it iterable
        axs_cont = [axs_cont]
    for i, var in enumerate(continuous_vars):
        axs_cont[i].set_title(f'{var} Distribution')
        sns.violinplot(data=df, x=var, ax=axs_cont[i], inner=None, orient="h",
                       color='lightgray')  # Violin plot without inner annotations
        sns.boxplot(data=df, x=var, ax=axs_cont[i], width=0.1, fliersize=3, whis=1.5, orient="h",
                    color='blue')  # Superimposed thin boxplot
        # Calculate and annotate skewness and kurtosis
        skw = skew(df[var].dropna())
        kurt = kurtosis(df[var].dropna())
        axs_cont[i].text(0.95, 0, f'Skew: {skw:.2f}\nKurt: {kurt:.2f}', ha='right', va='bottom',
                         transform=axs_cont[i].transAxes)
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


def plot_feature_clusters(df: pd.DataFrame,
                          columns_to_plot: Optional[List[str]] = None,
                          color_labels: Optional[pd.Series] = None,
                          sample_size: int = 100,
                          ) -> Figure:
    """
    Visualizes the features projected in 2D using PCA and t-SNE for dimensionality reduction on specified columns,
    with optional data subsampling and coloring based on provided labels.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns_to_plot (Optional[List[str]]): List of column names to include in the analysis, typically the continuous numerical ones. Uses all columns if None.
        color_labels (Optional[pd.Series]): Series containing labels for coloring the scatter plot points.
        sample_size (int): The number of samples to take from X and (optionally) Y to plot in each scatterplot. Default is 100.

    Returns:
        matplotlib.figure.Figure: The figure object containing PCA and t-SNE scatter plots.
    """
    # Select columns for analysis
    data = df if columns_to_plot is None else df[columns_to_plot]

    # Sample data if necessary
    if sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)
        color_labels = subsample_regular_interval(df=color_labels, sample_size=sample_size)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # PCA Analysis
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)[:2]

    # t-SNE Analysis
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X_scaled)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # PCA Plot
    ax1.set_title(
        f'PCA - Top 2 Components ({explained_variance[0]:.2f}, {explained_variance[1]:.2f} Explained Variance)')
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=color_labels)
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.grid(True)
    # t-SNE Plot
    ax2.set_title('t-SNE')
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, c=color_labels)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.grid(True)
    plt.tight_layout()

    return fig


def plot_pairwise_scatterplots(data: pd.DataFrame,
                               columns_to_plot: Optional[List[str]] = None,
                               target_columns: Optional[List[str]] = None,
                               color_labels: Optional[pd.Series] = None,
                               color_interpretation: Optional[str] = None,
                               sample_size: int = 100) -> plt.Figure:
    """
    Plot pairwise scatter plots for the dataframe `data`, optionally focusing on interactions between specified
    target columns and the rest of the selected data columns.

    Args:
        data (pd.DataFrame): The complete data matrix including any potential target columns.
        columns_to_plot (Optional[List[str]]): List of column names to consider for pairwise comparisons. If provided,
                                               restricts the data to these columns for all plotting operations.
        target_columns (Optional[List[str]]): List of column names to be treated as targets. If provided, scatter plots will be
                                              generated only between these target columns and other columns in `columns_to_plot`.
                                              Target columns not in `columns_to_plot` are ignored.
        color_labels (Optional[pd.Series]): Series containing labels for coloring the scatter plot points.
        color_interpretation (Optional[str]): Name of the variable rendered by the color.
        sample_size (int): The number of samples to take from `data` to plot in each scatterplot. Default is 100.

    Returns:
        plt.Figure: A matplotlib figure object containing the pairwise scatterplots.

    Example usage:
        # Assuming 'df' is your DataFrame containing both features and potential target columns.
        df = X.concat(Y)

        # To plot all combinations of columns
        fig = plot_pairwise_scatterplots(X, sample_size=100)  # Plot feature-feature scatterplots
        fig = plot_pairwise_scatterplots(df, sample_size=100)  # Plot feature-feature as well as feature-target scatterplots
        fig = plot_pairwise_scatterplots(df, columns_to_plot=["col1", "col7", "col8"], sample_size=100)
        fig = plot_pairwise_scatterplots(df, columns_to_plot=["col1", "col7", "col8"], color_labels=df["Target Class"], color_interpretation="Target Class", sample_size=100)  # Can color samples according to the target class, which must be a column of the dataset (not necessarily plotted).

        # To plot only specific features vs target
        fig_pairplots = plot_pairwise_scatterplots(df, columns_to_plot=[col for col in df.columns if col not in ["Target 1", "Target 2"]], target_columns=["Target 1", "Target 2"], sample_size=100)
        fig_pairplots = plot_pairwise_scatterplots(df, columns_to_plot=[col for col in df.columns if col not in ["Target 1"]], target_columns=["Target 1"], color_labels=Y["Target 1"], color_interpretation="Target 1", sample_size=100)   # Can color samples according to the (unique) target class
    """
    # Check the provided columns_to_plot and target_columns
    if columns_to_plot is None:
        columns_to_plot = data.columns.tolist()
    # Filter out non-existent columns_to_plot
    columns_to_plot = [col for col in columns_to_plot if col in data.columns]
    # Filter out invalid target_columns (not found in columns_to_plot)
    if target_columns is not None:
        target_columns = [col for col in target_columns if col in columns_to_plot]

    # Partition columns into non_target_columns (x) and target_columns (y)
    if target_columns is not None:
        non_target_columns = [col for col in columns_to_plot if col not in target_columns]
    else:
        non_target_columns = columns_to_plot

    # Check if color_labels have the same size of the data
    if len(color_labels) != len(data):
        color_labels = None

    # Sample data if necessary
    if sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)
        if color_labels is not None:
            color_labels = subsample_regular_interval(df=color_labels, sample_size=sample_size)

    # Prepare pairplot arguments depending on the presence of explicit target_columns and color_labels
    pairplot_kwargs = {}
    if target_columns is None:
        pairplot_kwargs["data"] = data[non_target_columns]
    else:
        pairplot_kwargs["data"] = data[non_target_columns + target_columns]
        pairplot_kwargs["y_vars"] = non_target_columns
        pairplot_kwargs["x_vars"] = target_columns
    if color_labels is not None:
        pairplot_kwargs["data"]["Color"] = color_labels  # Add column to data
        pairplot_kwargs["hue"] = "Color"

    # Generate pairwise scatter plots
    fig_pairplot = sns.pairplot(**pairplot_kwargs, plot_kws={'alpha': 0.5})
    # Add suptitle
    title = f"Pairwise Scatter Plots"
    if target_columns:
        title = title + "\nTarget Columns: " + ", ".join(target_columns)
    if color_labels is not None and color_interpretation is not None:
        title = title + f"\nThe Color Represents: {color_interpretation}"
    plt.suptitle(title)
    # Tilt y-labels to avoid overlap
    for ax in fig_pairplot.axes.flatten():
        if ax is not None:
            ax.set_ylabel(ax.get_ylabel(), rotation=30)
            ax.yaxis.get_label().set_horizontalalignment('right')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    return fig_pairplot.fig
