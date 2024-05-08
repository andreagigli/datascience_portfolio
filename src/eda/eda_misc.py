from functools import partial
from typing import Optional, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, f_regression, f_classif, r_regression, chi2
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.utils.my_dataframe import subsample_regular_interval
from src.utils.my_statstest import spearman_score_func


COLORMAPS = {
    'sequential': 'viridis',
    'diverging': 'coolwarm',
    'categorical': 'colorblind'
}


def check_outliers(data: pd.DataFrame,
                   columns_of_interest: List[str],
                   sample_size: Optional[int] = None,
                   profile_plots: bool = True) -> Tuple[pd.Series, Optional[Figure], Optional[Figure]]:
    """
    Detects and profiles outliers in the specified continuous columns using Isolation Forest.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        columns_of_interest (List[str]): List of column names to analyze for outliers, usually all the continuous columns.
        sample_size (Optional[int]): If specified, selects sample_size samples evenly spaced within the dataset.
        profile_plots (bool): If True, generates violin plots for profiling outliers.

    Returns:
        pd.Series: Series indicating outliers (1 for outlier, 0 for non-outlier).
        Two figures (optional): Figure handles for the visualization of outliers in 2d projctions and in violinplots.
    """
    # Sample data if necessary
    if sample_size and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    print("\n### Outliers in the continuous features ###\n")
    print("Detecting outliers with Random Isolation Forest:")
    rif = IsolationForest(n_estimators=100, random_state=0).fit(data[columns_of_interest])
    outliers = rif.predict(data[columns_of_interest]) == -1
    outliers = pd.Series(outliers.astype(int), name='Outlier', index=data.index)
    print(f"Number of identified outliers: {outliers.sum()}\n")

    print("Outliers profiling: are these actual outliers or valid extreme data points?")
    # Create temporary dataframe for outlier profiling
    data_tmp = data[columns_of_interest].copy()
    data_tmp["Outlier"] = outliers.values
    data_tmp["Sample colors"] = data["Good Risk"].values
    data_tmp.loc[outliers == 1, "Sample colors"] = data["Good Risk"].max() + 1

    print("Descriptive statistics of outlies:")
    print(data.loc[outliers == 1, columns_of_interest].describe())
    print("Descriptive statistics of non-outlies:")
    print(data.loc[outliers == 0, columns_of_interest].describe())

    fig_outliers_clusters = None
    fig_outliers_violins = None
    if profile_plots:
        print("Visualize outliers in a 2D projection of the continuous features.")
        fig_outliers_clusters = plot_clusters_2d(data_tmp, columns_to_plot=columns_of_interest, color_labels=data_tmp["Sample colors"])

        print("Visualize distribution of outliers vs non-outliers.")
        fig_outliers_violins = plot_grouped_violinplots(data_tmp, target_column="Outlier", columns_of_interest=columns_of_interest, overlay_stripplot=True)

    return outliers, fig_outliers_clusters, fig_outliers_violins


def compute_mutual_information(data: pd.DataFrame,
                               columns_of_interest: Optional[List[str]] = None,
                               target: Optional[str] = None,
                               discrete_features_mask: Optional[List[bool]] = None,
                               sample_size: Optional[int] = None,
                               plot_heatmap: bool = False,
                               include_diagonal: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Figure], Optional[Figure]]:
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
        discrete_features_mask (Optional[List[bool]]): Mask indicating whether each feature in columns_of_interest is discrete. If None, assumes all columns are discrete.
        sample_size (Optional[int]): If specified, selects sample_size samples evenly spaced within the dataset.
        plot_heatmap (bool): If True, plots a heatmap of the mutual information results.
        include_diagonal (bool): If False, excludes the diagonal in heatmap where feature and target are the same. Default is False.

    Returns:
        Tuple containing:
            - pd.DataFrame: MI scores for features considered as continuous targets.
            - pd.DataFrame: MI scores for features considered as discrete targets.
            - Optional[plt.Figure]: Heatmap for continuous features if generated.
            - Optional[plt.Figure]: Heatmap for discrete features if generated.

    Example uses:
        Mutual information between all features and a target column, with all columns assumed discrete: compute_mutual_information(data=df, target="target_col", sample_size=1000, plot_heatmap=True, include_diagonal=True)
        Mutual information between all features and a target column, with only some columns being discrete: compute_mutual_information(data=df, target="target_col", discrete_feature_mask=mask_discrete_cols, sample_size=1000, plot_heatmap=True, include_diagonal=True)
        Mutual information between all pairs of features, with only some columns being discrete:  compute_mutual_information(data=df, discrete_feature_mask=mask_discrete_cols, sample_size=1000, plot_heatmap=True, include_diagonal=True)
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
    if discrete_features_mask is None:  # If None, assume all columns discrete
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
            selector = SelectKBest(score_func=score_func_disc, k="all")  # Use sklearn selector to quickly compute all the pairwise scores between y and all the X
            selector.fit(X, y)
            results_disc = pd.DataFrame({'feature_name': selector.feature_names_in_, "target_name": target, "MI": selector.scores_})
        else:  # Continuous target
            selector = SelectKBest(score_func=score_func_cont, k="all")
            selector.fit(X, y)
            results_cont = pd.DataFrame(
                {'feature_name': selector.feature_names_in_, "target_name": target, "MI": selector.scores_})
    else:
        # Calculate MI between each column, chosen as a temporary target, and the rest, choosing the mutual information criterion based on the nature of such temporary target column
        for current_target_col in columns_of_interest:  # Discrete current_target_col
            X = data[columns_of_interest]
            y = data[current_target_col]
            if discrete_features_mask[columns_of_interest.index(current_target_col)]:
                selector = SelectKBest(score_func=score_func_disc, k="all")
                selector.fit(X, y)
                results_disc = pd.concat([results_disc, pd.DataFrame(
                    {'feature_name': selector.feature_names_in_, "target_name": current_target_col,
                     "MI": selector.scores_})], ignore_index=True)
            else:  # Continuous current_target_col
                selector = SelectKBest(score_func=score_func_cont, k="all")
                selector.fit(X, y)
                results_cont = pd.concat([results_cont, pd.DataFrame(
                    {'feature_name': selector.feature_names_in_, "target_name": current_target_col,
                     "MI": selector.scores_})], ignore_index=True)

    # Plotting heatmaps if required
    fig_heatmap_mi_cont, fig_heatmap_mi_disc = None, None
    if plot_heatmap:
        # Determine the number of plots based on the data availability
        num_plots = (not results_cont.empty) + (not results_disc.empty)
        # Create subplots: one row, with plots side by side
        fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 5))
        # Ensure axes is iterable (in case there is only one plot)
        if num_plots == 1:
            axes = [axes]
        current_ax = 0

        if not results_cont.empty:
            ax1 = axes[current_ax]
            ax1.set_title('Mutual Information for Continuous Target')
            tmp_results = results_cont if include_diagonal else results_cont[results_cont['feature_name'] != results_cont['target_name']]  # Do not plot scores pertaining the relationship of a feature with itself because this can mess up the visual scale of the heatmap
            heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="MI")
            sns.heatmap(heatmap_data, annot=True, cmap=COLORMAPS["sequential"], fmt=".2g", ax=ax1)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, horizontalalignment="right")
            current_ax += 1  # Move to the next subplot

        if not results_disc.empty:
            ax2 = axes[current_ax]
            ax2.set_title('Mutual Information for Discrete Target')
            tmp_results = results_disc if include_diagonal else results_disc[results_disc['feature_name'] != results_disc['target_name']]
            heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="MI")
            sns.heatmap(heatmap_data, annot=True, cmap=COLORMAPS["sequential"], fmt=".2g", ax=ax2)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, horizontalalignment="right")

        plt.tight_layout()

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
        sample_size (Optional[int]): If specified, selects sample_size samples evenly spaced within the dataset.
        plot_heatmap (bool): If True, plots a heatmap of the results.
        include_diagonal (bool): If False, excludes the diagonal in heatmap where feature and target are the same. Default is False.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[plt.Figure]]:
            - DataFrame with calculated scores.
            - Optional Figure object if a heatmap is generated.

    Example uses:
        Pearson correlation between all pairs of features: compute_relationship(data=df, score_func="pearson", sample_size=1000, plot_heatmap=True, include_diagonal=True)
        Spearman correlation between all pairs of continuous features and a target column: compute_relationship(data=df, score_func="spearman", columns_of_interest=cols_continuous, target="target_col", sample_size=1000, plot_heatmap=True, include_diagonal=True)
        Chi2 correlation between all pairs of discrete features: compute_relationship(data=df, score_func="chi2", columns_of_interest=cols_discrete, sample_size=1000, plot_heatmap=True, include_diagonal=True)
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
        'spearman': spearman_score_func,  # Using an sklearn-compatible custom Spearman scorer
    }.get(score_func, None)
    if score_func is None:
        raise ValueError("Scoring function not supported. Choose from 'f_classif', 'f_regression', 'mutual_info_classif', 'mutual_info_regression', 'chi2', 'pearson', 'spearman'.")

    # Sample the data if necessary
    if sample_size and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    results = pd.DataFrame()

    if target:
        # Compute the relationship between the target and other columns
        X = data[columns_of_interest]
        y = data[target]
        selector = SelectKBest(score_func, k='all')  # Use sklearn selector to quickly compute all the pairwise scores between y and all the X
        selector.fit(X, y)
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
            selector = SelectKBest(score_func, k='all')
            selector.fit(X, y)
            results = pd.concat([results, pd.DataFrame({'feature_name': columns_of_interest,
                                                        'target_name': column,
                                                        'score': selector.scores_,
                                                        'p_value': getattr(selector, 'pvalues_', None)})])

    # Plotting heatmap if required and results are not empty
    fig_heatmap_relationship = None
    if plot_heatmap and not results.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title(f'Heatmap of {score_func.__name__}')
        tmp_results = results if include_diagonal else results[results['feature_name'] != results['target_name']]  # Do not plot scores pertaining the relationship of a feature with itself because this can mess up the visual scale of the heatmap
        heatmap_data = tmp_results.pivot(index="feature_name", columns="target_name", values="score")
        if not target:
            # Only plot lower triangle if no target is provided
            mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=+1)
            sns.heatmap(heatmap_data, mask=mask, annot=True, cmap=COLORMAPS["sequential"], ax=ax, fmt=".2g")
        else:
            sns.heatmap(heatmap_data, annot=True, cmap=COLORMAPS["sequential"], ax=ax, fmt=".2g")
        plt.xticks(rotation=30, horizontalalignment="right")
        plt.tight_layout()

    return results, fig_heatmap_relationship


def plot_clusters_2d(data: pd.DataFrame,
                     columns_to_plot: Optional[List[str]] = None,
                     color_labels: Optional[pd.Series] = None,
                     sample_size: int = 100) -> Figure:
    """
    Visualizes the features projected in 2D using PCA and t-SNE for dimensionality reduction on specified columns,
    with optional data subsampling and coloring based on provided labels.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        columns_to_plot (Optional[List[str]]): List of column names to include in the analysis, typically the continuous numerical ones. Uses all columns if None.
        color_labels (Optional[pd.Series]): Series containing labels for coloring the scatter plot points.
        sample_size (int): The number of samples to take from X and (optionally) Y to plot in each scatterplot. Default is 100.

    Returns:
        matplotlib.figure.Figure: The figure object containing PCA and t-SNE scatter plots.
    """
    # Select columns for analysis
    data = data if columns_to_plot is None else data[columns_to_plot]

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


def plot_data_distribution(data: pd.DataFrame,
                           columns_of_interest: Optional[List[str]] = None,
                           discrete_features_mask: Optional[List[bool]] = None,) -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates a single figure with vertical violin plots for continuous variables, including skewness and kurtosis,
    and another figure with count plots for discrete variables in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        columns_of_interest (List[str], optional): List of columns to include in the plots. Defaults to all columns.
        discrete_features_mask (Optional[List[bool]]): Mask indicating whether each feature in columns_of_interest is discrete. If None, assumes all columns are discrete.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing the handles to the figures for continuous and discrete variables respectively.
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
    if discrete_features_mask is None:  # If None, assume all columns discrete
        discrete_features_mask = [False] * len(columns_of_interest)

    # Validate the discrete_features_mask size
    if len(discrete_features_mask) != len(columns_of_interest):
        raise ValueError("discrete_features_mask must match the length of columns_of_interest.")

    # Determine which columns are continuous and which are discrete
    continuous_vars = [col for col, is_discrete in zip(columns_of_interest, discrete_features_mask) if not is_discrete]
    discrete_vars = [col for col, is_discrete in zip(columns_of_interest, discrete_features_mask) if is_discrete]

    # Create figure for continuous variables
    fig_continuous, axs_continuous = plt.subplots(nrows=len(continuous_vars), figsize=(10, 5 * len(continuous_vars)))
    plt.suptitle("Distribution of the Continuous Variables")
    if len(discrete_vars) == 1:  # Handle case of single subplot by making it iterable
        axs_continuous = [axs_continuous]
    for i, var in enumerate(continuous_vars):
        axs_continuous[i].set_title(f'{var} Distribution')
        sns.violinplot(data=data, x=var, ax=axs_continuous[i], inner=None, orient="h", color='lightgray')  # Violin plot without inner annotations
        sns.boxplot(data=data, x=var, ax=axs_continuous[i], width=0.1, fliersize=3, whis=1.5, orient="h", color='blue')  # Superimposed thin boxplot
        # Calculate and annotate skewness and kurtosis
        skw = skew(data[var].dropna())
        kurt = kurtosis(data[var].dropna())
        axs_continuous[i].text(0.95, 0, f'Skew: {skw:.2f}\nKurt: {kurt:.2f}', ha='right', va='bottom', transform=axs_continuous[i].transAxes)
        axs_continuous[i].set_xlabel("")
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Adjust the layout to fit the title

    # Create figure for discrete variables
    fig_discrete, axs_discrete = plt.subplots(nrows=len(discrete_vars), figsize=(10, 5 * len(discrete_vars)))
    plt.suptitle("Distribution of the Discrete Variables")
    if len(discrete_vars) == 1:  # Handle case of single subplot
        axs_discrete = [axs_discrete]
    for i, var in enumerate(discrete_vars):
        sns.countplot(x=data[var], ax=axs_discrete[i] if len(discrete_vars) > 1 else axs_discrete)
        axs_discrete[i].set_title(f'{var} Count Plot')
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    return fig_continuous, fig_discrete


def plot_grouped_stripplots(data: pd.DataFrame,
                            target_column: str,
                            columns_of_interest: Optional[List[str]] = None,
                            sample_size: int = None,
                            jitter: Optional[Union[float, bool]] = True):
    """
    Plots grouped stripplots for specified categorical columns of a DataFrame against a categorical target column.

    This visualization helps in understanding the distribution of categorical variables across different categories
    of the target variable. It can be thought of as an equivalent to a grouped violinplot for discrete-discrete
    relationships (as opposed to continuous-discrete relationships).

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column against which the distributions will be plotted.
        columns_of_interest (list, optional): List of column names from the DataFrame to include in the plots. If None, defaults to all columns.
        sample_size (int, optional): If specified, selects sample_size samples evenly spaced within the dataset.
        jitter (float or bool, optional): Allows to customize the jitter of the stripplot. If not provided, uses a default amount.

    Returns:
        plt.Figure: A matplotlib figure object containing the grouped plots.
    """
    # Use all columns if none are given
    if columns_of_interest is None:
        columns_of_interest = data.columns.tolist()
    else:
        # Filter columns
        columns_of_interest = list(columns_of_interest)  # Ensure they are a list
        columns_of_interest = [col for col in columns_of_interest if col in data.columns]
        if len(columns_of_interest) == 0:
            columns_of_interest = data.columns.tolist()

    if target_column not in data.columns:
        raise ValueError("Argument target_column must be a valid column of data.")

    if sample_size is not None and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    num_plots = len(columns_of_interest)
    fig, axes = plt.subplots(ncols=num_plots, figsize=(4 * num_plots, 8))  # Adjust figure size as needed
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot case
    for ax, feature in zip(axes, columns_of_interest):
        sns.stripplot(data=data, y=feature, hue=target_column, ax=ax, dodge=True, jitter=jitter, alpha=0.6, palette=COLORMAPS["categorical"])
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel('')
        ax.set_ylabel('Values')
        ax.xaxis.set_ticks([])  # Optionally remove x-axis ticks for cleanliness if needed
        # Adjust the legend
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[0], ylim[1] * 1.1)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1, 1), title=target_column)

    plt.tight_layout()

    return fig


def plot_grouped_violinplots(data: pd.DataFrame,
                             target_column: str,
                             columns_of_interest: Optional[List[str]] = None,
                             sample_size: int = None,
                             overlay_stripplot: bool = False):
    """
    Plots grouped violin plots for specified continuous columns of a DataFrame against a categorical target column.

    This visualization helps in understanding the distribution of continuous variables across different categories
    of the target variable, which can be particularly useful for exploring potential influences or biases in features
    relative to the target.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        target_column (str): The name of the target column against which the distributions will be plotted.
        columns_of_interest (list, optional): List of column names from the DataFrame to include in the violin plots. If None, defaults to all columns.
        sample_size (int, optional): If specified, selects sample_size samples evenly spaced within the dataset.
        overlay_stripplot (bool, optional): If True, overlays strip plots on the violin plots.

    Returns:
        plt.Figure: A matplotlib figure object containing the grouped violin plots.
    """
    # Use all columns if none are given
    if columns_of_interest is None:
        columns_of_interest = data.columns.tolist()
    else:
        # Filter columns
        columns_of_interest = list(columns_of_interest)  # Ensure they are a list
        columns_of_interest = [col for col in columns_of_interest if col in data.columns]
        if len(columns_of_interest) == 0:
            columns_of_interest = data.columns.tolist()

    if target_column not in data.columns:
        raise ValueError("Argument target_column must be a valid column of data.")

    if sample_size is not None and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    num_plots = len(columns_of_interest)
    fig, axes = plt.subplots(ncols=num_plots, figsize=(4 * num_plots, 8))  # Adjust figure size as needed
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot case
    for ax, feature in zip(axes, columns_of_interest):
        sns.violinplot(data=data, x=target_column, y=feature, hue=target_column, ax=ax, orient='v', split=True, palette=COLORMAPS["categorical"])

        if overlay_stripplot:
            sns.stripplot(data=data, x=target_column, y=feature, hue=target_column, dodge=False, jitter=True, palette="dark:black", ax=ax, alpha=0.6)
            # Remove the second legend created by stripplot
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:len(labels) // 2], labels[:len(labels) // 2])

        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel('')
        ax.set_ylabel('Values')
        ax.xaxis.set_ticks([])  # Optionally remove x-axis ticks for cleanliness if needed

    plt.tight_layout()

    return fig


def plot_jittered_scatterplot(data: pd.DataFrame,
                              x: str,
                              y: str,
                              hue: Optional[str] = None,
                              jitter: Union[float, bool] = True,
                              alpha: float = 0.5,
                              sample_size: int = None):
    """
    Plots a jittered scatterplot between two categorical columns with an optional hue from a third categorical column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        x (str): The name of the column for the x-axis.
        y (str): The name of the column for the y-axis.
        hue (str, optional): The name of the column for hue. If None, no hue is used.
        jitter (float or bool, optional): Amount of jitter (only along the categorical axis) to apply. If True, uses a default amount.
        alpha (float, optional): The transparency level of the points. Default is 0.5.
        sample_size (int, optional): If specified, selects sample_size samples evenly spaced within the dataset.

    Returns:
        plt.Figure: A matplotlib figure object containing the scatterplot.
    """
    if sample_size is not None and sample_size < len(data):
        data = subsample_regular_interval(df=data, sample_size=sample_size)

    plt.figure(figsize=(10, 6))
    sns.stripplot(data=data, x=x, y=y, hue=hue, jitter=jitter, alpha=alpha, dodge=True, palette=COLORMAPS["categorical"])

    if hue:
        ylim = plt.gca().get_ylim()
        plt.gca().set_ylim(ylim[0], ylim[1] * 1.1)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1, 1), title=hue)

    plt.title(f'Jittered Scatterplot of {y} vs {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()

    return plt.gcf()


def plot_pairplots(data: pd.DataFrame,
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
        sample_size (int): If specified, selects sample_size samples evenly spaced within the dataset.

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
        pairplot_kwargs["palette"] = COLORMAPS['categorical']  # Set the colormap for hue

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
