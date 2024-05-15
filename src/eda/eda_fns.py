import numpy as np
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from src.eda.eda_misc import plot_data_distribution, plot_clusters_2d, plot_pairplots, \
    compute_mutual_information, compute_relationship, plot_grouped_violinplots, plot_grouped_stripplots, \
    plot_jittered_scatterplot, check_outliers
from src.utils.my_dataframe import custom_info
from src.utils.my_math import spot_seasonal_cycles


def eda_gcrdb(data: DataFrame) -> None:
    """
    Performs exploratory data analysis (EDA) on the provided dataset to prepare it for more effective modeling and insights derivation.

    Objectives and Techniques:
    - Features sanity, types, and values: Evaluate if the features contain nan values, and attempt to distinguish continuous from discrete ones.
        Discrete features could be natively numerical or arising from encoding of boolean or categorical variables.
        Use df.info() and df.nunique() to check the number of nans, the feature type and the number of unique values each feature assumes.
    - Distribution and Outliers: Evaluate the distribution of features and the target. The aim is assessing the data is well-conditioned for learning (scale, statistical assumptions, no outliers).
        Use histograms, boxplots, violinplots, countplots, and shape statistics to assess distribution.
        Use Tukey thresholding or isolation forests for univariate and multivariate outlier detection.
    - Existing Clusters: Use scatterplots, optionally preceeded by dimensionality reduction techniques like PCA or t-SNE, to visualize and assess data separability for classification and expected number of clusters.
    - Feature-Target Relationships: Characterize relationships between features and the target in terms of strength and nature to inform feature engineering and model selection.
        Perform a quick assessment, use pairplots and feature ranking based on mutual information. The pairplot, or specific scatter plots, is also important for the next steps.
        To assess linearity/monotonicity in regression, use the Pearson and Spearman correlation coefficients for continuous features or grouped boxplots for discrete ones.
        To assess homoscedasticity in regression, check for fan-shaped patterns in the pairplot (homoscedasticity involves feat-residuals, but feat-target can be a model-agnostic indication).
        To assess linear class separability in binary classification (or one-hot encoded target class), use statistical feature selection based on ANOVA f-score if the feature is continuous or chi2 if the feature is discrete.
        To assess linear class separability in multiclass classification (label-encoded target class), just check the pairplot.
    - Feature Interactions: Investigate relationships between features to inform feature engineering or gain insights into the underlying processes.
        To assess feature redundancy, check if the pearson corrcoef or the cramer V approach 1 for continuous-continuous or discrete-discrete relationships, respectively.
        To evaluate feature combination or gain insights on underlying process, check feature pairs that have high spearman corrcoef or cramer V for continuous-continuous or discrete-discrete relationships, respectively.
        To identify and explain complex phenomena, check for abrupt trend changes in the scatterplot.

    Args:
        data (DataFrame): A pandas DataFrame containing the data to be analyzed.

    Returns:
        Outputs various visualizations that illustrate the data's characteristics and summaries of statistical tests to guide further data preprocessing and feature engineering decisions.
    """
    # Distinguish continuous from discrete columns, in this case based on a heuristic thresholding
    print(f"Number of nans, type, and number of unique values for each column of the DataFrame: \n{custom_info(data)}")
    cols_continuous = data.nunique().loc[data.nunique() > 8].index.to_list()
    cols_discrete = data.nunique().loc[data.nunique() <= 8].index.to_list()

    # Feature Sanity Check
    print("\n### Feature Sanity Check ###\n")
    print("Evaluating data types and missing values:\n")
    print(f"{custom_info(data)}\n")  # Assumes custom_info is a function returning formatted string

    # Feature distributions
    print("\n### Feature distributions ###\n")
    print("Visualizing distribution of continuous and discrete variables:")
    plot_data_distribution(data, discrete_features_mask=[col in cols_discrete for col in data.columns])

    # Feature outliers
    print("\n### Outliers in the continuous features ###\n")
    outliers_mask, _, _ = check_outliers(data, columns_of_interest=cols_continuous, sample_size=100, profile_plots=True, color_labels=data["good_risk"])

    # Feature clusters
    print("\n### Feature cluster ###\n")
    print("Plot a 2D projections of the continuous features to identify natural clusters.")
    plot_clusters_2d(data, columns_to_plot=cols_continuous, color_labels=data["good_risk"])

    # Feature-Target Relationships
    print("\n### Feature-Target Relationships ###\n")
    print("Pairplot of features vs target ('good_risk').\nUseful to inform feature engineering and model selection.\n")
    plot_pairplots(data, target_columns=["good_risk"], color_labels=data["good_risk"], sample_size=100)

    print("Grouped stripplots plots of discrete features vs target ('good_risk').\nUseful to inform feature engineering and model selection.\n")
    plot_grouped_stripplots(data, target_column="good_risk", columns_of_interest=cols_discrete, sample_size=500, jitter=0.2)

    print("Grouped boxplots of continuous features vs target ('good_risk').\nUseful to inform feature engineering and model selection.\n")
    plot_grouped_violinplots(data, target_column="good_risk", columns_of_interest=cols_continuous, sample_size=100)

    print("General relationships between features and target ('good_risk').\nMutual information is used to explore nonlinear relationships regardless of their discrete or continuous type.\nUseful to inform feature engineering and model selection.\n")
    # This function handles automatically the correct mutual information criterion depending on the target type and the 'discrete_features' argument depending on the feature type
    compute_mutual_information(data, columns_of_interest=data.columns, target="good_risk", discrete_features_mask=[col in cols_discrete for col in data.columns], plot_heatmap=True, include_diagonal=False)

    # # For regression tasks, one can check the existence of linear or monotonic feature-target relationships. This is not the case in this analysis.
    # print("Linear relationships between features and target ('...').\nUseful to inform selection of regression models.")
    # compute_relationship(gcr, score_func="pearson", columns_of_interest=gcr.columns, target="good_risk", sample_size=1000, plot_heatmap=True, include_diagonal=True)
    #
    # print("Monotonic relationships between features and target ('...').\nUseful to inform the adoption of simple linearizing features.")
    # compute_relationship(gcr, score_func="spearman", columns_of_interest=gcr.columns, target="good_risk", sample_size=1000, plot_heatmap=True, include_diagonal=True)

    # Feature-Feature Relationships
    print("\n### Feature-Feature Relationships ###\n")
    print("Pairplot of feature vs feature.\nUseful to explain underlying processes and spot edge conditions, inform the feature elimination or feature joining.")
    # For classification tasks, use column_labels to color the samples
    plot_pairplots(data, columns_to_plot=[col for col in data.columns.to_list() if col != "good_risk"], color_labels=data["good_risk"], color_interpretation="good_risk", sample_size=100)

    print("General relationships between pairs of features.\nComputed with mutual information and reported separately for continuous 'target features' and for discrete 'target features'.\nUseful to explain underlying processes.")
    compute_mutual_information(data, columns_of_interest=data.columns, discrete_features_mask=[col in cols_discrete for col in data.columns], plot_heatmap=True, include_diagonal=False)

    print("Detail relationships between interesting pairs of discrete features.\nThis analysis is necessary because the pointplot alone is not informative enough for discrete-discrete relationships.")
    plot_jittered_scatterplot(data=data, x="checking_account", y="saving_accounts", hue="good_risk", sample_size=500)

    print("Linear relationships between pairs of continuous features.\nUseful to inform feature elimination.")
    compute_relationship(data, score_func="pearson", columns_of_interest=cols_continuous, sample_size=1000, plot_heatmap=True, include_diagonal=True)

    print("Monotonic relationships between pairs of continuous features.\nUseful to inform feature combination.")
    compute_relationship(data, score_func="spearman", columns_of_interest=cols_continuous, sample_size=1000, plot_heatmap=True, include_diagonal=True)

    print("\n--- EDA Completed ---\n")


def eda_m5salesdb(sales: DataFrame):
    """
    Performs exploratory data analysis (EDA) on one or more datasets provided as arguments.

    Args:
        *args: Variable length argument list. Each argument can be a Pandas DataFrame, a series, or any
               data structure suitable for analysis. The function can also accept arguments specifying
               details about the analysis to be performed, such as column names of interest, types of
               plots to generate, or specific statistics to compute.

    Returns:
        The function's return value would typically include summaries of the analyses performed, such as
        printed output of statistical tests, matplotlib figures or seaborn plots visualizing aspects of the
        data, or even a list or dictionary summarizing key findings. The specific return type and structure
        would depend on the implementation of the EDA tasks within the function.

    Notes:
        The sales dataset comprises:
        * the target variable (no. sold items)
        * categorical aggregation levels (state, store, item category, item department)
        * the price of each item
        * misc information such as special events

        Some interesting eda questions may be:
        * Characterize the number of items for wrt different aggregation levels. This means no. items wrt state, no. items
        wrt shop, no. items wrt category, no. items wrt shop and category
        * Characterizing the target variable wrt different aggregation layers. This means sales wrt state, sales wrt shop,
        sales wrt category, sales wrt shop and category
        * Identify potential trends in the target variable. This means sales over time, sales over time and over a
        periodic interval (heatmap)
    """

    # region Characterize items

    print("\n\nCharacterize the number of unique items across different dimensions")
    items_per_state = sales.groupby('state_id')['item_id'].nunique()
    items_per_shop = sales.groupby('store_id')['item_id'].nunique()
    items_per_category = sales.groupby('cat_id')['item_id'].nunique()
    items_per_shop_and_category = sales.groupby(['state_id', 'store_id', 'cat_id'])['item_id'].nunique().reset_index()
    items_per_shop_and_category = items_per_shop_and_category.rename(columns={"item_id": "unique_items"})
    print("Number of unique items per state:\n", items_per_state)
    print("\nNumber of unique items per shop (store):\n", items_per_shop)
    # print("\nNumber of unique items per category:\n", items_per_category)
    # print("\nNumber of unique items per shop and category combination:\n", items_per_shop_and_category)

    # # Create a bar plot showing the number of unique items for each shop and category combination.
    # plt.figure(figsize=(12, 8))
    # sns.barplot(x='store_id', y='unique_items', hue='cat_id', data=items_per_shop_and_category)
    # plt.title('Number of Unique Items per Shop and Category')
    # plt.xlabel('Store ID')
    # plt.ylabel('Number of Unique Items')
    # plt.xticks(rotation=45)
    # plt.legend(title='Category')
    # plt.tight_layout()

    # Create a sunburst chart using Plotly to visualize the hierarchical relationship of unique items across states,
    # stores, and categories. The 'path' defines the hierarchy levels, 'values' define the sizes of the sectors,
    # and 'color' reflects the count, with a gradient scale for visual distinction.
    items_per_shop_and_category['label'] = (items_per_shop_and_category['cat_id'] +
                                            ' (' + items_per_shop_and_category['unique_items'].astype(str) + ')')
    fig = px.sunburst(
        items_per_shop_and_category,
        path=['state_id', 'store_id', 'label'],
        values='unique_items',
        title='Sunburst Chart of Unique Items per Shop and Category',
        color='unique_items',
        color_continuous_scale='Blues'  # Optional: use a color scale to represent counts
    )
    plt.show(block=False)
    # fig.write_image("sunburst_chart.png")  # Save plotly figure as static image using kaleido package

    # endregion

    # region Characterize target variable

    # Explore the total number of sold items per day per store
    daily_sales_per_store = sales.groupby(["state_id", "store_id", "date"])['sold'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=daily_sales_per_store, x='store_id', y='sold', hue="state_id")
    plt.title('Distribution of Total Sold Items each Day per Store')
    plt.xlabel('Shop')
    plt.ylabel('Total Sold Items')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # Explore the historical evolution of the daily revenue per store. The daily revenue is the daily_sum(item_cost*item_sold_pieces)
    daily_revenue_per_store = sales.groupby(['store_id', 'state_id', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(
        name='total_revenue')  # this ensures that the store_id, state_id and date are columns of the new df and the aggregated column is names total_revenue (reset_index of pd.Series class)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_revenue_per_store, x='date', y='total_revenue', hue="state_id", style="store_id")
    plt.title('Historical Evolution of Daily Revenue per Store')
    plt.xlabel('Shop')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # Explore the historical evolution of the daily revenue per state and item category
    daily_revenue_per_cat_per_store = sales.groupby(['store_id', 'state_id', 'cat_id', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(
        name='total_revenue')  # this reset ensures that the store_id, state_id and date are columns of the new df and the aggregated column is names total_revenue (reset_index of pd.Series class)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_revenue_per_cat_per_store, x='date', y='total_revenue', hue="state_id", style="cat_id")
    plt.title('Historical Evolution of Daily Revenue per State and Item Category')
    plt.xlabel('Shop')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.show(block=False)

    # ... and the historical evoluation of daily revenue per state, store and item category
    g = sns.relplot(
        data=daily_revenue_per_cat_per_store,
        x='date', y='total_revenue',
        row='state_id',
        hue='store_id',  # Differentiates lines by store within each plot
        style='cat_id',  # Differentiates lines by category within each plot
        kind='line',  # Specifies that we want lineplots
        facet_kws={'sharey': False, 'sharex': True},
        height=3, aspect=2,  # Controls the size of each subplot
    )
    plt.suptitle("Historical Evolution of Daily Revenue per State, Store and Item Category")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    g.fig.tight_layout()
    plt.show(block=False)

    # Focusing on one specific store: historical evolution of daily revenue per item category, with indication of special promotions (SNAP)
    specific_store_revenue = daily_revenue_per_cat_per_store[daily_revenue_per_cat_per_store['store_id'] == "CA_1"]
    snap_dates = sales[sales['snap_CA'] == 1]['date'].drop_duplicates()
    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(data=specific_store_revenue, x='date', y='total_revenue', hue='cat_id', style='cat_id')
    plt.title('Historical Evolution of Daily Revenue per Category for Store CA_1, with SNAP Promotions')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.legend(title='Item Category')
    ax.vlines(x=snap_dates, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='grey', alpha=0.2)
    plt.tight_layout()
    plt.show(block=False)

    # Focusing on one specific store: distribution of the daily revenue for food items (which are influenced by SNAP), in SNAP vs non-SNAP days
    daily_revenue_per_cat_per_store_snap = sales.groupby(['store_id', 'state_id', 'cat_id', 'snap_CA', 'date']).apply(
        lambda group: (group['sold'] * group['sell_price']).sum()
    ).reset_index(name='total_revenue')
    specific_store_revenue_snap = daily_revenue_per_cat_per_store_snap[
        daily_revenue_per_cat_per_store['store_id'] == "CA_1"]
    plt.figure(figsize=(14, 7))
    ax = sns.violinplot(data=specific_store_revenue_snap, x='snap_CA', y='total_revenue', hue='cat_id')
    plt.title('Distribution of Daily Revenue per Item Category in SNAP vs. Non-SNAP Days for Store CA_1 ')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.xticks(ticks=[0, 1], labels=["Non-SNAP", "SNAP"], rotation=0)
    plt.legend(title='Item Category')
    plt.tight_layout()
    plt.show(block=False)

    # Find potential periods in the target variable
    print(
        "\n\nPotential cycles in the target variable (dominant freq (in no. days), corresponding normalized power in PSD):")
    average_sales_per_day = sales.groupby('d')['sold'].mean()
    dominant_cycles = spot_seasonal_cycles(
        average_sales_per_day)  # a list of tuples [(f1, pow1), (f2, pow2), ...] with pow1 >= pow2
    total_power = sum([power for _, power in dominant_cycles])
    dominant_cycles = [(freq, power / total_power) for freq, power in dominant_cycles]
    print('\n'.join(map(str, dominant_cycles[:5])))

    plt.figure()
    plt.title('Dominant Cycles in the Average Sales')
    freqs = [freq for freq, _ in dominant_cycles]
    normalized_powers = [power for _, power in dominant_cycles]
    plt.plot(normalized_powers, '-o', markersize=5, linewidth=1, markerfacecolor='blue')
    plt.ylabel('Normalized Power')
    # Add text annotations for frequencies up to the cutoff index
    cumsum_normalized_powers = np.cumsum(normalized_powers)
    for i, p in enumerate(normalized_powers):
        if cumsum_normalized_powers[i] < 0.5:
            plt.text(i + 0.1, p + 0.001, f'{freqs[i]:.2f}', fontsize=8, ha='center')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # Focusing on one specific store: historical evolution of daily revenue detailed over two time scales (daily=weekday, and weekly=week_number)
    specific_store_revenue.loc[:, 'weekday'] = specific_store_revenue['date'].dt.day_name()
    specific_store_revenue.loc[:, 'week_number'] = specific_store_revenue['date'].dt.isocalendar().week
    # If you want to check if aggregation is necessary for the values of the pivot table, use
    # (specific_store_revenue.groupby(['weekday', 'week_number']).size() == 1).all(). In this case, aggregation is
    # necessary over the cat_id.
    pivot_table = specific_store_revenue.pivot_table(index='weekday', columns='week_number', values='total_revenue',
                                                     aggfunc='sum')
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.loc[weekdays, :]  # This ensures the columns are in the correct order
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis', annot=False, fmt=".2f", linewidths=.5)
    plt.title('Weekly Revenue Distribution for Store CA_1')
    plt.xlabel('Day of the Week')
    plt.ylabel('Week Number of the Year')
    plt.tight_layout()
    plt.show(block=False)

    # endregion


def eda_passthrough(*args, **kwargs):
    return None
