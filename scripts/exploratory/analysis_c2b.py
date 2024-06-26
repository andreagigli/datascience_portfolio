import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import randint, loguniform, skew, kurtosis, ks_2samp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, PredictionErrorDisplay, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

from src.eda.eda_misc import check_outliers, evaluate_imputation, plot_pairplots, compute_relationship
from src.utils.my_math import spot_seasonal_cycles

# region

"""
### Summary of the Problem and Dataset Preparation

#### Problem Statement
The goal of this case study is to predict the conversion rate (c2b) for each advertiser and hotel combination on the next day, August 11, using the provided datasets. We are given data corresponding to August 1 to August 10 for various hotel-advertiser combinations. This involves understanding the data, identifying patterns and anomalies, and building a predictive model to estimate the conversion rate for each hotel-advertiser combination.

#### Provided Datasets
1. **hotels.csv**: Contains the following properties of hotels:
   - `hotel_id`
   - `stars`
   - `rating`
   - `n_reviews`
   - `city_id`
2. **metrics.csv**: Contains metrics related to hotel and advertiser combinations:
   - `hotel_id`
   - `advertiser_id`
   - `ymd` (date)
   - `n_clickouts`
   - `n_bookings`
"""


"""
## Load data.

Load the provided datasets.
"""
# The path to the dataset directory
dataset_path = "../../data/external/biddb/"

# Load data from specified CSV files into DataFrames for features and target variables.
hotels_df = pd.read_csv(os.path.join(dataset_path, "hotels.csv"))
metrics_df = pd.read_csv(os.path.join(dataset_path, "metrics.csv"))


"""
## Inspect the data

This section covers the preprocessing steps including inspection, merging, imputing missing values, handling missing time points, and modifying/creating variables.
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Inspect the structure of the given DataFrames.
print("Shape:")
print(hotels_df.shape)
print(metrics_df.shape)
print("\nHead:")
print(hotels_df.head())
print(metrics_df.head())
print("\nInfo:")
print(hotels_df.info(show_counts=True))
print(metrics_df.info(show_counts=True))


# Determine the nature of the data
print("Number of unique values:")
print(hotels_df.nunique())
print(metrics_df.nunique())
print("Descriptive statistics of numerical columns:")
print(hotels_df.describe())
print(metrics_df.describe())


# Determine the presence of missing values
print("The number of missing values in each column is:")
print(hotels_df.isna().sum(axis=0))
print(metrics_df.isna().sum(axis=0))

print("Rows with missing data in 'hotels_df':")
print(hotels_df.loc[hotels_df.isna().any(axis=1)])

# TODO: Determine if there are unobserved timepoints. (move here from below)


"""
## Preprocess the data

This section covers the preprocessing steps including merging datasets, imputing missing values, handling missing time points, and modifying/creating variables.
"""

# Merge the hotels and metrics DataFrames on the "hotel_id" column.
data = pd.merge(hotels_df, metrics_df, on="hotel_id", how="inner")

# Sort rows and reorder columns for ease of evaluation
data = data.sort_values(by=["hotel_id", "advertiser_id", "ymd"]).reset_index(drop=True)
data = data[['ymd', 'hotel_id', 'advertiser_id', 'city_id', 'stars', 'rating', 'n_reviews', 'n_clickouts', 'n_bookings']]

# Create conversion rate variable c2b
data["c2b"] = data["n_bookings"] / data["n_clickouts"]

selector = SelectKBest(score_func=mutual_info_regression, k="all")
for target_feature in ["rating", "n_reviews"]:
    features = ["city_id", "stars", "n_clickouts", "n_bookings"]
    selector.fit(data.dropna()[features], data.dropna()[target_feature])
    mi_scores_df = pd.DataFrame({'Feature': features, 'Mutual Information': selector.scores_})
    print(f"Mutual information between each feature and '{target_feature}':")
    print(mi_scores_df)
data["rating"] = data.groupby(["city_id"])["rating"].transform(lambda x: x.fillna(x.mean(skipna=True)))
data["n_reviews"] = data.groupby(["city_id"])["n_reviews"].transform(lambda x: x.fillna(x.mean(skipna=True)))

# Convert ymd to datetime
data["ymd"] = pd.to_datetime(data["ymd"], format='%Y%m%d')

# Optimize data types for memory efficiency
data[['hotel_id', 'advertiser_id', 'city_id']] = data[['hotel_id', 'advertiser_id', 'city_id']].astype('category')
data["stars"] = data["stars"].astype("int8")
data["rating"] = data["rating"].astype("float16")
data["n_reviews"] = data["n_reviews"].astype("int")
data["n_clickouts"] = data["n_clickouts"].astype("float32")
data["n_bookings"] = data["n_bookings"].astype("float32")
data["c2b"] = data["c2b"].astype("float32")

# The preprocessed data constitutes the inputs matrix
X = data
del data

# Define the target matrix Y as the value of c2b for the following day. Simply shift c2b by one day based on ymd.
def compute_c2b_tomorrow(group):
    # Create a date range from the min to the max date in the group
    date_range = pd.date_range(start=group['ymd'].min(), end=group['ymd'].max())
    tmp_df = pd.DataFrame(date_range, columns=['ymd'])
    # Merge the group with the accessory dataframe to fill in missing dates
    tmp_df = tmp_df.merge(group[['ymd', 'c2b']], on='ymd', how='left')
    # Shift the c2b values
    tmp_df['c2b_tomorrow'] = tmp_df['c2b'].shift(-1)
    # Merge the shifted c2b values back with the original group
    merged_group = group.merge(tmp_df[['ymd', 'c2b_tomorrow']], on='ymd', how='left')
    return merged_group["c2b_tomorrow"]

Y = X.groupby(['hotel_id', 'advertiser_id'], group_keys=False).apply(compute_c2b_tomorrow).reset_index(drop=True)
Y = pd.DataFrame(Y, columns=['c2b_tomorrow'])





"""
## Exploratory Data Analysis
"""
print(f"Preprocessed dataset 'data': \n{X.head(10)}")
print(X.info())

# Define auxiliary variables
cols_numerical = ["rating", "n_reviews", "n_clickouts", "n_bookings", "c2b"]
print(f"Numerical columns: {cols_numerical}")

cols_categorical = [col for col in X.columns if col not in cols_numerical]
print(f"Categorical columns: {cols_categorical}")

# Remove samples with nan target for certain EDA tasks
X_no_nan = X.drop(index=Y[Y['c2b_tomorrow'].isna()].index).reset_index(drop=True)
Y_no_nan = Y.dropna(axis=0).reset_index(drop=True)

# Subsampled data for certain EDA tasks. It is sensible to only subsample the data with targets different from zero.
sample_size = 100
if sample_size and sample_size < len(X):
    X_sampled = X_no_nan.iloc[::len(X)//sample_size].reset_index(drop=True)
    Y_sampled = Y_no_nan.iloc[::len(X)//sample_size].reset_index(drop=True)
else:
    X_sampled = X
    Y_sampled = Y

# Remove ymd from the data temporarily for certain EDA tasks
X_sampled_no_ymd = X_sampled.drop(columns=["ymd"])




# # Create figure for continuous variables
# fig_continuous, axs_continuous = plt.subplots(nrows=len(cols_numerical), figsize=(10, 5 * len(cols_numerical)))
# plt.suptitle("Distribution of the Continuous Variables")
#
# if len(cols_numerical) == 1:
#     axs_continuous = [axs_continuous]
#
# for i, var in enumerate(cols_numerical):
#     axs_continuous[i].set_title(f'{var} Distribution')
#     sns.violinplot(data=X, x=var, ax=axs_continuous[i], inner=None, orient="h", color='lightgray')
#     sns.boxplot(data=X, x=var, ax=axs_continuous[i], width=0.1, fliersize=3, whis=1.5, orient="h", color='blue')
#     skw = skew(X[var].dropna())
#     kurt = kurtosis(X[var].dropna())
#     axs_continuous[i].text(0.95, 0, f'Skew: {skw:.2f}\nKurt: {kurt:.2f}', ha='right', va='bottom', transform=axs_continuous[i].transAxes)
#     axs_continuous[i].set_xlabel("")
# plt.tight_layout(rect=(0, 0, 1, 0.95))
#
# # Create figure for discrete variables
# fig_discrete, axs_discrete = plt.subplots(nrows=len(cols_categorical), figsize=(10, 5 * len(cols_categorical)))
# plt.suptitle("Distribution of the Discrete Variables")
#
# if len(cols_categorical) == 1:
#     axs_discrete = [axs_discrete]
#
# for i, var in enumerate(cols_categorical):
#     sns.countplot(x=X[var], ax=axs_discrete[i] if len(cols_categorical) > 1 else axs_discrete)
#     axs_discrete[i].set_title(f'{var} Count Plot')
# plt.tight_layout(rect=(0, 0, 1, 0.95))
#
# """
# The features n_reviews, n_clickouts, n_bookings, c2b are strongly skewed to the right. Feature transformation is advised. Target (c2b_tomorrow) transformation is also in order, paying attention to back-transform both prediction and ground truth before performance evaluation.
# """
#
#
#
#
# print("\n### Outliers in the continuous features ###\n")
# print("Detecting outliers with Random Isolation Forest:")
#
# # Detect outliers
# rif = IsolationForest(n_estimators=100, random_state=0).fit(X_sampled[cols_numerical])
# outliers = rif.predict(X_sampled[cols_numerical]) == -1
# outliers = pd.Series(outliers.astype(int), name='Outlier', index=X_sampled.index)
# print(f"Random Isolation Forest identified {outliers.sum()} among {X_sampled.shape[0]} samples.\n")
#
# print("Outliers profiling: are these actual outliers or valid extreme data points?")
#
# # print("Descriptive statistics of outliers:")
# # print(X_sampled.loc[outliers == 1, cols_numerical].describe())
# # print("Descriptive statistics of non-outliers:")
# # print(X_sampled.loc[outliers == 0, cols_numerical].describe())
#
# # Compute 2D projections of the continuous features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_sampled[cols_numerical])
# # PCA Analysis
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
# explained_variance = np.cumsum(pca.explained_variance_ratio_)[:2]
# # t-SNE Analysis
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(X_scaled)
#
# # Plot PCA and t-SNE data projections with highlighted outliers
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# plt.suptitle(f"2D projection of Outliers vs Non-Outliers (Random Isolation Forest). \nEstimated Outliers percentage: {outliers.sum() / X_sampled.shape[0] * 100}%.")
# # PCA Plot
# ax1.set_title(f'PCA - Top 2 Components ({explained_variance[0]:.2f}, {explained_variance[1]:.2f} Explained Variance)')
# ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=outliers.values)
# ax1.set_xlabel('Component 1')
# ax1.set_ylabel('Component 2')
# ax1.grid(True)
# # t-SNE Plot
# ax2.set_title('t-SNE')
# ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7, c=outliers.values)
# ax2.set_xlabel('Dimension 1')
# ax2.set_ylabel('Dimension 2')
# ax2.grid(True)
# plt.tight_layout()
#
# # Visualize distribution of outliers vs non-outliers
# fig, axs = plt.subplots(nrows=len(cols_numerical), figsize=(10, 3 * len(cols_numerical)))
# plt.suptitle("Distribution of Outliers vs Non-Outliers (Random Isolation Forest). \nEstimated Outliers percentage: {outliers.sum() / X_sampled.shape[0] * 100}%.")
#
# if len(cols_numerical) == 1:
#     axs = [axs]
#
# for i, var in enumerate(cols_numerical):
#     axs[i].set_title(f'{var} Distribution by Outlier Status')
#     sns.violinplot(data=X_sampled, x=outliers, y=var, ax=axs[i], split=True, hue=outliers, palette='muted')
#     sns.stripplot(data=X_sampled[cols_numerical], x=outliers, y=var, ax=axs[i], color='black', size=2, jitter=True)
# plt.tight_layout(rect=(0, 0, 1, 0.95))
#
# """While certain differences in the median values of the Non-Outliers vs Outliers groups can be observed for the columns "rating" and "n_reviews", the difference in median is most striking for the "n_bookings" (and consequently c2b) columns. It appears that most non-zero "n_bookings" are considered outliers. Arguably, this result may be driven by the zero-inflated nature of the column "n_booking", and these samples should be considered valid points (i.e., Non-Outliers)."""
#
#
#
# print("\n### Feature-Target Relationships ###\n")
# print(f"Pairplot of features vs target ('c2b_tomorrow').\nUseful to inform feature engineering and model selection.\n")
# # Generate pairwise scatter plots
# fig_pairplot = sns.pairplot(data=pd.concat([X_sampled, Y_sampled], axis=1),
#                             x_vars=X_sampled.columns.difference(["ymd"]),
#                             y_vars=["c2b_tomorrow"],
#                             plot_kws={"alpha": 0.5})
# plt.suptitle(f"Feature-Target Scatter Plots\nTarget: c2b_tomorrow", y=1.02)
# plt.tight_layout(rect=[0, 0, 0.95, 0.95])
#
#
#
#
# def compute_and_plot_correlation(data: pd.DataFrame, target: pd.Series, method: str = 'pearson'):
#     """
#     Computes and plots the correlation heatmap between numerical features and the target.
#
#     Args:
#         data (pd.DataFrame): The DataFrame containing the numerical features.
#         target (pd.Series): The target variable.
#         method (str): The correlation method to use ('pearson' or 'spearman').
#
#     Returns:
#         pd.DataFrame: DataFrame containing the correlation coefficients.
#     """
#     # Compute correlation of each column of data with the target
#     corr_with_target = data.apply(lambda x: x.corr(target, method=method))
#
#     # Convert to DataFrame for heatmap plotting
#     corr_df = pd.DataFrame(corr_with_target, columns=[f'{method}_correlation'])
#
#     # Plotting correlation heatmap
#     plt.figure(figsize=(10, 1))
#     sns.heatmap(corr_df.T, annot=True, cmap="viridis", cbar=False, fmt=".2f")
#     plt.title(f"{method.capitalize()} correlation with target")
#     plt.xticks(rotation=30, horizontalalignment="right")
#     plt.tight_layout()
#
#     return corr_df
#
# # Compute and plot Pearson correlation
# print(f"Pearson correlations with the target ({'c2b_tomorrow'}):")
# pearson_corr_df = compute_and_plot_correlation(X_sampled_no_ymd[cols_numerical], Y_sampled['c2b_tomorrow'], method='pearson')
#
# # Compute and plot Spearman correlation
# print(f"Spearman correlations with the target ({'c2b_tomorrow'}):")
# spearman_corr_df = compute_and_plot_correlation(X_sampled_no_ymd[cols_numerical], Y_sampled['c2b_tomorrow'], method='spearman')
#
#
# """Characterize the zero-inflation of c2b"""
# # Calculate the overall zero-inflation percentage
# overall_zero_inflation = (X['c2b'] == 0).mean() * 100
#
# def compute_zero_inflation_ratio(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
#     """
#     Computes the zero-inflation ratio for the 'c2b' column within specified groups.
#
#     Args:
#         data (pd.DataFrame): The DataFrame containing the data.
#         group_col (str): The column name to group by.
#
#     Returns:
#         pd.DataFrame: A DataFrame containing the group, zero counts, total counts, and zero-inflation ratio.
#     """
#     zero_counts = data[data['c2b'] == 0].groupby(group_col).size().reset_index(name='zero_count')
#     total_counts = data.groupby(group_col).size().reset_index(name='total_count')
#     merged_counts = pd.merge(total_counts, zero_counts, on=group_col, how='left')
#     merged_counts['zero_count'] = merged_counts['zero_count'].fillna(0)
#     merged_counts['zero_inflation_ratio'] = merged_counts['zero_count'] / merged_counts['total_count']
#     return merged_counts
#
# # Compute zero inflation ratios for each group
# zero_inflation_ymd = compute_zero_inflation_ratio(X, 'ymd')
# zero_inflation_hotel = compute_zero_inflation_ratio(X, 'hotel_id')
# zero_inflation_stars = compute_zero_inflation_ratio(X, 'stars')
# zero_inflation_advertiser = compute_zero_inflation_ratio(X, 'advertiser_id')
# zero_inflation_city = compute_zero_inflation_ratio(X, 'city_id')
#
#
# # Plot the zero inflation ratios
# fig, axs = plt.subplots(5, 1, figsize=(10, 2*5), sharex=False)
# fig.suptitle(f'{overall_zero_inflation:.2f}% of the observed c2b are zero.\nPossible bias in such zero-inflation can be characterized group-wise.', fontsize=16)
#
# sns.lineplot(data=zero_inflation_ymd, x='ymd', y='zero_inflation_ratio', marker='o', ax=axs[0])
# axs[0].set_title('Ratio of Zero-Valued c2b per ymd')
# axs[0].set_xticks(zero_inflation_ymd['ymd'])
# axs[0].set_xticklabels(zero_inflation_ymd['ymd'].dt.strftime('%Y-%m-%d'), rotation=45)
#
# sns.lineplot(data=zero_inflation_hotel, x='hotel_id', y='zero_inflation_ratio', marker='o', ax=axs[1])
# axs[1].set_title('Ratio of Zero-Valued c2b per Hotel')
#
# sns.lineplot(data=zero_inflation_stars, x='stars', y='zero_inflation_ratio', marker='o', ax=axs[2])
# axs[2].set_title('Ratio of Zero-Valued c2b per Stars')
#
# sns.lineplot(data=zero_inflation_advertiser, x='advertiser_id', y='zero_inflation_ratio', marker='o', ax=axs[3])
# axs[3].set_title('Ratio of Zero-Valued c2b per Advertiser')
#
# sns.lineplot(data=zero_inflation_city, x='city_id', y='zero_inflation_ratio', marker='o', ax=axs[4])
# axs[4].set_title('Ratio of Zero-Valued c2b per City')
#
# plt.tight_layout(rect=[0, 0, 1, 0.97])
#
#
#
#
# # Identify the most dominant seasonal cycles using FFT
# # Compute average c2b per day
# average_c2b_per_day = X.groupby('ymd')['c2b'].mean().values
#
# # Perform FFT and get frequencies and amplitudes
# amplitudes = np.abs(np.fft.rfft(average_c2b_per_day))
# frequencies = np.fft.rfftfreq(len(average_c2b_per_day), d=1)  # d = 1 indicates daily data
#
# # Exclude the zero frequency (trend component)
# frequencies = frequencies[1:]
# amplitudes = amplitudes[1:]
#
# # Calculate the periods from frequencies and pair them with amplitudes
# periods_amplitudes = [(1 / freq, amp) for freq, amp in zip(frequencies, amplitudes) if freq != 0]
#
# # Sort the periods and amplitudes by amplitude in descending order
# periods_amplitudes.sort(key=lambda x: x[1], reverse=True)
#
# # Normalize power by dividing each amplitude by the total power
# total_power = sum([power for _, power in periods_amplitudes])
# normalized_periods_amplitudes = [(freq, power / total_power) for freq, power in periods_amplitudes]
#
# # Print the top 5 dominant cycles
# print("\n\nPotential cycles in the target variable (dominant freq (in no. days), corresponding normalized power in PSD):")
# print('\n'.join(map(str, normalized_periods_amplitudes[:5])))
#
# # Visualize the dominant cycles
# plt.figure()
# plt.title('Dominant Cycles in the Average c2b')
# freqs = [freq for freq, _ in normalized_periods_amplitudes]
# normalized_powers = [power for _, power in normalized_periods_amplitudes]
# plt.plot(freqs, normalized_powers, 'o', markersize=5, linewidth=1, markerfacecolor='blue')
# plt.ylabel('Normalized Power')
# plt.xlabel('Period (in days)')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# """One observes cyclical components with a frequency of 5 days (normalized power = 0.38) and 3 days (normalized power = 0.31). It is advisable to use lag and windowed features with these durations."""
#
#
#
#
#
# """Characterize the distribution of the unobserved samples"""
# # List unique dates
# unique_dates = pd.date_range(start=X['ymd'].min(), end=X['ymd'].max())
#
# # Compute theoretical number of samples
# n_combinations = X[['hotel_id', 'advertiser_id']].drop_duplicates().shape[0]
#
# # Function to compute missing ratio
# def compute_missing_ratio(df, group_col, total_possible):
#     observed_counts = df.groupby(group_col).size().reset_index(name='observed')
#     observed_counts['total'] = total_possible
#     observed_counts['missing_ratio'] = 1 - (observed_counts['observed'] / observed_counts['total'])
#     return observed_counts
#
# # Compute missing ratios for ymd, hotel_id, stars, advertiser_id, and city_id
# total_possible_ymd = n_combinations
# total_possible_hotel = X['hotel_id'].nunique() * len(unique_dates)
# total_possible_stars = X['hotel_id'].nunique() * len(unique_dates)
# total_possible_advertiser = X['hotel_id'].nunique() * len(unique_dates)
# total_possible_city = X['hotel_id'].nunique() * len(unique_dates)
#
# # Grouped data
# ymd_grouped = X.groupby('ymd')
# hotel_grouped = X.groupby('hotel_id')
# stars_grouped = X.groupby('stars')
# advertiser_grouped = X.groupby('advertiser_id')
# city_grouped = X.groupby('city_id')
#
# # Compute missing ratios for each group
# missing_ratios_ymd = compute_missing_ratio(X, 'ymd', total_possible_ymd)
# missing_ratios_hotel = compute_missing_ratio(X, 'hotel_id', total_possible_hotel)
# missing_ratios_stars = compute_missing_ratio(X, 'stars', total_possible_stars)
# missing_ratios_advertiser = compute_missing_ratio(X, 'advertiser_id', total_possible_advertiser)
# missing_ratios_city = compute_missing_ratio(X, 'city_id', total_possible_city)
#
# # Calculate the overall unobserved percentage
# total_possible_samples = n_combinations * len(unique_dates)
# observed_samples = X.dropna(subset=['n_clickouts', 'n_bookings']).shape[0]  # Adjust columns as needed
# overall_unobserved_percentage = (1 - (observed_samples / total_possible_samples)) * 100
#
# # Plot the missing ratios
# fig, axs = plt.subplots(5, 1, figsize=(14, 30), sharex=False)
# fig.suptitle(f'{overall_unobserved_percentage:.2f}% of the data samples are unobserved.\nPossible bias in such missing instances can be characterized group-wise.', fontsize=16)
#
# sns.lineplot(ax=axs[0], data=missing_ratios_ymd, x='ymd', y='missing_ratio', marker='o')
# axs[0].set_title('Ratio of Missing Instances per ymd')
# axs[0].set_xticklabels(axs[0].get_xticks(), rotation=45)
#
# sns.lineplot(ax=axs[1], data=missing_ratios_hotel, x='hotel_id', y='missing_ratio', marker='o')
# axs[1].set_title('Ratio of Missing Instances per Hotel')
#
# sns.lineplot(ax=axs[2], data=missing_ratios_stars, x='stars', y='missing_ratio', marker='o')
# axs[2].set_title('Ratio of Missing Instances per Stars')
#
# sns.lineplot(ax=axs[3], data=missing_ratios_advertiser, x='advertiser_id', y='missing_ratio', marker='o')
# axs[3].set_title('Ratio of Missing Instances per Advertiser')
# axs[3].set_xticklabels(axs[3].get_xticks(), rotation=90)
#
# sns.lineplot(ax=axs[4], data=missing_ratios_city, x='city_id', y='missing_ratio', marker='o')
# axs[4].set_title('Ratio of Missing Instances per City')
# axs[4].set_xticklabels(axs[4].get_xticks(), rotation=90)
#
# plt.tight_layout(rect=[0, 0, 1, 0.97])
# """TODO COMMENT: about 50% of the samples are not observed.
# IMPLICATION: Imputation risks to introduce too much noise. One has two options: use imputation controlling for further exacerbation of zero-inflation or only imputing the missing values for august 10, which can be used as target for the development set (use X on august 9 to predict c2b_tomorrow from august 10) and as inputs for the test set (use X on august 10 to compute c2b_tomorrow from august 11).  """
#





"""IMPUTATION"""

# Exclude the original 'ymd' column from the columns to be imputed
impute_cols = ['n_clickouts', 'n_bookings']
all_features = ['ymd', 'advertiser_id', 'city_id', 'stars', 'rating', 'n_reviews', 'n_clickouts', 'n_bookings']

# Prepare the DataFrame for imputation
complete_data = pd.DataFrame()
all_dates = pd.DataFrame(X["ymd"].unique(), columns=["ymd"])

for (hotel_id, advertiser_id), group in X.groupby(['hotel_id', 'advertiser_id'], observed=True):
    group_complete = group.merge(all_dates, on='ymd', how='outer')
    group_complete['hotel_id'] = hotel_id
    group_complete['advertiser_id'] = advertiser_id
    group_complete['city_id'] = group['city_id'].iloc[0]
    group_complete['stars'] = group['stars'].iloc[0]
    group_complete['rating'] = group['rating'].iloc[0]
    group_complete['n_reviews'] = group['n_reviews'].iloc[0]
    complete_data = pd.concat([complete_data, group_complete], ignore_index=True)

complete_data['imputed_day'] = complete_data[impute_cols].isna().any(axis=1).astype("uint8")

# Convert 'ymd' to numeric format YYYYMMDD
complete_data['ymd'] = complete_data['ymd'].dt.strftime('%Y%m%d').astype(int)

def impute_group(group):
    imputer = KNNImputer(n_neighbors=2, weights="distance")
    imputed_data = imputer.fit_transform(group[all_features])
    imputed_df = pd.DataFrame(imputed_data, columns=all_features, index=group.index)
    group[impute_cols] = imputed_df[impute_cols]
    return group

complete_data = complete_data.groupby(['hotel_id']).apply(impute_group).reset_index(drop=True)

# Convert 'ymd' back to datetime format
complete_data['ymd'] = pd.to_datetime(complete_data['ymd'].astype(str), format='%Y%m%d')

# Recompute the c2b for the imputed samples
complete_data["c2b"] = complete_data["n_bookings"] / complete_data["n_clickouts"]

# Create the target variable c2b_tomorrow by shifting c2b (predict c2b_tomorrow using information from day t)
complete_target = complete_data[['ymd', 'hotel_id', 'advertiser_id', 'c2b']].copy()
complete_target["c2b_tomorrow"] = complete_target.groupby(["hotel_id", "advertiser_id"], observed=True)["c2b"].shift(-1).fillna(-1)

# Create an indicator 'imputed_day_tomorrow' that the 'c2b_tomorrow' was imputed, by shifting 'imputed_day'
complete_data['imputed_day_tomorrow'] = complete_data.groupby(["hotel_id", "advertiser_id"], observed=True)["imputed_day"].shift(-1).fillna(1).astype("uint8")

# Optimize data types of new columns for memory efficiency
complete_data[['imputed_day', 'imputed_day_tomorrow']] = complete_data[['imputed_day', 'imputed_day_tomorrow']].astype('category')

# Sort columns in complete_data
complete_data = complete_data.sort_values(by=["hotel_id", "advertiser_id", "ymd"]).reset_index(drop=True)
complete_data = complete_data[['ymd', 'hotel_id', 'advertiser_id', 'city_id', 'stars', 'rating', 'n_reviews', 'n_clickouts', 'n_bookings', 'imputed_day', 'c2b']]

# Sort columns in complete_target
complete_target = complete_target.sort_values(by=["hotel_id", "advertiser_id", "ymd"]).reset_index(drop=True)
complete_target = complete_target[['c2b_tomorrow']]

# Set X to the complete data
X = complete_data

# Override Y with the target DataFrame
Y = complete_target

del complete_data, complete_target



def evaluate_imputation(data: pd.DataFrame, variable: str, imputed_mask: str) -> None:
    """
    Evaluates the imputation of a specified variable by comparing the distributions
    of observed and imputed values. Creates overlapping histograms using Seaborn
    and writes the results of the Kolmogorov-Smirnov test in the title of the image.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        variable (str): The name of the variable to evaluate.
        imputed_mask (str): Column name in the DataFrame indicating which values are imputed.
    """
    imputed_mask = data[imputed_mask]

    observed_values = data.loc[imputed_mask == 0, variable]
    imputed_values = data.loc[imputed_mask == 1, variable]

    # Perform Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(observed_values, imputed_values)

    # Create overlapping histograms
    plt.figure(figsize=(12, 6))
    sns.histplot(observed_values, bins=30, color='blue', label='Observed', stat="density")
    sns.histplot(imputed_values, bins=30, color='orange', label='Imputed', stat="density")
    plt.xlabel(variable)
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'Distribution of Observed and Imputed {variable}\nKS test statistic={ks_stat:.4f}, p-value={p_value:.4e}')

# Evaluate imputation for 'n_clickouts' and 'n_bookings'
evaluate_imputation(X, 'n_clickouts', 'imputed_day')
evaluate_imputation(X, 'n_bookings', 'imputed_day')























"""
## Feature Extraction

This section covers the feature extraction steps. Features are extracted from the given dataset to create a design matrix (X)
and a target matrix (Y).

The feature extraction performs the following steps:
1. Transform certain features to reduce right skewness (Yeo-Johnson transformation to stretch lower values to the right).
2. Groups the data by `hotel_id` and `advertiser_id`. For each group, copies existing features for the current day,
computes lag features for conversion rate (c2b) for specified lag periods (e.g., 1, 3, 5, 7 days), computes rolling
window statistics (mean) for specified window sizes (e.g., 3, 5, 7 days).
3. Adds mean encodings for the c2b by hotel, by advertiser, by city, and by stars.
4. Returns two data frames: one with input features (X) and the other with target value c2b_tomorrow (Y).
"""
# # Apply power transformation (Yeo-Johnson) to the skewed features. This stretches the distribution towards higher values.
# # skewed_features = ["n_reviews", "n_clickouts", "n_bookings", "c2b"]
# skewed_features = ["n_reviews"]  # Do not transform n_clickouts and n_bookings because yeo-johnson may mess up a zero-inflated distribution even more
# pt_features = PowerTransformer(method='yeo-johnson', standardize=False)
# X[skewed_features] = pt_features.fit_transform(X[skewed_features])
# X[skewed_features] = X[skewed_features].applymap(lambda x: 0 if abs(x) < 1e-10 else x)  # Post-process to set very small negative values to zero

# Correct mild negative skewness of "rating" with stretching transformation
X["rating"] = np.power(X["rating"], 2)

# Correct positive skewness of "n_reviews" with squashing transformation
X["n_reviews"] = np.log1p(X["n_reviews"])



# Add weekday column as a seasonality indicator
X['weekday'] = X['ymd'].dt.dayofweek  # Monday = 0, Sunday = 6


# Group the X by `hotel_id` and `advertiser_id`
grouped = X.groupby(['hotel_id', 'advertiser_id'])

# Create lag features for conversion rate (c2b) for specified lag periods (e.g., 1, 3, 7 days)
lags = [1, 2, 3, 5, 6, 7]
for lag in lags:
    X[f'c2b_lag_{lag}'] = grouped['c2b'].shift(lag)

# Compute rolling window statistics (mean) for specified window sizes (e.g., 3, 7 days)
windows = [2, 3, 4, 5, 6, 7]
for window in windows:
    X[f'c2b_roll_mean_{window}'] = grouped['c2b'].rolling(window=window).mean().reset_index(level=['hotel_id', 'advertiser_id'], drop=True)

# Mean encodings for the c2b by hotel, by advertiser, by city, and by stars
X['c2b_mean_hotel'] = X.groupby('hotel_id')['c2b'].transform('mean')
X['c2b_mean_advertiser'] = X.groupby('advertiser_id')['c2b'].transform('mean')
X['c2b_mean_city'] = X.groupby('city_id')['c2b'].transform('mean')
X['c2b_mean_stars'] = X.groupby('stars')['c2b'].transform('mean')# Add weekday column as a seasonality indicator
X['weekday'] = X['ymd'].dt.dayofweek  # Monday = 0, Sunday = 6


# Group the X by `hotel_id` and `advertiser_id`
grouped = X.groupby(['hotel_id', 'advertiser_id'])

# Create lag features for conversion rate (c2b) for specified lag periods (e.g., 1, 3, 7 days)
lags = [1, 2, 3, 5, 6, 7]
for lag in lags:
    X[f'c2b_lag_{lag}'] = grouped['c2b'].shift(lag)

# Compute rolling window statistics (mean) for specified window sizes (e.g., 3, 7 days)
windows = [2, 3, 4, 5, 6, 7]
for window in windows:
    X[f'c2b_roll_mean_{window}'] = grouped['c2b'].rolling(window=window).mean().reset_index(level=['hotel_id', 'advertiser_id'], drop=True)

# Mean encodings for the c2b by hotel, by advertiser, by city, and by stars
X['c2b_mean_hotel'] = X.groupby('hotel_id')['c2b'].transform('mean')
X['c2b_mean_advertiser'] = X.groupby('advertiser_id')['c2b'].transform('mean')
X['c2b_mean_city'] = X.groupby('city_id')['c2b'].transform('mean')
X['c2b_mean_stars'] = X.groupby('stars')['c2b'].transform('mean')

# Display the first few rows of the feature DataFrame to verify the extraction
print("Features DataFrame (X):")
print(X.head())

print("\nTarget DataFrame (Y):")
print(Y.head())


"""
## Split Data into Training, Development, and Test Sets.

The dataset is split into training, development, and test sets based on specific dates:
1. Training Set: Data from August 8. Note that data from August 8-th includes lag and windowed features of up to 7 days prior.
2. Development Set: Data from August 9. Used to evaluate the model on the prediction obtained for August 10, for which I have the ground truth.  
3. Test Set: Data from August 10. Used to predict the c2b for August 11. This will not be evaluated.

Validation set is not explicitly extracted; K-Fold cross-validation will be used on the training set for hyperparameter optimization.
"""
# Define the dates for the splits
train_dates = [pd.Timestamp("2023-08-08")]  # The first useful training day depends on the largest lag/window_size used in the feature extraction.
dev_dates = [pd.Timestamp("2023-08-09")]
test_dates = [pd.Timestamp("2023-08-10")]

# Split the data based on the defined dates
X_train = X[X["ymd"].isin(train_dates)]
Y_train = Y[X["ymd"].isin(train_dates)]
ymd_X_train = X_train["ymd"]
X_train = X_train.drop(columns=["ymd"])

X_dev = X[X["ymd"].isin(dev_dates)]
Y_dev = Y[X["ymd"].isin(dev_dates)]
ymd_X_dev = X_dev["ymd"]
X_dev = X_dev.drop(columns=["ymd"])

X_test = X[X["ymd"].isin(test_dates)]
Y_test = Y[X["ymd"].isin(test_dates)]
ymd_X_test = X_test["ymd"]
X_test = X_test.drop(columns=["ymd"])

# Check each matrix shape and absence of nan
print("Training set:")
print(f"X_train shape: {X_train.shape}")
print(f"X_train contains nans: {X_train.isna().any().any()}")  # nan may appear if the chosen lag/window_size exceeds the first training day - 1
print(f"Y_train shape: {Y_train.shape}")

print("\nDevelopment set:")
print(f"X_dev shape: {X_dev.shape}")
print(f"X_dev contains nans: {X_dev.isna().any().any()}")
print(f"Y_dev shape: {Y_dev.shape}")

print("\nTest set:")
print(f"X_test shape: {X_test.shape}")
print(f"X_test contains nans: {X_test.isna().any().any()}")
print(f"Y_test shape: {Y_test.shape}")


# endregion

"""Model Tuning and Training

....
"""
# def zero_inflated_mae(y_true: np.array, y_pred: np.array):
#     # Split the data into two subsets: ground truth equals zero and ground truth greater than zero
#     y_true_zero, y_pred_zero = y_true[y_true == 0], y_pred[y_true == 0]
#     y_true_gt_zero, y_pred_gt_zero = y_true[y_true > 0], y_pred[y_true > 0]
#
#     # Calculate the MSE for each subset
#     mse_zero = mean_absolute_error(y_true_zero, y_pred_zero)
#     mse_gt_zero = mean_absolute_error(y_true_gt_zero, y_pred_gt_zero)
#
#     # Average the two MSE values
#     avg_mse = (mse_zero + mse_gt_zero) / 2
#
#     return avg_mse
# neg_zero_inflated_mae_scorer = make_scorer(zero_inflated_mae, greater_is_better=False)
#
# metric_fn = zero_inflated_mae  # mean_absolute_error
# metric_name = "ZIMAE"  # "MAE"
# metric_score = neg_zero_inflated_mae_scorer  # "neg_mean_absolute_error"

# def zero_inflated_nrmse_nmae(y_true: np.array, y_pred: np.array) -> float:
#     """
#     Evaluates the prediction error for zero-inflated data by combining normalized MAE for zero-valued y_true and
#     normalized RMSE for non-zero-valued y_true. The normalizing factor is the range of y_true. The weights are computed
#     based on the ratio of zero-valued y_true in the data.
#     The combined error is computed as:
#     combined_error = weight_nmae * NMAE_zero + (1 - weight_nmae) * NRMSE_nonzero
#     where weight_nmae = len(y_true_zero) / len(y_true).
#     Args:
#         y_true (np.array): True values.
#         y_pred (np.array): Predicted values.
#     Returns:
#         float: Combined normalized error.
#     """
#     y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
#
#     # Split the data into two subsets: ground truth equals zero and ground truth greater than zero
#     y_true_zero, y_pred_zero = y_true[y_true == 0], y_pred[y_true == 0]
#     y_true_nonzero, y_pred_nonzero = y_true[y_true > 0], y_pred[y_true > 0]
#
#     # Calculate the MAE for the zero subset
#     mae_zero = mean_absolute_error(y_true_zero, y_pred_zero) if len(y_true_zero) > 0 else 0
#
#     # Calculate the RMSE for the non-zero subset
#     rmse_nonzero = root_mean_squared_error(y_true_nonzero, y_pred_nonzero) if len(y_true_nonzero) > 0 else 0
#
#     # Normalize both metrics if the range of y_true is greater than zero
#     range_y_true = np.ptp(y_true)
#     if range_y_true > 0:
#         nmae_zero = mae_zero / range_y_true
#         nrmse_nonzero = rmse_nonzero / range_y_true
#     else:
#         nmae_zero = mae_zero
#         nrmse_nonzero = 0
#
#     # Compute weights
#     weight_nmae = len(y_true_zero) / len(y_true)
#     weight_nrmse = 1 - weight_nmae
#
#     # Combine the two errors using computed weights
#     combined_error = weight_nmae * nmae_zero + weight_nrmse * nrmse_nonzero
#     return combined_error
#
#
# # Create the scorer for use in scikit-learn
# neg_zero_inflated_nrmse_nmae_scorer = make_scorer(zero_inflated_nrmse_nmae, greater_is_better=False)
#
# # Set the metric function and name for convenience
# metric_fn = zero_inflated_nrmse_nmae
# metric_name = "ZI_NRMSE_NMAE"
# metric_score = neg_zero_inflated_nrmse_nmae_scorer


# def symmetric_mean_percentage_error(y_true: np.array, y_pred: np.array):
#     # Ensure the inputs are numpy arrays for element-wise operations
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#
#     numerator = np.abs(y_true - y_pred)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#
#     # Use np.where to handle cases where the denominator is zero
#     # If the denominator is zero, set the sMAPE component to zero
#     smape = np.where(denominator == 0, 0, numerator / denominator)
#
#     smape = np.mean(smape) * 100
#
#     return smape
# neg_symmetric_mean_percentage_error_scorer = make_scorer(symmetric_mean_percentage_error, greater_is_better=False)
#
# metric_fn = symmetric_mean_percentage_error
# metric_name = "sMAPE"
# metric_score = neg_symmetric_mean_percentage_error_scorer

# def range_normalized_mean_absolute_error(y_true: np.array, y_pred: np.array):
#     mae = mean_absolute_error(y_true, y_pred)
#     rnmae = mae / np.ptp(y_true)
#     return rnmae
# neg_range_normalized_mean_absolute_error_scorer = make_scorer(range_normalized_mean_absolute_error, greater_is_better=False)
#
# metric_fn = range_normalized_mean_absolute_error
# metric_name = "RNMAE"
# metric_score = neg_range_normalized_mean_absolute_error_scorer

metric_fn = mean_absolute_error
metric_name = "MAE"
metric_score = "neg_mean_absolute_error"

# metric_fn = root_mean_squared_error
# metric_name = "RMSE"
# metric_score = "neg_root_mean_squared_error"

# Compute baseline performance with a DummyRegressor
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, Y_train)
Y_train_baseline_pred = baseline_model.predict(X_train)
Y_dev_baseline_pred = baseline_model.predict(X_dev)
Y_test_baseline_pred = baseline_model.predict(X_test)
# Compute baseline performance for training, development, and test sets
baseline_train_perf = metric_fn(Y_train.values.squeeze(), Y_train_baseline_pred)
baseline_train_rmse = root_mean_squared_error(Y_train.values.squeeze(), Y_train_baseline_pred)
baseline_train_r2 = r2_score(Y_train.values.squeeze(), Y_train_baseline_pred)
baseline_dev_perf = metric_fn(Y_dev.values.squeeze(), Y_dev_baseline_pred)
baseline_dev_rmse = root_mean_squared_error(Y_dev.values.squeeze(), Y_dev_baseline_pred)
baseline_dev_r2 = r2_score(Y_dev.values.squeeze(), Y_dev_baseline_pred)

class ClippedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_regressor=None):
        self.base_regressor = base_regressor if base_regressor is not None else Ridge()

    def fit(self, X, y):
        self.base_regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.base_regressor.predict(X)
        return np.clip(predictions, a_min=0, a_max=None)

    def get_params(self, deep=True):
        return {'base_regressor': self.base_regressor}

    def set_params(self, **params):
        base_regressor_params = {k.split('__', 1)[1]: v for k, v in params.items() if k.startswith('base_regressor__')}
        self.base_regressor.set_params(**base_regressor_params)
        return self

# Define the model
model = ClippedRegressor(base_regressor=lgb.LGBMRegressor())

# Define the hyperparameters to optimize

# # param_distributions for Random Forest
# param_distributions = {
#     'base_regressor__n_estimators': randint(100, 500),  # Number of trees in the forest
#     'base_regressor__max_depth': randint(10, 30),  # Maximum depth of the tree
# }


# Create kfold split based on hotel_id
group_kfold = GroupKFold(n_splits=5)
groups = X_train['hotel_id']
splits = list(group_kfold.split(X_train, Y_train, groups=groups))

# param_distributions for LGBMRegressor
param_distributions = {
    'base_regressor__num_leaves': randint(20, 32),  #
    'base_regressor__min_child_samples': randint(20, 30),  # No. samples in the leaves. Higher values contrast overfitting.
    'base_regressor__learning_rate': loguniform(0.01, 0.1),
    'base_regressor__n_estimators': randint(100, 300),
}

# Define the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=40,  # Number of parameter settings that are sampled
    refit=True,
    cv=splits,
    scoring=metric_score,
    verbose=1,  # Verbosity level
    random_state=0,
)

# Perform the random search on the training data
print("Starting Randomized Search for Hyperparameter Optimization...")
random_search.fit(X_train, Y_train)

# Summary of the best estimator
best_model = random_search.best_estimator_
print("\nBest hyperparameters found:")
print(random_search.best_params_)
print(f"Best training set negative {metric_name}: {random_search.best_score_}")






def evaluate_random_search(random_search):
    # Print best hyperparameters and score
    print("\nBest hyperparameters found:")
    print(random_search.best_params_)
    print(f"Best cross-validation score: {random_search.best_score_}")

    # Convert cv_results_ to DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)

    # Display top 10 configurations
    top_results = results_df.sort_values(by='rank_test_score').head(10)
    print("\nTop 10 hyperparameter combinations:")
    print(top_results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

    # Plot the results
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    for param in param_cols:
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=results_df, x=param, y='mean_test_score', hue='rank_test_score', palette='viridis_r')
        scatter.set_title(f'Hyperparameter Tuning: {param}')
        scatter.set_xlabel(param)
        scatter.set_ylabel('Mean Test Score')
        scatter.legend(title='Rank', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Additional heatmap if only two hyperparameters
    if len(param_distributions) == 2:
        # Extract the hyperparameters and corresponding scores
        param1 = list(param_distributions.keys())[0]
        param2 = list(param_distributions.keys())[1]
        param1_values = np.array(results_df['param_' + param1].astype(float))
        param2_values = np.array(results_df['param_' + param2].astype(float))
        scores = np.array(results_df['mean_test_score'].astype(float))

        data_plot = pd.DataFrame({
            param1: param1_values,
            param2: param2_values,
            'score': scores
        })
        # Create a grid for interpolation
        grid_x, grid_y = np.mgrid[min(param1_values):max(param1_values):100j,
                         min(param2_values):max(param2_values):100j]
        # Interpolate the scores
        grid_z = griddata((param1_values, param2_values), scores, (grid_x, grid_y), method='cubic')
        # Plot the interpolated heatmap using imshow
        plt.figure(figsize=(10, 8))
        plt.imshow(grid_z.T, extent=(min(param1_values), max(param1_values), min(param2_values), max(param2_values)),
                   origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label=f'Negative {metric_name}')
        plt.title(f"Hyperparameter Optimization Results\n{param1} vs {param2}")
        plt.xlabel(param1)
        plt.ylabel(param2)
        # Overlay the tested hyperparameter pairs
        plt.scatter(param1_values, param2_values, c='red', s=50, edgecolor='black',
                    zorder=5)  # Add dots for tested hyperparameters

# Example usage
evaluate_random_search(random_search)




# Compute predictions for X_train, X_dev and X_test
Y_train_pred = best_model.predict(X_train)
Y_dev_pred = best_model.predict(X_dev)
Y_test_pred = best_model.predict(X_test)




# Evaluate the model on the training set
print("\nEvaluating the model on the training set...")
train_perf = metric_fn(Y_train, Y_train_pred)
train_rmse = root_mean_squared_error(Y_train, Y_train_pred)
train_r2 = r2_score(Y_train, Y_train_pred)
print(f"Training set baseline performance: {metric_name} = {baseline_train_perf}, RMSE = {baseline_train_rmse}, R2 = {baseline_train_r2}")
print(f"Training set performance: {metric_name} = {train_perf}, RMSE = {train_rmse}, R2 = {train_r2}")

# Evaluate the model on the development set
print("\nEvaluating the model on the development set...")
dev_perf = metric_fn(Y_dev, Y_dev_pred)
dev_rmse = root_mean_squared_error(Y_dev, Y_dev_pred)
dev_r2 = r2_score(Y_dev, Y_dev_pred)
print(f"Development set baseline performance: {metric_name} = {baseline_dev_perf}, RMSE = {baseline_dev_rmse}, R2 = {baseline_dev_r2}")
print(f"Development set performance: {metric_name} = {dev_perf}, RMSE = {dev_rmse}, R2 = {dev_r2}")


# Plotting the results as scatterplots
plt.figure(figsize=(15, 5))

# Training set plot
plt.subplot(1, 3, 1)
plt.scatter(Y_train, Y_train_pred, alpha=0.5, label='Predicted')
plt.scatter(Y_train, Y_train_baseline_pred, alpha=0.3, color='magenta', label='Baseline')
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--', lw=2, label='Ideal')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Training Set\n{metric_name}: {train_perf:.2f}, Baseline {metric_name}: {baseline_train_perf:.2f}")
plt.legend()

# Development set plot
plt.subplot(1, 3, 2)
plt.scatter(Y_dev, Y_dev_pred, alpha=0.5, label='Predicted')
plt.scatter(Y_dev, Y_dev_baseline_pred, alpha=0.3, color='magenta', label='Baseline')
plt.plot([Y_dev.min(), Y_dev.max()], [Y_dev.min(), Y_dev.max()], 'r--', lw=2, label='Ideal')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Development Set\n{metric_name}: {dev_perf:.2f}, Baseline {metric_name}: {baseline_dev_perf:.2f}")
plt.legend()

# Test set plot
plt.subplot(1, 3, 3)
plt.scatter(Y_test, Y_test_pred, alpha=0.5, label='Predicted')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Test Set")
plt.legend()

plt.tight_layout()



# Plotting the results as lineplots
plt.figure(figsize=(15, 5))

# Training set plot
plt.subplot(1, 3, 1)
plt.plot(np.array(Y_train), label="Actual")
plt.plot(np.array(Y_train_pred), label="Predicted", color='orange')
plt.plot(np.array(Y_train_baseline_pred), 'r--', label="Baseline")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Training Set\n{metric_name}: {train_perf:.2f}, Baseline {metric_name}: {baseline_train_perf:.2f}")
plt.legend()

# Development set plot
plt.subplot(1, 3, 2)
plt.plot(np.array(Y_dev), label="Actual")
plt.plot(np.array(Y_dev_pred), label="Predicted", color='orange')
plt.plot(np.array(Y_dev_baseline_pred), 'r--', label="Baseline")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Development Set\n{metric_name}: {dev_perf:.2f}, Baseline {metric_name}: {baseline_dev_perf:.2f}")
plt.legend()

# Test set plot
plt.subplot(1, 3, 3)
plt.plot(np.array(Y_test_pred), label="Predicted", color='orange')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Test Set")
plt.legend()

plt.tight_layout()




# Plotting the results as scatter plots using PredictionErrorDisplay
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
PredictionErrorDisplay.from_predictions(Y_train.values.squeeze(), Y_train_pred, ax=axs[0], scatter_kwargs={'alpha':0.5})
axs[0].set_title(f"Training Set\n{metric_name}: {train_perf:.2f}")

PredictionErrorDisplay.from_predictions(Y_dev.values.squeeze(), Y_dev_pred, ax=axs[1], scatter_kwargs={'alpha':0.5})
axs[1].set_title(f"Development Set\n{metric_name}: {dev_perf:.2f}")

plt.tight_layout()





# Get feature importance from the best model
"""Split Importance: Each time a feature is used to split a node in a tree, it contributes to reducing the loss function (e.g., mean squared error in regression tasks). The importance of a feature is computed as the total gain across all splits where the feature is used. This gain measures how much using the feature at that point improves the model."""
best_lgb_model = best_model.base_regressor

# Extract feature importance
importance = best_lgb_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance')
plt.show()

# Print the DataFrame to check the ranked feature importance
print("Feature Importance:")
print(importance_df)
















# Create DataFrames for actual and predicted values with corresponding ids
Y_train_df = pd.DataFrame({
    'ymd': ymd_X_train.values,
    'hotel_id': X_train['hotel_id'].values,
    'advertiser_id': X_train['advertiser_id'].values,
    'actual': Y_train.squeeze(),
    'predicted': Y_train_pred.squeeze(),
})

Y_dev_df = pd.DataFrame({
    'ymd': ymd_X_dev.values,
    'hotel_id': X_dev['hotel_id'].values,
    'advertiser_id': X_dev['advertiser_id'].values,
    'actual': Y_dev.squeeze(),
    'predicted': Y_dev_pred.squeeze(),
})

Y_test_df = pd.DataFrame({
    'ymd': ymd_X_test.values,
    'hotel_id': X_test['hotel_id'].values,
    'advertiser_id': X_test['advertiser_id'].values,
    'actual': Y_test.squeeze(),
    'predicted': Y_test_pred.squeeze(),
})


# Save the DataFrames to CSV files
output_folder = f"../../outputs/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving results to {output_folder}")
Y_train_df.to_csv(os.path.join(output_folder, "train_predictions.csv"), index=False)
Y_dev_df.to_csv(os.path.join(output_folder, "dev_predictions.csv"), index=False)
Y_test_df.to_csv(os.path.join(output_folder, "test_predictions.csv"), index=False)

plt.show()