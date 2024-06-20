import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from scipy.signal import periodogram
import seaborn as sns

from datetime import datetime

from scipy.stats import randint, loguniform
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import PowerTransformer

from src.eda.eda_misc import plot_data_distribution, check_outliers, evaluate_imputation, plot_pairplots, compute_relationship
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
data = data[["ymd", "hotel_id", "advertiser_id", "city_id", "stars", "rating", "n_reviews", "n_clickouts", "n_bookings"]]  # Reorder columns


# Impute missing values for "rating" and "n_reviews" using groupwise mean based on "city_id".
selector = SelectKBest(score_func=mutual_info_regression, k="all")
data_without_na = data.dropna()
for target_feature in ["rating", "n_reviews"]:
    features = ["city_id", "stars", "n_clickouts", "n_bookings"]
    selector.fit(data_without_na[features], data_without_na[target_feature])
    mi_scores_df = pd.DataFrame({'Feature': features, 'Mutual Information': selector.scores_})
    print(f"Mutual information between each feature and '{target_feature}':")
    print(mi_scores_df)
data["rating"] = data.groupby(["city_id"])["rating"].transform(lambda x: x.fillna(x.mean(skipna=True)))
data["n_reviews"] = data.groupby(["city_id"])["n_reviews"].transform(lambda x: x.fillna(x.mean(skipna=True)))


# Impute missing time points for each hotel-advertiser pair.
complete_data = pd.DataFrame()

all_dates = pd.DataFrame(data["ymd"].unique(), columns=["ymd"])
for (hotel_id, advertiser_id), group in data.groupby(['hotel_id', 'advertiser_id']):
    group_complete = group.merge(all_dates, on='ymd', how='outer')
    group_complete['hotel_id'] = hotel_id
    group_complete['advertiser_id'] = advertiser_id
    group_complete['city_id'] = group['city_id'].iloc[0]
    group_complete['stars'] = group['stars'].iloc[0]
    group_complete['rating'] = group['rating'].iloc[0]
    group_complete['n_reviews'] = group['n_reviews'].iloc[0]
    complete_data = pd.concat([complete_data, group_complete], ignore_index=True)

impute_cols = ['n_clickouts', 'n_bookings']
all_features = ['ymd', 'advertiser_id', 'city_id', 'stars', 'rating', 'n_reviews', 'n_clickouts', 'n_bookings']
complete_data['imputed_day_t'] = complete_data[impute_cols].isna().any(axis=1).astype("uint8")
def impute_group(group):
    imputer = KNNImputer(n_neighbors=3, weights="distance")
    imputed_data = imputer.fit_transform(group[all_features])
    imputed_df = pd.DataFrame(imputed_data, columns=all_features, index=group.index)
    group[impute_cols] = imputed_df[impute_cols]
    return group
complete_data = complete_data.groupby(['hotel_id']).apply(impute_group).reset_index(drop=True)

data = complete_data
del complete_data

# # Evaluate imputation for 'n_clickouts' and 'n_bookings'
# evaluate_imputation(data, 'n_clickouts', 'imputed_day_t')
# evaluate_imputation(data, 'n_bookings', 'imputed_day_t')


# Convert ymd to datetime
data["ymd"] = pd.to_datetime(data["ymd"], format='%Y%m%d')


# Create conversion rate variable
data["c2b"] = data["n_bookings"] / data["n_clickouts"]

# Create the target variable c2b_t+1 by shifting c2b (predict c2b_t+1 using information from day t)
data["c2b_t+1"] = data.groupby(["hotel_id", "advertiser_id"])["c2b"].shift(-1).fillna(-1)
# Create an indicator 'imputed_day_t+1' that the 'c2b_t+1' was imputed
data["imputed_day_t+1"] = data.groupby(["hotel_id", "advertiser_id"])["imputed_day_t"].shift(-1).fillna(1).astype("uint8")


# Sort columns
data = data.sort_values(by=["hotel_id", "advertiser_id", "ymd"])
data = data[['ymd', 'hotel_id', 'advertiser_id', 'city_id', 'stars', 'rating', 'n_reviews', 'n_clickouts', 'n_bookings',
             'imputed_day_t', 'c2b', 'imputed_day_t+1', 'c2b_t+1']]


# Optimize data types for memory efficiency
cat_to_num_converted_columns = ['hotel_id', 'advertiser_id', 'city_id', 'imputed_day_t', 'imputed_day_t+1']
data[cat_to_num_converted_columns] = data[cat_to_num_converted_columns].astype('category')

data["stars"] = data["stars"].astype("int8")
data["rating"] = data["rating"].astype("float16")
data["n_reviews"] = data["n_reviews"].astype("int")
data["n_clickouts"] = data["n_clickouts"].astype("float32")
data["n_bookings"] = data["n_bookings"].astype("float32")
data["c2b"] = data["c2b"].astype("float32")
data["c2b_t+1"] = data["c2b_t+1"].astype("float32")


print("Preprocessed DataFrame:")
print(data.head())



"""
## Exploratory Data Analysis
"""
# # Distinguish numerical and categorical columns
# cols_numerical = ["stars", "rating", "n_reviews", "n_clickouts", "n_bookings", "c2b", "c2b_t+1"]
# cols_categorical = [col for col in data.columns if col not in cols_numerical]
#
# data_no_ymd = data.drop(columns=["ymd"])
#
#
# # Feature distributions
# print("\n### Feature distributions ###\n")
# print("Visualizing distribution of continuous and discrete variables:")
# plot_data_distribution(data, discrete_features_mask=[col in cols_categorical for col in data.columns])
# """
# The features n_reviews, n_clickouts, n_bookings, c2b are strongly skewed to the right. Feature transformation is advised. Target (c2b_t+1) transformation is also in order, paying attention to back-transform both prediction and ground truth before performance evaluation.
# """
#
# # Feature outliers
# print("\n### Outliers in the continuous features ###\n")
# outliers_mask, _, _ = check_outliers(data, columns_of_interest=cols_numerical, sample_size=100, profile_plots=True)
# """
# About 19% of points are found to be outliers. Outlier points seem to exhibit more skewed distributions in the features n_reviews, n_clickouts, n_bookings, c2b. This appears to relate to the feature skewness and may be attenuated with proper feature transformation.
# """
#
# # Feature-Target Relationships
# target = "c2b_t+1"
#
# print("\n### Feature-Target Relationships ###\n")
# print(f"Pairplot of features vs target ('{target}').\nUseful to inform feature engineering and model selection.\n")
# plot_pairplots(data, target_columns=[target], sample_size=100)
#
# # For regression tasks, one can check the existence of linear or monotonic feature-target relationships.
# print("Linear relationships between features and target ('...').\nUseful to inform selection of regression models.")
# compute_relationship(data, score_func="pearson", columns_of_interest=cols_numerical, target=target, sample_size=1000, plot_heatmap=True, include_diagonal=True)
#
# print("Monotonic relationships between features and target ('...').\nUseful to inform the adoption of simple linearizing features.")
# compute_relationship(data, score_func="spearman", columns_of_interest=cols_numerical, target=target, sample_size=1000, plot_heatmap=True, include_diagonal=True)
# """
# The feature c2b has medium positive correlation with the target (pearson=0.42, spearman=0.51) with the target.
# The n_bookings has a medium positive monotonic relationship (spearman=0.52) with the target.
# The n_clickouts has a mild positive monotonic relationship (spearman=0.31) with the target.
# """
#
# # Feature-Feature Relationships
# print("\n### Feature-Feature Relationships ###\n")
# print("Pairplot of feature vs feature.\nUseful to explain underlying processes and spot edge conditions, inform the feature elimination or feature joining.")
# plot_pairplots(data, columns_to_plot=[col for col in data.columns.to_list() if col != target], sample_size=100)
#
# """Identify seasonal cycles"""
# # Compute average c2b per day
# average_c2b_per_day = data.groupby('ymd')['c2b'].mean()
#
# # Find potential periods in the target variable
# print("\n\nPotential cycles in the target variable (dominant freq (in no. days), corresponding normalized power in PSD):")
# dominant_cycles = spot_seasonal_cycles(average_c2b_per_day)  # a list of tuples [(f1, pow1), (f2, pow2), ...] with pow1 >= pow2
# total_power = sum([power for _, power in dominant_cycles])
# dominant_cycles = [(freq, power / total_power) for freq, power in dominant_cycles]
# print('\n'.join(map(str, dominant_cycles[:5])))
#
# # Visualize the dominant cycles
# plt.figure()
# plt.title('Dominant Cycles in the Average c2b')
# freqs = [freq for freq, _ in dominant_cycles]
# normalized_powers = [power for _, power in dominant_cycles]
# plt.plot(freqs, normalized_powers, 'o', markersize=5, linewidth=1, markerfacecolor='blue')
# plt.ylabel('Normalized Power')
# plt.xlabel('Period (in days)')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()
# """One observes cyclical components with a frequency of 5 days (normalized power = 0.38) and 3 days (normalized power = 0.31). It is advisable to use lag and windowed features with these durations."""



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
4. Returns two data frames: one with input features (X) and the other with target value c2b_t+1 (Y).
"""
# Apply power transformation (Yeo-Johnson) to the skewed features. This stretches the distribution towards higher values.
skewed_features = ["n_reviews", "n_clickouts", "n_bookings", "c2b", "c2b_t+1"]
pt_features = PowerTransformer(method='yeo-johnson', standardize=False)
data[skewed_features] = pt_features.fit_transform(data[skewed_features])
data[skewed_features] = data[skewed_features].applymap(lambda x: 0 if abs(x) < 1e-10 else x)  # Post-process to set very small negative values to zero
# Apply a separate transformation to the target. The fitting doesn't change, but I can keep this pt_target to later back-transform only the target.
skewed_target = ["c2b_t+1"]
pt_target = PowerTransformer(method='yeo-johnson', standardize=False)
data[skewed_target] = pt_target.fit_transform(data[skewed_target])
data[skewed_target] = data[skewed_target].applymap(lambda x: 0 if abs(x) < 1e-10 else x)

# Group the data by `hotel_id` and `advertiser_id`
grouped = data.groupby(['hotel_id', 'advertiser_id'])

# Create lag features for conversion rate (c2b) for specified lag periods (e.g., 1, 3, 7 days)
lags = [1, 3, 5, 7]
for lag in lags:
    data[f'c2b_lag_{lag}'] = grouped['c2b'].shift(lag)

# Compute rolling window statistics (mean) for specified window sizes (e.g., 3, 7 days)
windows = [3, 5, 7]
for window in windows:
    data[f'c2b_roll_mean_{window}'] = grouped['c2b'].rolling(window=window).mean().reset_index(level=['hotel_id', 'advertiser_id'], drop=True)

# Mean encodings for the c2b by hotel, by advertiser, by city, and by stars
data['c2b_mean_hotel'] = data.groupby('hotel_id')['c2b'].transform('mean')
data['c2b_mean_advertiser'] = data.groupby('advertiser_id')['c2b'].transform('mean')
data['c2b_mean_city'] = data.groupby('city_id')['c2b'].transform('mean')
data['c2b_mean_stars'] = data.groupby('stars')['c2b'].transform('mean')

# Separate the features (X) and the target (Y)
X = data.drop(columns=['c2b_t+1'])
Y = data['c2b_t+1']

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
train_date = pd.Timestamp("2023-08-08")
dev_date = pd.Timestamp("2023-08-09")
test_date = pd.Timestamp("2023-08-10")

# Split the data based on the defined dates
X_train = X[data["ymd"] == train_date]
Y_train = Y[data["ymd"] == train_date]
ymd_X_train = X_train["ymd"]
X_train = X_train.drop(columns=["ymd"])

X_dev = X[data["ymd"] == dev_date]
Y_dev = Y[data["ymd"] == dev_date]
ymd_X_dev = X_dev["ymd"]
X_dev = X_dev.drop(columns=["ymd"])

X_test = X[data["ymd"] == test_date]
Y_test = Y[data["ymd"] == test_date]
ymd_X_test = X_test["ymd"]
X_test = X_test.drop(columns=["ymd"])

# Display the shapes of the splits to verify
print("Training set:")
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")

print("\nDevelopment set:")
print(f"X_dev shape: {X_dev.shape}")
print(f"Y_dev shape: {Y_dev.shape}")

print("\nTest set:")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")


# endregion

"""Model Tuning and Training

....
"""
# Compute baseline performance with a DummyRegressor
baseline_model = DummyRegressor(strategy="mean")
baseline_model.fit(X_train, Y_train)
Y_train_baseline_pred = baseline_model.predict(X_train)
Y_dev_baseline_pred = baseline_model.predict(X_dev)
Y_test_baseline_pred = baseline_model.predict(X_test)
# Back-transform the baseline predictions
Y_train_back = pt_target.inverse_transform(Y_train.values.reshape(-1, 1))
Y_dev_back = pt_target.inverse_transform(Y_dev.values.reshape(-1, 1))
Y_test_back = pt_target.inverse_transform(Y_test.values.reshape(-1, 1))
Y_train_baseline_pred_back = pt_target.inverse_transform(Y_train_baseline_pred.reshape(-1, 1))
Y_dev_baseline_pred_back = pt_target.inverse_transform(Y_dev_baseline_pred.reshape(-1, 1))
Y_test_baseline_pred_back = pt_target.inverse_transform(Y_test_baseline_pred.reshape(-1, 1))
# Compute baseline MAE for training, development, and test sets
baseline_train_mae_back = mean_absolute_error(Y_train_back, Y_train_baseline_pred_back)
baseline_dev_mae_back = mean_absolute_error(Y_dev_back, Y_dev_baseline_pred_back)
baseline_test_mae_back = mean_absolute_error(Y_test_back, Y_test_baseline_pred_back)

# # Define the model
# model = lgb.LGBMRegressor()
#
# param_distributions = {
#     'num_leaves': randint(20, 150),  # Adjust to a reasonable range
#     'learning_rate': loguniform(1e-3, 1e-1),  # Learning rate
#     'n_estimators': randint(50, 500),  # Number of boosting rounds
#     'min_gain_to_split': loguniform(1e-5, 1e-2),  # Lower the minimum gain to split
#     'min_data_in_leaf': randint(10, 50),  # Ensure minimum data in each leaf
#     'min_child_weight': loguniform(1e-4, 1e-2)  # Adjust the minimum child weight
# }

# Define the model
model = RandomForestRegressor(criterion="friedman_mse")

# Define the hyperparameters to optimize
param_distributions = {
    'n_estimators': randint(100, 500),  # Number of trees in the forest
    'max_depth': randint(10, 30),  # Maximum depth of the tree
}

# Create kfold split based on hotel_id
group_kfold = GroupKFold(n_splits=3)
groups = X_train['hotel_id']
splits = list(group_kfold.split(X_train, Y_train, groups=groups))

# Define the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,  # Number of parameter settings that are sampled
    cv=splits,
    scoring="neg_mean_absolute_error",
    verbose=3,  # Verbosity level
    random_state=0,
)

# Perform the random search on the training data
print("Starting Randomized Search for Hyperparameter Optimization...")
random_search.fit(X_train, Y_train)

# Summary of the best estimator
best_model = random_search.best_estimator_
print("\nBest hyperparameters found:")
print(random_search.best_params_)
print(f"Best training set negative MAE: {random_search.best_score_}")



# If only two hyperparameters were optimized, plot a heatmap of the scores.
results = random_search.cv_results_
if len(param_distributions) == 2:
    # Extract the hyperparameters and corresponding scores
    param1 = list(param_distributions.keys())[0]
    param2 = list(param_distributions.keys())[1]
    param1_values = np.array(results['param_' + param1].data, dtype=float)
    param2_values = np.array(results['param_' + param2].data, dtype=float)
    scores = np.array(results['mean_test_score'], dtype=float)

    data_plot = pd.DataFrame({
        param1: param1_values,
        param2: param2_values,
        'score': scores
    })

    # Create a grid for interpolation
    grid_x, grid_y = np.mgrid[min(param1_values):max(param1_values):100j, min(param2_values):max(param2_values):100j]

    # Interpolate the scores
    grid_z = griddata((param1_values, param2_values), scores, (grid_x, grid_y), method='cubic')

    # Plot the interpolated heatmap using imshow
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_z.T, extent=(min(param1_values), max(param1_values), min(param2_values), max(param2_values)),
               origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Negative MAE')
    plt.title(f"Hyperparameter Optimization Results\n{param1} vs {param2}")
    plt.xlabel(param1)
    plt.ylabel(param2)

    # Overlay the tested hyperparameter pairs
    plt.scatter(param1_values, param2_values, c='red', s=50, edgecolor='black',
                zorder=5)  # Add dots for tested hyperparameters



# Train the model with the best hyperparameters on the full training set
print("\nTraining the best model on the full training set...")
best_model.fit(X_train, Y_train)
Y_train_pred = best_model.predict(X_train)
Y_train_pred = np.clip(Y_train_pred, a_min=0, a_max=None)

# Compute predictions for X_dev and X_test
Y_dev_pred = best_model.predict(X_dev)
Y_dev_pred = np.clip(Y_dev_pred, a_min=0, a_max=None)

Y_test_pred = best_model.predict(X_test)
Y_test_pred = np.clip(Y_test_pred, a_min=0, a_max=None)

# Back-transform Y
Y_train_back = pt_target.inverse_transform(Y_train.values.reshape(-1, 1))
Y_train_pred_back = pt_target.inverse_transform(Y_train_pred.reshape(-1, 1))
Y_dev_back = pt_target.inverse_transform(Y_dev.values.reshape(-1, 1))
Y_dev_pred_back = pt_target.inverse_transform(Y_dev_pred.reshape(-1, 1))
Y_test_back = pt_target.inverse_transform(Y_test.values.reshape(-1, 1))
Y_test_pred_back = pt_target.inverse_transform(Y_test_pred.reshape(-1, 1))

# Create DataFrames for actual and predicted values with corresponding ids
Y_train_back_df = pd.DataFrame({
    'ymd': ymd_X_train.values,
    'hotel_id': X_train['hotel_id'].values,
    'advertiser_id': X_train['advertiser_id'].values,
    'actual': Y_train_back.squeeze(),
    'predicted': Y_train_pred_back.squeeze(),
})

Y_dev_back_df = pd.DataFrame({
    'ymd': ymd_X_dev.values,
    'hotel_id': X_dev['hotel_id'].values,
    'advertiser_id': X_dev['advertiser_id'].values,
    'actual': Y_dev_back.squeeze(),
    'predicted': Y_dev_pred_back.squeeze(),
})

Y_test_back_df = pd.DataFrame({
    'ymd': ymd_X_test.values,
    'hotel_id': X_test['hotel_id'].values,
    'advertiser_id': X_test['advertiser_id'].values,
    'actual': Y_test_back.squeeze(),
    'predicted': Y_test_pred_back.squeeze(),
})

# Evaluate the model on the training set with back-transformed values
print("\nEvaluating the model on the training set...")
train_mae_back = mean_absolute_error(Y_train_back, Y_train_pred_back)
print(f"Back-transformed development set MAE score: {train_mae_back}")

# Evaluate the model on the development set with back-transformed values
print("\nEvaluating the model on the development set...")
dev_mae_back = mean_absolute_error(Y_dev_back, Y_dev_pred_back)
print(f"Back-transformed development set MAE score: {dev_mae_back}")

# Plotting the results as scatterplots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(Y_train_back, Y_train_pred_back, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Training Set\nMAE: {train_mae_back:.2f}")

# Development set plot
plt.subplot(1, 3, 2)
plt.scatter(Y_dev_back, Y_dev_pred_back, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Development Set\nMAE: {dev_mae_back:.2f}")

# Test set plot
plt.subplot(1, 3, 3)
plt.scatter(Y_test_back, Y_test_pred_back, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Test Set")

plt.tight_layout()


# Plotting the results as lineplots
plt.figure(figsize=(15, 5))

# Training set plot
plt.subplot(1, 3, 1)
plt.plot(Y_train_back, label="Actual")
plt.plot(Y_train_pred_back, label="RF Predicted", color='orange')
plt.plot(Y_train_baseline_pred_back, 'r--', label="Baseline")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Training Set\nRF MAE: {train_mae_back:.2f}, Baseline MAE: {baseline_train_mae_back:.2f}")
plt.legend()

# Development set plot
plt.subplot(1, 3, 2)
plt.plot(Y_dev_back, label="Actual")
plt.plot(Y_dev_pred_back, label="RF Predicted", color='orange')
plt.plot(Y_dev_baseline_pred_back, 'r--', label="Baseline")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Development Set\nRF MAE: {dev_mae_back:.2f}, Baseline MAE: {baseline_dev_mae_back:.2f}")
plt.legend()

# Test set plot
plt.subplot(1, 3, 3)
plt.plot(Y_test_pred_back, label="RF Predicted", color='orange')
plt.plot(Y_test_baseline_pred_back, 'r--', label="Baseline")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title(f"Test Set")
plt.legend()

plt.tight_layout()


# Save the DataFrames to CSV files
output_folder = f"../../outputs/reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
os.makedirs(output_folder, exist_ok=True)
print(f"Saving results to {output_folder}")
Y_train_back_df.to_csv(os.path.join(output_folder, "train_predictions.csv"), index=False)
Y_dev_back_df.to_csv(os.path.join(output_folder, "dev_predictions.csv"), index=False)
Y_test_back_df.to_csv(os.path.join(output_folder, "test_predictions.csv"), index=False)

plt.show()