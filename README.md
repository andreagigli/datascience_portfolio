# Kaggle challenge: M5 Sales Forecasting 

## Description
This analysis is based on the Kaggle sales forecasting challenge using the M5 sales dataset (https://www.kaggle.com/c/m5-forecasting-accuracy). 


## M5 Sales Forecasting Challenge Overview

### Objective

The primary goal of the M5 sales forecasting challenge is to predict the daily sales for various products sold across ten Walmart stores located in three different US states. 
One is expected to forecast the unit sales of 3,049 products, categorized into 3 main categories and 7 departments, over a period of 28 days. 


### Data Description

The dataset for this challenge consists of the following files provided on Kaggle:

- **sales_train_evaluation.csv**: This file includes the historical daily unit sales data per product and store from day 1 (d_1) to day 1941 (d_1941). 

- **sell_prices.csv**: This file contains information about the price of the products sold per store and week from day 1 to day 1969. 

- **calendar.csv**: The calendar file includes details about the dates on which the products are sold from day 1 to day 1969, including day of the week, month, year, and any special events or holidays that might affect sales. 


### Sequential Forecasting Strategy

The Kaggle challenge requires predicting the sales volumes for days 1942 to 1969 (28 days) using only information until day 1941.
A sequential forecasting procedure is applied. 
Here's a quick overview of the data and approach:

#### Data Structure
- **Design Matrix (X)**: Each row corresponds to daily sales data for a unique item-shop combination, identified by `id` and `d` columns. 
                         The row includes item descriptors, daily sales (`sold` column), and historical sales data.
- **Target (Y)**: Designed to predict the sales units for the day immediately following each entry in X. 
                  This means every row in Y is aligned with a row in X but represents the sales for the next day. 
                  Essentially, Y acts as a one-day-forward version of the `sold` column in X.

#### Training, Validation, and Test Sets
- **Training Set**: Data up to day 1912 for model training.
- **Validation Set**: Days 1912 to 1940 for model tuning.
- **Test Set**: Days 1941 to 1969 for forecasting, with day 1941 included to initiate the sequential prediction process.
                Note that Kaggle does not provide the ground truth sales for days >= 1942, therefore Y_test is filled with zero-valued placeholders in all rows corresponding to days >= 1941.

#### Forecasting and Evaluation
- **Sequential Forecasting**: Starting with initial sales data for an item-shop combination on a specific day, the model forecasts the next day's sales. 
                              This prediction is then utilized to update the model's features for the subsequent day. 
                              This process is iterated to allow a chain of forecasts across multiple future days. 
- **Evaluation**: 
  - The evaluation focuses on the days 1942 to 1969. Since Y is structured to represent the following day's sales, what is sent to the Kaggle competition are the predictions in the rows of Y_test corresponding to the days 1941-1968 in X_test.
  - A direct evaluation with goodness of fit metrics is of course possible on the train and validation sets, although this is not representative of realistic model performance.


## Try It Out-of-the-Box

You can run an example analysis, where house prices predictions are performed on the California Housing Dataset:

```
cd path/to/portfolio_ML_datascience 
git checkout -b sales_prediction
python analysis_exampledb.py --data_path ./data/external/m5salesdb/ --data_loading_fn load_m5salesdb --model sklearn_compatible_LGBMRegressor --hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}" --hopt_n_rndcv_samplings 5 --hopt_subsampling_fn subsample_train_m5salesdb --hopt_subsampling_rate 1.0 --preprocessing_fn preprocess_m5salesdb --eda_fn eda_m5salesdb --feature_extraction_fn features_m5salesdb --split_fn split_m5salesdb --prediction_fn predict_m5salesdb --look_back_days_sequential_prediction 380 --evaluation_fn evaluate_m5salesdb --log_level INFO --random_seed 0 --save_output --output_data_dir ./data/processed/ --output_model_dir ./models/ --output_reports_dir ./outputs/reports/ --output_figures_dir ./outputs/figures/
```


## License

GNU General Public License (GPL).
