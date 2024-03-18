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


## Project Structure

```
portfolio_ML_datascience/
│
├── data/                   # Data used in the project
│   ├── raw/                # Unprocessed, original data files
│   ├── processed/          # Cleaned and pre-processed data
│   └── external/           # Data from external sources
│       └── exampledb/      # Example databases for the project
│           └── california_housing.csv  # California housing example dataset
│
├── scripts/                # Python scripts for analysis and reporting
│   ├── exploratory/        # Scripts for initial data exploration and analysis
│   │   └── analysis_template.py  # Example data analysis script
│   └── report/             # Scripts for generating final reports or presentations
│
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── data/               # Scripts to download or generate data
│   ├── features/           # Scripts to transform raw data into features for modeling
│   ├── models/             # Scripts and modules defining various machine learning models
│   ├── optimization/       # Hyperparameter optimization scripts for model tuning
│   ├── evaluation/         # Scripts for evaluating models and analyzing their performance
│   └── utils/              # Utility scripts and helper functions
│
├── models/                 # Model files and summaries
│   ├── trained_models/     # Serialized trained model objects
│   └── model_summaries/    # Text files or summaries describing model performance
│
├── outputs/                # Output files (e.g., predictions, figures)
│   ├── figures/            # Generated graphics and plots
│   └── reports/            # Generated analysis reports or summaries
│
├── config/                 # Configuration files, like model parameters and environmental variables
├── docs/                   # Project documentation
├── tests/                  # Automated tests for the project
│   └── __init__.py
│
├── __init__.py	        		        
├── requirements.txt        # List of dependencies for the project
├── setup.py                # Setup script for installing the project package
├── .gitignore	
├── README.md               # Overview and instructions for the project	
└── LICENSE
```

## Installation

### Cloning and customizing the package

1. **Clone the project repository**:
	- Use Git to clone the repository or download it manually from the project's webpage.
	```
	git clone https://github.com/andreagigli/portfolio_ML_datascience.git
	```


### Setting Up the Anaconda Environment

1. **Create a new Anaconda environment**:
    ```
    conda create -n portfolio_ML_datascience_env python=3.10
    ```

2. **Activate the environment**:
    ```
    conda activate portfolio_ML_datascience_env
    ```

3. **Install necessary packages**:
    - To ensure compatibility and manageability, install pip within your Conda environment:
      ```
      conda install pip
      ```
    - Install `pipreqs`, a tool for generating a `requirements.txt` file based on your project's imports:
      ```
      pip install pipreqs
      ```

### Installing the Project Package

1. **Generate the `requirements.txt` file (optional)**:
    - If your project does not already have a `requirements.txt`, or if you want to update it, run `pipreqs`:
      ```
	  cd portfolio_ML_datascience
      pipreqs --force . 
      ```

2. **Install the package**:
    - For a standard installation:
      ```
      pip install .
      ```
    - For an editable installation:
      ```
      pip install -e .
      ```
      The editable installation allows you to modify the code and see the changes without reinstalling the package.
	  

## Usage

To conduct a data analysis within this framework, follow these steps for each new project. Here the database `newdb` is used as an example.


### Set Up New Project

1. **Create a New Branch**
For each new project, initiate a new branch from the main repository. This branch will house all project-specific scripts, data, and analyses.
```
git checkout -b project_newdb
```

2. **Prepare Data Directory**
Store your new database in `data/external/newdb/`.

3. **Set Up Analysis Script**
Copy `scripts/exploratory/analysis_exampledb.py` to `scripts/exploratory/analysis_newdb.py` as a starting template.


### Implement Custom Modules

Custom modules can be implemented for a variety of operations: 
- **Data Loading**: `src.data.data_newdb.load_data`.
- **Data Preprocessing**: `src.data.data_newdb.preprocess_data`.
- **Exploratory Data Analysis (EDA)**: `src.eda.eda_newdb.eda`.
- **Feature Extraction**: `src.features.features_newdb.extract_features`.
- **Data Splitting**: `src.data.data_newdb.split_data`.
- **Hyperparameter Optimization Subsampling**: `src.data.data_newdb.subsample_items` for subsampling training and validation sets during hyperparameter optimization.
- **Model Training**: sklearn model or custom one, e.g., `sklearn.ensemble.RandomForestClassifier`.
- **Model Evaluation**: `src.evaluation.evaluate_newdb.evaluate`.
- **Custom Model**: If no scikit-learn, pytorch or tensorflow modules meet one's need, one can implement a custom model `src.models.custom_model.CustomModel`. In this case, it is recommended to adhere to scikit-learn's estimator interface (by defining fit(), predict(), and score() methods).

### Customize Analysis Script

Import all the newly written modules into the analysis script `scripts/exploratory/analysis_newdb.py`:
```python
import sklearn.svm.SVC
import src.models.custom_model.CustomModel
import src.data.data_newdb
import src.data.preprocess_data
import src.data.data_newdb.split_data
import src.data.data_newdb.subsample_data
import src.eda.eda_newdb 
import src.features.features_newdb
import src.prediction.predict_newdb 
import src.evaluation.evaluate_newdb
```

Associate the imported modules with their command-line identifier, to allow for dynamic function mapping based on command-line arguments:
```python
MODELS = {
    ...
    "sklearn_SVC": sklearn.svm.SVC,
    "sklearn_compatible_CustomModule": src.models.custom_model.CustomModel,  # ensure the id starts with "sklearn_compatible_" if the CustomModel is compatible with scikit-learn's estimator interface 
}

DATA_LOADING_FNS = {
    ...
    "load_newdb": src.data.data_newdb.load_data,
}

PREPROCESSING_FNS = {
    ...
    "preprocess_newdb": src.data.data_newdb.preprocess_data,
}

EDA_FNS = {
    ...
    "eda_newdb": src.eda.eda_newdb.eda,
}

FEATURE_EXTRACTION_FNS = {
    ...
    "features_newdb": src.features.features_newdb.extract_features,
}

SPLITTING_FNS = {
    ...
    "split_newdb": src.data.data_newdb.split_data,
}

HOPT_SUBSAMPLING_FNS = {
    ...
    "subsample_newdb": src.data.data_newdb.subsample_items,
}

PREDICTION_FNS = {
    ...
    "predict_newdb": src.prediction.predict_newdb.predict,
}

EVALUATION_FNS = {
    ...
    "evaluate_newdb": src.evaluation.evaluate_newdb.evaluate,
}

```


### Run

1. **Execute the analysis script with this command**:  
```
python analysis_newdb.py
--data_path data/external/newdb/
--data_loading_fn load_newdb
--model sklearn_RandomForestClassifier
--hparams "{\"sklearn_SVC__C\": \"loguniform(0.1, 10)\", \"sklearn_SVC__max_iter\": 200}"
--hopt_n_rndcv_samplings 10
--hopt_subsampling_fn subsample_newdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_newdb
--eda_fn eda_newdb
--feature_extraction_fn features_newdb
--split_fn split_newdb
--prediction_fn predict_newdb
--evaluation_fn evaluate_newdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir data/processed/
--output_model_dir models/
--output_reports_dir outputs/reports/
--output_figures_dir outputs/figures/
```


2. **Output Organization and Processed Data Handling**
- Outputs such as reports, figures, and model summaries are stored in directories named after a unique run identifier (typically a timestamp), e.g., `outputs/reports/run_id/` for reports and figures, `models/model_summaries/run_id/` and `models/trained_models/run_id/` for trained models. 
- Extracted features (processed data) should be stored in `data/processed/newdb/` for quick reloading. This path can be specified through the corresponding command line argument --precomputed_features_path.



## License

GNU General Public License (GPL).
