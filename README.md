# ML and Data Science Portfolio

## Description
A portfolio of machine learning and data science projects using my own ML and Data Science Python package.
This analysis is based on the Kaggle sales prediction challenge using the M5 sales dataset (https://www.kaggle.com/c/m5-forecasting-accuracy). 


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

Place your data analysis scripts within the scripts/exploratory for initial exploration or scripts/report for final reports.
Implement new functionalities (data loading, preprocessing, eda, feature extraction, splitting, prediction, evaluation) in dedicated modules within the src/ directory or utilize existing modules from the src/ directory. 
Store new datasets in dedicated directories within data/external, and their processed version can be stored in dedicated directories within data/processed.
Save trained models in models/trained_models and your analysis outputs, like figures and reports, in their respective directories within outputs.
Ensure the models and output folders are named with a unique run identifier, typically a timestamp, to facilitate organization and traceability.


## License

GNU General Public License (GPL).
