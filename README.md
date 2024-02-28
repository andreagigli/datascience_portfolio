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

- **sell_prices.csv**: This file contains information about the price of the products sold per store and date. 

- **calendar.csv**: The calendar file includes details about the dates on which the products are sold, including day of the week, month, year, and any special events or holidays that might affect sales. 


### Sequential forecasting strategy

To predict sales volumes over the following 28 days, a sequential forecasting is applied. 
Starting with initial sales data for an item and shop on a specific day, the model forecasts the next day's sales. 
This prediction is then utilized to update the model's features for the subsequent day. 
This process is iterated to allow a chain of forecasts across multiple future days. 
Training covers data up to day 1912, with days 1912 to 1940 used for validation, and days 1941 to 1969 for testing.


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

Place your data analysis scripts within the scripts/exploratory for initial exploration or scripts/report for final reports. Implement or utilize source code functionalities from the src directory as needed. New datasets should be stored in dedicated directories within data/external. Save trained models in models/trained_models and your analysis outputs, like figures and reports, in their respective directories within outputs. Ensure the models and output folders are named with a unique run identifier, typically a timestamp, to facilitate organization and traceability.


## License

GNU General Public License (GPL).
