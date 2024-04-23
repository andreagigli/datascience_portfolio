# Data Science Use Case: Residual Margin Risk Management in Car Leasing

## Project Description
This repository provides a summary of theoretical and practical findings on residual margin risk management in vehicle leasing.
A data science project was not possible as no publicly available data set was found for this task. 

Reference articles include:
* [Kim, Mihye, et al. "Trustworthy residual vehicle value prediction for auto finance." AI Magazine 44.4 (2023): 394-405.](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12136)
* [Pirotte, Hugues, and Céline Vaessen. "Residual value risk in the leasing industry: A European case." European Journal of Finance 14.2 (2008): 157-177.](https://www.tandfonline.com/doi/abs/10.1080/13518470701705637)


## Overview on Car Leasing

Leasing is a financial arrangement where a car company (lessor) buys a vehicle and leases it to a customer, who can choose to buy the vehicle at the end of the lease term by paying the residual value.

### Car Company's Financial Goals:
The car company calculates the yearly payment for the leased vehicle to cover specific financial goals:
- **Cover Depreciation**: To recover the loss in value of the vehicle over the lease term. The expected depriciation is estimated using machine learning, in terms of residual value. 
- **Cover Interest Costs**: To cover the cost of capital used to purchase the vehicle.
- **Make Profit**: To ensure profitability of the leasing agreement.

The car company then evaluates the profitability of a car taking into account if the car's lease is competitive in terms of yearly payments and residual value (for final sale) for the desired level of profit. 

### Computation of Residual Value and Yearly Payment Calculation:
To calculate the yearly payments, the car company first predicts the expected residual value (V_r) using machine learning. This prediction defines the total expected depreciation (D = V_0 - V_r, with V_0 being the initial value) and the yearly depreciation component (D/T) of the payment.

Then, the yearly payment can be calculated as follows: 
P_t = Yearly Depreciation + Yearly Interest + Yearly Profit = (D/T) + (V_0 - \sum_{i=1}^{t-1} P_i) * r + Profit Component

Where:
- **Yearly Depreciation (D/T)**: The amount the vehicle value decreases each year.
- **Yearly Interest (V_0 - \sum_{i=1}^{t-1} P_i) * r**: Interest on the remaining balance, with r being the interest rate.
- **Profit Component**: Adjusted per the company’s profit strategy, either as a fixed amount or a percentage of the yearly payment.

### Example:
For a vehicle with an initial value of (V_0 = 30000€), expected total depreciation (D = 12000€) over a 4-year lease term (T = 4), and an interest rate (r) of 5%:
- **Depreciation Component**: (12000/4 = 3000€) per year.
- **Interest Component** for the first year: (30000 - 0) * 0.05 = 1500€.
- **Profit Component**: Assuming a fixed amount of 500€ per year.

Thus, the first year's payment (P_1) would total (3000€ + 1500€ + 500€ = 5000€). This process can be repeated for all T years of the lease and the overall payments and residual values are compared to those of similar cars and to those of the same car by other lessors to evaluate the car's profitability. 


## Estimation of the Residual Value with Machine Learning

A regression model is used to forecast the residual value (V_r) at the end of the lease using information collected at the beginning of the lease. Each input sample corresponds to the initiation of a new lease.  The features included in these input samples consist of vehicle specifications (manufacturer, model, year, engine size, new car price MSRP, duration of the lease), historical transaction prices up to the lease start (mileage, age of the vehicle, previous usage types), and market indicators relevant at the time the lease begins (used cars sales volume, new car sales volume, frequency of used car searches on platforms, interest rates, gross domestic product, consumer confidence indices).


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

1. **Navigate to the project's root directory**:
      ```
	  cd path/to/portfolio_ML_datascience/
      ```

2. **Generate the `requirements.txt` file (optional)**:
    - If your project does not already have a `requirements.txt`, or if you want to update it, run `pipreqs`:
      ```
	  cd portfolio_ML_datascience/
      pipreqs --force . 
      ```

3. **Install the package**:
    - For a standard installation:
      ```
      pip install .
      ```
    - For an editable installation:
      ```
      pip install -e .
      ```
      The editable installation allows you to modify the code and see the changes without reinstalling the package.
	  

## License

GNU General Public License (GPL).
