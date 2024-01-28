# Custom Python package template for Data Science and ML projects

## Description
This template serves as a foundational structure for machine learning and data science projects. It provides an organized framework for managing various aspects of a project, including data storage, script management, model training, and evaluation, ensuring a modular and scalable approach to data science and machine learning development.

## Project Structure

```
my_ml_package/
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
	git clone https://github.com/andreagigli/my_ml_package.git
	```

2. **Rename the package**
	- The name of this template package is `my_ml_package`. Please replace it with the desired name in `setup.py` as well as in all the instructions below.


### Setting Up the Anaconda Environment

1. **Create a new Anaconda environment**:
    ```
    conda create -n my_ml_package_env python=3.10
    ```

2. **Activate the environment**:
    ```
    conda activate my_ml_package_env
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
	  cd my_ml_package
      pipreqs .
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