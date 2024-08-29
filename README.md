# Data Science Portfolio



## Description

This package provides a custom modular framework for managing multiple data analysis projects. 
New data science projects are initiated in isolated branches, leveraging both pre-built and customizable Python modules.



## Project Structure

```
portfolio_ML_datascience/
│
├── config/                 # Configuration files, like model parameters and environmental variables
│
├── data/                   # Data used in the project
│   ├── raw/                # Unprocessed, original data files
│   ├── processed/          # Cleaned and pre-processed data
│   └── external/           # Data from external sources
│       └── exampledb/      # Example databases for the project
│
├── docs/                   # Project documentation
│
├── models/                 # Model files and summaries
│   ├── dl_runs/            # Deep Learning run outputs
│   ├── model_summaries/    # Text files or summaries describing model performance
│   └── trained_models/     # Serialized trained model objects
│
├── outputs/                # Output files (e.g., predictions, figures)
│   ├── figures/            # Generated figures and plots
│   └── reports/            # Generated analysis reports and plots
│
├── scripts/                # Python scripts for analysis and reporting
│   ├── exploratory/        # Scripts for initial data exploration and analysis
│   │   └── analysis_template.py  # Data analysis script. This is the ENTRY POINT for launching different analyses based on CLI calls.
│   └── report/             # Scripts for generating final reports or presentations
│
├── src/                    # Source code for the project
│   ├── data/               # Modules to download or generate data
│   │   ├── load_data_fns.py        # Contains functions to load datasets
│   │   ├── preprocess_data_fns.py  # Contains functions to preprocess datasets
│   │   └── split_data_fns.py       # Contains functions to split datasets into training, validation, and test sets
│   ├── eda/                # Modules for exploratory data analysis
│   │   ├── eda_fns.py              # Functions for performing exploratory data analysis
│   │   └── eda_misc.py             # Miscellaneous EDA-related functions
│   ├── evaluation/         # Modules for evaluating models and analyzing their performance
│   │   └── evaluate_fns.py         # Functions for model evaluation
│   ├── features/           # Modules to transform raw data into features for modeling
│   │   └── features_fns.py         # Functions for feature engineering
│   ├── models/             # Modules defining various machine learning models
│   │   └── custom_models.py        # Custom model definitions
│   ├── optimization/       # Hyperparameter optimization modules for model tuning
│   │   └── hopt_subsampling_fns.py # Functions for subsampling training and validation sets during hyperparameter optimization
│   ├── prediction/         # Modules for performing inference
│   │   └── prediction_fns.py       # Functions for generating predictions
│   ├── utils/              # Utility modules and helper functions
│   ├── pipeline/           # Main pipeline scripts and modules
│   ├── __init__.py
│   ├── data_science_pipeline.py    # Main script to run the data science pipeline
│   └── register_functions.py       # Script to register functions dynamically
│
├── tests/                  # Automated tests for the project
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
    - If the installation in the next step fails due to conflicting versions, try removing the version constraints from the generated requirements.txt file with the following command:
      ```
      sed -i 's/==.*//' requirements.txt 
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
	  

## Try It Out-of-the-Box

You can run an example analysis, where house prices predictions are performed on the California Housing Dataset:

```
cd path/to/portfolio_ML_datascience
python ./scripts/exploratory/analysis_template.py --data_path ./data/external/exampledb/california_housing.csv --data_loading_fn load_exampledb --model sklearn_RandomForestRegressor --hparams "{\"sklearn_RandomForestRegressor__n_estimators\": \"randint(20, 200)\", \"sklearn_RandomForestRegressor__max_depth\": 10}" --hopt_n_rndcv_samplings 5 --hopt_subsampling_fn hopt_subsampling_passthrough --hopt_subsampling_rate 1.0 --preprocessing_fn preprocess_passthrough --eda_fn eda_passthrough --feature_extraction_fn features_exampledb --split_fn split_train_test --split_ratio "80 20" --n_folds 3 --prediction_fn predict_sklearn --evaluation_fn evaluate_exampledb --log_level INFO --random_seed 0 --save_output --output_data_dir ./data/processed/ --output_model_dir ./models/ --output_reports_dir ./outputs/reports/ --output_figures_dir ./outputs/figures/
```


## Conduct a New Analysys

To conduct a data analysis within this framework, follow these steps for each new project. Here the database `newdb` is used as an example.


### Set Up New Project

1. **Create a New Branch**
For each new project, initiate a new branch from the main repository. This branch will house all project-specific functions, data, and optionally a dedicated script within scripts/exploratory/.
```
git checkout -b project_newdb
```

2. **Prepare Data Directory**
Store your new database in `data/external/newdb/`.

3. **(Optional) Set Up Analysis Script**
If `scripts/exploratory/analysis_exampledb.py`, with an appropriate CLI call does not serve your purposes, create a dedicated analysis script. To do so, create `scripts/exploratory/analysis_template_newdb.py`.


### Implement Custom Modules

Custom modules can be implemented for a variety of operations: 
- **Data Loading**: Function `load_data_newdb` in `src.data.load_data_fns`.
- **(Optional) Data Preprocessing**: Function `preprocess_data_newdb` in `src.data.preprocess_data_fns`. Optional because script `scripts/exploratory/analysis_exampledb.py` uses function `preprocess_data_passthrough` from `src.data.preprocess_data_fns` by default.
- **(Optional) Exploratory Data Analysis (EDA)**: Function `eda_newdb` in `src.eda.eda_fns`. Optional because script `scripts/exploratory/analysis_exampledb.py` uses function `eda_passthrough` from `src.eda.eda_fns` by default.
- **Feature Extraction**: Function `extract_features_newdb` in `src.features.features_fns`.
- **Data Splitting**: Function `split_data_newdb` in `src.data.split_data_fns`. Already available options are `split_train_test` and `split_train_val_test`.
- **(Optional) Hyperparameter Optimization Subsampling**: Function `hopt_subsampling_newdb` in `src.optimization.hopt_subsampling_fns` for subsampling training and validation sets during hyperparameter optimization. Optional because script `scripts/exploratory/analysis_exampledb.py` uses function `hopt_subsampling_passthrough` from `src.optimization.hopt_subsampling` by default.
- **Prediction Function**: If custom prediction routine is desired, implement function `predict_newdb` in `src.prediction.prediction_fns`. Already available options are `predict_sklearn` and `predict_zeros`.
- **Model Evaluation**: Function `evaluate_newdb` in `src.evaluation.evaluate_fns`.
- **(Optional) Custom Model**: If no scikit-learn, PyTorch, or TensorFlow modules meet your needs, you can implement a custom model `CustomModel` in `src.models.custom_models`. It is recommended to adhere to scikit-learn's estimator interface by defining `fit()`, `predict()`, and `score()` methods.

`predict_zeros`, `predict_sklearn`


### Customize Analysis Script

Add all newly written functions and classes to the module `register_functions.py` for integration with the analysis pipeline. This allows for dynamic function mapping based on command-line arguments.

```python
   from src.data.load_data_fns import ..., load_data_newdb
   from src.data.preprocess_data_fns import ..., preprocess_data_newdb
   from src.data.split_data_fns import ..., split_data_newdb
   from src.eda.eda_fns import ..., eda_newdb
   from src.evaluation.evaluation_fns import ..., evaluate_newdb
   from src.features.features_fns import ..., extract_features_newdb
   from src.models.custom_models import ..., CustomModel
   from src.optimization.hopt_subsampling_fns import ..., hopt_subsampling_newdb
   from src.prediction.prediction_fns import ..., predict_newdb
   from src.utils.my_argparsing import FunctionRegistry

   def register_all_functions(function_registry: FunctionRegistry):
       # Data Loading Functions
       ...
       function_registry.register("load_newdb", load_data_newdb)

       # Preprocessing Functions
       ...
       function_registry.register("preprocess_newdb", preprocess_data_newdb)

       # EDA Functions
       ...
       function_registry.register("eda_passthrough", eda_passthrough)

       # Feature Extraction Functions
       ...
       function_registry.register("features_newdb", extract_features_newdb)

       # Splitting Functions
       ...
       function_registry.register("split_newdb", split_data_newdb)

       # Prediction Functions
       ...
       function_registry.register("predict_newdb", predict_newdb)

       # Evaluation Functions
       ...
       function_registry.register("evaluate_newdb", evaluate_newdb)

       # Models
       ...
       function_registry.register("mymodel", CustomModel)

       # Data Transformers
       ...

       # Random Distribution Functions
       ...

       # Hyperparameter Optimization Subsampling Functions
       ...
       function_registry.register("hopt_subsampling_newdb", hopt_subsampling_newdb)
```

### Notes on Running a New Analysis

Take this into account when defining the command line instructions for a new analysis:

* **Data Splitting Modality**: The data splitting modality can be regulated through the arguments `split_fn`, `split_ratio`, and `n_folds`. 
  - For scenarios without hyperparameter optimization (hopt) use:
    ```
    --split_fn split_train_test 
    --split_ratio "80 20"
    ```
  - For hyperparameter optimization use either of the following:
    - K-fold cross-validation:
      ```
      --split_fn split_train_test 
      --split_ratio "80 20" 
      --n_folds 3
      ```
    - Fixed split:
      ```
      --split_fn split_train_val_test 
      --split_ratio "70 15 15"
      ```

* **Hyperparameter Optimization (hopt)**: Hopt is triggered if there are "distribution strings" among the provided `hparams` values. 
  - Example triggering hopt:
    ```
    --hparams "{\"sklearn_RandomForestRegressor__n_estimators\": \"randint(20, 200)\", \"sklearn_RandomForestRegressor__max_depth\": 10}"
    ```
  - Example not triggering hopt:
    ```
    --hparams "{\"sklearn_RandomForestRegressor__n_estimators\":150, \"sklearn_RandomForestRegressor__max_depth\": 10}"
    ```
  - During hopt, the argument `--hopt_n_rndcv_samplings` determines the number of random samplings of the hyperparameters to be optimized, and `--hopt_subsampling_fn` along with `--hopt_subsampling_rate` are used to subsample the training data to save on computation.

* **Precomputed Features**: To reload a version of the dataset where the features have been already computed, replace
  ```
  --data_path ../../data/external/m5salesdb/ 
  --data_loading_fn load_m5salesdb
  ```   
  with
  ```
  --precomputed_features_path ../../data/processed/newdb/
  ```

* **Outputs**: To save outputs, include the following parameters:
  ```
  --save_output 
  --output_data_dir ../../data/processed/ 
  --output_model_dir ../../models/ 
  --output_reports_dir ../../outputs/reports/ 
  --output_figures_dir ../../outputs/figures/
  ```
  
  - Outputs such as reports, figures, and model summaries are stored in directories named after a unique run identifier (typically a timestamp), e.g., `outputs/reports/run_id/` for reports and figures, `models/model_summaries/run_id/` and `models/trained_models/run_id/` for trained models. 
  - Extracted features (processed data) should be stored in `data/processed/newdb/` for quick reloading. This path can be specified through the corresponding command line argument --precomputed_features_path.

    
### Example CLI Calls

- This CLI call launches a custom analysis on a new dataset `newdb`. 
```
python analysis_template.py
--data_path data/external/newdb/
--data_loading_fn load_newdb
--preprocessing_fn preprocess_newdb
--eda_fn eda_newdb
--feature_extraction_fn features_newdb
--split_fn split_newdb
--model sklearn_RandomForestClassifier
--hparams "{\"sklearn_SVC__C\": \"loguniform(0.1, 10)\", \"sklearn_SVC__max_iter\": 200}"
--hopt_n_rndcv_samplings 10
--hopt_subsampling_fn hopt_subsampling_newdb
--hopt_subsampling_rate 1.0
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


- This CLI call launches a straightforward analysis pipeline on a new dataset (no preprocessing, no eda, 80-20 split, no hopt, sklearn prediction):
```
python analysis_newdb.py 
--data_path data/external/newdb/ 
--data_loading_fn load_newdb 
--preprocessing_fn preprocess_passthrough 
--eda_fn eda_passthrough 
--feature_extraction_fn features_newdb 
--split_fn split_train_test 
--split_ratio "80 20" 
--model sklearn_RandomForestClassifier 
--prediction_fn predict_sklearn 
--evaluation_fn evaluate_newdb 
--log_level INFO 
--random_seed 0
```


- This CLI call launches a straightforward analysis pipeline on precomputed features (no preprocessing, no eda, 80-20 split, no hopt, sklearn prediction):
```
python analysis_newdb.py 
--precomputed_features_path data/processed/newdb/ 
--preprocessing_fn preprocess_passthrough 
--eda_fn eda_passthrough 
--feature_extraction_fn features_newdb 
--split_fn split_train_test 
--split_ratio "80 20" 
--model sklearn_RandomForestClassifier 
--prediction_fn predict_sklearn 
--evaluation_fn evaluate_newdb 
--log_level INFO 
--random_seed 0
```


## License

GNU General Public License (GPL).
