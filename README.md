# Data Science Portfolio



## Description

This package provides a modular framework for managing multiple data analysis projects. 
Users can initiate projects in isolated branches, leveraging both pre-built and customizable Python modules.



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
│   └── reports/            # Generated analysis reports and plots
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

1. Navigate to the project's root directory:
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
	  

## Try It Out-of-the-Box

You can run an example analysis, where house prices predictions are performed on the California Housing Dataset:

```
cd path/to/portfolio_ML_datascience
python analysis_exampledb.py --data_path ../../data/external/exampledb/california_housing.csv --data_loading_fn load_exampledb --model sklearn_RandomForestRegressor --hparams "{\"sklearn_RandomForestRegressor__n_estimators\": \"randint(20, 200)\", \"sklearn_RandomForestRegressor__max_depth\": 10}" --hopt_n_rndcv_samplings 5 --hopt_subsampling_fn subsample_passthrough --hopt_subsampling_rate 1.0 --preprocessing_fn preprocess_passthrough --eda_fn eda_passthrough --feature_extraction_fn features_exampledb --split_fn split_train_test --split_ratio "80 20" --n_folds 3 --prediction_fn predict_sklearn --evaluation_fn evaluate_exampledb --log_level INFO --random_seed 0 --save_output --output_data_dir ../../data/processed/ --output_model_dir ../../models/ --output_reports_dir ../../outputs/reports/ --output_figures_dir ../../outputs/figures/
```


## Conduct a New Analysys

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
import src.models.custom_model
import src.data.data_newdb
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


### Notes on Running a New Analysis

Take these notes into account when defining the command line instructions for a new analysis:

* **Functions that Must Be Implemented**: For a new dataset, `newdb`, you MUST implement and specify functions for:
  ```
  --data_loading_fn load_newdb`
  --feature_extraction_fn features_newdb`
  --evaluation_fn evaluate_newdb`
  ```

* **Functions that Can Be Re-Implemented**: Default functions are available for other steps:
  ```
  --hopt_subsampling_fn subsample_passthrough
  --preprocessing_fn preprocess_passthrough
  --eda_fn eda_passthrough
  ```
  - The `--split_fn` argument must be given a function_id, such as `split_train_test`, `split_train_val_test`, or a custom one. 
  - The `--prediction_fn` argument must also be given a function_id, such as `predict_zeros`, `predict_sklearn` (for sklearn-compatible models), or a custom one. 

* **Data Splitting Modality**: The data splitting modality can be regulated through the arguments `split_fn`, `split_ratio`, and `n_folds`. 
  - For scenarios without hyperparameter optimization (hopt):
    ```
    --split_fn split_train_test 
    --split_ratio "80 20"
    ```
  - For hyperparameter optimization, utilize:
    - K-fold cross-validation:
      ```
      --split_fn split_train_test 
      --split_ratio "80 20" 
      --n_folds 3
      ```
    - Or a fixed split:
      ```
      --split_fn split_train_val_test 
      --split_ratio "70 15 15"
      ```

* **Hyperparameter Optimization (hopt)**: Hopt is triggered if there are distribution strings among the provided `hparams` values. 
  - Example triggering hopt:
    ```
    --hparams "{\"sklearn_RandomForestRegressor__n_estimators\": \"randint(20, 200)\", \"sklearn_RandomForestRegressor__max_depth\": 10}"
    ```
  - Example not triggering hopt:
    ```
    --hparams "{\"sklearn_RandomForestRegressor__n_estimators\":150, \"sklearn_RandomForestRegressor__max_depth\": 10}"
    ```
  - During hopt, the arguments `--hopt_n_rndcv_samplings` determine the number of random samplings of the hyperparameters to be optimized, and `--hopt_subsampling_fn` along with `--hopt_subsampling_rate` are used to subsample the training data to save on computation.

* **Precomputed Features**: To reload precomputed features, provide:
  ```
  --precomputed_features_path ../../data/processed/newdb/
  ```
  instead of the standard:
  ```
  --data_path ../../data/external/m5salesdb/ 
  --data_loading_fn load_m5salesdb
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

    
### Example Use Cases

- This example launches a custom analysis on a new dataset `newdb`. 
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


- This example demonstrates a straightforward analysis pipeline (custom load data, no preprocessing, eda, custom feature_extraction, split_train_test, no hopt, sklearn prediction, custom evaluation):
```
python analysis_newdb.py 
--data_path data/external/newdb/ 
--data_loading_fn load_newdb 
--model sklearn_RandomForestClassifier 
--preprocessing_fn preprocess_passthrough 
--eda_fn eda_passthrough 
--feature_extraction_fn features_newdb 
--split_fn split_train_test 
--split_ratio "80 20" 
--prediction_fn predict_sklearn 
--evaluation_fn evaluate_newdb 
--log_level INFO 
--random_seed 0
```


- This example demonstrates a more complex analysis pipeline (reload precomputed features, custom preprocessing, eda, feature extraction, split, hopt, prediction, evaluation):
```
python analysis_newdb.py 
--precomputed_features_path data/processed/newdb/ 
--model sklearn_RandomForestClassifier 
--hparams "{\"sklearn_RandomForestRegressor__n_estimators\": \"randint(20, 200)\", \"sklearn_RandomForestRegressor__max_depth\": 10}"
--hopt_n_rndcv_samplings 5 
--hopt_subsampling_fn subsample_newdb
--hopt_subsampling_rate 1.0 
--preprocessing_fn preprocess_newdb 
--eda_fn eda_newdb 
--feature_extraction_fn extract_features_newdb 
--split_fn split_train_val_test 
--split_ratio "70 15 15" 
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



## License

GNU General Public License (GPL).
