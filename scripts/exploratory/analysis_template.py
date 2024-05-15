"""
Example CLI calls.

--- SALES PREDICTION ON M5 SALES DATASET ---
PREFERRED: using LGBMRegressor (handles large datasets, without need to one-hot encode cathegorical variables)
python analysis_exampledb.py
--data_path ./data/external/m5salesdb/
--data_loading_fn load_m5salesdb
--model sklearn_compatible_LGBMRegressor
--hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ./data/processed/
--output_model_dir ./models/
--output_reports_dir ./outputs/reports/
--output_figures_dir ./outputs/figures/

IF FEATURES WERE PRECOMPUTED:
python analysis_exampledb.py
--precomputed_features_path ./data/processed/m5salesdb_debug_100/
--model sklearn_compatible_LGBMRegressor
--hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ./data/processed/
--output_model_dir ./models/
--output_reports_dir ./outputs/reports/
--output_figures_dir ./outputs/figures/

FASTER, FOR DEBUG:
python analysis_exampledb.py
--precomputed_features_path ./data/processed/m5salesdb_debug_10/
--model sklearn_compatible_LGBMRegressor
--hparams "{\"sklearn_compatible_LGBMRegressor__num_leaves\": \"randint(20, 200)\", \"sklearn_compatible_LGBMRegressor__learning_rate\": \"loguniform(0.001, 1)\", \"sklearn_compatible_LGBMRegressor__n_estimators\": 1000}"
--hopt_n_rndcv_samplings 5
--hopt_subsampling_fn subsample_train_m5salesdb
--hopt_subsampling_rate 1.0
--preprocessing_fn preprocess_m5salesdb
--eda_fn eda_m5salesdb
--feature_extraction_fn features_m5salesdb
--split_fn split_m5salesdb
--prediction_fn predict_m5salesdb
--look_back_days_sequential_prediction 380
--evaluation_fn evaluate_m5salesdb
--log_level INFO
--random_seed 0
--save_output
--output_data_dir ./data/processed/
--output_model_dir ./models/
--output_reports_dir ./outputs/reports/
--output_figures_dir ./outputs/figures/
"""
import logging
import os

from src.data_science_pipeline import run_pipeline
from src.register_functions import register_all_functions
from src.utils.my_argparsing import FunctionRegistry, parse_data_science_arguments
from src.utils.my_os import ensure_dir_exists


def main():
    # Step 1: Register all functions
    function_registry = FunctionRegistry()
    register_all_functions(function_registry)

    # Step 2: Parse command-line arguments
    parsed_args = parse_data_science_arguments(function_registry)

    # Step 3: Set up logging
    if parsed_args.save_output:
        output_reports_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")
        ensure_dir_exists(output_reports_dir)
        log_fname = os.path.join(output_reports_dir, f"log_{parsed_args.run_id}.log")
        logging.basicConfig(level=parsed_args.log_level.upper(),
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            filename=log_fname,
                            filemode='w')
    else:
        logging.basicConfig(level=parsed_args.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Step 4: Run the pipeline
    logger.info("Starting the machine learning pipeline.")
    run_pipeline(parsed_args, function_registry)


if __name__ == "__main__":
    main()
