import inspect
import json
import logging
import os
import pickle
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.models.models_misc import init_reload_model
from src.optimization.custom_sk_validators import PredefinedSplit
from src.utils.my_argparsing import FunctionRegistry, string_to_distribution, escape_quotes_in_curly_brackets
from src.utils.my_os import ensure_dir_exists


def run_pipeline(parsed_args, function_registry: FunctionRegistry):
    """
        Executes the machine learning pipeline.

        This function dynamically loads data loading, preprocessing, feature extraction,
        and evaluation functions based on the provided arguments. It initializes or reloads
        the specified model, performs data splitting, model training (including hyperparameter
        optimization if specified), and evaluates the model.

        Parameters:
        parsed_args (argparse.Namespace): The namespace object containing command-line arguments.
        function_registry (FunctionRegistry): The function registry instance to retrieve available functions.

        Note:
        The function handles different scenarios such as using sklearn models, TensorFlow models,
        reusing pre-trained models, and saving outputs including models, reports, and figures.
        """
    # Check if the logger exists, if not, create a default one
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logging.basicConfig(level=parsed_args.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Determine required functions
    logger.info("Initializing functions and model...")
    load_data_fn = function_registry.get_function(parsed_args.data_loading_fn)
    preprocess_fn = function_registry.get_function(parsed_args.preprocessing_fn)
    eda_fn = function_registry.get_function(parsed_args.eda_fn)
    extract_features_fn = function_registry.get_function(parsed_args.feature_extraction_fn)
    split_data_fn = function_registry.get_function(parsed_args.split_fn)
    aux_split_params = {}
    hopt_subsampling_fn = function_registry.get_function(parsed_args.hopt_subsampling_fn)
    predict_fn = function_registry.get_function(parsed_args.prediction_fn)
    aux_predict_params = {}
    evaluate_fn = function_registry.get_function(parsed_args.evaluation_fn)
    aux_eval_params = {}

    # Save the extract_feature_fn into the predict parameters as it may be necessary for some custom prediction function, such as sequential prediction
    aux_predict_params["extract_features_fn"] = extract_features_fn

    # Initialize model or reload existing one
    model = init_reload_model(parsed_args, function_registry)

    # Load data and extract features, or reload previously computed features
    if parsed_args.precomputed_features_path is None:
        # Load data
        logger.info("Loading data and extracting features...")
        dataset = load_data_fn(parsed_args.data_path, debug=False)  # Set debug=True to select a subset of the items (faster computation)
        if not isinstance(dataset, tuple):
            dataset = (dataset,)

        # Preprocess data
        dataset = preprocess_fn(*dataset)
        if not isinstance(dataset, tuple):
            dataset = (dataset,)

        # Exploratory data analysis
        eda_fn(*dataset)

        # Extract features
        X, Y = extract_features_fn(*dataset)  # X and Y are expected to be pd.DataFrame

        if parsed_args.save_output:
            logger.info("Storing computed features...")
            dbname = os.path.basename(
                parsed_args.data_path.rstrip('/'))  # Extracts filename from data_path without extension
            output_dir = os.path.join(parsed_args.output_data_dir, dbname)
            os.makedirs(output_dir, exist_ok=True)
            X.to_pickle(os.path.join(output_dir, "X.pkl"))
            Y.to_pickle(os.path.join(output_dir, "Y.pkl"))
        del dataset

    else:
        # Reload the pre-computed features
        X = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "X.pkl"))
        Y = pd.read_pickle(os.path.join(parsed_args.precomputed_features_path, "Y.pkl"))

    # Parse split arguments
    if parsed_args.split_ratio is not None:  # Parse the split_ratio if provided
        split_ratios = [float(item) for item in
                        parsed_args.split_ratio.split()]  # Format correctness was already checked
        if len(split_ratios) != 2 and len(split_ratios) != 3:
            raise ValueError("split_ratio must include two or three integers for train-test or train-val-test percentages.")
        aux_split_params["train_prc"] = split_ratios[0]
        # Assign val_prc only if there are 3 values, otherwise set it to 0 or None based on your logic preference
        aux_split_params["val_prc"] = split_ratios[1] if len(split_ratios) == 3 else 0
        # Assign test_prc based on the number of ratios provided
        aux_split_params["test_prc"] = split_ratios[2] if len(split_ratios) == 3 else split_ratios[1]
    else:
        aux_split_params["train_prc"] = None
        aux_split_params["val_prc"] = None
        aux_split_params["test_prc"] = None
    aux_split_params["n_folds"] = parsed_args.n_folds
    aux_split_params['stratified'] = parsed_args.stratified_kfold
    aux_split_params["random_seed"] = parsed_args.random_seed
    aux_split_params["look_back_days_sequential_prediction"] = parsed_args.look_back_days_sequential_prediction

    # Split data
    (X_train, Y_train,
     X_val, Y_val,
     X_test, Y_test,
     cv_indices,
     optional_split_info) = split_data_fn(X, Y, **aux_split_params)
    # Accumulate potential split parameters to the predict parameters as they may be useful for the prediction function
    if optional_split_info:
        aux_predict_params.update(optional_split_info)

    # Optimize hyperparameters and train model, distinguishing between sklearn and pytorch/tensorflow pipeline
    if "sklearn" in function_registry.get_function(parsed_args.model).__module__:
        # Ignore specific warnings relating to sparse columns warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Allowing arbitrary scalar fill_value in SparseDtype is deprecated")
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but RBFSampler was fitted with feature names")
        warnings.filterwarnings("ignore", message="X does not have valid feature names, but Ridge was fitted with feature names")
        warnings.filterwarnings("ignore", message="pandas.DataFrame with sparse columns found.It will be converted to a dense numpy array.")

        # Construct the data transformers list dynamically
        transformer_instances = []
        for transformer_name in parsed_args.data_transformers:
            TransformerClass = function_registry.get_function(transformer_name)
            if TransformerClass is not None:
                valid_params = inspect.signature(TransformerClass.__init__).parameters
                # If hparams are passed in the command line arguments, filter them to find hparams valid and fixed for the current transformer
                if parsed_args.hparams is not None:
                    transformer_params = {pname.split('__')[1]: pvalue for pname, pvalue in
                                          json.loads(parsed_args.hparams).items()
                                          if pname.startswith(transformer_name)
                                          and pname.split('__')[1] in valid_params
                                          and not isinstance(pvalue, str)}
                else:
                    transformer_params = {}
                # If transformer accepts a random_state and parsed_args.random_seed is provided, add it
                if 'random_state' in valid_params and parsed_args.random_seed is not None:
                    transformer_params['random_state'] = parsed_args.random_seed
                # Initialize the transformer with fixed parameters
                transformer_instances.append((transformer_name, TransformerClass(**transformer_params)))

        # Define an sklearn Pipeline
        pipeline_steps = []
        pipeline_steps.extend(transformer_instances)
        pipeline_steps.append((parsed_args.model, model))

        pipeline = Pipeline(pipeline_steps)

        # Set up hyperparams optimization if valid hyperparameters are provided
        optimization_needed = False
        param_distributions = {}
        if parsed_args.hparams is not None:
            raw_param_distributions = json.loads(parsed_args.hparams)

            distr_pattern = r'^(uniform|loguniform|randint)\(\d+(\.\d+)?, \d+(\.\d+)?\)$'
            for estimname_paramname, distr_str in raw_param_distributions.items():
                # Check if the parameter value is a proper "distribution" string
                # (not well-formed distribution strings or numeric values are ignored by hopt)
                if isinstance(distr_str, str) and re.match(distr_pattern, distr_str):
                    # Split the parameter name to get the component and parameter
                    estimator, paramname = estimname_paramname.split("__")
                    # Check if the component is the model or one of the transformers
                    if estimator == parsed_args.model and paramname in model.get_params().keys():
                        param_distributions[estimname_paramname] = string_to_distribution(distr_str, function_registry)
                    elif estimator in dict(transformer_instances) and paramname in dict(transformer_instances)[estimator].get_params().keys():
                        param_distributions[estimname_paramname] = string_to_distribution(distr_str, function_registry)

                    optimization_needed = True

                # Note: Parameters not matching the pattern are ignored

        # Either perform hyperparams optimization with refit=True OR just fit the model
        cv_results_df = None
        if optimization_needed:
            # Case 1: Use predefined k-fold validation splits if no explicit validation set is provided
            if X_val is None and Y_val is None and cv_indices is not None:
                cv = cv_indices
                # Optionally subsample the training set and shallow copy it into X_tmp and Y_tmp for consistency with case 2
                X_tmp, Y_tmp, cv_indices = hopt_subsampling_fn(
                    X_train,
                    Y_train,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )
                n_folds = len(cv_indices)

            # Case 2: Use a predefined validation set when provided, ignoring cv_indices
            elif X_val is not None and Y_val is not None and cv_indices is None:
                # Subsample training and validation data to reduce model training time or manage computational resources
                X_train_subsampled, Y_train_subsampled, _ = hopt_subsampling_fn(
                    X_train,
                    Y_train,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )
                X_val_subsampled, Y_val_subsampled, _ = hopt_subsampling_fn(
                    X_val,
                    Y_val,
                    cv_indices=cv_indices,
                    subsampling_rate=parsed_args.hopt_subsampling_rate,
                    random_seed=parsed_args.random_seed,
                )

                # Combine training and validation datasets for use in scikit-learn's model selection tools
                X_tmp = pd.concat([X_train_subsampled, X_val_subsampled], ignore_index=True)
                Y_tmp = pd.concat([Y_train_subsampled, Y_val_subsampled], ignore_index=True)

                # Creating one explicit fold is not possible out of the box with sklearn (cv_indices must contain at
                # least two different train/val splits). For this reason, a custom validator PredefinedSplit was defined
                # to allow for a single train/val split.
                n_folds = 1
                val_fold = [0] * len(X_train_subsampled) + [1] * len(X_val_subsampled)  # Indicates which samples are from the training set (0) and which are from the validation set (1)
                del X_train_subsampled, Y_train_subsampled, X_val_subsampled, Y_val_subsampled
                cv = PredefinedSplit(test_fold=np.array(val_fold))

            else:
                raise ValueError("Only one is expected to be not None between cv_indices and (X_val, Y_val).")

            logger.info(
                f"Optimizing model hyperparameters ({parsed_args.hopt_n_rndcv_samplings} samplings * {n_folds} folds and fitting the model...")
            search = RandomizedSearchCV(pipeline,
                                        param_distributions,
                                        n_iter=parsed_args.hopt_n_rndcv_samplings,
                                        refit=False,
                                        # Avoid automatic refitting because hopt may be performed on subsampled data
                                        cv=cv,
                                        random_state=parsed_args.random_seed,
                                        return_train_score=True,  # May slow down the execution
                                        verbose=3)
            search.fit(X_tmp.squeeze(), Y_tmp.squeeze())  # search.fit(convert_df_to_sparse_matrix(X_tmp), convert_df_to_sparse_matrix(Y_tmp)) for sparse fitting (debug it)

            # Collect the cv_results_
            if hasattr(search, "cv_results_"):
                cv_results_df = pd.DataFrame(search.cv_results_)
                float_columns = cv_results_df.select_dtypes(include=['float']).columns
                cv_results_df[float_columns] = cv_results_df[float_columns].round(4)
            else:
                cv_results_df = None

            # Collect the best_params_ and instantiate a new pipeline with optimized values
            if hasattr(search, 'best_params_'):
                best_params = search.best_params_

                best_pipeline = pipeline
                for pname, pvalue in best_params.items():
                    step, param = pname.split('__')
                    setattr(best_pipeline.named_steps[step], param, pvalue)
                pipeline = best_pipeline  # Shade optimized_pipeline behind the name model for unified use in case of no hopt
            else:
                raise AttributeError(
                    "Failed to find the best parameters. The model fitting process did not complete successfully.")

        # Fit the model (either optimized or not)
        logger.info("Fitting the model...")
        pipeline.fit(X_train.squeeze(), Y_train.squeeze())

        model = pipeline  # For consistency with other libraries, simply call the Sklearn pipeline "model"

    elif "tensorflow" in function_registry.get_function(parsed_args.model):
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None

    elif "torch" in function_registry.get_function(parsed_args.model):
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None

    else:
        optimization_needed = True
        Y_pred = None
        Y_train_pred = None
        cv_results_df = None

    # Compute model predictionsce    logger.info("Computing model predictions...")
    Y_pred, Y_train_pred, optional_predictions = predict_fn(
        model,
        X_test.squeeze(),
        Y_test.squeeze(),  # Only relevant for prediction_fn = predict_zeros
        X_train.squeeze(),
        Y_train.squeeze(),  # Only relevant for prediction_fn = predict_zeros
        **aux_predict_params,
    )

    # Accumulate potential extra predictions into aux_eval_params, which may be useful for complete model evaluation
    if optional_predictions is not None:
        aux_eval_params.update(optional_predictions)

    # Evaluate model predictions
    logger.info("Evaluating model predictions...")

    if isinstance(Y_test, pd.DataFrame):
        target_names = Y_test.columns.tolist()
    elif isinstance(Y_test, pd.Series):
        target_names = [Y_test.name]
    else:
        target_names = None

    scores, figs = evaluate_fn(
        Y_test.squeeze(),
        Y_pred,
        model,
        target_names,
        Y_train.squeeze(),
        Y_train_pred,
        **aux_eval_params,
    )

    # Save results
    if parsed_args.save_output:
        logger.info("Storing results...")
        ensure_dir_exists(parsed_args.output_data_dir)
        model_summaries_dir = os.path.join(parsed_args.output_model_dir, "model_summaries")
        ensure_dir_exists(model_summaries_dir)
        trained_models_dir = os.path.join(parsed_args.output_model_dir, "trained_models")
        ensure_dir_exists(trained_models_dir)
        output_reports_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")  # Done before
        ensure_dir_exists(output_reports_dir)
        output_figures_dir = os.path.join(parsed_args.output_reports_dir, f"report_{parsed_args.run_id}")
        ensure_dir_exists(parsed_args.output_figures_dir)

        model_path = os.path.join(trained_models_dir, f"trained_model_{parsed_args.run_id}.pkl")
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)

        if cv_results_df is not None:
            cv_results_path = os.path.join(model_summaries_dir, f"cv_results_{parsed_args.run_id}.csv")
            cv_results_df.to_csv(cv_results_path, index=True)

        report_path = os.path.join(output_reports_dir, f"scores_{parsed_args.run_id}.csv")
        scores.to_csv(report_path, index=True)

        for fig_name, fig in figs.items():
            fig_path = os.path.join(output_figures_dir, f"{fig_name}_{parsed_args.run_id}.png")
            fig.savefig(fig_path)
            plt.close(fig)  # Close the figure after saving to free up memory

        # Collect analysis information in readable form
        script_call = "python " + " ".join(sys.argv)
        script_call = script_call.replace("\\", "/")
        script_call = script_call.replace(f'--split_ratio {parsed_args.split_ratio}',
                                          f'--split_ratio "{parsed_args.split_ratio}"')
        script_call = script_call.replace(f'--hparams {parsed_args.hparams}', f'--hparams "{parsed_args.hparams}"')
        script_call = escape_quotes_in_curly_brackets(script_call)

        # Convert relative paths to absolute paths
        matches = re.findall(re.compile(r'\.\.?/[^ ]+'), script_call)
        for match in matches:
            abs_path = os.path.abspath(match)
            script_call = script_call.replace(match, abs_path)

        # Format model steps' information and hyperparameters
        if "sklearn" in function_registry.get_function(parsed_args.model).__module__:
            model_steps_info = {step[0]: step[1].__class__.__name__ for step in model.steps}
            final_hyperparams = {step[0]: step[1].get_params() for step in model.steps}
        elif "tensorflow" in function_registry.get_function(parsed_args.model):
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO
        elif "torch" in function_registry.get_function(parsed_args.model):
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO
        else:
            model_steps_info = None  # TODO
            final_hyperparams = None  # TODO

        # Store information as a text file. Do not use JSON because it messes up with necessary escapes.
        experiment_info = {
            "run_id": parsed_args.run_id,
            "data_path": parsed_args.data_path,
            "dataset_shape": str(X.shape),
            "precomputed_features_path": parsed_args.precomputed_features_path,
            "data_loading_fn": parsed_args.data_loading_fn,
            "model": parsed_args.model,
            "data_transformers": parsed_args.data_transformers,
            "hparams": parsed_args.hparams,
            "hopt_n_rndcv_samplings": parsed_args.hopt_n_rndcv_samplings,
            "hopt_subsampling_fn": parsed_args.hopt_subsampling_fn,
            "hopt_subsampling_rate": parsed_args.hopt_subsampling_rate,
            "reuse_model": "True" if parsed_args.reuse_model else "False",
            "preprocessing_fn": parsed_args.preprocessing_fn,
            "eda_fn": parsed_args.eda_fn,
            "feature_extraction_fn": parsed_args.feature_extraction_fn,
            "split_fn": parsed_args.split_fn,
            "split_ratio": parsed_args.split_ratio,
            "n_folds": parsed_args.n_folds,
            "stratified_kfold": "True" if parsed_args.stratified_kfold else "False",
            "evaluation_fn": parsed_args.evaluation_fn,
            "log_level": parsed_args.log_level,
            "random_seed": parsed_args.random_seed,
            "save_output": "True" if parsed_args.save_output else "False",
            "output_data_dir": parsed_args.output_data_dir,
            "output_model_dir": parsed_args.output_model_dir,
            "output_reports_dir": parsed_args.output_reports_dir,
            "output_figures_dir": parsed_args.output_figures_dir,
            "model_steps_info": model_steps_info,
            "initial_hyperparameters": parsed_args.hparams,
            "final_hyperparameters": final_hyperparams,
            "optimization_performed": "True" if optimization_needed else "False",
            "performance_metrics": scores.index.tolist(),
            "script_call": " ".join(sys.argv),
            "additional_information": "None",
        }
        # The file is saved with the run_id as part of the filename for easy identification.
        with open(os.path.join(output_reports_dir, f"experiment_details_{parsed_args.run_id}.txt"), "w") as file:
            for key, value in experiment_info.items():
                file.write(f"{key}: {value}\n\n")

