from lightgbm import LGBMRegressor, LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.data.load_data_fns import load_data_exampledb, load_data_m5salesdb, load_data_gcrdb
from src.data.preprocess_data_fns import preprocess_data_passthrough, preprocess_data_m5salesdb, preprocess_data_gcrdb
from src.data.split_data_fns import split_data_train_test, split_data_train_val_test, split_data_m5salesdb, \
    split_data_passthrough
from src.eda.eda_fns import eda_passthrough, eda_m5salesdb, eda_gcrdb
from src.evaluation.evaluation_fns import evaluate_exampledb, evaluate_m5salesdb, evaluate_gcrdb, evaluate_passthrough
from src.features.extract_features_fns import extract_features_exampledb, extract_features_m5salesdb, \
    extract_features_gcrdb
from src.models.custom_linear_regressor import CustomModel
from src.optimization.hopt_subsampling import hopt_subsampling_passthrough, hopt_subsampling_m5salesdb
from src.prediction.prediction_fns import predict_sklearn, predict_m5salesdb, predict_zeros
from src.utils.my_argparsing import FunctionRegistry


def register_all_functions(function_registry: FunctionRegistry):
    # Data Loading Functions
    function_registry.register("load_exampledb", load_data_exampledb)
    function_registry.register("load_m5salesdb", load_data_m5salesdb)
    function_registry.register("load_gcrdb", load_data_gcrdb)

    # Preprocessing Functions
    function_registry.register("preprocess_passthrough", preprocess_data_passthrough)
    function_registry.register("preprocess_m5salesdb", preprocess_data_m5salesdb)
    function_registry.register("preprocess_gcrdb", preprocess_data_gcrdb)

    # EDA Functions
    function_registry.register("eda_exampledb", eda_passthrough)
    function_registry.register("eda_m5salesdb", eda_m5salesdb)
    function_registry.register("eda_gcrdb", eda_gcrdb)
    function_registry.register("eda_passthrough", eda_passthrough)

    # Feature Extraction Functions
    function_registry.register("features_exampledb", extract_features_exampledb)
    function_registry.register("features_m5salesdb", extract_features_m5salesdb)
    function_registry.register("features_gcrdb", extract_features_gcrdb)

    # Splitting Functions
    function_registry.register("split_m5salesdb", split_data_m5salesdb)
    function_registry.register("split_train_test", split_data_train_test)
    function_registry.register("split_train_val_test", split_data_train_val_test)
    function_registry.register("split_passthrough", split_data_passthrough)

    # Prediction Functions
    function_registry.register("predict_sklearn", predict_sklearn)
    function_registry.register("predict_m5salesdb", predict_m5salesdb)
    function_registry.register("predict_zeros", predict_zeros)

    # Evaluation Functions
    function_registry.register("evaluate_exampledb", evaluate_exampledb)
    function_registry.register("evaluate_m5salesdb", evaluate_m5salesdb)
    function_registry.register("evaluate_gcrdb", evaluate_gcrdb)
    function_registry.register("evaluate_passthrough", evaluate_passthrough)

    # Models
    function_registry.register("sklearn_LinearRegression", LinearRegression)
    function_registry.register("sklearn_Ridge", Ridge)
    function_registry.register("sklearn_SVC", SVC)
    function_registry.register("sklearn_RandomForestRegressor", RandomForestRegressor)
    function_registry.register("sklearn_HistGradientBoostingRegressor", HistGradientBoostingRegressor)
    function_registry.register("sklearn_compatible_LGBMRegressor", LGBMRegressor)
    function_registry.register("sklearn_compatible_LGBMClassifier", LGBMClassifier)
    function_registry.register("mymodel", CustomModel)

    # Data Transformers
    function_registry.register("sklearn_RBFSampler", RBFSampler)
    function_registry.register("sklearn_StandardScaler", StandardScaler)
    function_registry.register("sklearn_MinMaxScaler", MinMaxScaler)

    # Random Distribution Functions
    function_registry.register("loguniform", loguniform)
    function_registry.register("randint", randint)
    function_registry.register("uniform", uniform)

    # Hyperparameter Optimization Subsampling Functions
    function_registry.register("hopt_subsampling_m5salesdb", hopt_subsampling_m5salesdb)
    function_registry.register("hopt_subsampling_passthrough", hopt_subsampling_passthrough)
