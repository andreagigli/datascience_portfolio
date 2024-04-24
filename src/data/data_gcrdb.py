import warnings

import pandas as pd
from pandas import DataFrame

from src.utils.my_dataframe import downcast
from src.utils.my_statstest import evaluate_relationship_with_cat


def load_data(fpath: str, *args, **kwargs) -> DataFrame:
    """
    Loads data from a specified CSV file into DataFrames for features and target variables.

    Args:
        fpath (str): The file path to the CSV file containing the dataset.
        *args and **kwargs: Only included for compatibility with other load_data functions. Will be ignored.

    Returns:
        TODO
    """
    gcr = pd.read_csv(fpath)
    gcr = gcr.rename(columns={"Unnamed: 0": "Id"})

    return gcr


def preprocess_data(gcr: pd.DataFrame) -> pd.DataFrame:
    """
    TODO
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # region Inspect dataset 'gcr'

    print("\n\n\n*******************************************")
    print("************** Inspect 'gcr' **************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(gcr.shape)
    print("Head:")
    print(gcr.head())
    print("Info:")
    print(gcr.info(show_counts=True))

    print("\n************* Missing data *************")
    print(f"The number of missing values in each column is:")
    print(gcr.isna().sum(axis=0))

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(gcr.nunique())
    print("Description of numerical columns:")
    print(gcr.describe())

    # endregion

    # region Data processing

    print("\n\nPreprocess dataframe")

    print("Discard the Id column")
    gcr = gcr.drop(labels=["Id"], axis=1)

    print("Convert 'Risk' to a numeric column 'Good Risk' with values 1 - good and 0 - bad.")
    gcr["Risk"] = (gcr["Risk"] == "good").astype('int')
    gcr = gcr.rename(columns={"Risk": "Good Risk"})

    print("Cast specific columns to categorical type")
    columns_to_convert = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Duration']  # Note that binary columns must be left numerical
    gcr[columns_to_convert] = gcr[columns_to_convert].astype('category')

    # Cast data to most adequate types in order to save memory
    print("Downcast data to save memory")
    gcr = downcast(gcr)

    print("Impute missing values of the columns 'Saving accounts' and 'Checking account' with global their modes.")
    gcr["Saving accounts"] = gcr["Saving accounts"].fillna(gcr["Saving accounts"].mode()[0])
    gcr["Checking account"] = gcr["Checking account"].fillna(gcr["Checking account"].mode()[0])

    """
    Note: Considering group-wise instead of global missing value imputation could potentially enhance the accuracy.
    This would involve grouping the data based on a correlated categorical feature and using the mean/median/mode specific to each group for imputation.
    Potential correlations between an uncorrupted categorical feature and a corrupted categorical or numerical feature can be revealed by chi2 tests or anova tests respectively. 
    
    Example: Evaluate correlation between "Saving account" (categorical) and other categorical features, then perform group-wise value imputation according to the most correlated one. 
    
    print("To evaluate the possibility of group-wise missing value imputation, evaluate the relationship between features "
          "with missing values and categorical features without missing values.")
    features_with_missing_values = ["Saving accounts", "Checking account"]
    for tested_feature in features_with_missing_values:
        chi_results, anova_results = evaluate_relationship_with_cat(gcr, tested_feature, verbose=True)

    # I noticed that the mode values of "Checking account" change significantly across different "Purpose" groups. Let's 
    # compare the global and group-wise modes. 
    
    global_mode = gcr['Checking account'].mode()[0]
    print(f"Global mode for 'Checking account':\n{global_mode}")

    print("Group-wise mode for 'Checking account' by 'Purpose':")
    group_modes = gcr.groupby('Purpose')['Checking account'].agg(lambda x: pd.Series.mode(x)[0]).reset_index(name='Mode')
    print(group_modes)
        
    # One can impute missing values of "Checking account" "Purpose"-wise as opposed to globally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gcr["Checking account"] = gcr.groupby("Purpose")["Checking account"].transform(lambda x: x.fillna(x.mode()[0]))
    """

    # endregion

    return gcr