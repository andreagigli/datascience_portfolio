import os

import pandas as pd
from pandas import DataFrame

from src.utils.my_dataframe import downcast


def load_data(dpath: str, *args, **kwargs) -> DataFrame:
    """
    Loads data from a specified CSV file into DataFrames for features and target variables.

    Args:
        dpath (str): The file path to the directory containing the dataset.
        *args and **kwargs: Only included for compatibility with other load_data functions. Will be ignored.

    Returns:
        TODO
    """
    gcr = pd.read_csv(os.path.join(dpath, "gcrdb.csv"))
    gcr = gcr.rename(columns={"Unnamed: 0": "id"})

    # Standardize column names so that they contain only lower case letters and no spaces.
    gcr.columns = gcr.columns.str.replace(' ', '_').str.lower()

    return gcr


def preprocess_data(gcr: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given DataFrame by performing several data wrangling tasks.

    Detailed Steps:
        1. Inspect the structure and properties of the DataFrame including shape, head, data types, and summary statistics.
        2. Drop the 'id' column as it is typically not useful for modeling.
        3. Convert the 'risk' column to a numeric binary column 'good_risk'.
        4. Cast specific columns to categorical types.
        5. Downcast numerical data to more memory-efficient types.
        6. Impute missing values for 'saving_accounts' and 'checking_account' using their global modes.
        7. Optionally, explore group-wise imputation for more accuracy.

    Parameters:
        gcr (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with modified and imputed data.
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

    print("Discard the id column")
    gcr = gcr.drop(labels=["id"], axis=1)

    print("Impute missing values of the columns 'saving_accounts' and 'checking_account' with global their modes.")
    gcr["saving_accounts"] = gcr["saving_accounts"].fillna(gcr["saving_accounts"].mode()[0])
    gcr["checking_account"] = gcr["checking_account"].fillna(gcr["checking_account"].mode()[0])

    print("Convert 'risk' to a numeric column 'good_risk' with values 1 - good and 0 - bad.")
    gcr["risk"] = (gcr["risk"] == "good").astype('int')
    gcr = gcr.rename(columns={"risk": "good_risk"})
    bool_to_num_converted_cols = ["good_risk"]

    print("Cast specific columns to categorical type")
    cat_to_num_converted_columns = ['job', 'sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']  # Note that binary columns must be left numerical
    gcr[cat_to_num_converted_columns] = gcr[cat_to_num_converted_columns].astype('category')
    print("Replace category values with their numerical codes")
    # Store the correspondance {catcode: catvalue} for future analyses. TODO: return and store this mapping.
    catcode_catvalue_mappings = {}
    for col in cat_to_num_converted_columns:
        catcode_catvalue_mappings[col] = dict(enumerate(gcr[col].cat.categories))
        gcr[col] = gcr[col].cat.codes

    # Cast data to most adequate types in order to save memory
    print("Downcast data to save memory")
    gcr = downcast(gcr)

    """
    Note: Considering group-wise instead of global missing value imputation could potentially enhance the accuracy.
    This would involve grouping the data based on a correlated categorical feature and using the mean/median/mode specific to each group for imputation.
    Potential correlations between an uncorrupted categorical feature and a corrupted categorical or numerical feature can be revealed by chi2 tests or anova tests respectively. 
    
    Example: Evaluate correlation between "saving_account" (categorical) and other categorical features, then perform group-wise value imputation according to the most correlated one. 
    
    print("To evaluate the possibility of group-wise missing value imputation, evaluate the relationship between features "
          "with missing values and categorical features without missing values.")
    features_with_missing_values = ["saving_accounts", "checking_account"]
    for tested_feature in features_with_missing_values:
        chi_results, anova_results = evaluate_relationship_with_cat(gcr, tested_feature, verbose=True)

    # I noticed that the mode values of "checking_account" change significantly across different "purpose" groups. Let's 
    # compare the global and group-wise modes. 
    
    global_mode = gcr['checking_account'].mode()[0]
    print(f"Global mode for 'checking_account':\n{global_mode}")

    print("Group-wise mode for 'checking_account' by 'purpose':")
    group_modes = gcr.groupby('purpose')['checking_account'].agg(lambda x: pd.Series.mode(x)[0]).reset_index(name='Mode')
    print(group_modes)
        
    # One can impute missing values of "checking_account" "purpose"-wise as opposed to globally.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gcr["checking_account"] = gcr.groupby("purpose")["checking_account"].transform(lambda x: x.fillna(x.mode()[0]))
    """

    # Print a summary of the preprocessed dataset
    print("\n" + "=" * 60)
    print("INTERPRETATION OF THE POSTPROCESSED DATAFRAME 'gcr'")
    print("=" * 60)
    for col in gcr.columns:
        if col in bool_to_num_converted_cols:
            print(f"- {col} (boolean)")
        elif col in cat_to_num_converted_columns:
            mapping_info = ", ".join([f"{code}-{val}" for code, val in catcode_catvalue_mappings[col].items()])
            print(f"- {col} (catcode numeric): {mapping_info}")
        else:
            print(f"- {col} ({gcr[col].dtype})")
    print("=" * 60 + "\n")
    # endregion

    return gcr
