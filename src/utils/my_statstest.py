import warnings

import pandas as pd
import pingouin as pg

from pandas import DataFrame
from typing import Optional, Tuple


def evaluate_relationship_with_cat(df: DataFrame, target_col: str, verbose: bool = False,
                                   ) -> Tuple[Optional[DataFrame], Optional[DataFrame]]:
    """
    Evaluates the relationship between a specified target column (either categorical or numerical) and other
    categorical columns in the dataset that do not have missing values, using Chi-Square and ANOVA tests.
    This is typically used to determine potential group-wise strategies for imputing missing values within the
    target feature based on the values of related categorical features.

    The Chi-Square Test compares two categorical variables to see if ... more intuitive.
    The ANOVA Test determines whether there are significant differences between the means of three or more groups.
    It can be used to evaluate the presence of relationship between a numerical and a target variable by partitioning
    the numerical variable into groups based on the values of the categorical variable.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - target_col (str): The name of the column with missing values or to be analyzed.
    - verbose (bool): If True, prints the detailed results and the list of categorical columns evaluated.

    Returns:
    - Tuple of Optional[pandas.DataFrames]: (chi_square_results, anova_results)
      - chi_square_results: DataFrame with Chi-Square test results if target_col is categorical.
      - anova_results: DataFrame with ANOVA test results if target_col is numerical.
    """
    # Ensure data does not include rows where the target column is NaN
    df = df.dropna(subset=[target_col])

    # Define dataframes for results
    chi_square_results = []
    chi_square_results_df = None
    anova_results = []
    anova_results_df = None

    # List of categorical columns without missing values, not including the target column
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if
                        df[col].isnull().sum() == 0 and col != target_col]  # TODO: you can simplify using isna??

    # If target column is categorical, perform Chi-Square test else an ANOVA test
    if df[target_col].dtype.name in ['category', 'object']:
        for col in categorical_cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Temporarily suppress warnings
                _, _, chi2_stats = pg.chi2_independence(data=df, x=target_col, y=col)
            p_value = chi2_stats['pval'][0]
            chi_square_results.append({'Column': col, 'P-value': p_value})
        chi_square_results_df = pd.DataFrame(chi_square_results).sort_values('P-value')
        if verbose:
            print(f"Evaluating relationships between {target_col} (categorical) and other categorical features without missing values (Chi-Square Test):")
            if chi_square_results_df is not None and not chi_square_results_df.empty:
                print(chi_square_results_df)
            else:
                print("No Chi-Square Test Results available.")

    else:
        for col in categorical_cols:
            aov = pg.anova(data=df, dv=target_col, between=col, detailed=True)
            p_value = aov.at[0, 'p-unc']  # Extracting the p-value
            anova_results.append({'Column': col, 'P-value': p_value})
        anova_results_df = pd.DataFrame(anova_results).sort_values('P-value')
        if verbose:
            print(f"Evaluating relationships between {target_col} (numerical) and other categorical features without missing values (ANOVA Test):")
            if anova_results_df is not None and not anova_results_df.empty:
                print(anova_results_df)
            else:
                print("No ANOVA Test Results available.")

    return chi_square_results_df, anova_results_df