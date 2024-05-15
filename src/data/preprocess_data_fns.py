import re
import warnings

import numpy as np
import pandas as pd

from src.utils.my_dataframe import downcast


def preprocess_data_gcrdb(gcr: pd.DataFrame) -> pd.DataFrame:
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
    cat_to_num_converted_columns = ['job', 'sex', 'housing', 'saving_accounts', 'checking_account',
                                    'purpose']  # Note that binary columns must be left numerical
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


def preprocess_data_m5salesdb(sales: pd.DataFrame, sell_prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses sales, sell prices, and calendar datasets for time series analysis.

    Steps:
    1. Inspect datasets for content, structure, and data integrity.
    2. Correct anomalies and standardize formats.
    3. Merge datasets into a single long-format dataframe.

    Inputs:
    - sales: DataFrame of daily sales data.
    - sell_prices: DataFrame of item prices.
    - calendar: DataFrame of date information.

    Parameters:
    - sales (pd.DataFrame): Daily sales data.
    - sell_prices (pd.DataFrame): Item price data.
    - calendar (pd.DataFrame): Date information data.

    Returns:
    - sales (pd.DataFrame): A unified dataset ready for feature engineering and subsequent analysis.
    """
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', None)

    # region Inspect dataset 'sales'
    print("\n\n\n*******************************************")
    print("************* Inspect 'sales' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(sales.shape)
    print("There are more than 10 columns to be printed: attempt grouping patterned columns with same dtype.")
    # I know that this dataset has a group of patterned columns with the same type.
    # TODO: the presence of patterned columns with same dtype should be automatic
    patterned_columns = [col for col in sales.columns if re.match('d_\d+', col)]
    # same_type = all(sales[col].dtype == sales[patterned_columns[0]].dtype for col in patterned_columns)
    print("All patterned columns have the same type, let's group them for the purpose of understanding the dataset.\n")
    independent_cols = [col for col in sales.columns if col not in patterned_columns]
    sales_view = sales[independent_cols + [patterned_columns[0]]]
    sales_view = sales_view.rename(
        columns={patterned_columns[0]: f"PATT: {patterned_columns[0]} ... {patterned_columns[-1]}"})
    del independent_cols, patterned_columns
    print("Head:")
    print(sales_view.head())
    print("Info:")
    print(sales_view.info(show_counts=True))

    print("\n************* Missing data *************")
    print(f"Missing values: None, (see db.info() above)")

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(sales_view.nunique())
    print("Value counts of categorical column:")
    print(sales_view['id'].value_counts())
    print("Value counts of categorical column:")
    print(sales_view['item_id'].value_counts())
    print("Value counts of categorical column:")
    print(sales_view['store_id'].value_counts())
    print("Description of numerical columns:")
    print(sales_view.describe())

    # endregion

    # region Inspect dataset 'sell_prices'
    print("\n\n\n*******************************************")
    print("************* Inspect 'sell_prices' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(sell_prices.shape)
    print("Head:")
    print(sell_prices.head())
    print("Info:")
    print(sell_prices.info(show_counts=True))

    print("\n************* Missing data *************")
    print("Missing values: None, (see db.info() above)")

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(sell_prices.nunique())
    print("Value counts of categorical column:")
    print(sell_prices['item_id'].value_counts())
    print("Value counts of categorical column:")
    print(sell_prices['store_id'].value_counts())
    print("Description of numerical columns:")
    print(sell_prices.describe())

    # endregion

    # region Inspect dataset 'calendar'
    print("\n\n\n*******************************************")
    print("************* Inspect 'calendar' *************")
    print("*******************************************")
    print("\n************* Dataset structure *************")
    print("Shape:")
    print(calendar.shape)
    print("Head:")
    print(calendar.head())
    print("Info:")
    print(calendar.info(show_counts=True))

    print("\n************* Missing data *************")
    print(
        "The db.info() above reveals suspect missing values in cols ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']. Remember: the db has",
        calendar.shape[0], "entries in total.")

    print("Study the NaN entries with db.isna().sum():")
    print(calendar.isna().sum())
    print("Study the NaN entries with db.unique() and db.value_counts():")
    print("Unique values in 'event_type_1':")
    print(calendar['event_type_1'].unique())
    print("Value counts for 'event_type_1':")
    print(calendar['event_type_1'].value_counts())
    print("Unique values in 'event_type_2':")
    print(calendar['event_type_2'].unique())
    print("Value counts for 'event_type_2':")
    print(calendar['event_type_2'].value_counts())

    print("Are the detected NaN entries supposed to be NaN?")
    # Some NaN entries may convey useful information (Missing Not At Random, MNAR), as opposed to just being
    # missing datapoints. For example, they may convey the absence of a special condition, a default state, or else.
    # In this case, the NaN must be expressed with a usable value of the same type of the other entries in the column.
    print("I find that the '<NA>' entries in the calendar actually represent a default event class 'NoEvent',"
          "and therefore must NOT be expressed as a NaN.")
    calendar[[
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"]] = calendar[["event_name_1", "event_type_1", "event_name_2", "event_type_2"]].fillna("NoEvent")

    print("Are there entries that must become NaN?")
    # Online datasets may express NaN entries in non-standard ways such as "Not available" for strings or
    # -999 for numbers. In that case, pandas might miss those entries and explicit casting to pd.NA
    # (this is the current standard) is necessary. If so, df = df.replace({'Not available': pd.NA, -999: pd.NA}).

    print("Standardize all NaN entries.")
    # NaN values should be expressed in a standard way such as '<NA>' for strings or object columns
    # (although 'None' or 'NaN' are also used) or NaN for numeric columns. For maximum compatibility,
    # redefine all the NaN entries with pd.NA. If so, df = df.fillna(pd.NA).

    print("\n************* Data description *************")
    print("Number of unique values:")
    print(calendar.nunique())
    print("Value counts of categorical column:")
    print(calendar['weekday'].value_counts())
    print("Description of numerical columns:")
    print(calendar.describe())

    print("\n************* Sanity of temporal information *************")
    # Cast the date field into datetime format
    calendar['date'] = pd.to_datetime(calendar['date'])

    print("Check for duplicate date entries:")
    print(f"There are {'no ' if not calendar['date'].duplicated().any() else ''}duplicates in the 'date' column.")
    print(f"There are {'no ' if not calendar['d'].duplicated().any() else ''}duplicates in the 'd' column.\n")

    print("Ensure that fields 'date' and 'd' are monotonically increasing:")
    if not calendar['date'].is_monotonic_increasing:
        print("The 'date' column is not monotonically increasing. Sorting the DataFrame...")
        calendar.sort_values('date', inplace=True)
    else:
        print("The 'date' column is monotonically increasing.")
    assert calendar['d'].str.extract('(\d+)').astype(int).diff().dropna().gt(0).all().all(), \
        "The 'd' column is not monotonically increasing. This column does not logically correspond to the dates."

    print("\n************* Characteristics of temporal information *************")
    print(
        f"The start and end dates in the calendar are {calendar['date'].min()} and {calendar['date'].max()}, respectively.")

    print("Time gaps:")
    date_gaps = calendar['date'].diff().dt.days[1:]
    if date_gaps.nunique() != 1:
        print(f"The entries of calendar['date'] are equispaced. The gap is always {date_gaps.unique()[0]} day(s).")
    else:
        print(f"The entries of calendar['date'] are not equispaced. Here's some info about the gap distribution:")
        print(date_gaps.describe())

        """ One could also get a visual representation of the date information with these plots.

        # Plot data sample date on x axis
        plt.figure(figsize=(10, 6))
        plt.title('Date of each data sample')
        plt.plot(calendar['date'], np.zeros(len(calendar['date'])), marker='o', linestyle='none')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)

        # Histogram of date gaps
        plt.figure(figsize=(10, 6))
        date_gaps.plot(kind='hist', bins=30, figsize=(10, 6), title='Distribution of Date Gaps')
        plt.xlabel('Gap to next day (days)')
        plt.ylabel('Frequency')
        plt.show(block=False)

        # Cumulative distribution of date gaps
        cumulative_distribution = np.cumsum(date_gaps)
        plt.figure(figsize=(10, 6))
        plt.plot(calendar['date'][1:], cumulative_distribution, marker='.', linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Gap Days')
        plt.title('Cumulative Distribution of Date Gaps Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)

        """

    # endregion

    # region Join datasets

    print("\n\nPrepare to join sales, sell_prices, calendar datasets")

    print("Downcast data to save memory")
    sales = downcast(sales)
    calendar = downcast(calendar)
    sell_prices = downcast(sell_prices)

    """
    Joining together sell price and date information for multiple items at multiple time points involves transforming 
    all dataframes into a long format. In this format, each row corresponds to one item at one time point, organizing 
    the data to effectively represent temporal sequences for each item with both static and dynamic features:
        item1, staticFeature1, ..., sf10, timePoint1
        item1, staticFeature1, ..., sf10, timePoint2
        ...
        item1, staticFeature1, ..., sf10, timePoint1946
        item2, staticFeature1, ..., sf10, timePoint1
        ...
        item3000, staticFeature1, ..., sf10, timePoint1945
        item3000, staticFeature1, ..., sf10, timePoint1946

    For Sklearn models, feature engineering transforms the temporal data into a format where each row represents a 
    comprehensive set of features for one item, incorporating lagged and rolling window calculations to encapsulate 
    temporal dependencies within a single observation. This approach facilitates the use of regression or 
    classification algorithms that do not inherently process sequential data:
        item1, staticFeature1, ..., sf10, lagFeature1, ..., lf1946, rollingFeature1, rf2
        ...
        item3000, staticFeature1, ..., sf10, lagFeature1, ..., lf1946, rollingFeature1, rf2
    In this case, data rows encapsulate all necessary historical context, allowing for rows shuffling as each sample
    is treated independently.

    Partitioning the data into smaller sequences for Sklearn is an option for focusing on shorter temporal dependencies,
    leading to multiple rows per item, each encapsulating a portion of the item's temporal sequence:
        item1, staticFeature1, ..., sf10, lagFeature1, ..., lf900, rollingFeature1_1-900, rf2_1-900
        item1, staticFeature1, ..., sf10, lagFeature901, ..., lf1946, rollingFeature1_901-1946, rf2_901-1946
        ...
        item3000, staticFeature1, ..., sf10, lagFeature1, ..., lf900, rollingFeature1_1-900, rf2_1-900
        item3000, staticFeature1, ..., sf10, lagFeature901, ..., lf1946, rollingFeature1_901-1946, rf2_901-1946
    In this case, shuffling the rows requires caution as each row is part of a larger temporal segments. One must not 
    shuffle rows pertaining to the same item as they belong to a common temporal segment. Nonetheless, one can shuffle
    blocks of rows pertaining to different items. 

    For LSTMs and other sequence models, the data must maintain its sequential integrity, formatted into a 3D array 
    (n_items x n_timePoints x n_features) to allow these models to leverage temporal order and dependencies directly.
    In this case, it is possible to shuffle the rows, as each row pertains to a different item.
    """
    # Ensure all dataframes are in long format with each row representing a unique item-timePoint
    print("Melt sales dataframe")
    sales = pd.melt(sales,
                    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                    var_name="d",
                    value_name="sold",
                    ignore_index=True,
                    )

    # Integrate the calendar dates into (notice "left") the sales dataframe based on d
    print("Merge sales and calendar dataframes")
    sales = pd.merge(sales, calendar, on="d", how="left")

    # Integrate the sell prices into the sales dataframe based on the common fields
    print("Merge sales and sell_prices dataframes")
    sales = pd.merge(sales, sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # endregion

    # region Data processing

    print("\n\nPreprocess dataframes")

    print("Simplify sales['id']")
    if sales['id'].str.endswith('_evaluation').all():
        sales['id'] = sales['id'].str.replace('_evaluation', '')

    print("Convert the day into a number")
    sales["d"] = sales["d"].apply(lambda s: s.split("_")[1]).astype(np.int16)

    print("Align weekday and wday columns (monday=1, tuesday=2, ...)")
    weekday_to_num = {'Monday': 1,
                      'Tuesday': 2,
                      'Wednesday': 3,
                      'Thursday': 4,
                      'Friday': 5,
                      'Saturday': 6,
                      'Sunday': 7}
    sales['wday'] = sales['weekday'].map(weekday_to_num)

    print("Cast specific columns to categorical type")
    columns_to_convert = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1',
                          'event_name_2', 'event_type_2']  # Note that binary columns must be left numerical
    sales[columns_to_convert] = sales[columns_to_convert].astype('category')

    # Cast data to most adequate types in order to save memory
    print("Downcast data to save memory")
    sales = downcast(sales)

    # Check residual NaN values and handle them through elimination or samples or imputation
    print("Check if there are any remaining columns with NaN: ")
    sales.isna().any()
    print(f"Impute NaN values for the column sell_price as the median sell_price for the item")
    # Step 1: Attempt group-wise Median Imputation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sales['sell_price'] = sales.groupby("id")['sell_price'].transform(lambda x: x.fillna(x.median()))
    # Step 2: Global Median Imputation for remaining NaNs (for whom the sell_price might be NaN for the entire group)
    sales['sell_price'] = sales['sell_price'].fillna(sales['sell_price'].median())

    # Sort the dataframe so that the temporal evolution of the target variable is immediately evident for each item
    sales = sales.sort_values(by=["id", "d"]).reset_index(drop=True)  # Sort by the item and then by the day

    # endregion

    return sales


def preprocess_data_passthrough(*args, **kwargs):
    return (args, kwargs) if kwargs else args
