import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple


def id_explan_columns(data: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """A function which returns a list of explanation columns along with which ones of them are populated with non null values

    Args:
        data: a pandas dataset output from a prediction explanations job

    Returns:
        explanation_columns: A list of explanation column names
        populated_explanation_col_numbers: A list of numbers associated with prediciton explanation columns with non null values"""

    number_of_rows = data.shape[0]
    pattern_base = re.compile("EXPLANATION_(.*?)")
    explanation_columns = [col for col in data.columns if pattern_base.match(col)]
    pattern = re.compile("EXPLANATION_(.*?)_FEATURE_NAME")

    populated_explanation_col_numbers = [
        pattern.match(col)[1]
        for col in explanation_columns
        if pattern.match(col) and data[col].isna().sum() != number_of_rows
    ]
    return explanation_columns, populated_explanation_col_numbers


def return_explanations_flat(data: pd.DataFrame) -> pd.DataFrame:
    """A function returns a dataframe containing the two columns for each (important) feature.
    Specifically the features value and strength of its contribution to the overall prediction.
    It does this by iterating through each row of the dataset - pulling out the prediction explanations and their strength
    then either recording the feature strength in the appropriate row or creating a new row if that feature has not been seen before

    Args:
        data: a pandas dataset output from a prediction explanations job

    Returns:
        data: a pandas dataframe without the dditional explanation columns from the
              prediction explanations job and columns added for each feature indicating the strength of its contribution"""
    explanation_columns, populated_explanation_col_numbers = id_explan_columns(data)

    for i, row in tqdm(data.iterrows()):
        for col_num in populated_explanation_col_numbers:
            feature_name = row[f"EXPLANATION_{str(col_num)}_FEATURE_NAME"]
            # TODO does this work on a copy or a window
            if f"{feature_name}_EXPLANATION_STRENGTH" in data.columns:
                data.at[i, f"{feature_name}_EXPLANATION_STRENGTH"] = row[
                    f"EXPLANATION_{str(col_num)}_STRENGTH"
                ]
            elif row[f"EXPLANATION_{str(col_num)}_FEATURE_NAME"] == np.nan:
                pass
            else:
                data[f"{feature_name}_EXPLANATION_STRENGTH"] = np.nan
                data.at[i, f"{feature_name}_EXPLANATION_STRENGTH"] = row[
                    f"EXPLANATION_{str(col_num)}_STRENGTH"
                ]

    return data.drop(explanation_columns, axis=1)


def return_melted_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """A function returns a dataframe containing three columns a strength column and a feature name column and an origional row number column
    (to connect it back to the initial prediciton requests if needed) suitable for a variety of business intellegence tools

    Args:
        data: a pandas dataset output from a prediction explanations job

    Returns:
        data: a dataframe with three columns suitable for use in business intellegence tools"""
    explanation_columns, populated_explanation_col_numbers = id_explan_columns(data)

    explan_feature_name_cols = []
    explan_feature_str_cols = []

    for col_num in populated_explanation_col_numbers:
        explan_feature_name_cols.append(f"EXPLANATION_{str(col_num)}_FEATURE_NAME")
        explan_feature_str_cols.append(f"EXPLANATION_{str(col_num)}_STRENGTH")

    data["orig_row_num"] = data.index
    pattern = re.compile("[0-9]+")
    def_trim_function = lambda x: pattern.findall(x)[0]

    data_melted_feat_name = data.melt(
        id_vars=["orig_row_num"],
        value_vars=explan_feature_name_cols,
        value_name="feature_name",
        var_name="variable_number",
    )

    data_melted_feat_str = data.melt(
        id_vars=["orig_row_num"],
        value_vars=explan_feature_str_cols,
        value_name="feature_strength",
        var_name="variable_number",
    )

    data_melted_feat_name["variable_number"] = data_melted_feat_name[
        "variable_number"
    ].map(def_trim_function)
    data_melted_feat_str["variable_number"] = data_melted_feat_str[
        "variable_number"
    ].map(def_trim_function)

    return data_melted_feat_name.merge(
        data_melted_feat_str, on=["orig_row_num", "variable_number"]
    ).drop(["variable_number"], inplace=True, axis=1)
