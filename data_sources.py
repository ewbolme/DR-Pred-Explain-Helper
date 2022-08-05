import pandas as pd


def get_from_csv(input_filepath: str) -> pd.DataFrame:
    """
    reads in data from a csv and creates a pandas dataframe

    args:
        input_filepath (str): a string containing the file to read into the pipeline
    """
    return pd.read_csv(input_filepath)
