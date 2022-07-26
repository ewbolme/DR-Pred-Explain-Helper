import pandas as pd

def get_from_csv(input_filepath: str) -> pd.DataFrame:
    return pd.read_csv(input_filepath)


# def get_from_db() -> pd.DataFrame:
#    pass