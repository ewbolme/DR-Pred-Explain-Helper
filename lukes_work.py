import pandas as pd
from datetime import datetime
from pathlib import Path
import re
from functools import lru_cache
import datarobot as dr
from tempfile import NamedTemporaryFile
import sys

def process_explanation_file(file_path, data_as_of_date, out_path):
    base_data = pd.read_csv(file_path, converters={'Date': pd.to_datetime})
    explain_columns = [col for col in base_data.columns if col.startswith("EXPLANATION")]
    ranger = 1
    grouper = base_data[base_data.Date == data_as_of_date].groupby('Store_ID')
    # name, df = grouper.__iter__().__next__()
    # process_row(name, df)
    frames = [process_row(name, row) for  name, row in grouper.__iter__()]
    all_data = pd.concat(frames, axis=0)
    all_data.to_csv(out_path, index=False)

def create_target_current_file( file_in, target, data_as_of_date, out_path, training_dataset):
    file_to_read = file_in
    base_data = pd.read_csv(file_to_read, parse_dates=True, infer_datetime_format=True)
    if not isinstance(data_as_of_date, datetime):
        dt = datetime.fromisoformat(data_as_of_date)
    else:
        dt = data_as_of_date

    dataset = dr.Dataset.get(training_dataset)
    f = NamedTemporaryFile('w')
    dataset.get_file(f.name)
    out_prediction_file = add_actuals(base_data, f.name, target)
    filter_to_forecast = base_data[base_data.Date == base_data.Date.min()]
    target_column_name = f"{target} (actual)_PREDICTION"
    filter_to_forecast = filter_to_forecast[['Store_ID', target_column_name]].rename(columns={
       target_column_name: fix_column_name(target)
    })
    filter_to_forecast['score_type'] = 'Current Score'
    
    filter_to_forecast['cluster'] = pd.qcut(filter_to_forecast[fix_column_name(target)], q=5, duplicates='drop', labels=False).apply(fix_bin_names)
    # 
    delta_data = prepare_delta_files(dt, f.name, target).to_frame()
    delta_data.rename(columns={target: fix_column_name(target)}, inplace=True)
    delta_data['score_type'] = 'WoW Delta'
    delta_data['cluster'] = pd.qcut(delta_data[fix_column_name(target)], q=5, duplicates='drop', labels=False).apply(fix_bin_names)
    pd.concat([filter_to_forecast.set_index('Store_ID'), delta_data]).to_csv(out_path, index=True)
    out_prediction_file.to_csv(file_in, index=False)
    f.close()

def add_actuals(prediction_frame: pd.DataFrame, training_file_name: NamedTemporaryFile, target):
    training_data = pd.read_csv(training_file_name, parse_dates=True, infer_datetime_format=True)[['Store_ID','Date','ROW_ID', target]].dropna()
    prediction_frame['Actual Value'] = pd.NA
    training_data.rename(columns={target: f"Actual Value"}, inplace=True)
    return pd.concat([prediction_frame, training_data], axis=0).sort_values(['Store_ID', 'Date'])


def fix_column_name(col: str):
    new_col = col.replace(" ", "_")
    new_col = new_col.lower()
    return f"{new_col}_value"

def fix_bin_names(namer): 
    return namer + 1

def process_row(name, row, predict_type="Current"):
    output_frame = {
        "Feature Name": [],
        "Feature Importance - Now Casting - Strength": [],
        "Feature Importance - Now Casting - Current Value": [],
        "Feature Type": [],
        "Store_ID": [],
        "Actionable": [],
        "Delta_or_Current": []

    }
    for r in range(50):
        col = r + 1
        feature_name, f_type, actionable = feature_extract(row[f"EXPLANATION_{str(col)}_FEATURE_NAME"].values[0])
        output_frame['Feature Name'].append(feature_name)
        output_frame["Feature Importance - Now Casting - Strength"].append(row[f"EXPLANATION_{str(col)}_STRENGTH"].values[0])
        output_frame["Feature Importance - Now Casting - Current Value"].append(row[f"EXPLANATION_{str(col)}_ACTUAL_VALUE"].values[0])
        output_frame["Store_ID"].append(name)
        output_frame["Actionable"].append(actionable)
        output_frame["Feature Type"].append(f_type)
        output_frame["Delta_or_Current"].append(predict_type)
    return pd.DataFrame(output_frame)

@lru_cache
def feature_extract(in_col):
    if "[" in in_col and "(" in  in_col:
        feature = re.findall('''\[(.*?)\]''', in_col)[0]
        f_type = re.findall('''\((.*?)\)''', in_col)[0]
        actionable = False
    else:
        feature = in_col
        f_type = 'Actual'
        actionable = True
    return (feature, f_type, actionable)


def prepare_delta_files(last_date_of_data, training_dataset_file_name, target):
    current_week = last_date_of_data

    training_data = pd.read_csv(training_dataset_file_name, parse_dates=True, infer_datetime_format=True)
    training_data = training_data.replace(-1, pd.NA).replace("-1", pd.NA)
    r = training_data.sort_values(['Store_ID', 'Date'], ascending=True).groupby('Store_ID')[target].agg(lambda s: s.dropna().diff().tail(1))
    del training_data
    return r


if __name__ == '__main__':
    process_explanation_file(sys.argv[1], sys.argv[2], sys.argv[3])



##########################################################

#%%
from numpy import result_type
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

result_data = pd.read_csv("result-629e16618a6af76b7e19b50e(1).csv").assign(row_id=lambda df: df.index)
explanation_columns = [ col for col in result_data.columns if col.startswith("EXPLANATION_")]
other_columns = [col for col in result_data.columns if col not in explanation_columns]
input_data = pd.read_csv("/Users/luke.shulman/Downloads/DR Demo Insurance Exposure Scoring_629a6bf7e1c91164fe23d74f.csv")
#%%
# engine = create_engine("mssql+pyodbc://brent:P@ssword123@MYMSSQL")
# engine = engine.connect()
# # engine.execute("select * from demo;")

# %%
def process_row(row):
    output_frame = {
        "Feature Name": [],
        "Feature Importance-Strength": [],
        "Feature Importance-Current Value": [],
        "ROW_ID": [],
    }
    row = row._asdict()
    for r in range(4):
        col = r + 1

        output_frame["Feature Name"].append(
            row[f"EXPLANATION_{col}_FEATURE_NAME"]
        )
        output_frame["Feature Importance-Strength"].append(
            row[f"EXPLANATION_{col}_STRENGTH"]
        )
        output_frame["Feature Importance-Current Value"].append(
            row[f"EXPLANATION_{col}_ACTUAL_VALUE"]
        )
        output_frame["ROW_ID"].append(row["row_id"])
    return pd.DataFrame(output_frame)


frames = [process_row(row) for row in result_data.itertuples()]
pe_data = pd.concat(frames, axis=0)
# %%
import re
from shapely import wkt
import numpy as np


def latlng(row, latitude=False):
    w = row.wkt
    if isinstance(w, str):
        point = wkt.loads(row.wkt)
        if latitude:
            return point.y
        else:
            return point.x
    else:
        return None




out_data = result_data[other_columns]

out_data.to_csv("results/prediction_data.csv", index=False)
# out_data.to_sql("incurred_cars_prediction_data", con=engine, if_exists='replace')
outp = Path("sql_schema.sql")

with open(outp, 'w') as p:
    p.write(pd.io.sql.get_schema(out_data.reset_index(), name="prediction_data"))
    p.write('''
    __________________
    ''')

    p.write(pd.io.sql.get_schema(pe_data.reset_index(), name="explanation_data"))
    p.write('''
    __________________
    ''')
    p.write(pd.io.sql.get_schema(input_data.reset_index(), name="input_data_claims"))

print()

pe_data.to_csv("results/explanation_data.csv", index=False)
# pe_data.to_sql("incurred_cars_explanation_data", con=engine, if_exists='replace')

# %%


