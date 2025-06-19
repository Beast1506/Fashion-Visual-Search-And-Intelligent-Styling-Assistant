import pandas as pd
import ast
import json
import os
import logging

def safe_parse(val, default):
    if pd.isnull(val):
        return default
    try:
        return json.loads(val.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return default

def load_and_clean_csv(filepath):
    if not os.path.exists(filepath):
        logging.warning(f"CSV file not found: {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    dict_cols = ['selling_price', 'mrp', 'style_attributes']
    list_cols = ['feature_list', 'pdp_images_s3']
    for col in dict_cols:
        df[col] = df[col].apply(lambda x: safe_parse(x, {}))
    for col in list_cols:
        df[col] = df[col].apply(lambda x: safe_parse(x, []))
    return df

dresses = load_and_clean_csv('dresses_bd_processed_data.csv')
jeans = load_and_clean_csv('jeans_bd_processed_data.csv')
