import os
import requests
import pandas as pd
import ast
import json
from tqdm import tqdm

def download_images(df, image_col, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row[image_col]
        product_id = row['product_id']
        if not isinstance(url, str) or not url.startswith('http'):
            print(f"Skipping invalid URL for product {product_id}")
            continue
        ext = url.split('.')[-1]
        if len(ext) > 5 or '/' in ext:
            ext = 'jpg'  # Default to jpg if extension is weird
        filename = f"{product_id}.{ext}"
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def safe_parse(val, default):
    if pd.isnull(val):
        return default
    try:
        # Try JSON first
        return json.loads(val.replace("'", '"'))
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return default

def load_and_clean_csv(filepath):
    df = pd.read_csv(filepath)
    dict_cols = ['selling_price', 'mrp', 'style_attributes']
    list_cols = ['feature_list', 'pdp_images_s3']
    for col in dict_cols:
        df[col] = df[col].apply(lambda x: safe_parse(x, {}))
    for col in list_cols:
        df[col] = df[col].apply(lambda x: safe_parse(x, []))
    return df

if __name__ == "__main__":
    dresses = load_and_clean_csv('dresses_bd_processed_data.csv')
    jeans = load_and_clean_csv('jeans_bd_processed_data.csv')
    print("Downloading dress images...")
    download_images(dresses, 'feature_image_s3', 'images/dresses')
    print("Downloading jeans images...")
    download_images(jeans, 'feature_image_s3', 'images/jeans')
    print("Done.")