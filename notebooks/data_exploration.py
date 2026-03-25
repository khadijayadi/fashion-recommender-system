import pandas as pd
import ast
from tqdm import tqdm
import os


metadata_path = "data/raw/meta_Clothing_Shoes_and_Jewelry.json"
reviews_path = "data/raw/Clothing_Shoes_and_Jewelry_5.json"

# inspect the raw metadata dataset
metadata_sample = []

with open(metadata_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f)):
        line = line[line.find("{"):]
        metadata_sample.append(ast.literal_eval(line))
        if i == 5000:
            break

metadata_df = pd.DataFrame(metadata_sample)

print(metadata_df.head())
print(metadata_df.columns)
print(metadata_df.shape)

#cleaning the dataset 

metadata_df = metadata_df[['asin', 'title', 'description', 'brand', 'categories', 'price', 'related', 'imUrl']].copy()

metadata_df['description'] = metadata_df['description'].fillna('')
metadata_df['description'] = metadata_df['description'].apply(
    lambda x: ' '.join(x) if isinstance(x, list) else str(x)
)

metadata_df['product_text'] = metadata_df['title'] + ' ' + metadata_df['description']
metadata_df['product_text'] = metadata_df['product_text'].str.strip()
metadata_df = metadata_df[metadata_df['product_text'] != '']
metadata_df = metadata_df.reset_index(drop=True)

# check the result
print("\n--- CLEANED METADATA ---")
print(metadata_df[['asin', 'title', 'description', 'product_text']].head())
print("Cleaned metadata shape:", metadata_df.shape)


metadata_df.to_csv("data/processed/cleaned_metadata_sample.csv", index=False)

# now we inspect the raw reviews dataset 
reviews_sample = []

with open(reviews_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f)):
        if "{" in line:
            line = line[line.find("{"):]
        reviews_sample.append(ast.literal_eval(line))
        if i == 5000:
            break

reviews_df = pd.DataFrame(reviews_sample)

print("\n--- RAW REVIEWS ---")
print(reviews_df.head())
print(reviews_df.columns)
print(reviews_df.shape)

 # cleaning the dataset 

reviews_df = reviews_df[['reviewerID', 'asin', 'overall', 'reviewText', 'summary', 'reviewTime']].copy()

reviews_df['reviewText'] = reviews_df['reviewText'].fillna('')
reviews_df['summary'] = reviews_df['summary'].fillna('')

reviews_df = reviews_df.dropna(subset=['reviewerID','asin','overall']).reset_index(drop=True)

print("\n--- CLEANED REVIEWS ---")
print(reviews_df.head())


reviews_df.to_csv("data/processed/cleaned_reviews_sample.csv", index=False)

# merge metadata and reviews in a single dataset called merged_df
merged_df = pd.merge(
    reviews_df,
    metadata_df,
    on="asin",
    how="inner"
)

print("\n--- MERGED DATASET ---")
print(merged_df.head())

print("\nMerged dataset shape:", merged_df.shape)

print("Unique users:", merged_df['reviewerID'].nunique())
print("Unique products:", merged_df['asin'].nunique())

print("\nMissing values check:")
print(merged_df[['asin','reviewerID','overall','product_text']].isna().sum())

merged_df.to_csv("data/processed/merged_sample.csv", index=False)

