import pandas as pd
import numpy as np

# Load the cleaned reviews dataset
reviews_df = pd.read_csv("data/processed/cleaned_reviews_sample.csv")

print(reviews_df.shape)
print(reviews_df[['asin', 'reviewText', 'overall']].head())

negative_words = [
    "bad",
    "poor",
    "cheap",
    "uncomfortable",
    "broken",
    "defective",
    "small",
    "tight",
    "loose",
    "returned",
    "disappointed",
    "terrible",
    "waste",
    "rip",
    "tear",
    "itchy",
    "weak",
    "hard",
    "problem",
    "wrong"
]

def count_negative_words(text):
    text = str(text).lower()
    count = 0

    for word in negative_words:
        if word in text:
            count += 1

    return count

reviews_df['negative_score'] = reviews_df['reviewText'].apply(count_negative_words)

print(reviews_df[['asin', 'reviewText', 'negative_score']].head(10))

def weighted_dissatisfaction(row):
    base_score = row['negative_score']
    rating = row['overall']

    if rating <= 2:
        return base_score + 2
    elif rating == 3:
        return base_score + 1
    else:
        return base_score
    
reviews_df['weighted_negative_score'] = reviews_df.apply(weighted_dissatisfaction, axis=1)

print(reviews_df[['asin', 'overall', 'negative_score', 'weighted_negative_score']].head(10))

# create a penalty score based on the dissatisfaction score and the number of reviews for each product
#
product_dissatisfaction = reviews_df.groupby('asin').agg(
    dissatisfaction_score=('weighted_negative_score', 'mean'),
    review_count=('reviewText', 'count')
).reset_index()

print(product_dissatisfaction.head())
print(product_dissatisfaction.shape)

max_score = product_dissatisfaction['dissatisfaction_score'].max()

if max_score != 0:
    product_dissatisfaction['dissatisfaction_norm'] = (
        product_dissatisfaction['dissatisfaction_score'] / max_score
    )
else:
    product_dissatisfaction['dissatisfaction_norm'] = 0
    
#inspect the worst products for better Analysis 
worst_products = product_dissatisfaction.sort_values(
    by='dissatisfaction_score',
    ascending=False
)

print("\n--- MOST DISSATISFYING PRODUCTS ---")
print(worst_products.head(10))


product_dissatisfaction.to_csv(
    "data/processed/product_dissatisfaction_scores.csv",
    index=False
)

