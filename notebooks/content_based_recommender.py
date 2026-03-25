import os
import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


merged_df = pd.read_csv("data/processed/merged_sample.csv")

products_df = merged_df[
    ["asin", "title", "product_text", "brand", "categories", "price", "related", "imUrl"]
].drop_duplicates(subset="asin").reset_index(drop=True)

# Data cleaning 
for col in ["title", "product_text", "brand", "categories"]:
    if col in products_df.columns:
        products_df[col] = products_df[col].fillna("").astype(str)

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


products_df["category_text"] = products_df["title"]
products_df["category_text_clean"] = products_df["category_text"].apply(clean_text)

# prepare fo TF-IDF
products_df["combined_text"] = (
    products_df["title"] + " " +
    products_df["brand"] + " " +
    products_df["product_text"]
)
products_df["combined_text_clean"] = products_df["combined_text"].apply(clean_text)


def contains_any_token(text: str, keywords: list[str]) -> bool:
    tokens = set(clean_text(text).split())
    for kw in keywords:
        kw_tokens = clean_text(kw).split()
        if all(token in tokens for token in kw_tokens):
            return True
    return False


def assign_simple_category(text: str) -> str:
    text = clean_text(text)

    if contains_any_token(text, ["watch", "wristwatch", "chronograph"]):
        return "watch"

    if contains_any_token(text, ["bag", "backpack", "purse", "handbag", "tote", "satchel", "clutch", "crossbody"]):
        return "bag"

    if contains_any_token(text, [
        "sneaker", "shoe", "shoes", "boot", "boots", "sandal", "sandals",
        "slipper", "slippers", "loafer", "loafers", "heel", "heels", "pump", "pumps"
    ]):
        return "shoes"

    if contains_any_token(text, [
        "shirt", "t shirt", "t-shirt", "tee", "top", "blouse",
        "sweater", "hoodie", "jacket", "cardigan", "tank"
    ]):
        return "top"

    if contains_any_token(text, [
        "jean", "jeans", "pants", "trouser", "trousers",
        "shorts", "skirt", "leggings", "capri", "tutu"
    ]):
        return "bottom"

    return "other"

products_df["simple_category"] = products_df["title"].fillna("").astype(str).apply(assign_simple_category)

print("\n--- CATEGORY COUNTS ---")
print(products_df["simple_category"].value_counts())

print("\n--- SAMPLE PRODUCTS PER CATEGORY ---")
for cat in ["shoes", "watch", "bag", "top", "bottom", "other"]:
    print(f"\nCategory: {cat}")
    print(products_df[products_df["simple_category"] == cat][["title"]].head(5))


# TF-IDF Matrix
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2
)

tfidf_matrix = tfidf.fit_transform(products_df["combined_text_clean"])

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)

def recommend_from_query_with_category(
    user_query: str,
    target_category: str,
    top_n: int = 5,
    min_score: float = 0.0
):
    filtered_products = products_df[products_df["simple_category"] == target_category].copy()

    if filtered_products.empty:
        return pd.DataFrame()

    filtered_vectors = tfidf.transform(filtered_products["combined_text_clean"])
    query_vector = tfidf.transform([clean_text(user_query)])
    sim_scores = cosine_similarity(query_vector, filtered_vectors).flatten()

    filtered_products["content_score"] = sim_scores
    filtered_products = filtered_products[filtered_products["content_score"] >= min_score]
    filtered_products = filtered_products.sort_values("content_score", ascending=False).head(top_n)

    return filtered_products[["asin", "title", "brand", "simple_category", "content_score"]]

# test queries  for different categories 
print("\n--- QUERY TEST: shoes ---")
print(recommend_from_query_with_category("black casual sneakers", "shoes", top_n=5))

print("\n--- QUERY TEST: watch ---")
print(recommend_from_query_with_category("black watch", "watch", top_n=5))

print("\n--- QUERY TEST: bag ---")
print(recommend_from_query_with_category("leather handbag", "bag", top_n=5))




os.makedirs("models", exist_ok=True)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
joblib.dump(tfidf_matrix, "models/tfidf_matrix.pkl")
joblib.dump(products_df, "models/products_dataframe.pkl")
