import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


tfidf = joblib.load("models/tfidf_vectorizer.pkl")
products_df = joblib.load("models/products_dataframe.pkl")
svd_model = joblib.load("models/svd_collaborative_model.pkl")
merged_df = pd.read_csv("data/processed/merged_sample.csv")
dissatisfaction_df = pd.read_csv("data/processed/product_dissatisfaction_scores.csv")
ratings_df = merged_df[["reviewerID", "asin", "overall"]].dropna().copy()


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in ["title", "product_text", "brand"]:
    if col in products_df.columns:
        products_df[col] = products_df[col].fillna("").astype(str)

if "combined_text_clean" not in products_df.columns:
    products_df["combined_text"] = (
        products_df["title"] + " " +
        products_df["brand"] + " " +
        products_df["product_text"]
    )
    products_df["combined_text_clean"] = products_df["combined_text"].apply(clean_text)

if "simple_category" not in products_df.columns:
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

# add dissatisfaction scores to products_df
products_df = products_df.merge(
    dissatisfaction_df[["asin", "dissatisfaction_norm"]],
    on="asin",
    how="left"
)
products_df["dissatisfaction_norm"] = products_df["dissatisfaction_norm"].fillna(0)

#score normalization function
def minmax(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    if series.max() == series.min():
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def hybrid_recommend(
    user_id: str,
    user_query: str,
    target_category: str = None,
    top_n: int = 5,
    alpha = 0.8,
    beta = 0.2,
    gamma: float = 0.2,
    min_content_threshold = 0.0,
):
    candidate_products = products_df.copy()

    if target_category is not None and target_category != "all":
        candidate_products = candidate_products[
            candidate_products["simple_category"] == target_category
        ].copy()

    if candidate_products.empty:
        return pd.DataFrame()

    query_vector = tfidf.transform([clean_text(user_query)])
    candidate_vectors = tfidf.transform(candidate_products["combined_text_clean"])

    candidate_products["content_score"] = cosine_similarity(
        query_vector,
        candidate_vectors
    ).flatten()

    candidate_products = candidate_products[
        candidate_products["content_score"] >= min_content_threshold
    ].copy()

    if candidate_products.empty:
        return pd.DataFrame()

    collab_scores = []
    for asin in candidate_products["asin"]:
        pred = svd_model.predict(user_id, asin)
        collab_scores.append(pred.est)

    candidate_products["collab_score"] = collab_scores

    candidate_products["content_score_norm"] = minmax(candidate_products["content_score"])
    candidate_products["collab_score_norm"] = minmax(candidate_products["collab_score"])

    candidate_products["hybrid_score"] = (
        alpha * candidate_products["content_score_norm"] +
        beta * candidate_products["collab_score_norm"]
    )

    candidate_products["final_score"] = (
        candidate_products["hybrid_score"] -
        gamma * candidate_products["dissatisfaction_norm"]
    )

    result = candidate_products.sort_values(
        by="final_score",
        ascending=False
    ).head(top_n)

    cols = [
        "asin", "title", "brand", "simple_category",
        "content_score", "collab_score",
        "content_score_norm", "collab_score_norm",
        "hybrid_score", "dissatisfaction_norm", "final_score"
    ]

    if "imUrl" in result.columns:
        cols.append("imUrl")

    return result[cols]

# just for checkup 
print("\n--- CATEGORY COUNTS IN products_df ---")
print(products_df["simple_category"].value_counts())


# testing  
test_user = ratings_df.iloc[0]["reviewerID"]

print("\n--- TEST: shoes ---")
print(hybrid_recommend(
    user_id=test_user,
    user_query="black casual sneakers",
    target_category="shoes",
    top_n=5
))

print("\n--- TEST: watch ---")
print(hybrid_recommend(
    user_id=test_user,
    user_query="black watch",
    target_category="watch",
    top_n=5
))

print("\n--- TEST: bag ---")
print(hybrid_recommend(
    user_id=test_user,
    user_query="leather handbag",
    target_category="bag",
    top_n=5
))