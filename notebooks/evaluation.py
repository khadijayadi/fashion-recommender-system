import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split


MODELS_DIR = "models"
DATA_DIR = "data/processed"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load saved models and data
tfidf = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load(f"{MODELS_DIR}/tfidf_matrix.pkl")
products_df = joblib.load(f"{MODELS_DIR}/products_dataframe.pkl")
svd_model = joblib.load(f"{MODELS_DIR}/svd_collaborative_model.pkl")
merged_df = pd.read_csv(f"{DATA_DIR}/merged_sample.csv")
ratings_df = merged_df[["reviewerID", "asin", "overall"]].dropna().copy()

# Prepare test set for CF evaluation
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["reviewerID", "asin", "overall"]], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Evaluation metrics 
predictions = svd_model.test(testset)

print("\n--- COLLABORATIVE FILTERING EVALUATION ---")
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

#Rating distribution plot
plt.figure(figsize=(6, 4))
ratings_df["overall"].value_counts().sort_index().plot(kind="bar")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/rating_distribution.png")
plt.close()

# Content similarity distribution plot
query = "black casual sneakers"
query_vec = tfidf.transform([query])

scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

plt.figure(figsize=(6, 4))
plt.hist(scores, bins=30)
plt.title("Content Similarity Distribution")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/content_score_distribution.png")
plt.close()

# the Hybrid model score calculation
def minmax(series: pd.Series) -> np.ndarray:
    if series.max() == series.min():
        return np.zeros(len(series))
    return (series - series.min()) / (series.max() - series.min())

def hybrid_scores(user_id: str, query_text: str, alpha: float = 0.8, beta: float = 0.2) -> pd.DataFrame:
    df = products_df.copy()

    query_vector = tfidf.transform([query_text])

    if "combined_text_clean" in df.columns:
        product_vectors = tfidf.transform(df["combined_text_clean"])
    else:
        
        fallback_text = (
            df["title"].fillna("").astype(str) + " " +
            df["brand"].fillna("").astype(str) + " " +
            df["product_text"].fillna("").astype(str)
        )
        product_vectors = tfidf.transform(fallback_text)

    df["content_score"] = cosine_similarity(query_vector, product_vectors).flatten()

    collab_scores = []
    for asin in df["asin"]:
        pred = svd_model.predict(user_id, asin)
        collab_scores.append(pred.est)

    df["collab_score"] = collab_scores
    df["content_norm"] = minmax(df["content_score"])
    df["collab_norm"] = minmax(df["collab_score"])
    df["hybrid_score"] = alpha * df["content_norm"] + beta * df["collab_norm"]

    return df.sort_values(by="hybrid_score", ascending=False).head(20)

# hybrid score different component plot for a sample user and query
user_id = ratings_df.iloc[0]["reviewerID"]
recs = hybrid_scores(user_id, "black casual sneakers")

ax = recs[["content_score", "collab_score", "hybrid_score"]].plot(
    kind="bar",
    figsize=(10, 5)
)
ax.set_title("Hybrid Score Components")
ax.set_xlabel("Recommended Items")
ax.set_ylabel("Score")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/hybrid_scores.png")
plt.close()

# category distribution plot
plt.figure(figsize=(6, 4))
products_df["simple_category"].value_counts().plot(kind="bar")
plt.title("Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/category_distribution.png")
plt.close()


recs.to_csv(f"{RESULTS_DIR}/sample_recommendations.csv", index=False)

# Ranking metrics :

# 1. Precision@K, Recall@K
def precision_recall_at_k(predictions_list, k: int = 5, threshold: float = 4.0):
    
    user_est_true = defaultdict(list)

    for pred in predictions_list:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    precisions = []
    recalls = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant = sum(true_r >= threshold for (_, true_r) in user_ratings)
        n_relevant_k = sum(true_r >= threshold for (_, true_r) in top_k)

        precision = n_relevant_k / k if k > 0 else 0
        recall = n_relevant_k / n_relevant if n_relevant > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return float(np.mean(precisions)), float(np.mean(recalls))
#2. NDCG@K
def ndcg_at_k(predictions_list, k: int = 5):
    """
    Computes mean NDCG@K using true ratings as graded relevance.
    """
    user_est_true = defaultdict(list)

    for pred in predictions_list:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    ndcgs = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        dcg = 0.0
        for i, (_, true_r) in enumerate(top_k):
            dcg += (2**true_r - 1) / math.log2(i + 2)

        ideal = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:k]
        idcg = 0.0
        for i, (_, true_r) in enumerate(ideal):
            idcg += (2**true_r - 1) / math.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    return float(np.mean(ndcgs))

#3. CTR@K approximation since there are no real click logs, we use held-out ratings as a proxy for clicks.
def ctr_at_k(predictions_list, k: int = 5, threshold: float = 4.0):
    """
    Approximate CTR@K:
    item is treated as a 'click' if the held-out true rating >= threshold.
    This is a proxy because real click logs are unavailable.
    """
    user_est_true = defaultdict(list)

    for pred in predictions_list:
        user_est_true[pred.uid].append((pred.est, pred.r_ui))

    ctrs = []

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        clicks = sum(true_r >= threshold for (_, true_r) in top_k)
        ctrs.append(clicks / k)

    return float(np.mean(ctrs))

precision, recall = precision_recall_at_k(predictions, k=5, threshold=4.0)
ndcg = ndcg_at_k(predictions, k=5)
ctr = ctr_at_k(predictions, k=5, threshold=4.0)

print("\n--- RANKING METRICS ---")
print(f"Precision@5: {precision:.4f}")
print(f"Recall@5:    {recall:.4f}")
print(f"NDCG@5:      {ndcg:.4f}")
print(f"CTR@5*:      {ctr:.4f}")


metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "Precision@5", "Recall@5", "NDCG@5", "CTR@5*"],
    "Value": [rmse, mae, precision, recall, ndcg, ctr]
})
metrics_df.to_csv(f"{RESULTS_DIR}/evaluation_metrics.csv", index=False)

