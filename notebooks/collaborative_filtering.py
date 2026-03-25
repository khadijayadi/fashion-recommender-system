import os
import joblib
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# Data loading
merged_df = pd.read_csv("data/processed/merged_sample.csv")

ratings_df = merged_df[["reviewerID", "asin", "overall"]].dropna().copy()
ratings_df["overall"] = ratings_df["overall"].astype(float)

print("Ratings shape:", ratings_df.shape)
print(ratings_df.head())

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[["reviewerID", "asin", "overall"]], reader)

# split the data into train/test sets with an 80/20 ratio
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# train the SVD model
svd_model = SVD(random_state=42)
svd_model.fit(trainset)




def recommend_for_user(user_id: str, top_n: int = 5):
    all_items = ratings_df["asin"].unique()
    rated_items = ratings_df[ratings_df["reviewerID"] == user_id]["asin"].unique()
    unrated_items = [item for item in all_items if item not in rated_items]

    predictions_list = []
    for item_id in unrated_items:
        pred = svd_model.predict(user_id, item_id)
        predictions_list.append((item_id, pred.est))

    predictions_list = sorted(predictions_list, key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = pd.DataFrame(predictions_list, columns=["asin", "predicted_rating"])

    product_info = merged_df[["asin", "title", "brand"]].drop_duplicates(subset="asin")

    recommendations = recommendations.merge(
        product_info,
        on="asin",
        how="left"
    )

    return recommendations[["asin", "title", "brand", "predicted_rating"]]

test_user = ratings_df.iloc[0]["reviewerID"]
print("\nTest user:", test_user)
print(recommend_for_user(test_user, top_n=5))



os.makedirs("models", exist_ok=True)
joblib.dump(svd_model, "models/svd_collaborative_model.pkl")
