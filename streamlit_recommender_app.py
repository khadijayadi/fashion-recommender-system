import os
import re
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

st.set_page_config(
    page_title="AI Fashion Recommender",
    page_icon="🛍️",
    layout="wide",
)

# layout and styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fafafa;
    }

    .hero-box {
        background: linear-gradient(135deg, #111827 0%, #374151 100%);
        color: white;
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }

    .panel-box {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: #111827;
    }

    .meta {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.55rem;
    }

    .score-pill {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 0.6rem;
    }

    .section-label {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        color: #111827;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = "/Users/khadijaayadi/fashion-recommender-system"
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
PRODUCTS_PATH = os.path.join(MODELS_DIR, "products_dataframe.pkl")
SVD_PATH = os.path.join(MODELS_DIR, "svd_collaborative_model.pkl")
MERGED_PATH = os.path.join(DATA_DIR, "merged_sample.csv")
DISSATISFACTION_PATH = os.path.join(DATA_DIR, "product_dissatisfaction_scores.csv")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

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

def minmax(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    if series.max() == series.min():
        return pd.Series(np.full(len(series), 0.5), index=series.index)
    return (series - series.min()) / (series.max() - series.min())

@st.cache_data(show_spinner=False)
def fetch_image_from_url(url: str):
    if pd.isna(url):
        return None

    url = str(url).strip()
    if not url or url.lower() == "nan":
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.amazon.com/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception:
        return None

def render_missing_image(height: int = 260):
    st.markdown(
        f"""
        <div style="
            height:{height}px;
            border-radius:16px;
            background:#f3f4f6;
            display:flex;
            align-items:center;
            justify-content:center;
            border:1px solid #e5e7eb;
            color:#6b7280;
            font-weight:600;">
            Image unavailable
        </div>
        """,
        unsafe_allow_html=True,
    )

# outfit rules 

COMPATIBILITY_RULES = {
    "shoes": ["bottom", "top", "bag"],
    "bag": ["shoes", "top"],
    "top": ["bottom", "shoes", "bag"],
    "bottom": ["top", "shoes"],
    "watch": ["top", "bottom"],
    "other": ["top", "bottom", "shoes"],
}


@st.cache_resource
def load_resources():
    tfidf = joblib.load(TFIDF_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    products_df = joblib.load(PRODUCTS_PATH)
    svd_model = joblib.load(SVD_PATH)
    merged_df = pd.read_csv(MERGED_PATH)
    dissatisfaction_df = pd.read_csv(DISSATISFACTION_PATH)

    products_df = products_df.copy()

    for col in ["title", "brand", "product_text"]:
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
        products_df["simple_category"] = products_df["title"].apply(assign_simple_category)

    products_df = products_df.merge(
        dissatisfaction_df[["asin", "dissatisfaction_norm"]],
        on="asin",
        how="left",
    )
    products_df["dissatisfaction_norm"] = products_df["dissatisfaction_norm"].fillna(0)

    ratings_df = merged_df[["reviewerID", "asin", "overall"]].dropna().copy()
    title_to_index = pd.Series(products_df.index, index=products_df["title"]).drop_duplicates()

    return tfidf, tfidf_matrix, products_df, svd_model, ratings_df, title_to_index

# hybrid recommender model 
def hybrid_recommend(
    user_id: str,
    user_query: str,
    target_category: Optional[str],
    top_n: int,
    alpha: float,
    beta: float,
    gamma: float,
    min_content_threshold: float,
    tfidf,
    products_df: pd.DataFrame,
    svd_model,
):
    candidate_products = products_df.copy()

    if target_category and target_category != "all":
        candidate_products = candidate_products[
            candidate_products["simple_category"] == target_category
        ].copy()

    if candidate_products.empty:
        return pd.DataFrame()

    query_vector = tfidf.transform([clean_text(user_query)])
    candidate_vectors = tfidf.transform(candidate_products["combined_text_clean"])
    candidate_products["content_score"] = cosine_similarity(
        query_vector, candidate_vectors
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
    ).head(top_n).copy()

    columns_to_return = [
        "asin", "title", "brand", "simple_category",
        "content_score", "collab_score", "hybrid_score",
        "dissatisfaction_norm", "final_score"
    ]

    if "imUrl" in result.columns:
        columns_to_return.insert(3, "imUrl")

    return result[columns_to_return]

# outfit recommender 
def recommend_outfit_items(
    main_product_title: str,
    top_n_per_category: int,
    products_df: pd.DataFrame,
    title_to_index,
    tfidf_matrix
):
    if main_product_title not in title_to_index:
        return pd.DataFrame()

    main_idx = title_to_index[main_product_title]
    main_product = products_df.iloc[main_idx]
    main_category = main_product["simple_category"]
    compatible_categories = COMPATIBILITY_RULES.get(main_category, [])

    outfit_results = []

    for category in compatible_categories:
        category_products = products_df[
            products_df["simple_category"] == category
        ].copy()

        if category_products.empty:
            continue

        category_indices = category_products.index.tolist()
        main_vector = tfidf_matrix[main_idx]
        category_vectors = tfidf_matrix[category_indices]
        similarity_scores = cosine_similarity(main_vector, category_vectors).flatten()

        category_products["similarity_score"] = similarity_scores
        category_products = category_products.sort_values(
            by="similarity_score",
            ascending=False
        ).head(top_n_per_category)

        cols = ["asin", "title", "brand", "simple_category", "similarity_score"]
        if "imUrl" in category_products.columns:
            cols.insert(3, "imUrl")

        outfit_results.append(category_products[cols])

    if outfit_results:
        return pd.concat(outfit_results).reset_index(drop=True)

    return pd.DataFrame()

# Load models and data with error handling
try:
    tfidf, tfidf_matrix, products_df, svd_model, ratings_df, title_to_index = load_resources()
except Exception as e:
    st.error(f"Could not load models/data: {e}")
    st.stop()

user_ids = sorted(ratings_df["reviewerID"].dropna().unique().tolist())

allowed_categories = ["shoes", "watch", "bag", "top", "bottom", "other"]
available_categories = ["all"] + [c for c in allowed_categories if c in products_df["simple_category"].unique()]

# UI Layout

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">AI Fashion Recommender</div>
        <p class="hero-subtitle">You bring the vision, we’ll bring the rest.</p>    
        </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.8], gap="large")

with left_col:
    st.markdown('<div class="section-label">Recommendation Inputs</div>', unsafe_allow_html=True)

    user_id = st.selectbox("User ID", options=user_ids)

    user_query = st.text_input(
        "What are you looking for?",
        value="",
        placeholder="Type your fashion query here..."
    )

    default_cat = available_categories.index("shoes") if "shoes" in available_categories else 0
    target_category = st.selectbox(
        "Category",
        options=available_categories,
        index=default_cat
    )

    top_n = st.slider("Number of recommendations", min_value=1, max_value=8, value=5)

    with st.expander("Advanced scoring settings"):
        alpha = st.slider("Content relevance weight", 0.0, 1.0, 0.8, 0.05)
        beta = st.slider("Collaborative weight", 0.0, 1.0, 0.2, 0.05)
        gamma = st.slider("Dissatisfaction penalty", 0.0, 0.5, 0.2, 0.05)
        min_content_threshold = st.slider("Minimum content relevance", 0.0, 0.20, 0.0, 0.01)

    run_button = st.button("Generate Recommendations", use_container_width=True)
    

with right_col:
    st.markdown('<div class="section-label">Recommendation Results</div>', unsafe_allow_html=True)
    st.caption(f"Query: {user_query} • Category: {target_category}")
        
    if run_button:
        recs = hybrid_recommend(
            user_id=user_id,
            user_query=user_query,
            target_category=target_category,
            top_n=top_n,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            min_content_threshold=min_content_threshold,
            tfidf=tfidf,
            products_df=products_df,
            svd_model=svd_model,
        )

        if recs.empty:
            st.warning("No strong matches found. Try another query or category.")
        else:
            st.caption(f"Showing top {len(recs)} recommendations")

            for i, row in recs.reset_index(drop=True).iterrows():
                img_col, text_col = st.columns([1, 1.6], gap="medium")

                with img_col:
                    img = fetch_image_from_url(row["imUrl"]) if "imUrl" in recs.columns else None
                    if img is not None:
                        st.image(img, use_container_width=True)
                    else:
                        render_missing_image(250)

                with text_col:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">#{i+1} {row['title']}</div>
                            <div class="meta">
                                Brand: {row['brand'] if pd.notna(row['brand']) and str(row['brand']).strip() != '' else 'Unknown'}<br>
                                Category: {row['simple_category']}<br>
                                ASIN: {row['asin']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander("Show technical details"):
                        st.write(f"Final score: {row['final_score']:.3f}")
                        st.write(f"Content score: {row['content_score']:.3f}")
                        st.write(f"Collaborative score: {row['collab_score']:.3f}")
                        st.write(f"Dissatisfaction penalty: {row['dissatisfaction_norm']:.3f}")

            st.markdown("---")
            st.markdown('<div class="section-label">Optional Outfit Completion</div>', unsafe_allow_html=True)

            selected_title = st.selectbox(
                "Select one recommendation to build an outfit",
                options=recs["title"].tolist()
            )

            outfit_df = recommend_outfit_items(
                main_product_title=selected_title,
                top_n_per_category=2,
                products_df=products_df,
                title_to_index=title_to_index,
                tfidf_matrix=tfidf_matrix,
            )

            if outfit_df.empty:
                st.info("No outfit suggestions available for this item.")
            else:
                outfit_cols = st.columns(min(3, len(outfit_df)))

                for idx, (_, item) in enumerate(outfit_df.iterrows()):
                    with outfit_cols[idx % len(outfit_cols)]:
                        img = fetch_image_from_url(item["imUrl"]) if "imUrl" in outfit_df.columns else None
                        if img is not None:
                            st.image(img, use_container_width=True)
                        else:
                            render_missing_image(220)

                        st.markdown(f"**{item['simple_category'].title()}**")
                        st.caption(item["title"])
                        
    else:
        st.info("Enter a query, choose a category, and click Generate Recommendations.")

    

st.markdown("---")
st.caption("Hybrid recommender system built with Streamlit, TF-IDF, SVD, and review-based penalty scoring.")