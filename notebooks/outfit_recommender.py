import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

tfidf = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
products_df = joblib.load("models/products_dataframe.pkl")



def assign_simple_category(text):
    text = str(text).lower()

    if any(word in text for word in ['sneaker', 'shoe', 'boots', 'boot', 'sandals', 'slipper', 'loafer']):
        return 'shoes'
    elif any(word in text for word in ['scarf', 'shawl']):
        return 'scarf'
    elif any(word in text for word in ['dress', 'gown']):
        return 'dress'
    elif any(word in text for word in ['bag', 'backpack', 'purse', 'handbag']):
        return 'bag'
    elif any(word in text for word in ['watch']):
        return 'watch'
    elif any(word in text for word in ['shirt', 't-shirt', 'tee', 'top']):
        return 'top'
    elif any(word in text for word in ['jean', 'pants', 'trouser', 'shorts']):
        return 'bottom'
    else:
        return 'other'
    
products_df['simple_category'] = products_df['product_text'].apply(assign_simple_category)

print(products_df['simple_category'].value_counts())

title_to_index = pd.Series(products_df.index, index=products_df['title']).drop_duplicates()


compatibility_rules = {
    'shoes': ['bottom', 'top', 'bag', 'belt'],
    'dress': ['shoes', 'bag', 'scarf'],
    'bag': ['shoes', 'dress', 'top'],
    'top': ['bottom', 'shoes', 'bag'],
    'bottom': ['top', 'shoes'],
    'scarf': ['top', 'dress', 'bag'],
    'watch': ['top', 'bottom'],
    'other': ['top', 'bottom', 'shoes']
}

def recommend_outfit_items(main_product_title, top_n_per_category=2):
    
    if main_product_title not in title_to_index:
        return f"Product '{main_product_title}' not found."

    
    main_idx = title_to_index[main_product_title]
    main_product = products_df.iloc[main_idx]
    main_category = main_product['simple_category']

    
    compatible_categories = compatibility_rules.get(main_category, [])

    
    main_vector = tfidf_matrix[main_idx]

    outfit_results = []

    for category in compatible_categories:
        category_products = products_df[products_df['simple_category'] == category].copy()

        if category_products.empty:
            continue

        category_indices = category_products.index.tolist()
        category_vectors = tfidf_matrix[category_indices]

        similarity_scores = cosine_similarity(main_vector, category_vectors).flatten()

        category_products['similarity_score'] = similarity_scores
        category_products = category_products.sort_values(
            by='similarity_score',
            ascending=False
        ).head(top_n_per_category)

        outfit_results.append(category_products[['asin', 'title', 'brand', 'simple_category', 'similarity_score']])

    if outfit_results:
        return pd.concat(outfit_results).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=['asin', 'title', 'brand', 'simple_category', 'similarity_score'])
    
    

print("\n--- OUTFIT RECOMMENDATIONS ---")
print(recommend_outfit_items(
    "Converse Unisex Chuck Taylor All Star Hi Top Black Monochrome Sneaker",
    top_n_per_category=2
))

