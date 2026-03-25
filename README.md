# AI Fashion Recommender — End-to-End Recommendation System

This repository contains an end-to-end recommender system developed for B198c7 class. 
The system combines content-based filtering and collaborative filtering to generate personalized and query-driven fashion recommendations. It also includes evaluation, visualization, and an interactive Streamlit interface.

---

## Project Overview

The aim of this project is to design and analyze a hybrid Fashion recommendation system  based on Amazon product and review data provided by university of California , San Diego 

## Raw Data

The `data/raw/` folder is intentionally kept empty in this repository.

To run the pipeline, download the following files using this link https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html and place them inside `data/raw/`:

- `Clothing_Shoes_and_Jewelry_5.json`
- `meta_Clothing_Shoes_and_Jewelry.json`

These files can be downloaded under the part (Clothing,Shoes and Jewelry).


The system integrates:

- Content-based filtering using TF-IDF
- Collaborative filtering using SVD (matrix factorization)
- Hybrid ranking combining both approaches
- Review-based dissatisfaction penalty
- Outfit recommendation module
- Evaluation with ranking and prediction metrics
- Interactive Streamlit application

---

## How to run the pipeline :

### 1. data_exploration.py 
Load the data, perform preprocessing and feature engineering, and prepare the dataset for modeling.

### 2. content_based_recommender.py 
Build the TF-IDF representation and compute content similarity between products.

### 3. collaborative_filtering.py
Train the SVD model on user-item interactions and learn user preferences.

### 4. hybrid_model.py
Combine content-based and collaborative scores into a hybrid ranking.

### 5. run the extensions of the hybrid model 
Apply outfit_recommender.py and review_based_extension.py

### 6. evaluation.py 
Compute evaluation metrics and generate plots (saved in `/results`).

### 7. run the streamlit_recommender_app.py 
Launch the interface to interact with the recommender system.

```bash
streamlit run streamlit_recommender_app.py



=======
# fashion-recommender-system
>>>>>>> 4d1044060f6298438bb2fa0131322a9a8f677fe9
