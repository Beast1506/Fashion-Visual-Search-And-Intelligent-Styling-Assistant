import asyncio
import sys

if sys.platform.startswith("win"):
    # For Windows, use the following to avoid event loop issues with Streamlit and asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

import streamlit as st
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import os
try:
    import faiss
except ImportError:
    faiss = None
from Visual_search_engine import build_faiss_index, search
from data_preprocessing import load_and_clean_csv
import importlib.util
import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dynamically import recommend_outfit from the file with a space in its name
spec = importlib.util.spec_from_file_location("Outfit_recommendation_engine", "Outfit_recommendation _engine.py")
OutfitRecEngine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(OutfitRecEngine)
recommend_outfit = OutfitRecEngine.recommend_outfit

st.title("Fashion Visual Search & Styling Assistant")

# --- UI Layout ---
st.markdown("""
<style>
.uploaded-img {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 1rem;
}
.section-header {
    font-size: 1.2rem;
    font-weight: bold;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
    color: #4F8BF9;
}
.divider {
    border-top: 1px solid #e0e0e0;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

if faiss is None:
    st.warning("faiss library is not installed. Please install faiss to enable visual search.")

@st.cache_resource
def get_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval().to(device)
    return model

@st.cache_resource
def get_faiss_index(embeddings_npy):
    return build_faiss_index(embeddings_npy)

# Load FAISS index and filenames for dresses (default)
model = get_model()
dress_index, dress_filenames = get_faiss_index('dresses_embeddings.npy')
if dress_index is None:
    st.error("Failed to load FAISS index for dresses. Please check the embeddings file.")
jeans_index, jeans_filenames = get_faiss_index('jeans_embeddings.npy')
if jeans_index is None:
    st.error("Failed to load FAISS index for jeans. Please check the embeddings file.")

# Load dataframes for recommendations
dresses_df = load_and_clean_csv('dresses_bd_processed_data.csv')
jeans_df = load_and_clean_csv('jeans_bd_processed_data.csv')

# Align DataFrames with available embeddings/images
_dress_ids = [fname.split('.')[0] for fname in dress_filenames] if dress_filenames else []
if 'product_id' in dresses_df.columns:
    dresses_df = dresses_df[dresses_df['product_id'].isin(_dress_ids)].reset_index(drop=True)
else:
    dresses_df = dresses_df.copy()
_jeans_ids = [fname.split('.')[0] for fname in jeans_filenames] if jeans_filenames else []
if 'product_id' in jeans_df.columns:
    jeans_df = jeans_df[jeans_df['product_id'].isin(_jeans_ids)].reset_index(drop=True)
else:
    jeans_df = jeans_df.copy()

# Remove duplicate model loading and use only the cached model
# Feature extraction setup (same as in Feature_extraction.py)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
def extract_embedding_pil(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_t).squeeze().cpu().numpy()
    return embedding.flatten()

# --- Helper functions for multi-modal similarity ---
def normalize_prices(prices):
    scaler = MinMaxScaler()
    return scaler.fit_transform(np.array(prices).reshape(-1, 1)).flatten()

def get_metadata_features(df):
    # Example: use normalized price and category as features
    prices = [list(x.values())[0] if isinstance(x, dict) and x else 0 for x in df['selling_price']]
    norm_prices = normalize_prices(prices)
    categories = df['category_id'].astype(str).astype('category').cat.codes.values
    return np.stack([norm_prices, categories], axis=1)

def multi_modal_search(query_emb, query_meta, emb_matrix, meta_matrix, filenames, top_k=5, alpha=0.7):
    # alpha: weight for visual similarity
    visual_dists = np.linalg.norm(emb_matrix - query_emb, axis=1)
    meta_dists = np.linalg.norm(meta_matrix - query_meta, axis=1)
    visual_dists = (visual_dists - visual_dists.min()) / (visual_dists.max() - visual_dists.min() + 1e-8)
    meta_dists = (meta_dists - meta_dists.min()) / (meta_dists.max() - meta_dists.min() + 1e-8)
    total_score = alpha * visual_dists + (1 - alpha) * meta_dists
    idxs = np.argsort(total_score)[:top_k]
    return [filenames[i] for i in idxs]

# --- Trend awareness: boost recent items ---
def get_trend_score(df):
    today = datetime.datetime.now().date()
    last_seen = pd.to_datetime(df['last_seen_date'], errors='coerce').dt.date
    days_ago = [(today - d).days if pd.notnull(d) else 9999 for d in last_seen]
    scaler = MinMaxScaler()
    return 1 - scaler.fit_transform(np.array(days_ago).reshape(-1, 1)).flatten()  # higher is more recent

# --- Personalization: store user history in session ---
if 'history' not in st.session_state:
    st.session_state['history'] = []

def update_user_history(product_id):
    st.session_state['history'].append(product_id)

def get_personalized_recs(df, top_k=3):
    if not st.session_state['history']:
        return df.sample(top_k)
    # Recommend items from brands/categories the user has searched for
    recent_ids = st.session_state['history'][-5:]
    recent_items = df[df['product_id'].isin(recent_ids)]
    brands = recent_items['brand'].unique()
    cats = recent_items['category_id'].unique()
    filtered = df[df['brand'].isin(brands) | df['category_id'].isin(cats)]
    if len(filtered) >= top_k:
        return filtered.sample(top_k)
    return df.sample(top_k)

# --- Precompute metadata features and trend scores ---
import logging

try:
    dress_data = np.load('dresses_embeddings.npy', allow_pickle=True).item()
    dress_emb_matrix = dress_data['embeddings']
    dress_meta_matrix = get_metadata_features(dresses_df)
    dress_trend = get_trend_score(dresses_df)
except Exception as e:
    logging.warning(f"Failed to load dresses_embeddings.npy: {e}")
    dress_emb_matrix = np.array([])
    dress_meta_matrix = np.array([])
    dress_trend = np.array([])
    dress_filenames = []

try:
    jeans_data = np.load('jeans_embeddings.npy', allow_pickle=True).item()
    jeans_emb_matrix = jeans_data['embeddings']
    jeans_meta_matrix = get_metadata_features(jeans_df)
    jeans_trend = get_trend_score(jeans_df)
except Exception as e:
    logging.warning(f"Failed to load jeans_embeddings.npy: {e}")
    jeans_emb_matrix = np.array([])
    jeans_meta_matrix = np.array([])
    jeans_trend = np.array([])
    jeans_filenames = []

uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.markdown('<div class="section-header">Uploaded Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="uploaded-img">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", width=180, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)
        # Extract embedding from uploaded image
        query_emb = extract_embedding_pil(img).astype('float32')
        # --- Category detection: dress or jeans? ---
        # Compute similarity to both dress and jeans embeddings
        dress_sim = np.linalg.norm(dress_emb_matrix - query_emb, axis=1).mean()
        jeans_sim = np.linalg.norm(jeans_emb_matrix - query_emb, axis=1).mean()
        if dress_sim < jeans_sim:
            # Uploaded image is more likely a dress
            main_label = 'Dress'
            emb_matrix = dress_emb_matrix
            meta_matrix = dress_meta_matrix
            filenames = dress_filenames
            df = dresses_df
            trend = dress_trend
            rec_label = 'Dresses'
        else:
            # Uploaded image is more likely jeans
            main_label = 'Jeans'
            emb_matrix = jeans_emb_matrix
            meta_matrix = jeans_meta_matrix
            filenames = jeans_filenames
            df = jeans_df
            trend = jeans_trend
            rec_label = 'Jeans'
        # For demo, use median price/category as query meta
        query_meta = np.median(meta_matrix, axis=0)
        # Multi-modal search (visual + metadata)
        top_k = 5
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">Visually Similar {rec_label}</div>', unsafe_allow_html=True)
        similar_files = multi_modal_search(query_emb, query_meta, emb_matrix, meta_matrix, filenames, top_k=top_k)
        sim_cols = st.columns(top_k)
        for i, fname in enumerate(similar_files):
            if main_label == 'Dress':
                img_path = os.path.join('images/dresses', fname)
            else:
                img_path = os.path.join('images/jeans', fname)
            if os.path.exists(img_path):
                with sim_cols[i]:
                    st.image(img_path, width=110, use_container_width=False)
                    st.caption(fname)
                update_user_history(fname.split('.')[0])
        # --- Outfit Recommendation: cross-category, ordinal best fit ---
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if main_label == 'Dress':
            st.markdown('<div class="section-header">Best Jeans to Pair (Outfit Recommendation)</div>', unsafe_allow_html=True)
            # Use color/style compatibility and trend for ordinal ranking
            dress_idx = filenames.index(similar_files[0]) if similar_files else 0
            dress_color = df.iloc[dress_idx]['feature_list'][0] if df.iloc[dress_idx]['feature_list'] else ''
            jeans_scores = []
            for i, row in jeans_df.iterrows():
                color_score = 1 if dress_color and dress_color in (row['feature_list'] or []) else 0
                jeans_scores.append((i, color_score + jeans_trend[i]))
            jeans_scores.sort(key=lambda x: x[1], reverse=True)
            best_jeans = [jeans_df.iloc[i] for i, _ in jeans_scores[:3]]
            outfit_cols = st.columns(3)
            for i, row in enumerate(best_jeans):
                img_path = os.path.join('images/jeans', row['product_id'] + '.jpg')
                if os.path.exists(img_path):
                    with outfit_cols[i]:
                        st.image(img_path, width=110, use_container_width=False)
                        st.caption(row['product_name'])
        else:
            st.markdown('<div class="section-header">Best Dresses to Pair (Outfit Recommendation)</div>', unsafe_allow_html=True)
            jeans_idx = filenames.index(similar_files[0]) if similar_files else 0
            jeans_color = df.iloc[jeans_idx]['feature_list'][0] if df.iloc[jeans_idx]['feature_list'] else ''
            dress_scores = []
            for i, row in dresses_df.iterrows():
                color_score = 1 if jeans_color and jeans_color in (row['feature_list'] or []) else 0
                dress_scores.append((i, color_score + dress_trend[i]))
            dress_scores.sort(key=lambda x: x[1], reverse=True)
            best_dresses = [dresses_df.iloc[i] for i, _ in dress_scores[:3]]
            outfit_cols = st.columns(3)
            for i, row in enumerate(best_dresses):
                img_path = os.path.join('images/dresses', row['product_id'] + '.jpg')
                if os.path.exists(img_path):
                    with outfit_cols[i]:
                        st.image(img_path, width=110, use_container_width=False)
                        st.caption(row['product_name'])
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">Personalized {rec_label} Recommendations</div>', unsafe_allow_html=True)
        personalized = get_personalized_recs(df, top_k=3)
        pers_cols = st.columns(3)
        for i, (_, row) in enumerate(personalized.iterrows()):
            if main_label == 'Dress':
                img_path = os.path.join('images/dresses', row['product_id'] + '.jpg')
            else:
                img_path = os.path.join('images/jeans', row['product_id'] + '.jpg')
            if os.path.exists(img_path):
                with pers_cols[i]:
                    st.image(img_path, width=110, use_container_width=False)
                    st.caption(row['product_name'])
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# --- API/Scalability comments ---
# To make this API-ready, move the core logic (embedding, search, recommend) into functions.
# Use FastAPI or Flask to expose endpoints for /search, /recommend, /personalize.
# For scale: deploy with Docker, use GPU inference, and run FAISS as a microservice.
# Use a load balancer and cache popular queries. For millions of products, use distributed FAISS.