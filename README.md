# Fashion Visual Search & Intelligent Styling Assistant

## Overview
This project is a scalable, API-ready fashion visual search and intelligent styling assistant. Users can upload a fashion image (dress or jeans) and receive:
- Visually similar products (same category)
- Outfit recommendations (cross-category, best-fit)
- Personalized recommendations

## Features
- Visual similarity search using deep image embeddings (ResNet + FAISS)
- Multi-modal similarity (visual + metadata)
- Trend awareness (recency boosting)
- Cross-category outfit recommendations (color/style/trend aware)
- Personalized suggestions based on user history
- Streamlit UI for easy interaction
- Modular, API-ready code structure

## Folder Structure
```
MLE Assignment/
├── UI.py
├── data_preprocessing.py
├── Feature_extraction.py
├── Visual_search_engine.py
├── Outfit_recommendation _engine.py
├── image_downloader.py
├── dresses_bd_processed_data.csv
├── jeans_bd_processed_data.csv
├── dresses_embeddings.npy
├── jeans_embeddings.npy
├── images/
│   ├── dresses/
│   └── jeans/
└── requirements.txt
```

## Setup Instructions
1. **Clone or extract the project folder.**
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   (If requirements.txt is missing, install: streamlit, torch, torchvision, faiss-cpu, pandas, scikit-learn, pillow, tqdm)
3. **Run the app:**
   ```
   streamlit run UI.py
   ```
4. **Upload a dress or jeans image to test the system.**

## How it Works
- The system detects if the uploaded image is a dress or jeans.
- Returns visually similar products from the same category.
- Recommends best-fit outfits from the other category (ordinal, color/trend aware).
- Shows personalized recommendations based on session history.

## Solution Design
- **Image Embeddings:** Extracted using ResNet50, stored as .npy files.
- **Similarity Search:** FAISS for fast nearest neighbor search.
- **Multi-modal:** Combines visual and metadata (price, category) features.
- **Trend Awareness:** Recency (last_seen_date) boosts recommendations.
- **Personalization:** Session-based, can be extended to user accounts.
- **API-Ready:** Core logic is modular for easy FastAPI/Flask deployment.

## Architecture Diagram
```
User ──> Streamlit UI ──> (Feature Extraction, Category Detection)
                        ├─> FAISS Visual Search (Dresses/Jeans)
                        ├─> Outfit Recommendation Engine
                        └─> Personalized Recommendation
Data: CSVs, Embeddings, Images
```

## Large Files
Due to GitHub's 100MB file limit, large data and embedding files are not included in this repository.



https://github.com/user-attachments/assets/511647d2-5a23-4da8-95fc-2cee33f0b66e


---

For questions or issues, contact the project owner.
