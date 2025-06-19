import numpy as np
import faiss
import numpy as np
import logging

def build_faiss_index(embeddings_npy):
    try:
        data = np.load(embeddings_npy, allow_pickle=True).item()
    except Exception as e:
        logging.warning(f"Failed to load embeddings file {embeddings_npy}: {e}")
        return None, []
    embeddings = data['embeddings'].astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, data.get('filenames', [])

def search(query_embedding, index, filenames, top_k=5):
    D, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [filenames[i] for i in I[0]]

# Example usage:
# index, filenames = build_faiss_index('dresses_embeddings.npy')
# similar = search(query_emb, index, filenames)