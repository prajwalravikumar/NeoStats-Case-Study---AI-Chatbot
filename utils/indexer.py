"""
FAISS index building and retrieval.
"""
import os
import faiss
import pickle
import numpy as np
from typing import List, Dict
from config.config import INDEX_DIR, EMBEDDING_DIM, TOP_K
from models.embeddings import embed_texts
import logging

logger = logging.getLogger(__name__)
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")

def build_index(chunks: List[Dict]):
    """Build FAISS index from chunks and save index + metadata."""
    texts = [c["text"] for c in chunks]
    embs = embed_texts(texts)
    mat = np.array(embs).astype("float32")
    # normalize for cosine via inner product
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)
    return True

def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        return None, None
    idx = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return idx, meta

def retrieve(query: str, top_k: int = TOP_K) -> List[str]:
    idx, meta = load_index()
    if idx is None:
        return []
    q_emb = np.array(embed_texts([query])).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = idx.search(q_emb, top_k)
    results = []
    for i in I[0]:
        if i < len(meta):
            results.append(meta[i]["text"])
    return results