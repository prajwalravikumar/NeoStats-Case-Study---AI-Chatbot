from typing import List
import logging
from config.config import EMBEDDING_DIM
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_texts(texts: List[str]) -> List[List[float]]:
        embs = model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embs]
except Exception as e:
    logger.exception("SentenceTransformer not available: %s", e)
    # fallback: zero vectors (not ideal)
    def embed_texts(texts: List[str]) -> List[List[float]]:
        return [[0.0]*EMBEDDING_DIM for _ in texts]