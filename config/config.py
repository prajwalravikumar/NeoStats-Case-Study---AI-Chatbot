import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS = os.getenv(
    "GROQ_FALLBACK_MODELS",
    "llama-3.3-70b-versatile,llama-3.1-8b-instant"
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
PDF_DIR = os.getenv("PDF_DIR", os.path.join(BASE_DIR, "pdfs"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
INDEX_DIR = os.path.join(DATA_DIR, "vectors")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
TOP_K = int(os.getenv("TOP_K", "4"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")