import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "km-documents")
VECTOR_DIMENSIONS = int(os.getenv("VECTOR_DIMENSIONS", "3072"))
FAISS_DIR = os.getenv("FAISS_DIR", "./app/db/indices")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/km_final_output.xlsx")

# ---- ETL / extraction ----
DOCS_INPUT_DIR = os.getenv("DOCS_INPUT_DIR", "./docs")
ETL_OUTPUT_DIR = os.getenv("ETL_OUTPUT_DIR", "./app/db/etl")
ETL_STATE_FILE = os.getenv("ETL_STATE_FILE", "./app/db/etl/state.json")
ETL_RECORDS_FILE = os.getenv("ETL_RECORDS_FILE", "./app/db/etl/records.jsonl")
INDEX_MAX_EMBED_CHARS = int(os.getenv("INDEX_MAX_EMBED_CHARS", "8000"))
CHUNK_LEVEL = os.getenv("CHUNK_LEVEL", "document")
EXTRACT_MIN_CHARS_PER_PAGE = int(os.getenv("EXTRACT_MIN_CHARS_PER_PAGE", "100"))
EXTRACT_MAX_GARBLED_RATIO = float(os.getenv("EXTRACT_MAX_GARBLED_RATIO", "0.2"))
EXTRACT_MIN_ALNUM_RATIO = float(os.getenv("EXTRACT_MIN_ALNUM_RATIO", "0.5"))
EXTRACT_MAX_IMAGE_ONLY_PAGE_RATIO = float(os.getenv("EXTRACT_MAX_IMAGE_ONLY_PAGE_RATIO", "0.5"))
DOCUMENT_AI_ENABLED = os.getenv("DOCUMENT_AI_ENABLED", "true").lower() in {"1", "true", "yes"}
DOCUMENT_AI_LOCATION = os.getenv("DOCUMENT_AI_LOCATION", "us")
DOCUMENT_AI_OCR_PROCESSOR_ID = os.getenv("DOCUMENT_AI_OCR_PROCESSOR_ID", "")
DOCUMENT_AI_LAYOUT_PROCESSOR_ID = os.getenv("DOCUMENT_AI_LAYOUT_PROCESSOR_ID", "")
DOCUMENT_AI_ONLINE_PAGE_LIMIT = int(os.getenv("DOCUMENT_AI_ONLINE_PAGE_LIMIT", "15"))

# ---- Backend selection ----
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "faiss")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini")
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")
VERTEX_ENABLE_FALLBACK = os.getenv("VERTEX_ENABLE_FALLBACK", "true").lower() in {"1", "true", "yes"}
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
GCP_STAGING_BUCKET = os.getenv("GCP_STAGING_BUCKET", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
VERTEX_COLLECTION = os.getenv("VERTEX_COLLECTION", "default_collection")
VERTEX_DATASTORE_ID = os.getenv("VERTEX_DATASTORE_ID", "")
VERTEX_SEARCH_ENGINE_ID = os.getenv("VERTEX_SEARCH_ENGINE_ID", "")
VERTEX_SEARCH_APP_ID = os.getenv("VERTEX_SEARCH_APP_ID", "")
VERTEX_EMBEDDING_MODEL = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")

# ---- Ranking (per-backend relevance tiers) ----
RANKING_TIER_THRESHOLDS = [
    float(v) for v in os.getenv("RANKING_TIER_THRESHOLDS", "63.5,60.0,55.5").split(",") if v
]
RANKING_RELEVANCE_FLOOR = float(os.getenv("RANKING_RELEVANCE_FLOOR", "55.5"))
