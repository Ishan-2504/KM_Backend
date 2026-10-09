import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "km-documents")
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH", "../data/km_final_output.xlsx")
