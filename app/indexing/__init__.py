"""Search backend abstraction.

Backends expose a common interface so the API layer (`/api/search`,
`/api/filters`, `/api/health`, chat RAG) is decoupled from where vectors
live:

  - `faiss`   local FAISS + Gemini embeddings (open-source default)
  - `vertex`  Vertex AI Search (Discovery Engine), with optional fallback
              to faiss when unreachable.
"""

from app.indexing.base import SearchBackend
from app.indexing.factory import get_search_backend

__all__ = ["SearchBackend", "get_search_backend"]