"""Backend factory: choose FAISS or Vertex AI Search from configuration."""

from __future__ import annotations

import threading

from app.config import (
    GCP_PROJECT_ID,
    SEARCH_BACKEND,
    VERTEX_DATASTORE_ID,
    VERTEX_ENABLE_FALLBACK,
    VERTEX_SEARCH_ENGINE_ID,
)
from app.indexing.base import SearchBackend
from app.indexing.faiss_backend import FaissBackend

_backend: SearchBackend | None = None
_backend_lock = threading.Lock()


def _build_backend() -> SearchBackend:
    mode = (SEARCH_BACKEND or "faiss").strip().lower()

    if mode == "vertex":
        configured = bool(GCP_PROJECT_ID and (VERTEX_SEARCH_ENGINE_ID or VERTEX_DATASTORE_ID))
        if not configured and VERTEX_ENABLE_FALLBACK:
            import logging

            logging.warning(
                "SEARCH_BACKEND=vertex but GCP_PROJECT_ID / VERTEX_* IDs are unset; "
                "falling back to faiss."
            )
            return FaissBackend()
        from app.indexing.vertex_backend import VertexSearchBackend

        return VertexSearchBackend()

    if mode != "faiss":
        import logging

        logging.warning(
            "Unknown SEARCH_BACKEND=%r, falling back to faiss.", SEARCH_BACKEND
        )
    return FaissBackend()


def get_search_backend() -> SearchBackend:
    """Return the configured backend (cached)."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _build_backend()
    return _backend


def reset_backend() -> None:
    """Clear the cached backend (useful in tests)."""
    global _backend
    with _backend_lock:
        _backend = None