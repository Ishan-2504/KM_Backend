"""Open-source backend: local FAISS index + Gemini embeddings (default)."""

from __future__ import annotations

from app.db.search_client import FaissSearchClient, get_search_client
from app.indexing.base import FilterDict
from app.services.embeddings import embed_single


class FaissBackend:
    """Thin adapter over `FaissSearchClient`, moving query embedding inside
    `search` so the interface is text-in, hits-out for every backend."""

    name = "faiss"

    def __init__(self, client: FaissSearchClient | None = None):
        self._client = client or get_search_client()

    def get_document_count(self) -> int:
        return self._client.get_document_count()

    def get_facets(
        self,
        filters: FilterDict | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        return self._client.get_facets(filters, fields)

    def search(
        self,
        *,
        query: str,
        filters: FilterDict | None = None,
        top_k: int = 50,
        select: list[str] | None = None,
    ) -> list[dict]:
        if query and query.strip():
            query_vector = embed_single(query)
            return self._client.search(
                query_vector=query_vector,
                filters=filters,
                top_k=top_k,
                select=select,
            )
        return self._client.search(filters=filters, top_k=top_k, select=select)