"""Common contract implemented by every search backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

FilterDict = dict[str, str | list[str]]


@runtime_checkable
class SearchBackend(Protocol):
    """Unified interface for vector + metadata search."""

    name: str

    def get_document_count(self) -> int:
        """Total number of indexed documents (used by /api/health)."""
        ...

    def get_facets(
        self,
        filters: FilterDict | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        """Return {field: [{"value": ..., "count": ...}]} for filter dropdowns,
        optionally narrowed by active filters."""
        ...

    def search(
        self,
        *,
        query: str,
        filters: FilterDict | None = None,
        top_k: int = 50,
        select: list[str] | None = None,
    ) -> list[dict]:
        """Return hits as dicts carrying the standard metadata keys plus a
        backend-native similarity score in `_score`.

        An empty/blank `query` performs metadata-only filtering.
        """
        ...