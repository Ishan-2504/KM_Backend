"""Vertex AI Search (Discovery Engine) backend.

Query path: SearchServiceClient.search against an unstructured data store or
an engine's default serving config. Metadata passed at import time becomes
`structData`, which powers filtering and facets.

All google client imports are lazy so the app runs (with FAISS fallback)
even when google-cloud-discoveryengine is not installed or no credentials
exist yet.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import (
    GCP_LOCATION,
    GCP_PROJECT_ID,
    VERTEX_COLLECTION,
    VERTEX_DATASTORE_ID,
    VERTEX_ENABLE_FALLBACK,
    VERTEX_LOCATION,
    VERTEX_SEARCH_ENGINE_ID,
)
from app.indexing.base import FilterDict


def _quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def build_filter_expression(filters: FilterDict | None) -> str | None:
    """Convert our {field: value|list} filters into a Discovery Engine filter
    expression. String equality is expressed as `field: ANY("v")`."""
    if not filters:
        return None
    parts: list[str] = []
    for field, value in filters.items():
        if value is None or value == "":
            continue
        if isinstance(value, (list, tuple, set)):
            values = [v for v in value if v not in ("", None)]
            if not values:
                continue
            args = ", ".join(_quote(v) for v in values)
            parts.append(f'{field}: ANY({args})')
        else:
            parts.append(f'{field}: ANY({_quote(value)})')
    return " AND ".join(parts) if parts else None


class VertexSearchBackend:
    name = "vertex"

    def __init__(self, client: Any | None = None):
        self._client = client  # created lazily on first RPC
        self._use_fallback = VERTEX_ENABLE_FALLBACK
        self._fallback = None  # created lazily
        self._serving_config: str | None = None

    # ------------------------------------------------------------- helpers

    def _require_configured(self) -> None:
        if not (GCP_PROJECT_ID and (VERTEX_SEARCH_ENGINE_ID or VERTEX_DATASTORE_ID)):
            raise RuntimeError(
                "Vertex backend is not configured: set GCP_PROJECT_ID and "
                "VERTEX_SEARCH_ENGINE_ID (or VERTEX_DATASTORE_ID)."
            )

    def _engine_client(self) -> Any:
        if self._client is None:
            from google.cloud import discoveryengine_v1 as de

            self._require_configured()
            self._client = de.SearchServiceClient()
        return self._client

    def _serving_config_path(self) -> str:
        if self._serving_config:
            return self._serving_config
        location = "global" if VERTEX_LOCATION == "global" else GCP_LOCATION
        base = (
            f"projects/{GCP_PROJECT_ID}/locations/{location}"
            f"/collections/{VERTEX_COLLECTION}"
        )
        if VERTEX_SEARCH_ENGINE_ID:
            path = (
                f"{base}/engines/{VERTEX_SEARCH_ENGINE_ID}/servingConfigs/"
                "default_serving_config"
            )
        elif VERTEX_DATASTORE_ID:
            path = f"{base}/dataStores/{VERTEX_DATASTORE_ID}/servingConfigs/default_search"
        else:
            raise RuntimeError("Vertex backend is not configured")
        self._serving_config = path
        return path

    def _fallback_backend(self):
        if self._fallback is None:
            from app.indexing.faiss_backend import FaissBackend

            self._fallback = FaissBackend()
        return self._fallback

    def _with_fallback(self, primary, fallback, *args, **kwargs):
        try:
            return primary(*args, **kwargs)
        except Exception as exc:
            if self._use_fallback:
                import logging

                logging.warning("Vertex backend error, falling back to faiss: %s", exc)
                return fallback(*args, **kwargs)
            raise

    # ------------------------------------------------------------- interface

    def get_document_count(self) -> int:
        try:
            client = self._engine_client()
            self._require_configured()
            # CountDocuments is exposed via the EnginesService in newer SDKs.
            # It can be long-running; fall back to 0 without creds.
            from google.cloud import discoveryengine_v1 as de

            count_client = de.EnginesServiceClient()
            request = de.CountDocumentsRequest(
                parent=f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
                f"/collections/{VERTEX_COLLECTION}"
                f"/dataStores/{VERTEX_DATASTORE_ID}"
                if VERTEX_DATASTORE_ID else
                f"projects/{GCP_PROJECT_ID}/locations/{GCP_LOCATION}"
                f"/collections/{VERTEX_COLLECTION}/engines/{VERTEX_SEARCH_ENGINE_ID}",
            )
            resp = count_client.count_documents(request)
            return int(resp.total_documents)
        except Exception as exc:  # pragma: no cover - depends on live GCP
            if self._use_fallback:
                import logging

                logging.warning("Vertex count failed, falling back to faiss: %s", exc)
                return self._fallback_backend().get_document_count()
            return 0

    def get_facets(
        self,
        filters: FilterDict | None = None,
        fields: list[str] | None = None,
    ) -> dict[str, list[dict]]:
        return self._with_fallback(
            self._get_facets_vertex,
            self._fallback_backend().get_facets,
            filters,
            fields,
        )

    def _get_facets_vertex(
        self,
        filters: FilterDict | None,
        fields: list[str] | None,
    ) -> dict[str, list[dict]]:
        client = self._engine_client()
        from google.cloud import discoveryengine_v1 as de

        fields = fields or []
        facet_specs = [
            de.SearchRequest.FacetSpec(
                facet_key=de.SearchRequest.FacetSpec.FacetKey(key=field),
                limit=200,
            )
            for field in fields
        ]

        request = de.SearchRequest(
            query="",
            serving_config=self._serving_config_path(),
            page_size=1,
            filter=build_filter_expression(filters) or None,
            facet_specs=facet_specs,
        )
        response = client.search(request)

        counts: dict[str, list[dict]] = {}
        for facet in response.facets:
            counts[facet.key] = [
                {"value": bucket.key, "count": bucket.count}
                for bucket in facet.facets
                if bucket.key
            ]
        return counts

    def search(
        self,
        *,
        query: str,
        filters: FilterDict | None = None,
        top_k: int = 50,
        select: list[str] | None = None,
    ) -> list[dict]:
        return self._with_fallback(
            self._search_vertex,
            self._fallback_backend().search,
            query=query,
            filters=filters,
            top_k=top_k,
            select=select,
        )

    def _search_vertex(
        self,
        *,
        query: str,
        filters: FilterDict | None,
        top_k: int,
        select: list[str] | None,
    ) -> list[dict]:
        client = self._engine_client()
        from google.cloud import discoveryengine_v1 as de

        request = de.SearchRequest(
            query=query,
            serving_config=self._serving_config_path(),
            page_size=min(top_k, 100),
            filter=build_filter_expression(filters) or None,
        )
        response = client.search(request)

        select = select or []
        hits: list[dict] = []
        for result in response.results:
            doc = result.document
            struct_data = dict(getattr(doc, "struct_data", {}) or {})
            derived = dict(getattr(doc, "derived_struct_data", {}) or {})
            run_scores = getattr(result, "relevance_score", None)
            score = float(run_scores) if run_scores is not None else 0.0

            hit: dict[str, Any] = dict(struct_data)
            if derived:
                snippet = _first_snippet(derived)
                if snippet:
                    hit["file_text"] = snippet
            if "document_name" not in hit:
                hit["document_name"] = getattr(doc, "id", "")
            if select:
                hit = {field: hit.get(field, "") for field in select}
            hit["_score"] = score
            hits.append(hit)
        return hits


def _first_snippet(derived: dict) -> str:
    """Extract the best text snippet from derived_struct_data."""
    try:
        snippets = derived.get("snippets") or []
        if snippets:
            return snippets[0].get("snippet", "")
        extractive = derived.get("extractive_answers") or []
        if extractive:
            return extractive[0].get("content", "") if isinstance(extractive[0], dict) else ""
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


# Re-export for CLI / import tooling
__all__ = ["VertexSearchBackend", "build_filter_expression", "_first_snippet"]