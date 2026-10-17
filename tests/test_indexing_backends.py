"""Tests for Phase 4: search backend abstraction (FAISS adapter, Vertex
filter/import tooling, factory + fallback). No GCP SDK or network required."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from app.config import VECTOR_DIMENSIONS
from app.etl.models import ETLRecord
from app.indexing import factory
from app.indexing.faiss_backend import FaissBackend
from app.indexing.vertex_backend import (
    _first_snippet,
    build_filter_expression,
)
from app.indexing.vertex_import import (
    _split_gcs_uri,
    prepare_vertex_records,
    write_staging_jsonl,
)
from app.indexing.vertex_backend import VertexSearchBackend


class _StubClient:
    def __init__(self):
        self.last: dict | None = None

    def get_document_count(self) -> int:
        return 7

    def get_facets(self, filters=None, fields=None):
        return {"industry": [{"value": "Retail", "count": 3}]}

    def search(self, **kwargs):
        self.last = kwargs
        return [{"document_name": "x", "_score": 1.0}]


# ---------------------------------------------------------------- faiss adapter

def test_faiss_backend_search_embeds_query(monkeypatch):
    stub = _StubClient()
    backend = FaissBackend(stub)
    monkeypatch.setattr(
        "app.indexing.faiss_backend.embed_single",
        lambda t: [0.1] * VECTOR_DIMENSIONS,
    )
    hits = backend.search(query="hello world")
    assert stub.last["query_vector"] is not None
    assert stub.last["top_k"] == 50
    assert hits[0]["_score"] == 1.0


def test_faiss_backend_search_blank_query_is_filter_only():
    stub = _StubClient()
    backend = FaissBackend(stub)
    backend.search(query="", filters={"industry": "Retail"})
    assert "query_vector" not in stub.last
    assert stub.last["filters"] == {"industry": "Retail"}


def test_faiss_backend_counts_and_facets():
    stub = _StubClient()
    backend = FaissBackend(stub)
    assert backend.get_document_count() == 7
    assert backend.get_facets(fields=["industry"])["industry"][0]["count"] == 3


# ---------------------------------------------------------------- vertex helpers

def test_build_filter_expression():
    assert build_filter_expression(None) is None
    assert build_filter_expression({"industry": ""}) is None
    expr = build_filter_expression({"industry": "Retail", "solution": ["A", "B"]})
    assert 'industry: ANY("Retail")' in expr
    assert 'solution: ANY("A", "B")' in expr
    assert expr.startswith("industry") and "AND" in expr


def test_first_snippet():
    derived = {"snippets": [{"snippet": "hello text"}], "extractive_answers": []}
    assert _first_snippet(derived) == "hello text"
    derived2 = {"snippets": [], "extractive_answers": [{"content": "extractive"}]}
    assert _first_snippet(derived2) == "extractive"
    assert _first_snippet({}) == ""


# ---------------------------------------------------------------- vertex import prep

def _record(i: int) -> ETLRecord:
    return ETLRecord(
        doc_id=f"id-{i}",
        document_name=f"Doc {i}",
        source_path=f"C:/docs/doc{i}.pdf",
        source_type="pdf",
        content_hash=f"h{i}",
        extractor="pymupdf",
        text=f"Extracted body text for doc {i}.",
        pages=[],
        metadata={"industry": "Retail"} if i == 0 else {},
    )


def test_prepare_vertex_records_maps_metadata_and_content():
    lines = prepare_vertex_records([_record(0)])
    assert len(lines) == 1
    d = lines[0]
    assert d["id"] == "id-0"
    assert d["structData"]["document_name"] == "Doc 0"
    assert d["structData"]["industry"] == "Retail"
    assert d["structData"]["content_hash"] == "h0"
    assert d["content"]["mimeType"] == "text/plain"
    decoded = base64.b64decode(d["content"]["data"]).decode("utf-8")
    assert "Extracted body text" in decoded


def test_write_staging_jsonl(tmp_path):
    lines = prepare_vertex_records([_record(0)])
    out = write_staging_jsonl(tmp_path / "sub" / "import.jsonl", lines)
    assert out.exists()
    parsed = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert parsed[0]["id"] == "id-0"


def test_split_gcs_uri():
    assert _split_gcs_uri("gs://my-bucket/folder/file.jsonl") == ("my-bucket", "folder/file.jsonl")
    try:
        _split_gcs_uri("https://example.com/x")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------- factory + fallback

def test_factory_defaults_to_faiss(monkeypatch):
    monkeypatch.setattr(factory, "SEARCH_BACKEND", "faiss")
    assert isinstance(factory._build_backend(), FaissBackend)


def test_factory_unknown_mode_falls_back_to_faiss(monkeypatch):
    monkeypatch.setattr(factory, "SEARCH_BACKEND", "nope")
    assert isinstance(factory._build_backend(), FaissBackend)


def test_factory_vertex_unconfigured_falls_back(monkeypatch):
    monkeypatch.setattr(factory, "SEARCH_BACKEND", "vertex")
    monkeypatch.setattr(factory, "GCP_PROJECT_ID", "")
    monkeypatch.setattr(factory, "VERTEX_DATASTORE_ID", "")
    monkeypatch.setattr(factory, "VERTEX_SEARCH_ENGINE_ID", "")
    monkeypatch.setattr(factory, "VERTEX_ENABLE_FALLBACK", True)
    assert isinstance(factory._build_backend(), FaissBackend)


def test_factory_vertex_configured_returns_vertex_backend(monkeypatch):
    monkeypatch.setattr(factory, "SEARCH_BACKEND", "vertex")
    monkeypatch.setattr(factory, "GCP_PROJECT_ID", "my-project")
    monkeypatch.setattr(factory, "VERTEX_SEARCH_ENGINE_ID", "engine-1")
    monkeypatch.setattr(factory, "VERTEX_ENABLE_FALLBACK", True)
    backend = factory._build_backend()
    assert isinstance(backend, VertexSearchBackend)
    assert backend.name == "vertex"
    # Construction must not require credentials (deferred GCP client).
    assert backend._client is None