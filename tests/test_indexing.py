"""Tests for Phase 2: indexing canonical ETL records into FAISS.

Uses a stub search client + stubbed embeddings so no network calls happen.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import VECTOR_DIMENSIONS
from app.etl.models import ETLRecord
from app.services import ingestion


class _StubClient:
    def __init__(self):
        self.docs: list[dict] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.docs = []

    def upload_documents(self, documents: list[dict]) -> None:
        self.docs.extend(documents)

    def get_document_count(self) -> int:
        return len(self.docs)


def _write_records(path: Path, n: int = 2, with_metadata: bool = False) -> None:
    lines = []
    for i in range(n):
        rec = ETLRecord(
            doc_id=f"etl-id-{i}",
            document_name=f"Doc {i}",
            source_path=f"C:/docs/doc{i}.md",
            source_type="md",
            content_hash=f"hash{i}",
            extractor="markdown",
            text=f"Body text content for document number {i}. " * 40,
            pages=[],
            metadata={
                "document_type": "knowledge_library",
                "industry": "Retail",
                "solution": "POS Modernisation",
            } if with_metadata else {},
            summary="",
        )
        lines.append(rec.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- load + build

def test_load_etl_records(tmp_path):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, n=3, with_metadata=True)

    records = ingestion.load_etl_records(str(records_path))
    assert len(records) == 3
    assert records[0]["doc_id"] == "etl-id-0"
    assert records[0]["document_name"] == "Doc 0"
    assert "Body text" in records[0]["file_text"]
    # metadata block is mapped onto ingest fields
    assert records[0]["industry"] == "Retail"
    assert records[0]["document_type"] == "knowledge_library"
    assert records[0]["client_name"] == ""


def test_load_etl_records_missing_file(tmp_path):
    try:
        ingestion.load_etl_records(str(tmp_path / "nope.jsonl"))
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_etl_composite_text_is_document_level(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, n=1)
    records = ingestion.load_etl_records(str(records_path))

    monkeypatch.setattr(ingestion, "INDEX_MAX_EMBED_CHARS", 20)
    composite = ingestion._build_etl_composite_text(records[0])

    assert composite.startswith("Doc 0")
    assert "Body text content" in composite
    assert composite.endswith("...")  # truncated to the bound
    assert len(composite) < 200


def test_build_documents_etl_fields(tmp_path):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, n=2, with_metadata=True)
    records = ingestion.load_etl_records(str(records_path))

    documents, texts = ingestion._build_documents(records, etl=True)
    assert len(documents) == 2
    assert len(texts) == 2
    doc = documents[0]
    assert doc["id"] == "etl-id-0"
    assert doc["document_name"] == "Doc 0"
    assert doc["file_text"].startswith("Body text")
    assert doc["industry"] == "Retail"
    assert doc["composite_text"] == texts[0]


def test_ingest_etl_uses_etl_source(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, n=2, with_metadata=True)

    stub = _StubClient()
    monkeypatch.setattr(ingestion, "ensure_index", lambda: None)
    monkeypatch.setattr(ingestion, "get_search_client", lambda: stub)
    monkeypatch.setattr(
        ingestion,
        "embed_texts",
        lambda texts: [[0.01] * VECTOR_DIMENSIONS for _ in texts],
    )

    assert ingestion.ingest_etl(str(records_path)) == 2
    assert stub.reset_calls == 1
    assert len(stub.docs) == 2
    assert stub.docs[0]["id"] == "etl-id-0"
    assert "composite_text" in stub.docs[0]
    assert "embedding" in stub.docs[0]