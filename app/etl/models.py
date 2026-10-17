"""Canonical data models for the ETL pipeline (folder -> extracted text records)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractionInfo(BaseModel):
    """Metadata about how a document was extracted."""

    extractor: str = ""            # extractor actually used (e.g. "pymupdf")
    routed_to: str = ""            # preferred extractor from routing (e.g. "document_ai_ocr")
    route_reason: str = ""         # why routing picked `routed_to`
    page_count: int = 0
    ocr_used: bool = False
    confidence: float | None = None
    tables: int = 0
    warnings: list[str] = Field(default_factory=list)


class PageText(BaseModel):
    """Text of a single page/slide/section with its 1-based index."""

    page: int
    text: str


class ETLRecord(BaseModel):
    """A single document after extraction - the handoff contract to indexing."""

    doc_id: str
    document_name: str
    source_path: str
    source_type: str                 # pdf | docx | pptx | md ...
    content_hash: str                # sha256 of the source file
    extractor: str = ""
    text: str = ""
    pages: list[PageText] = Field(default_factory=list)
    extraction: ExtractionInfo = Field(default_factory=ExtractionInfo)
    metadata: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    extracted_at: str = ""


class RunManifest(BaseModel):
    """Summary of a full ETL run, written alongside the JSONL records."""

    run_id: str
    started_at: str
    finished_at: str
    input_dir: str
    files_found: int = 0
    files_skipped: int = 0
    files_extracted: int = 0
    files_failed: int = 0
    total_chars: int = 0
    totals_per_extractor: dict[str, int] = Field(default_factory=dict)
    totals_per_source_type: dict[str, int] = Field(default_factory=dict)
    failures: list[dict] = Field(default_factory=list)