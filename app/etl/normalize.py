"""Normalize extractor output into canonical ETLRecords.

Metadata is intentionally left empty for Phase 1 - populating it from the
Excel workbook / manifest files is a follow-up step.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from app.etl.extractors.base import ExtractedDocument
from app.etl.models import ETLRecord, ExtractionInfo, PageText
from app.etl.routing import RoutingDecision
from app.models import METADATA_FIELDS


def _doc_id(relative_path: str) -> str:
    norm = relative_path.replace("\\", "/").lstrip("./")
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def build_record(
    *,
    source_path: Path,
    input_root: Path,
    content_hash: str,
    extracted: ExtractedDocument,
    routing: RoutingDecision,
    extractor_used: str,
    fallback_reason: str | None = None,
) -> ETLRecord:
    """Turn one extracted file into a canonical ETLRecord."""

    relative = source_path.relative_to(input_root.resolve()).as_posix()
    metadata = {field: "" for field in METADATA_FIELDS}

    warnings = list(extracted.warnings)
    if fallback_reason:
        warnings.insert(0, fallback_reason)

    extraction = ExtractionInfo(
        extractor=extractor_used,
        routed_to=routing.kind,
        route_reason=routing.reason,
        page_count=extracted.page_count,
        ocr_used=extracted.ocr_used,
        confidence=extracted.confidence,
        tables=extracted.tables,
        warnings=warnings,
    )

    return ETLRecord(
        doc_id=_doc_id(relative),
        document_name=source_path.stem or relative,
        source_path=source_path.resolve().as_posix(),
        source_type=source_path.suffix.lstrip(".").lower(),
        content_hash=content_hash,
        extractor=extractor_used,
        text=extracted.text,
        pages=[PageText(page=p["page"], text=p["text"]) for p in extracted.pages],
        extraction=extraction,
        metadata=metadata,
        summary="",
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )