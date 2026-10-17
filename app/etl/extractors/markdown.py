"""Markdown extractor - reads the file as UTF-8 text with minimal cleanup."""

from __future__ import annotations

from pathlib import Path

from app.etl.extractors.base import ExtractedDocument, ExtractorError


def extract_markdown(path: Path) -> ExtractedDocument:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise ExtractorError(f"Failed to read markdown file: {exc}") from exc

    warnings: list[str] = []
    if not text.strip():
        warnings.append("Markdown file is empty")
    return ExtractedDocument(
        text=text,
        pages=[],
        page_count=1,
        warnings=warnings,
        confidence=None,
        ocr_used=False,
    )