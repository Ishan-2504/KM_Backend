"""PyMuPDF extractor for text-based PDFs.

Used for "easy" PDFs that have a usable text layer. Documents routed to
Document AI (scanned / garbled / layout-heavy) still fall back here when
the Document AI path is not configured (Phase 1 keeps the pipeline
GCP-free).
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from app.etl.extractors.base import ExtractedDocument, ExtractorError

PAGE_TEXT_LIMIT = 50_000  # hard cap per page to bound memory


def _page_text(page: "fitz.Page") -> str:
    text = page.get_text("text") or ""
    if len(text) > PAGE_TEXT_LIMIT:
        text = text[:PAGE_TEXT_LIMIT]
    return text


def extract_pdf(path: Path) -> ExtractedDocument:
    """Extract text from a PDF. Image-only pages yield empty text."""
    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # corrupt / unsupported file
        raise ExtractorError(f"PyMuPDF failed to open PDF: {exc}") from exc

    try:
        chunks: list[str] = []
        pages: list[dict] = []
        empty_pages = 0
        for i, page in enumerate(doc.pages(), start=1):
            text = _page_text(page)
            pages.append({"page": i, "text": text})
            if text.strip():
                chunks.append(text)
            else:
                empty_pages += 1

        warnings: list[str] = []
        if empty_pages and len(doc) > 0:
            warnings.append(
                f"{empty_pages}/{len(doc)} pages yielded no text "
                "(likely scanned or image-only)"
            )

        return ExtractedDocument(
            text="\n\n".join(chunks),
            pages=pages,
            page_count=len(doc),
            warnings=warnings,
            confidence=None,
            ocr_used=False,
        )
    finally:
        doc.close()