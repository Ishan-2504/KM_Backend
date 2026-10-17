"""Extractor implementations."""

from app.etl.extractors.base import ExtractedDocument, ExtractorError
from app.etl.extractors.docx import extract_docx
from app.etl.extractors.markdown import extract_markdown
from app.etl.extractors.pdf import extract_pdf
from app.etl.extractors.pptx import extract_pptx

# Registry of extractors available in this build. Phase 3 adds Document AI.
AVAILABLE_EXTRACTORS: dict[str, object] = {
    "pymupdf": extract_pdf,
    "docx": extract_docx,
    "pptx": extract_pptx,
    "markdown": extract_markdown,
}

EXTRACTOR_BY_SOURCE_TYPE: dict[str, list[str]] = {
    "pdf": ["pymupdf"],
    "docx": ["docx"],
    "pptx": ["pptx"],
    "md": ["markdown"],
    "markdown": ["markdown"],
}

__all__ = [
    "AVAILABLE_EXTRACTORS",
    "EXTRACTOR_BY_SOURCE_TYPE",
    "ExtractedDocument",
    "ExtractorError",
    "extract_pdf",
    "extract_docx",
    "extract_pptx",
    "extract_markdown",
]