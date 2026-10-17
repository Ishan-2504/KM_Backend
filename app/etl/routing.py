"""Source-type routing: pick the best extractor for each file.

One level of routing is static (file extension -> extractor family).
PDFs get a second, quality-based routing so scanned / complex documents
go to Document AI instead of PyMuPDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import (
    DOCUMENT_AI_ENABLED,
    DOCUMENT_AI_LAYOUT_PROCESSOR_ID,
    DOCUMENT_AI_OCR_PROCESSOR_ID,
    EXTRACT_MAX_GARBLED_RATIO,
    EXTRACT_MAX_IMAGE_ONLY_PAGE_RATIO,
    EXTRACT_MIN_ALNUM_RATIO,
    EXTRACT_MIN_CHARS_PER_PAGE,
)
from app.etl.quality import QualityReport, analyze_pdf

DOCUMENT_AI_OCR = "document_ai_ocr"
DOCUMENT_AI_LAYOUT = "document_ai_layout"

# Extensions the ETL understands.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md", ".markdown"}


@dataclass
class RoutingDecision:
    kind: str          # "pymupdf" | "docx" | "pptx" | "markdown" | "document_ai_*"
    reason: str
    quality: QualityReport | None = None


def _pick_document_ai_processor() -> tuple[str, str]:
    """Return the preferred Document AI extractor kind. Prefers the Layout
    Parser when configured (tables / structure), else Enterprise OCR."""
    if DOCUMENT_AI_LAYOUT_PROCESSOR_ID:
        return DOCUMENT_AI_LAYOUT, "layout-heavy or low-quality text layer"
    return DOCUMENT_AI_OCR, "scanned or low-quality text layer"


def _document_ai_available() -> bool:
    return DOCUMENT_AI_ENABLED and bool(
        DOCUMENT_AI_OCR_PROCESSOR_ID or DOCUMENT_AI_LAYOUT_PROCESSOR_ID
    )


def route(path: Path, min_chars_per_page: int = EXTRACT_MIN_CHARS_PER_PAGE) -> RoutingDecision:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return RoutingDecision(kind="unsupported", reason=f"unsupported extension '{suffix}'")

    if suffix == ".pdf":
        quality = analyze_pdf(path, min_chars_per_page)
        reasons: list[str] = []

        if quality.page_count == 0:
            reasons.append("PDF could not be read")
        elif not quality.has_text_layer:
            reasons.append("no text layer (scanned or image-only)")

        if quality.has_text_layer and quality.total_chars > 0:
            if quality.avg_chars_per_page < min_chars_per_page:
                reasons.append(
                    f"low text density ({quality.avg_chars_per_page:.0f} chars/page "
                    f"< {min_chars_per_page})"
                )
            if quality.garbled_ratio > EXTRACT_MAX_GARBLED_RATIO:
                reasons.append(
                    f"garbled chars ({quality.garbled_ratio:.1%} > {EXTRACT_MAX_GARBLED_RATIO:.0%})"
                )
            if quality.alnum_ratio < EXTRACT_MIN_ALNUM_RATIO:
                reasons.append(
                    f"low alphanumeric ratio ({quality.alnum_ratio:.1%} < {EXTRACT_MIN_ALNUM_RATIO:.0%})"
                )
            if quality.image_only_ratio >= EXTRACT_MAX_IMAGE_ONLY_PAGE_RATIO:
                reasons.append(
                    f"{quality.image_only_pages}/{quality.page_count} pages image-only"
                )

        if reasons:
            if _document_ai_available():
                kind, docai_reason = _pick_document_ai_processor()
                return RoutingDecision(kind=kind, reason="; ".join(reasons) or docai_reason, quality=quality)
            # Phase 1: Document AI not configured -> PyMuPDF with warning.
            return RoutingDecision(
                kind="pymupdf",
                reason=f"{'; '.join(reasons)} (Document AI not configured, using PyMuPDF)",
                quality=quality,
            )

        return RoutingDecision(kind="pymupdf", reason="clean text layer", quality=quality)

    if suffix in {".docx"}:
        return RoutingDecision(kind="docx", reason="Word document")
    if suffix in {".pptx"}:
        return RoutingDecision(kind="pptx", reason="PowerPoint presentation")
    return RoutingDecision(kind="markdown", reason="Markdown file")