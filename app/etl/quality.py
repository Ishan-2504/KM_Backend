"""PDF quality analysis - decides whether a PDF has a usable text layer.

Used to route "easy" PDFs to PyMuPDF and "difficult" ones (scanned,
garbled, layout-heavy) to Document AI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

# Control / replacement characters indicate a broken text layer (bad ToUnicode
# maps, malformed encodings).
_GARBLED_CHARS = frozenset(
    chr(c) for c in range(0x00, 0x20)
) | frozenset("\ufffd\u000b\u000c")


@dataclass
class QualityReport:
    page_count: int = 0
    total_chars: int = 0
    avg_chars_per_page: float = 0.0
    garbled_chars: int = 0
    garbled_ratio: float = 0.0
    alnum_chars: int = 0
    alnum_ratio: float = 0.0
    image_only_pages: int = 0
    image_only_ratio: float = 0.0
    has_text_layer: bool = False


def _garbled_count(text: str) -> int:
    return sum(1 for ch in text if ch in _GARBLED_CHARS)


def _image_only_page(page: "fitz.Page", min_chars: int) -> bool:
    text = page.get_text("text") or ""
    has_images = len(page.get_images(full=True)) > 0
    return has_images and len(text.strip()) < min_chars


def analyze_pdf(path: Path, min_chars_per_page: int = 100) -> QualityReport:
    """Inspect a PDF's pages and report text-layer health."""
    report = QualityReport()

    try:
        doc = fitz.open(str(path))
    except Exception:
        return report  # unparseable - caller will route to Document AI

    try:
        report.page_count = len(doc)
        if report.page_count == 0:
            return report

        text_chars = 0
        garbled = 0
        alnum = 0
        image_only = 0
        total = 0

        for page in doc.pages():
            text = page.get_text("text") or ""
            if _image_only_page(page, min_chars_per_page):
                image_only += 1
            n = len(text)
            total += n
            if n:
                text_chars += n
                garbled += _garbled_count(text)
                alnum += sum(1 for ch in text if ch.isalnum())

        report.total_chars = text_chars
        report.avg_chars_per_page = text_chars / max(report.page_count, 1)
        report.alnum_chars = alnum
        report.alnum_ratio = alnum / total if total else 0.0
        report.garbled_chars = garbled
        report.garbled_ratio = garbled / total if total else 0.0
        report.image_only_pages = image_only
        report.image_only_ratio = image_only / report.page_count
        report.has_text_layer = text_chars > 0
        return report
    finally:
        doc.close()