"""Extractor contracts and shared helpers.

An extractor turns a single source file into an `ExtractedDocument`:
full text, optional per-page text, and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ExtractorError(Exception):
    """Raised when an extractor cannot process a file."""


@dataclass
class ExtractedDocument:
    """Normalized output of a single extractor."""

    text: str = ""
    pages: list[dict] = field(default_factory=list)  # [{"page": int, "text": str}]
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)
    confidence: float | None = None
    ocr_used: bool = False
    tables: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)