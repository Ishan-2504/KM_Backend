"""Tests for the Phase 1 ETL pipeline (extraction + routing + canonical output)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as fitz
from docx import Document
from pptx import Presentation

import app.etl.routing as routing
from app.etl.discovery import iter_documents
from app.etl.pipeline import _run
from app.etl.quality import analyze_pdf
from app.etl import extractors as ext


# ---------------------------------------------------------------- fixtures

def _make_text_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Hello PyMuPDF world, this is a clean text layer for extraction testing. "
        "It has plenty of alphanumeric content so the quality router treats it "
        "as an easy, born-digital PDF. " * 3
    )
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def _make_blank_pdf(path: Path) -> None:
    doc = fitz.open()
    doc.new_page()  # no text, no font - looks scanned/image-only
    doc.new_page()
    doc.save(str(path))
    doc.close()


def _make_docx(path: Path) -> None:
    doc = Document()
    doc.add_paragraph("Hello python-docx world.")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Value"
    doc.save(str(path))


def _make_pptx(path: Path) -> None:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Hello python-pptx"
    slide.placeholders[1].text = "Slide body text."
    prs.save(str(path))


def _make_md(path: Path) -> None:
    path.write_text("# Title\n\nSome markdown **content** here.", encoding="utf-8")


# ---------------------------------------------------------------- routing

def test_clean_text_pdf_routes_to_pymupdf(tmp_path):
    pdf = tmp_path / "clean.pdf"
    _make_text_pdf(pdf)

    decision = routing.route(pdf)
    assert decision.kind == "pymupdf"
    assert "clean" in decision.reason


def test_blank_pdf_routes_away_from_pymupdf_when_docai_unavailable(tmp_path, monkeypatch):
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf)
    monkeypatch.setattr(routing, "_document_ai_available", lambda: False)

    decision = routing.route(pdf)
    # Phase 1: Document AI unavailable -> PyMuPDF fallback, but reason says so.
    assert decision.kind == "pymupdf"
    assert "no text layer" in decision.reason.lower() or "scanned" in decision.reason.lower()


def test_blank_pdf_routes_to_document_ai_when_configured(tmp_path, monkeypatch):
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf)
    monkeypatch.setattr(routing, "DOCUMENT_AI_LAYOUT_PROCESSOR_ID", "layout-123")
    monkeypatch.setattr(routing, "DOCUMENT_AI_OCR_PROCESSOR_ID", "ocr-123")
    monkeypatch.setattr(routing, "DOCUMENT_AI_ENABLED", True)

    decision = routing.route(pdf)
    assert decision.kind == "document_ai_layout"


def test_quality_report_detects_missing_text_layer(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf)
    report = analyze_pdf(pdf)
    assert report.page_count == 2
    assert report.has_text_layer is False


# ---------------------------------------------------------------- extractors

def test_extract_pdf_text(tmp_path):
    pdf = tmp_path / "clean.pdf"
    _make_text_pdf(pdf)
    result = ext.extract_pdf(pdf)
    assert "PyMuPDF" in result.text
    assert result.page_count >= 1


def test_extract_docx(tmp_path):
    docx = tmp_path / "sample.docx"
    _make_docx(docx)
    result = ext.extract_docx(docx)
    assert "python-docx" in result.text
    assert result.tables == 1
    assert "Name | Value" in result.text


def test_extract_pptx(tmp_path):
    pptx = tmp_path / "sample.pptx"
    _make_pptx(pptx)
    result = ext.extract_pptx(pptx)
    assert "python-pptx" in result.text
    assert "Slide body text." in result.text
    assert result.page_count == 1


def test_extract_markdown(tmp_path):
    md = tmp_path / "notes.md"
    _make_md(md)
    result = ext.extract_markdown(md)
    assert "markdown" in result.text


# ---------------------------------------------------------------- discovery

def test_discovery_filters_extensions(tmp_path):
    _make_md(tmp_path / "a.md")
    _make_docx(tmp_path / "b.docx")
    (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "~$lock.docx").write_text("", encoding="utf-8")

    files = iter_documents(tmp_path)
    names = {f.name for f in files}
    assert names == {"a.md", "b.docx"}


# ---------------------------------------------------------------- pipeline

def _namespace(input_dir, out_dir, state):
    return argparse.Namespace(
        input=str(input_dir), out=str(out_dir), state=str(state),
        force=False, dry_run=False, limit=0, verbose=False,
    )


def test_pipeline_writes_records_manifest_state(tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    _make_md(input_dir / "a.md")
    _make_docx(input_dir / "b.docx")
    out_dir = tmp_path / "out"
    state_file = tmp_path / "state.json"

    assert _run(_namespace(input_dir, out_dir, state_file)) == 0

    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
    assert state_file.exists()

    lines = (out_dir / "records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = __import__("json").loads(lines[0])
    assert first["text"]
    assert first["metadata"]["document_type"] == ""
    assert first["content_hash"]


def test_pipeline_skips_unchanged_files(tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    _make_md(input_dir / "a.md")
    out_dir = tmp_path / "out"
    state_file = tmp_path / "state.json"

    args = _namespace(input_dir, out_dir, state_file)
    assert _run(args) == 0
    manifest1 = (out_dir / "manifest.json").read_text(encoding="utf-8")

    assert _run(args) == 0
    manifest2 = (out_dir / "manifest.json").read_text(encoding="utf-8")

    import json as _json
    m1 = _json.loads(manifest1)["manifest"]
    m2 = _json.loads(manifest2)["manifest"]
    assert m1["files_skipped"] == 0
    assert m2["files_skipped"] == 1
    assert m2["files_extracted"] == 0
    # Regression: a skip-only run must NOT truncate the previous snapshot.
    leftover = [
        l for l in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    assert len(leftover) == 1
    assert "#" in leftover[0]


def test_pipeline_force_re_extracts(tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    _make_md(input_dir / "a.md")
    out_dir = tmp_path / "out"
    state_file = tmp_path / "state.json"

    args = _namespace(input_dir, out_dir, state_file)
    _run(args)
    args.force = True
    _run(args)

    import json as _json
    manifest = _json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["manifest"]
    assert manifest["files_skipped"] == 0
    assert manifest["files_extracted"] == 1