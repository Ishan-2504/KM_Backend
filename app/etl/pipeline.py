"""ETL pipeline CLI.

Usage:
    python -m app.etl.pipeline --input docs --out app/db/etl

Walks `--input` for supported documents (pdf/docx/pptx/md), routes each to
the best extractor (PyMuPDF for easy PDFs; Document AI once configured for
scanned/complex ones), and writes canonical JSONL records + a run manifest.

The output `records.jsonl` is a *snapshot* of the input folder: unchanged
files keep their previous records, changed files are re-extracted, and
records for files that no longer exist are pruned.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from app.config import (
    DOCS_INPUT_DIR,
    ETL_OUTPUT_DIR,
    ETL_STATE_FILE,
)
from app.etl.discovery import iter_documents
from app.etl.extractors import AVAILABLE_EXTRACTORS
from app.etl.extractors.base import ExtractedDocument, ExtractorError
from app.etl.models import ETLRecord, RunManifest
from app.etl.normalize import build_record
from app.etl.routing import RoutingDecision, route
from app.etl.state import EtlState, file_sha256
from app.etl.writers import JsonlRecordWriter, new_manifest, write_manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="etl.pipeline",
        description="Extract text from a folder of documents into canonical records.",
    )
    parser.add_argument("--input", default=DOCS_INPUT_DIR, help="input folder (scanned recursively)")
    parser.add_argument("--out", default=ETL_OUTPUT_DIR, help="output dir for records.jsonl + manifest.json")
    parser.add_argument("--state", default=ETL_STATE_FILE, help="state ledger file for incremental runs")
    parser.add_argument("--force", action="store_true", help="re-extract even if unchanged")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be processed")
    parser.add_argument("--limit", type=int, default=0, help="stop after N files (0 = no limit)")
    parser.add_argument("--verbose", "-v", action="store_true", help="print per-file details")
    return parser.parse_args(argv)


def _resolve_decision(path: Path) -> RoutingDecision:
    """Pick the best extractor; fall back to PyMuPDF for PDFs when the
    Document AI route is unavailable (Phase 1 keeps the pipeline GCP-free)."""
    decision = route(path)
    if decision.kind.startswith("document_ai") and "document_ai" not in AVAILABLE_EXTRACTORS:
        return RoutingDecision(
            kind="pymupdf",
            reason=decision.reason,
            quality=decision.quality,
        )
    return decision


def _extract_one(path: Path, decision: RoutingDecision) -> tuple[ExtractedDocument, str, str | None]:
    """Run the chosen extractor; returns (extracted, extractor_used, fallback_reason)."""
    fallback_reason = None
    if decision.kind.startswith("document_ai"):
        # Document AI extractor arrives in Phase 3. Until then route to PyMuPDF
        # with a warning so the pipeline never hard-depends on GCP.
        fallback_reason = (
            "routed to Document AI but it is not configured in this build; "
            f"used PyMuPDF (route reason: {decision.reason})"
        )
        decision = RoutingDecision(kind="pymupdf", reason=decision.reason, quality=decision.quality)

    extractor = AVAILABLE_EXTRACTORS.get(decision.kind)
    if extractor is None:
        raise ExtractorError(f"No extractor available for route '{decision.kind}'")

    extracted = extractor(path)
    return extracted, decision.kind, fallback_reason


def _load_snapshot(records_path: Path) -> tuple[dict[str, ETLRecord], dict[str, str]]:
    """Load previous snapshot: {doc_id: record} and {source_path: doc_id}."""
    if not records_path.exists():
        return {}, {}
    records: dict[str, ETLRecord] = {}
    path_to_docid: dict[str, str] = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = ETLRecord.model_validate_json(line)
        records[rec.doc_id] = rec
        path_to_docid[rec.source_path] = rec.doc_id
    return records, path_to_docid


def _run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        return 2

    out_dir = Path(args.out)
    records_path = out_dir / "records.jsonl"
    manifest_path = out_dir / "manifest.json"

    # Never index our own output.
    state_path = Path(args.state)
    files = iter_documents(input_dir, exclusions=(out_dir, state_path))

    if not files:
        print("No supported documents found (pdf/docx/pptx/md).")
        return 0

    files = files[: args.limit] if args.limit > 0 else files
    print(f"{len(files)} file(s) to process.")
    if args.dry_run:
        for f in files:
            print(f"  {f}")
        return 0

    state = EtlState(state_path).load()
    manifest = new_manifest(input_dir)
    manifest.files_found = len(files)

    records, path_to_docid = _load_snapshot(records_path)

    for path in tqdm(files, desc="Extracting", unit="file"):
        key = path.resolve().as_posix()
        current_hash = file_sha256(path)

        if not args.force and state.unchanged(key, current_hash):
            manifest.files_skipped += 1
            doc_id = state.files.get(key, {}).get("doc_id") or path_to_docid.get(key)
            if doc_id:
                path_to_docid[key] = doc_id
            continue

        try:
            decision = _resolve_decision(path)
            extracted, extractor_used, fallback_reason = _extract_one(path, decision)
        except Exception as exc:
            manifest.files_failed += 1
            manifest.failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            if args.verbose:
                traceback.print_exc()
            continue

        record = build_record(
            source_path=path,
            input_root=input_dir,
            content_hash=current_hash,
            extracted=extracted,
            routing=RoutingDecision(kind=extractor_used, reason=decision.reason),
            extractor_used=extractor_used,
            fallback_reason=fallback_reason,
        )
        records[record.doc_id] = record
        path_to_docid[key] = record.doc_id
        state.record(key, current_hash, record.doc_id, record.extracted_at)
        manifest.files_extracted += 1
        manifest.total_chars += len(record.text)
        manifest.totals_per_extractor[record.extractor] = (
            manifest.totals_per_extractor.get(record.extractor, 0) + 1
        )
        manifest.totals_per_source_type[record.source_type] = (
            manifest.totals_per_source_type.get(record.source_type, 0) + 1
        )
        if args.verbose:
            print(f"  {path.name} -> {record.extractor} "
                  f"({len(record.text)} chars, routed to {decision.kind})")

    # Prune records whose source file is no longer present in the folder.
    current_docids = set(path_to_docid[key] for key in (p.resolve().as_posix() for p in files))
    pruned = [
        rid for rid in records if rid not in current_docids
    ]
    for rid in pruned:
        del records[rid]

    with JsonlRecordWriter(records_path) as writer:
        for rec in sorted(records.values(), key=lambda r: r.source_path):
            writer.write(rec)

    state.save()
    write_manifest(manifest_path, manifest)

    print("\nDone.")
    print(f"  found:      {manifest.files_found}")
    print(f"  extracted:  {manifest.files_extracted}")
    print(f"  skipped:    {manifest.files_skipped}")
    print(f"  failed:     {manifest.files_failed}")
    print(f"  chars:      {manifest.total_chars:,}")
    print(f"  records:    {len(records)}")
    print(f"  pruned:     {len(pruned)}")
    print(f"  records:    {records_path}")
    print(f"  manifest:   {manifest_path}")
    if manifest.failures:
        print("\nFailures:")
        for failure in manifest.failures[:20]:
            print(f"  - {failure['path']}: {failure['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(_run(_parse_args()))