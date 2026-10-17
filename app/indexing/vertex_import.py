"""Vertex AI Search import job.

Turns canonical ETL records into Discovery Engine import payloads (JSONL
with `structData` metadata + inline extracted text) and:
  - writes the local JSONL staging file (no GCP required)
  - uploads it to GCS and starts an ImportDocuments operation (GCP required)

The search backend reads this metadata back as `structData`, which powers
the same filter facets used by the FAISS path.

Usage:
    python -m app.indexing.vertex_import --records app/db/etl/records.jsonl
        --gcs gs://BUCKET/km-import.jsonl [--engine ID | --data-store ID]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from tqdm import tqdm

from app.config import (
    ETL_RECORDS_FILE,
    GCP_LOCATION,
    GCP_PROJECT_ID,
    GCP_STAGING_BUCKET,
    VERTEX_COLLECTION,
    VERTEX_DATASTORE_ID,
    VERTEX_SEARCH_ENGINE_ID,
)
from app.etl.models import ETLRecord

TEXT_PLAIN = "text/plain"


def load_records(records_path: str | None = None) -> list[ETLRecord]:
    path = Path(records_path or ETL_RECORDS_FILE)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    if not path.exists():
        raise FileNotFoundError(
            f"ETL records file not found: {path}. Run the ETL pipeline first."
        )
    records: list[ETLRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(ETLRecord.model_validate_json(line))
    return records


def prepare_vertex_records(records: list[ETLRecord]) -> list[dict]:
    """Map canonical records to Discovery Engine import lines.

    Metadata goes into `structData` so filters/facets work on Vertex the
    same way they do on FAISS. Extracted text is embedded inline as
    text/plain so Vertex indexes our text directly (no re-parsing).
    """
    lines: list[dict] = []
    for rec in records:
        struct_data = dict(rec.metadata) or {}
        struct_data.update(
            {
                "document_name": rec.document_name,
                "source_path": rec.source_path,
                "source_type": rec.source_type,
                "content_hash": rec.content_hash,
            }
        )
        lines.append(
            {
                "id": rec.doc_id,
                "structData": struct_data,
                "content": {
                    "mimeType": TEXT_PLAIN,
                    "data": base64.b64encode(rec.text.encode("utf-8")).decode("ascii"),
                },
            }
        )
    return lines


def write_staging_jsonl(path: Path, lines: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
    path.write_text(payload, encoding="utf-8")
    return path


# ------------------------------------------------------------------ GCP path

def run_gcs_import(
    staging_path: Path,
    gcs_uri: str,
    *,
    data_store_id: str | None = None,
    engine_id: str | None = None,
) -> str:
    """Upload JSONL to GCS and start an ImportDocuments operation.

    Returns the long-running operation name. Requires GCP credentials.
    """
    from google.cloud import discoveryengine_v1 as de
    from google.cloud import storage

    # 1) Upload staging file.
    bucket_name, blob_name = _split_gcs_uri(gcs_uri)
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(
        str(staging_path)
    )

    # 2) Build parent (engine preferred, else data store).
    ds = data_store_id or VERTEX_DATASTORE_ID
    eng = engine_id or VERTEX_SEARCH_ENGINE_ID
    location = "global" if not GCP_LOCATION else GCP_LOCATION
    if not GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is not configured")
    if eng:
        parent = (
            f"projects/{GCP_PROJECT_ID}/locations/{location}"
            f"/collections/{VERTEX_COLLECTION}/engines/{eng}"
        )
    elif ds:
        parent = (
            f"projects/{GCP_PROJECT_ID}/locations/{location}"
            f"/collections/{VERTEX_COLLECTION}/dataStores/{ds}"
        )
    else:
        raise RuntimeError(
            "No Vertex data store/engine configured: set VERTEX_SEARCH_ENGINE_ID "
            "or VERTEX_DATASTORE_ID."
        )

    # 3) Start the import operation.
    client = de.DocumentServiceClient()
    request = de.ImportDocumentsRequest(
        parent=parent,
        input_config=de.ImportDocumentsRequest.InputConfig(
            gcs_source=de.GcsSource(input_uris=[gcs_uri], data_schema="document"),
        ),
    )
    operation = client.import_documents(request)
    return operation.operation.name


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[len("gs://"):]
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {uri}")
    return bucket, blob


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vertex_import")
    parser.add_argument("--records", default=ETL_RECORDS_FILE, help="ETL records.jsonl")
    parser.add_argument("--staging", default="./app/db/etl/vertex_import.jsonl")
    parser.add_argument(
        "--gcs", default=None,
        help="gs:// staging URI to upload before importing (defaults to the "
        f"configured bucket: gs://{GCP_STAGING_BUCKET}/vertex_import.jsonl)",
    )
    parser.add_argument("--engine", default=None)
    parser.add_argument("--data-store", default=None)
    parser.add_argument("--prep-only", action="store_true", help="only write the staging JSONL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_records(args.records)
    if not records:
        print("No records loaded - nothing to import.")
        return 1

    print(f"Preparing {len(records)} records for Vertex AI Search...")
    lines = prepare_vertex_records(records)
    staging = write_staging_jsonl(Path(args.staging), lines)
    print(f"Staging JSONL written: {staging} ({len(lines)} documents)")

    if args.prep_only:
        return 0

    gcs_uri = args.gcs or f"gs://{GCP_STAGING_BUCKET}/vertex_import.jsonl"
    if "://" not in gcs_uri or not gcs_uri.startswith("gs://"):
        print(f"ERROR: invalid --gcs URI: {gcs_uri}. Use --prep-only to skip upload.")
        return 2

    print(f"Uploading to {gcs_uri} and starting import...")
    op_name = run_gcs_import(
        staging, gcs_uri, data_store_id=args.data_store, engine_id=args.engine,
    )
    print(f"Import started (operation): {op_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())