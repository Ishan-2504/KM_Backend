"""Offline ingestion: read Excel, embed composite text with Gemini, write FAISS + metadata."""

import hashlib
import sys
from pathlib import Path

from openpyxl import load_workbook
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import DATA_FILE_PATH
from app.db.search_client import ensure_index, get_search_client
from app.models import COLUMN_MAP, DOC_TYPE_MAP, METADATA_FIELDS, NULL_MARKERS
from app.services.embeddings import BATCH_SIZE, embed_texts

UPLOAD_CHUNK = 100


def _normalise_value(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s in NULL_MARKERS:
        return None
    return s


def _doc_id(name: str, idx: int) -> str:
    return hashlib.md5(f"{name}_{idx}".encode()).hexdigest()


def load_records(file_path: str | None = None) -> list[dict]:
    path = Path(file_path or DATA_FILE_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    wb = load_workbook(str(path), read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    headers = rows[0]
    records: list[dict] = []
    for row in rows[1:]:
        raw = {str(headers[i]): row[i] for i in range(len(headers))}
        record: dict = {}
        for src_col, dst_col in COLUMN_MAP.items():
            record[dst_col] = _normalise_value(raw.get(src_col))

        if record.get("document_type") in DOC_TYPE_MAP:
            record["document_type"] = DOC_TYPE_MAP[record["document_type"]]

        records.append(record)

    return records


def _build_composite_text(rec: dict) -> str:
    """Build a search-optimised composite string for embedding.

    Bios embed name, designation, and summary so role / skill queries match.
    KL docs embed name, industry, knowledge type, solution, and summary
    so structured metadata signals are captured in the vector.
    """
    doc_type = rec.get("document_type", "")
    if doc_type == "bio":
        parts = [
            rec.get("document_name", ""),
            rec.get("designation", ""),
            rec.get("summary", ""),
        ]
    else:
        parts = [
            rec.get("document_name", ""),
            rec.get("industry", ""),
            rec.get("sub_industry", ""),
            rec.get("knowledge_type", ""),
            rec.get("solution", ""),
            rec.get("summary", ""),
        ]
    return " | ".join(p for p in parts if p)


def ingest(file_path: str | None = None) -> int:
    ensure_index()
    records = load_records(file_path)
    client = get_search_client()
    client.reset()

    documents: list[dict] = []
    texts_to_embed: list[str] = []

    for idx, rec in enumerate(records):
        doc_id = _doc_id(rec.get("document_name", ""), idx)
        summary = rec.get("summary") or rec.get("document_name", "")
        composite = _build_composite_text(rec)

        doc = {
            "id": doc_id,
            "document_name": rec.get("document_name") or "",
            "summary": summary,
            "composite_text": composite,
            "file_text": rec.get("file_text") or "",
        }
        for field in METADATA_FIELDS:
            doc[field] = rec.get(field) or ""

        documents.append(doc)
        texts_to_embed.append(composite)

    print(f"Embedding {len(texts_to_embed)} documents in batches of {BATCH_SIZE}...")
    all_embeddings: list[list[float]] = []
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding"):
        batch = texts_to_embed[i : i + BATCH_SIZE]
        batch_emb = embed_texts(batch)
        all_embeddings.extend(batch_emb)

    for i, doc in enumerate(documents):
        doc["embedding"] = all_embeddings[i]

    print(f"Uploading {len(documents)} documents to FAISS...")
    for i in tqdm(range(0, len(documents), UPLOAD_CHUNK), desc="Uploading"):
        chunk = documents[i : i + UPLOAD_CHUNK]
        client.upload_documents(chunk)

    count = client.get_document_count()
    print(f"Ingestion complete. Index now has {count} documents.")
    return count


if __name__ == "__main__":
    ingest()
