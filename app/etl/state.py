"""State ledger: remembers per-file content hashes so unchanged files are
skipped on subsequent runs (incremental ETL)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

STATE_VERSION = 1


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class EtlState:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.files: dict[str, dict] = {}

    def load(self) -> "EtlState":
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data.get("version") == STATE_VERSION:
                    self.files = data.get("files", {})
            except (json.JSONDecodeError, OSError):
                self.files = {}
        return self

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": STATE_VERSION, "files": self.files}
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def unchanged(self, key: str, content_hash: str) -> bool:
        entry = self.files.get(key)
        return bool(entry and entry.get("hash") == content_hash)

    def record(self, key: str, content_hash: str, doc_id: str, extracted_at: str) -> None:
        self.files[key] = {
            "hash": content_hash,
            "doc_id": doc_id,
            "extracted_at": extracted_at,
        }