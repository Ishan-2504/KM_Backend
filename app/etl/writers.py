"""Output writers: JSONL records + run manifest."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.etl.models import ETLRecord, RunManifest


class JsonlRecordWriter:
    """Line-delimited JSON writer; one ETLRecord per line."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w", encoding="utf-8")

    def write(self, record: ETLRecord) -> None:
        self._f.write(record.model_dump_json() + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlRecordWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def new_manifest(input_dir: Path, run_id: str | None = None) -> RunManifest:
    return RunManifest(
        run_id=run_id or uuid.uuid4().hex[:12],
        started_at=datetime.now(timezone.utc).isoformat(),
        finished_at="",
        input_dir=str(input_dir.resolve()),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    manifest.finished_at = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"manifest": manifest.model_dump(), "schema_version": 1},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )