"""Folder discovery: walk an input directory and yield supported source files."""

from __future__ import annotations

from pathlib import Path

from app.etl.routing import SUPPORTED_EXTENSIONS

# Office lock files and common temp artifacts to skip.
_IGNORED_PREFIXES = ("~$", "._", ".~", "~")
_IGNORED_SUFFIXES = (".tmp", ".crdownload", ".part")


def iter_documents(input_dir: Path, exclusions: tuple[Path, ...] = ()) -> list[Path]:
    """Return supported files under `input_dir`, recursively, sorted.

    `exclusions` lets callers skip nested output/state directories so
    the ETL never tries to index its own output.
    """
    excluded = {p.resolve() for p in exclusions}
    root = input_dir.resolve()
    found: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(path.is_relative_to(exc) for exc in excluded):
            continue
        if any(path.name.startswith(p) for p in _IGNORED_PREFIXES):
            continue
        if path.name.endswith(_IGNORED_SUFFIXES):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        found.append(path)

    return found