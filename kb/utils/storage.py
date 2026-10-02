"""Atomic JSON storage, backups, content hashes and staging directories."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

def load_json(path: Path) -> list[dict[str, Any]]:
    """Read a list of records from a UTF-8 JSON file. Missing file means empty list."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write a list of records to disk as UTF-8 JSON, atomically.

    The knowledge base is the source of truth and the staging files are what an
    interrupted run resumes from, so a crash in the middle of a write must not
    leave a truncated file behind: the data goes to a temporary file first and
    replaces the target in one step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # One temporary file per process: two runs in parallel terminals may save
    # the same shared file at once.
    temporary_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def backup_file(path: Path, backup_dir: Path) -> Path | None:
    """Copy a file into the backup directory with a timestamp in its name."""
    if not path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = backup_dir / f"{path.stem}-{stamp}{path.suffix}"
    shutil.copy2(path, destination)
    return destination


def today() -> str:
    """Return the current date as an ISO string, stored on every entry."""
    return datetime.now().date().isoformat()


# ----- Hashes and staging --------------------------------------------------


def hash_file(path: Path) -> str:
    """Return the SHA-256 of a file, used to detect a re-uploaded dump."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """Return the SHA-256 of a text: a cache key independent of an entry's position."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def hash_entry(entry: dict[str, Any]) -> str:
    """Hash an entry by its question and answer: a verdict depends on both."""
    return hash_text(entry["question"] + "\n" + entry["answer"])


def staging_dir_for(source_path: Path, staging_root: Path) -> Path:
    """Return the staging directory of a source file, creating it if needed.

    The hash is part of the name so that two files with the same name, or a
    corrected re-upload of one, do not share intermediate files.
    """
    name = f"{source_path.stem}-{hash_file(source_path)[:8]}"
    directory = staging_root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory
