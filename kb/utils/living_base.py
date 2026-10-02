"""The files of a living base: its JSON, its Excel files and its ledger.

People review and edit the Excel file of a base, not its JSON. So every
script that changes a base starts with sync_from_excel: the newest
<name>_<date>.xlsx in the output folder becomes the base, when it was saved
after the JSON, and a pair deleted there does not come back with the next
dump.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from kb.utils import excel, storage
from kb.utils.logs import logger
from kb.utils.settings import RunSettings


def latest_excel(base: RunSettings) -> Path | None:
    """The newest Excel file this project wrote for the base, by the date in its name."""
    pattern = re.compile(rf"{re.escape(base.name)}_(\d{{4}}-\d{{2}}-\d{{2}})\.xlsx")
    dated = [
        (match.group(1), path)
        for path in base.output_dir.glob("*.xlsx")
        if (match := pattern.fullmatch(path.name))
    ]
    return max(dated)[1] if dated else None


def canonical(entry: dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, sort_keys=True)


def sync_from_excel(base: RunSettings) -> list[dict[str, Any]]:
    """Make the newest Excel file of the base its JSON, and return the entries."""
    entries = storage.load_json(base.base_json)
    path = latest_excel(base)
    if path is None:
        return entries
    # Every run writes the Excel right after the JSON, so an Excel older than
    # the JSON was not edited since; reading it back would undo a run that
    # stopped before its export.
    if base.base_json.exists() and path.stat().st_mtime < base.base_json.stat().st_mtime:
        return entries
    edited = excel.read_knowledge_base(path)
    # The sheet is sorted by category and question, the JSON is not.
    if sorted(map(canonical, edited)) == sorted(map(canonical, entries)):
        return entries
    known = {storage.hash_entry(entry) for entry in entries}
    kept = {storage.hash_entry(entry) for entry in edited}
    logger.info(
        "Synced base %s from %s: %d entries, %d removed or edited there",
        base.name,
        path.name,
        len(edited),
        len(known - kept),
    )
    save(base, edited)
    return edited


def save(base: RunSettings, entries: list[dict[str, Any]]) -> None:
    """Write the base JSON, the previous version backed up first."""
    backup_path = storage.backup_file(base.base_json, base.backup_dir)
    if backup_path:
        logger.info("Previous base backed up to %s", backup_path)
    storage.save_json(base.base_json, entries)


def export(base: RunSettings, entries: list[dict[str, Any]]) -> None:
    """Write today's Excel file of the base."""
    if entries:
        excel.write_knowledge_base(entries, base.output_file())
    else:
        logger.info("Knowledge base %s is empty, nothing to export", base.name)


# ----- Ledger --------------------------------------------------------------


def is_merged(base: RunSettings, file_hash: str) -> bool:
    """Tell whether this very file — a dump or a base — is already in the base."""
    return any(
        record["hash"] == file_hash
        for record in storage.load_json(base.processed_dumps_json)
    )


def register(
    base: RunSettings, path: Path, file_hash: str, updated: int, added: int
) -> None:
    """Record a merged file in the ledger, replacing an earlier record of it."""
    ledger = [
        record
        for record in storage.load_json(base.processed_dumps_json)
        if record["hash"] != file_hash
    ]
    ledger.append(
        {
            "file": path.name,
            "path": str(path),
            "hash": file_hash,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "updated": updated,
            "added": added,
        }
    )
    storage.save_json(base.processed_dumps_json, ledger)
