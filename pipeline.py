"""Entry point of the knowledge base pipeline.

Processes every dump listed in config.DUMP_PATHS that is not in the ledger yet,
in the order given, then rebuilds the Excel view from the living base.

Per dump: extract candidates, collapse duplicates inside the batch, merge the
batch into the base. The base is the single source of truth; the workbook is
only a view of it.

An interrupted run resumes: extract skips rows it already processed and
deduplicate reuses the verdicts it already received, both from the staging
directory of that dump.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import common
import config
import deduplicate
import export
import extract
import merge
from common import logger


def find_pending_dumps() -> list[tuple[Path, str]]:
    """List dumps that still have to be processed, keeping the configured order."""
    ledger = common.load_json(config.PROCESSED_DUMPS_JSON)
    known_hashes = {record["hash"] for record in ledger}

    pending: list[tuple[Path, str]] = []
    for path in config.DUMP_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Dump not found: {path}")
        file_hash = common.hash_file(path)
        if config.FORCE_REPROCESS or file_hash not in known_hashes:
            pending.append((path, file_hash))
        else:
            logger.info("Skipping %s, already processed", path.name)
    return pending


def register_dump(path: Path, file_hash: str, updated: int, added: int) -> None:
    """Record a processed dump in the ledger, replacing an earlier record of it."""
    ledger = [
        record
        for record in common.load_json(config.PROCESSED_DUMPS_JSON)
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
    common.save_json(config.PROCESSED_DUMPS_JSON, ledger)


def process_dump(path: Path, file_hash: str) -> None:
    """Run the three per-dump steps and record the result in the ledger."""
    staging_dir = common.staging_dir_for(path)

    logger.info("=== %s ===", path.name)
    extracted_path = extract.run(path, staging_dir)
    deduped_path = deduplicate.run(extracted_path, staging_dir)
    updated, added = merge.run(deduped_path, staging_dir)

    register_dump(path, file_hash, updated, added)


def main() -> None:
    common.configure_logging()
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)

    pending = find_pending_dumps()
    if not pending:
        logger.info("No new dumps to process")
    else:
        logger.info("Processing %d dumps", len(pending))
        for path, file_hash in pending:
            process_dump(path, file_hash)

    export.run()


if __name__ == "__main__":
    main()
