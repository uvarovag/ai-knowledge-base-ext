"""Scenario 1: grow the living knowledge base from dumps of support tickets.

Processes every dump listed in config.DUMP_PATHS that is not in the ledger yet,
in the order given, then rebuilds the Excel view from the base. Per dump:

1. filter and rewrite every ticket into an entry (filtering, rewriting);
2. collapse duplicates inside the batch (matching, dedup);
3. merge the batch into the base (merging).

The base is the single source of truth; the workbook is only a view of it.
An interrupted run resumes from the staging directory of the dump: processed
rows are skipped and verdicts already received are reused.

Usage:
    make run
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import config
from kb.steps import dedup, filtering, matching, merging, rewriting
from kb.utils import batch, entries, excel, logs, storage
from kb.utils.excel import SourcePair
from kb.utils.logs import logger

# ----- One ticket ----------------------------------------------------------


def process_ticket(llm: Any, pair: SourcePair, source_file: str) -> batch.Outcome:
    """Filter one ticket, rewrite it and validate the result."""
    verdict = filtering.run_filter(llm, pair)
    if verdict is None:
        return batch.Outcome(reason="filter_failed")

    topic = verdict.get("topic") if isinstance(verdict.get("topic"), str) else None
    had_private_data = verdict.get("no_private_data") is not True
    flag = filtering.failed_flag(verdict)
    if flag is not None:
        return batch.Outcome(reason=f"filter:{flag}", topic=topic)

    fields = rewriting.run_transform(llm, pair)
    if fields is None:
        return batch.Outcome(reason="transform_failed", topic=topic)

    # The question counts as source too: "0 поставщиков" asked about may be
    # written as a digit in the answer that said "равно нулю". A ticket number
    # from the question is removed by the rewriting prompt, not by this check.
    validation_error = entries.validate_entry(
        fields, pair.question + "\n" + pair.answer
    )
    if validation_error:
        return batch.Outcome(reason=validation_error, topic=topic, model_fields=fields)

    return batch.Outcome(
        entry=entries.new_entry(
            fields, source_file, pair.row_number, pair.source_columns
        ),
        topic=topic,
        had_private_data=had_private_data,
    )


# ----- One dump ------------------------------------------------------------


def extract_entries(dump_path: Path, staging_dir: Path) -> list[dict[str, Any]]:
    """Turn every ticket of a dump with both a question and an answer into an entry."""
    pairs = [
        pair
        for pair in excel.read_pairs(
            dump_path,
            config.QUESTION_COLUMNS,
            config.ANSWER_COLUMNS,
            config.SOURCE_EXTRA_COLUMNS,
        )
        if pair.question and pair.answer
    ]
    logger.info("Found %d rows with both question and answer", len(pairs))

    extracted, _ = batch.process_rows(
        pairs,
        lambda llm, pair: process_ticket(llm, pair, dump_path.name),
        staging_dir / "extracted.json",
        staging_dir / "rejected.json",
        f"Extract {dump_path.name}",
    )
    return extracted


def process_dump(path: Path, file_hash: str) -> None:
    """Extract, deduplicate and merge one dump, then record it in the ledger."""
    staging_dir = storage.staging_dir_for(path)
    logger.info("=== %s ===", path.name)

    extracted = extract_entries(path, staging_dir)

    cache_path = staging_dir / "verdicts.json"
    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    logger.info("Deduplicating %d extracted entries", len(extracted))
    deduplicated = dedup.collapse_duplicates(
        extracted,
        matching.find_candidate_pairs(extracted),
        dedup.VerdictCache(cache_path, extracted),
        "Dedup",
    )
    storage.save_json(staging_dir / "deduped.json", deduplicated)

    updated, added = merging.merge_into_base(deduplicated, staging_dir)
    register_dump(path, file_hash, updated, added)


# ----- Ledger --------------------------------------------------------------


def find_pending_dumps() -> list[tuple[Path, str]]:
    """List dumps that still have to be processed, keeping the configured order."""
    known_hashes = {
        record["hash"] for record in storage.load_json(config.PROCESSED_DUMPS_JSON)
    }
    pending: list[tuple[Path, str]] = []
    for path in config.DUMP_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Dump not found: {path}")
        file_hash = storage.hash_file(path)
        if config.FORCE_REPROCESS or file_hash not in known_hashes:
            pending.append((path, file_hash))
        else:
            logger.info("Skipping %s, already processed", path.name)
    return pending


def register_dump(path: Path, file_hash: str, updated: int, added: int) -> None:
    """Record a processed dump in the ledger, replacing an earlier record of it."""
    ledger = [
        record
        for record in storage.load_json(config.PROCESSED_DUMPS_JSON)
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
    storage.save_json(config.PROCESSED_DUMPS_JSON, ledger)


# ----- Entry point ---------------------------------------------------------


def main() -> None:
    logs.configure_logging()
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)

    pending = find_pending_dumps()
    if not pending:
        logger.info("No new dumps to process")
    for path, file_hash in pending:
        process_dump(path, file_hash)

    base = storage.load_json(config.KNOWLEDGE_BASE_JSON)
    if base:
        excel.write_knowledge_base(
            base, config.KNOWLEDGE_BASE_XLSX, config.SOURCE_EXTRA_COLUMNS
        )
    else:
        logger.warning("Knowledge base is empty, nothing to export")


if __name__ == "__main__":
    main()
