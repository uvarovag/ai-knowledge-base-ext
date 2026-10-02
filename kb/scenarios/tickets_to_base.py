"""Scenario 1: grow a living knowledge base from a dump of support tickets.

The TOML config names the base and the dump. The base is the one in
config.BASES_DIR/<name>/ whatever the dump is called, so every new dump
updates the same base. A dump already in the ledger of the base (matched by
file content) is skipped. Otherwise:

1. filter and rewrite every ticket into an entry (filtering, rewriting);
2. collapse duplicates inside the batch (matching, dedup);
3. merge the batch into the base (merging).

The base JSON is the single source of truth; the Excel file written to the
output folder of the config is only a view of it. An interrupted run resumes
from the staging directory of the dump: processed rows are skipped and
verdicts already received are reused.

Usage:
    make run CONFIG=configs/<base>.toml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from kb.steps import dedup, filtering, matching, merging, rewriting
from kb.utils import batch, entries, excel, gigachat, logs, settings, storage
from kb.utils.excel import SourcePair
from kb.utils.logs import logger
from kb.utils.settings import Domain, TicketsSettings

# ----- One ticket ----------------------------------------------------------


def process_ticket(
    llm: Any, domain: Domain, pair: SourcePair, source_file: str
) -> batch.Outcome:
    """Filter one ticket, rewrite it and validate the result."""
    verdict = filtering.run_filter(llm, domain, pair)
    if verdict is None:
        return batch.Outcome(reason="filter_failed")

    topic = verdict.get("topic") if isinstance(verdict.get("topic"), str) else None
    had_private_data = verdict.get("no_private_data") is not True
    flags = filtering.failed_flags(verdict)
    if len(flags) > config.MAX_DOUBTFUL_FLAGS:
        return batch.Outcome(reason=f"filter:{','.join(flags)}", topic=topic)

    fields = rewriting.run_transform(llm, domain, pair)
    if fields is None:
        return batch.Outcome(reason="transform_failed", topic=topic)

    # The question counts as source too: "0 поставщиков" asked about may be
    # written as a digit in the answer that said "равно нулю". A ticket number
    # from the question is removed by the rewriting prompt, not by this check.
    validation_error = entries.validate_entry(
        fields, pair.question + "\n" + pair.answer, domain
    )
    if validation_error:
        return batch.Outcome(reason=validation_error, topic=topic, model_fields=fields)

    return batch.Outcome(
        entry=entries.new_entry(
            fields,
            source_file,
            pair.row_number,
            pair.source_columns,
            doubtful=filtering.doubt_reason(flags),
        ),
        topic=topic,
        had_private_data=had_private_data,
    )


# ----- The dump ------------------------------------------------------------


def extract_entries(
    base: TicketsSettings, staging_dir: Path
) -> list[dict[str, Any]]:
    """Turn every ticket of the dump with both a question and an answer into an entry."""
    dump = base.dump
    pairs = [
        pair
        for pair in excel.read_pairs(
            dump.path, dump.question_columns, dump.answer_columns, dump.extra_columns
        )
        if pair.question and pair.answer
    ]
    logger.info("Found %d rows with both question and answer", len(pairs))

    extracted, _ = batch.process_rows(
        pairs,
        lambda llm, pair: process_ticket(llm, base.domain, pair, dump.path.name),
        staging_dir / "extracted.json",
        staging_dir / "rejected.json",
        f"Extract {dump.path.name}",
    )
    return extracted


def process_dump(base: TicketsSettings, file_hash: str) -> None:
    """Extract, deduplicate and merge the dump, then record it in the ledger."""
    staging_dir = storage.staging_dir_for(base.dump.path, base.staging_dir)
    logger.info("=== %s -> base %s ===", base.dump.path.name, base.name)

    extracted = extract_entries(base, staging_dir)

    cache_path = staging_dir / "verdicts.json"
    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    logger.info("Deduplicating %d extracted entries", len(extracted))
    deduplicated = dedup.collapse_duplicates(
        base.domain,
        extracted,
        matching.find_candidate_pairs(extracted),
        dedup.VerdictCache(cache_path, extracted),
        "Dedup",
    )
    storage.save_json(staging_dir / "deduped.json", deduplicated)

    merged, updated, added = merging.merge_into_base(
        base.domain,
        base.merge_strategy,
        storage.load_json(base.base_json),
        deduplicated,
        staging_dir,
    )
    backup_path = storage.backup_file(base.base_json, base.backup_dir)
    if backup_path:
        logger.info("Previous base backed up to %s", backup_path)
    storage.save_json(base.base_json, merged)
    register_dump(base, file_hash, updated, added)


# ----- Ledger --------------------------------------------------------------


def is_processed(base: TicketsSettings, file_hash: str) -> bool:
    """Tell whether this very file has already been merged into the base."""
    return any(
        record["hash"] == file_hash
        for record in storage.load_json(base.processed_dumps_json)
    )


def register_dump(
    base: TicketsSettings, file_hash: str, updated: int, added: int
) -> None:
    """Record the processed dump in the ledger, replacing an earlier record of it."""
    ledger = [
        record
        for record in storage.load_json(base.processed_dumps_json)
        if record["hash"] != file_hash
    ]
    ledger.append(
        {
            "file": base.dump.path.name,
            "path": str(base.dump.path),
            "hash": file_hash,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "updated": updated,
            "added": added,
        }
    )
    storage.save_json(base.processed_dumps_json, ledger)


# ----- Entry point ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow a base from a dump of tickets.")
    parser.add_argument("config", type=Path, help="TOML config of the base")
    base = settings.load(settings.TicketsSettings, parser.parse_args().config)

    with logs.run(base.log_dir, "run"):
        gigachat.require_certificates()
        if not base.dump.path.exists():
            raise SystemExit(f"Dump not found: {base.dump.path}")

        file_hash = storage.hash_file(base.dump.path)
        if is_processed(base, file_hash) and not config.FORCE_REPROCESS:
            logger.info(
                "%s is already in base %s, nothing to process",
                base.dump.path.name,
                base.name,
            )
        else:
            process_dump(base, file_hash)

        entries_of_base = storage.load_json(base.base_json)
        if entries_of_base:
            excel.write_knowledge_base(entries_of_base, base.output_file())
        else:
            logger.info("Knowledge base %s is empty, nothing to export", base.name)


if __name__ == "__main__":
    main()
