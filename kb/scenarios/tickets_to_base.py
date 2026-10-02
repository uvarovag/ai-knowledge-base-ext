"""Scenario 1: grow a living knowledge base from a dump of support tickets.

The TOML config names the base and the dump. The base is the one in
config.BASES_DIR/<name>/ whatever the dump is called, so the first dump
creates it and every next one updates it. The newest Excel file of the base,
with the reviewer's edits, becomes the base first (living_base). A dump
already in the ledger of the base (matched by file content) is skipped.
Otherwise:

1. filter and rewrite every ticket into an entry (filtering, rewriting);
2. collapse duplicates inside the batch (matching, dedup);
3. merge the batch into the base (merging).

The Excel file written to the output folder is what people review and edit;
the base JSON follows it. An interrupted run resumes
from the staging directory of the dump: processed rows are skipped and
verdicts already received are reused.

Usage:
    make run CONFIG=configs/<base>.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import config
from kb.steps import dedup, filtering, matching, merging, rewriting
from kb.utils import batch, entries, excel, gigachat, living_base, logs, settings, storage
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


def process_dump(
    base: TicketsSettings, entries_of_base: list[dict[str, Any]], file_hash: str
) -> list[dict[str, Any]]:
    """Extract, deduplicate and merge the dump, record it in the ledger, return the base."""
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
        base.domain, base.merge_strategy, entries_of_base, deduplicated, staging_dir
    )
    living_base.save(base, merged)
    living_base.register(base, base.dump.path, file_hash, updated, added)
    return merged


# ----- Entry point ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow a base from a dump of tickets.")
    parser.add_argument("config", type=Path, help="TOML config of the base")
    base = settings.load(settings.TicketsSettings, parser.parse_args().config)

    with logs.run(base.log_dir, "run"):
        gigachat.require_certificates()
        if not base.dump.path.exists():
            raise SystemExit(f"Dump not found: {base.dump.path}")

        entries_of_base = living_base.sync_from_excel(base)
        file_hash = storage.hash_file(base.dump.path)
        if living_base.is_merged(base, file_hash) and not config.FORCE_REPROCESS:
            logger.info(
                "%s is already in base %s, nothing to process",
                base.dump.path.name,
                base.name,
            )
        else:
            entries_of_base = process_dump(base, entries_of_base, file_hash)
        living_base.export(base, entries_of_base)


if __name__ == "__main__":
    main()
