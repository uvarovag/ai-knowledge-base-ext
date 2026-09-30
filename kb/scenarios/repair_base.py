"""Scenario 2: turn a poor knowledge base into a good one.

Reads a base — an Excel sheet with question and answer columns, such as a
knowledge_base.xlsx this project wrote — given on the command line, or
config.REPAIR_INPUT_PATH when none is, and:

1. repairs every entry (repairing): the answer holds the knowledge, so a
   vague question is rewritten to fit it, a partial answer completed from the
   question, both brought to the canonical format; only a row whose answer
   holds nothing to keep is left out;
2. collapses duplicates across the whole base (matching, dedup).

Nothing here asks whether an entry belongs in a base: somebody already put it
there. Writes the repaired base and the rows left out, with the reason, to
config.REPAIRED_BASE_DIR/<input file stem>/, so runs on different files, in
parallel terminals too, keep their results apart. The living base of
scenario 1 is not touched. An interrupted run resumes from its staging
directory.

Usage:
    make repair-base [FILE="path/to/base.xlsx"]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import config
from kb.steps import dedup, matching, repairing
from kb.utils import batch, entries, excel, logs, storage
from kb.utils.excel import SourcePair
from kb.utils.logs import logger


def repair_entry(llm: Any, pair: SourcePair, source_file: str) -> batch.Outcome:
    """Repair one entry of the input base and validate the result."""
    # A question can be rebuilt from an answer, but an answer cannot be made
    # up from a question alone: such a row is left out without a model call.
    if not pair.answer:
        return batch.Outcome(reason="empty_answer")

    repaired = repairing.run_repair(llm, pair)
    if repaired is None:
        return batch.Outcome(reason="repair_failed")
    if not repaired["complete"]:
        return batch.Outcome(reason=f"incomplete: {repaired['reason']}")

    validation_error = entries.validate_entry(
        repaired, pair.question + "\n" + pair.answer, config.REPAIR_MAX_ANSWER_WORDS
    )
    if validation_error:
        return batch.Outcome(reason=validation_error, model_fields=repaired)

    return batch.Outcome(
        entry=entries.new_entry(
            repaired, source_file, pair.row_number, pair.source_columns
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair and deduplicate a base.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=config.REPAIR_INPUT_PATH,
        help="Excel file of the base to repair (default: config.REPAIR_INPUT_PATH)",
    )
    path: Path = parser.parse_args().path

    logs.configure_logging()
    if not path.exists():
        raise FileNotFoundError(
            f"Base to repair not found: {path} "
            "(pass FILE=... to make or set REPAIR_INPUT_PATH in config.py)"
        )
    staging_dir = storage.staging_dir_for(path, prefix="repair-")
    pairs = excel.read_pairs(
        path,
        config.REPAIR_QUESTION_COLUMNS,
        config.REPAIR_ANSWER_COLUMNS,
        config.REPAIR_EXTRA_COLUMNS,
    )
    logger.info("Found %d rows with a question or an answer", len(pairs))

    repaired, rejected = batch.process_rows(
        pairs,
        lambda llm, pair: repair_entry(llm, pair, path.name),
        staging_dir / "repaired.json",
        staging_dir / "rejected.json",
        f"Repair {path.name}",
        max_tokens=config.REPAIR_MAX_TOKENS,
    )

    cache_path = staging_dir / "verdicts.json"
    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    logger.info("Deduplicating %d repaired entries", len(repaired))
    result = dedup.collapse_duplicates(
        repaired,
        matching.find_candidate_pairs(repaired),
        dedup.VerdictCache(cache_path, repaired),
        "Repair dedup",
    )

    output_dir = config.REPAIRED_BASE_DIR / path.stem
    storage.save_json(output_dir / "knowledge_base.json", result)
    excel.write_knowledge_base(
        result, output_dir / "knowledge_base.xlsx", config.REPAIR_EXTRA_COLUMNS
    )
    excel.write_rejected(rejected, output_dir / "rejected.xlsx")
    logger.info(
        "Repaired base: %d of %d rows kept, %d left out, %d after deduplication",
        len(repaired),
        len(pairs),
        len(rejected),
        len(result),
    )
    logger.info("Results are in %s", output_dir)


if __name__ == "__main__":
    main()
