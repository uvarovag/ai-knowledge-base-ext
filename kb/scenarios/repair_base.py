"""Scenario 2: turn a poor knowledge base into a good one.

Reads the base named in the [input] table of the TOML config — an Excel
sheet with question and answer columns, such as one this project wrote — and:

1. repairs every entry (repairing): the answer holds the knowledge, so a
   vague question is rewritten to fit it, a partial answer completed from the
   question, both brought to the canonical format; only a row whose answer
   holds nothing to keep is left out;
2. collapses duplicates across the whole base (matching, dedup).

Nothing here asks whether an entry belongs in a base: somebody already put it
there. Writes the repaired base and the rows left out, with the reason, as
two Excel files to the output folder of the config. The technical state lives
in config.REPAIRS_DIR/<name>/, so runs with different configs, in parallel
terminals too, keep apart. The living bases of scenario 1 are not touched. An
interrupted run resumes from its staging directory.

Usage:
    make repair-base CONFIG=configs/<name>-repair.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import config
from kb.steps import dedup, matching, repairing
from kb.utils import batch, entries, excel, logs, settings, storage
from kb.utils.excel import SourcePair
from kb.utils.logs import logger
from kb.utils.settings import Domain


def repair_entry(
    llm: Any, domain: Domain, pair: SourcePair, source_file: str
) -> batch.Outcome:
    """Repair one entry of the input base and validate the result."""
    # A question can be rebuilt from an answer, but an answer cannot be made
    # up from a question alone: such a row is left out without a model call.
    if not pair.answer:
        return batch.Outcome(reason="empty_answer")

    repaired = repairing.run_repair(llm, domain, pair)
    if repaired is None:
        return batch.Outcome(reason="repair_failed")
    if not repaired["complete"]:
        return batch.Outcome(reason=f"incomplete: {repaired['reason']}")

    validation_error = entries.validate_entry(
        repaired,
        pair.question + "\n" + pair.answer,
        domain,
        config.REPAIR_MAX_ANSWER_WORDS,
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
    parser.add_argument("config", type=Path, help="TOML config of the repair")
    repair = settings.load(settings.RepairSettings, parser.parse_args().config)
    source = repair.input

    logs.configure_logging()
    if not source.path.exists():
        raise FileNotFoundError(f"Base to repair not found: {source.path}")
    staging_dir = storage.staging_dir_for(source.path, repair.staging_dir)
    pairs = excel.read_pairs(
        source.path, source.question_columns, source.answer_columns, source.extra_columns
    )
    logger.info("Found %d rows with a question or an answer", len(pairs))

    repaired, rejected = batch.process_rows(
        pairs,
        lambda llm, pair: repair_entry(llm, repair.domain, pair, source.path.name),
        staging_dir / "repaired.json",
        staging_dir / "rejected.json",
        f"Repair {source.path.name}",
        max_tokens=config.REPAIR_MAX_TOKENS,
    )

    cache_path = staging_dir / "verdicts.json"
    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    logger.info("Deduplicating %d repaired entries", len(repaired))
    result = dedup.collapse_duplicates(
        repair.domain,
        repaired,
        matching.find_candidate_pairs(repaired),
        dedup.VerdictCache(cache_path, repaired),
        "Repair dedup",
    )

    excel.write_knowledge_base(result, repair.output_file())
    excel.write_rejected(rejected, repair.output_file("_rejected"))
    logger.info(
        "Repaired base: %d of %d rows kept, %d left out, %d after deduplication",
        len(repaired),
        len(pairs),
        len(rejected),
        len(result),
    )
    logger.info("Results are in %s", repair.output_dir)


if __name__ == "__main__":
    main()
