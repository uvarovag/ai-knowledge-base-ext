"""Resumable parallel processing of source rows into entries and rejections.

Both scenarios turn every row of a sheet into an entry or a rejection with one
or two model calls per row. This module owns what that takes besides the call
itself: running rows in parallel, saving progress so an interrupted run skips
the rows it already processed, and the final breakdown of rejection reasons.

A row that failed (the model never gave a usable reply) is not a verdict: it
is written to the rejections for the reviewer, but processed again on resume.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from kb.utils import gigachat, storage
from kb.utils.excel import SourcePair
from kb.utils.logs import ModelCall, ProgressBar, logger, recording_calls, render_item


@dataclass(slots=True)
class Outcome:
    """Result of one row: an entry, or the reason it was left out."""

    entry: dict[str, Any] | None = None
    reason: str | None = None
    topic: str | None = None
    had_private_data: bool = False
    # What the model wrote, for a row the code validation rejected: the
    # reviewer sees why, not only the reason code.
    model_fields: dict[str, Any] | None = None


Worker = Callable[[Any, SourcePair], Outcome]


def run_worker(
    worker: Worker, llm: Any, pair: SourcePair
) -> tuple[Outcome, list[ModelCall]]:
    """Run the worker on one row, with the model calls it made."""
    with recording_calls() as calls:
        try:
            outcome = worker(llm, pair)
        except Exception:
            logger.exception("Row %d: unhandled error", pair.row_number)
            outcome = Outcome(reason="unhandled_error")
    return outcome, calls


def is_failure(reason: str) -> bool:
    """Tell a row the model could not process from one it rejected."""
    return reason.endswith("_failed") or reason == "unhandled_error"


def log_summary(accepted: int, rejected: list[dict[str, Any]], total: int) -> None:
    """Print the counters and the breakdown of rejection reasons."""
    failed = sum(is_failure(record["reason"]) for record in rejected)
    logger.info(
        "Processed: %d accepted, %d rejected, %d failed, out of %d rows",
        accepted,
        len(rejected) - failed,
        failed,
        total,
    )
    reasons: dict[str, int] = {}
    for record in rejected:
        reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
        logger.info("  %-40s %d", reason, count)


def process_rows(
    pairs: list[SourcePair],
    worker: Worker,
    accepted_path: Path,
    rejected_path: Path,
    label: str,
    max_tokens: int = config.GIGACHAT_MAX_TOKENS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the worker over every row not processed yet; return entries and rejections.

    The worker gets a shared chat client, with max_tokens of output, and one
    row. Results are saved every config.SAVE_EVERY rows and at the end, sorted
    by source row. A row left out is printed above the progress bar with its
    model calls.
    """
    if config.FORCE_REPROCESS:
        entries: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
    else:
        entries = storage.load_json(accepted_path)
        rejected = [
            record
            for record in storage.load_json(rejected_path)
            if not is_failure(record.get("reason", "unknown"))
        ]
    processed_rows = {row for entry in entries for row in entry.get("source_rows", [])}
    processed_rows |= {record["source_row"] for record in rejected}

    pending = [pair for pair in pairs if pair.row_number not in processed_rows]
    if processed_rows:
        logger.info(
            "Resuming: %d rows already processed, %d left", len(processed_rows), len(pending)
        )

    def save() -> None:
        entries.sort(key=lambda entry: entry["source_rows"])
        rejected.sort(key=lambda record: record["source_row"])
        storage.save_json(accepted_path, entries)
        storage.save_json(rejected_path, rejected)

    private_data_seen = 0
    if pending:
        llm = gigachat.build_llm(max_tokens=max_tokens)
        bar = ProgressBar(len(pending), label)
        completed = 0
        # Only the workers run in parallel; every result is folded in here, on
        # the main thread, so the lists need no lock.
        with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
            futures = {
                executor.submit(run_worker, worker, llm, pair): pair for pair in pending
            }
            for future in as_completed(futures):
                pair = futures[future]
                outcome, calls = future.result()

                if outcome.entry is not None:
                    entries.append(outcome.entry)
                    private_data_seen += outcome.had_private_data
                    bar.advance(accepted=1)
                else:
                    record = {
                        "source_row": pair.row_number,
                        "reason": outcome.reason or "unknown",
                        "topic": outcome.topic,
                        "question": pair.question,
                        "answer": pair.answer,
                    }
                    if outcome.model_fields is not None:
                        record["model_question"] = outcome.model_fields.get("question")
                        record["model_answer"] = outcome.model_fields.get("answer")
                    rejected.append(record)
                    failed = is_failure(record["reason"])
                    bar.print(
                        render_item(
                            f"row {pair.row_number}", record["reason"], calls, failed
                        )
                    )
                    bar.advance(**{"failed" if failed else "rejected": 1})

                completed += 1
                if completed % config.SAVE_EVERY == 0:
                    save()
        bar.finish()
    save()

    log_summary(len(entries), rejected, len(pairs))
    if private_data_seen:
        logger.info(
            "  %d accepted answers contained private data before rewriting",
            private_data_seen,
        )
    return entries, rejected
