"""Scenario 1 maintenance: deduplicate the entries already in the living base.

The pipeline compares a new batch with itself and with the base, but never the
base with itself, so duplicates that slipped past the matching of earlier runs
stay until this script is run by hand. It runs the same model-adjudicated
grouping and two-step merge as dedup.py, over the whole base.

Candidates come from question embeddings and question trigrams; the category
is ignored (the model assigns it, and one question easily lands in two). With
config.EMBEDDING_BACKEND set to "none" the trigrams are the only signal and get
the low config.TRIGRAM_ONLY_CANDIDATE_THRESHOLD: this is a one-off clean-up,
so many model calls are acceptable where the pipeline could not afford them.

The base is backed up before anything is written. Verdicts are cached by
content in data/staging/base_dedupe/, so a run can be interrupted and resumed,
and a run over a base that changed since does not pick up stale verdicts.

Usage:
    make dedupe-base
"""

from __future__ import annotations

import config
from kb.steps import dedup, matching
from kb.utils import excel, logs, storage
from kb.utils.logs import logger

STAGING_DIR = config.STAGING_DIR / "base_dedupe"


def main() -> None:
    logs.configure_logging()

    entries = storage.load_json(config.KNOWLEDGE_BASE_JSON)
    if not entries:
        logger.info("Base is empty, nothing to deduplicate")
        return

    backup_path = storage.backup_file(config.KNOWLEDGE_BASE_JSON)
    logger.info("Backed up base to %s", backup_path)

    cache_path = STAGING_DIR / "verdicts.json"
    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()

    logger.info("Deduplicating %d base entries", len(entries))
    matrix = matching.embed_questions(entries, "Base dedup")
    by_embedding = matching.collect_embedding_pairs(matrix)
    trigram_threshold = (
        config.CANDIDATE_THRESHOLD
        if matrix is not None
        else config.TRIGRAM_ONLY_CANDIDATE_THRESHOLD
    )
    by_trigram = matching.collect_similar_pairs(
        entries, "question", trigram_threshold, trigram_threshold
    )
    candidates = sorted(by_embedding | by_trigram)
    logger.info(
        "Candidates: %d by embeddings, %d by question trigrams (threshold %.2f), "
        "%d in total",
        len(by_embedding),
        len(by_trigram),
        trigram_threshold,
        len(candidates),
    )

    result = dedup.collapse_duplicates(
        entries, candidates, dedup.VerdictCache(cache_path, entries), "Base dedup"
    )
    storage.save_json(config.KNOWLEDGE_BASE_JSON, result)
    excel.write_knowledge_base(
        result, config.KNOWLEDGE_BASE_XLSX, config.SOURCE_EXTRA_COLUMNS
    )


if __name__ == "__main__":
    main()
