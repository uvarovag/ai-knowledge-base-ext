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

The TOML config names the base, under config.BASES_DIR/<name>/, the folder
the cleaned Excel view goes to and the domain the merge prompts are told
about; it has no source file. The base is backed up right before it is
written. Verdicts are cached by content in the staging directory of the base,
so a run can be interrupted and resumed, and a run over a base that changed
since does not pick up stale verdicts.

Usage:
    make dedupe-base CONFIG=configs/<base>-dedupe.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from kb.steps import dedup, matching
from kb.utils import excel, logs, settings, storage
from kb.utils.logs import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate a living base.")
    parser.add_argument("config", type=Path, help="TOML config of the clean-up")
    base = settings.load(settings.DedupeSettings, parser.parse_args().config)

    with logs.run(base.log_dir, "dedupe-base"):
        entries = storage.load_json(base.base_json)
        if not entries:
            logger.info(
                "Base %s is empty or missing (%s), nothing to deduplicate",
                base.name,
                base.base_json,
            )
            return

        cache_path = base.staging_dir / "base_dedupe" / "verdicts.json"
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
            base.domain,
            entries,
            candidates,
            dedup.VerdictCache(cache_path, entries),
            "Base dedup",
        )
        backup_path = storage.backup_file(base.base_json, base.backup_dir)
        logger.info("Previous base backed up to %s", backup_path)
        storage.save_json(base.base_json, result)
        excel.write_knowledge_base(result, base.output_file())


if __name__ == "__main__":
    main()
