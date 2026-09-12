"""One-off deduplication of the entries already sitting in the living base.

deduplicate.py only collapses duplicates inside a single freshly extracted
batch, before it is merged into the base; entries that ended up in
knowledge_base.json through different dumps are never compared against each
other. This script runs the same model-adjudicated grouping as
deduplicate.run(), but over the current base, using ONLY question similarity
to find candidate pairs (answer-similarity candidates are skipped here since
the goal right now is collapsing near-identical questions, not catching
shared boilerplate answers).

The base is backed up before anything is written. Verdicts are cached in
data/staging/base_dedupe/, so an interrupted run can be resumed by running
this script again.

Usage:
    python dedupe_base.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import common
import config
import deduplicate
import export
import matching
from common import logger


def run() -> None:
    common.configure_logging()

    entries = common.load_json(config.KNOWLEDGE_BASE_JSON)
    if not entries:
        logger.info("Base is empty, nothing to deduplicate")
        return

    backup_path = common.backup_file(config.KNOWLEDGE_BASE_JSON)
    logger.info("Backed up base to %s", backup_path)

    staging_dir = config.STAGING_DIR / "base_dedupe"
    staging_dir.mkdir(parents=True, exist_ok=True)
    cache_path = staging_dir / "verdicts.json"

    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    cache = deduplicate.VerdictCache(cache_path)

    logger.info("Deduplicating %d base entries by question only", len(entries))
    candidates = sorted(
        matching.collect_similar_pairs(
            entries,
            "question",
            config.CANDIDATE_THRESHOLD,
            config.CANDIDATE_CROSS_CATEGORY_THRESHOLD,
        )
    )
    logger.info("Found %d candidate pairs to check with the model", len(candidates))

    llm = common.build_llm()
    relations: dict[tuple[int, int], str] = {}
    alternatives_count = 0
    contradiction_count = 0

    if candidates:
        relations, alternatives_count, contradiction_count = deduplicate.judge_pairs(
            llm, entries, candidates, cache, "Base dedup: comparing pairs"
        )

        missing = deduplicate.find_missing_pairs(entries, relations)
        if missing:
            logger.info("Checking %d indirectly confirmed pairs", len(missing))
            extra_relations, extra_alternatives, extra_contradictions = (
                deduplicate.judge_pairs(
                    llm, entries, missing, cache, "Base dedup: checking indirect pairs"
                )
            )
            relations.update(extra_relations)
            alternatives_count += extra_alternatives
            contradiction_count += extra_contradictions

    logger.info(
        "Model confirmed %d duplicate pairs (%d complementary), %d contradicting pairs",
        len(relations),
        alternatives_count,
        contradiction_count,
    )

    groups = deduplicate.build_groups(entries, relations)
    duplicate_groups = [group for group in groups if len(group.members) > 1]
    logger.info("Merging %d groups", len(duplicate_groups))

    merged_by_group: dict[int, list[dict]] = {}
    if duplicate_groups:
        bar = common.ProgressBar(len(duplicate_groups), "Base dedup: merging groups")
        with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
            futures = {
                executor.submit(
                    deduplicate.merge_group, llm, entries, group
                ): group.members[0]
                for group in duplicate_groups
            }
            for future in as_completed(futures):
                merged_by_group[futures[future]] = future.result()
                bar.advance(merged=1)
        bar.finish()

    result = []
    for group in groups:
        if len(group.members) == 1:
            result.append(entries[group.members[0]])
        else:
            result.extend(merged_by_group[group.members[0]])

    common.save_json(config.KNOWLEDGE_BASE_JSON, result)
    logger.info("Collapsed %d entries into %d", len(entries), len(result))

    export.run()


if __name__ == "__main__":
    run()
