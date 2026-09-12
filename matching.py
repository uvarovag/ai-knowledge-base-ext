"""Local similarity search used to pick candidate pairs before calling the model.

Two signals feed the candidate list. Similar questions are the obvious one.
Nearly identical answers are the second: support often pastes the same
instruction for what users describe as different problems, and without this
signal those entries never meet each other.

The category of an entry is produced by the model, so it is treated as a hint
rather than a hard filter: entries of different categories are still compared,
but they have to clear a higher similarity threshold.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import config

NON_WORD_PATTERN = re.compile(r"[^а-яёa-z0-9 ]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase the text and strip punctuation so trigrams stay comparable."""
    lowered = text.lower().replace("ё", "е")
    without_punctuation = NON_WORD_PATTERN.sub(" ", lowered)
    return WHITESPACE_PATTERN.sub(" ", without_punctuation).strip()


def build_trigrams(text: str) -> set[str]:
    """Return the set of character trigrams of a normalized text."""
    normalized = normalize(text)
    size = config.TRIGRAM_SIZE
    if len(normalized) <= size:
        return {normalized} if normalized else set()
    return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}


def build_postings(trigram_sets: list[set[str]]) -> dict[str, list[int]]:
    """Build an inverted index from a trigram to the entries containing it."""
    postings: dict[str, list[int]] = defaultdict(list)
    for index, trigrams in enumerate(trigram_sets):
        for trigram in trigrams:
            postings[trigram].append(index)
    return postings


def jaccard(first_size: int, second_size: int, shared: int) -> float:
    """Return the Jaccard similarity of two trigram sets from their sizes."""
    union = first_size + second_size - shared
    return shared / union if union else 0.0


def threshold_for(
    first: dict[str, Any],
    second: dict[str, Any],
    same_category_threshold: float,
    cross_category_threshold: float,
) -> float:
    """Return the similarity a pair has to reach, given their categories."""
    if first.get("category") == second.get("category"):
        return same_category_threshold
    return cross_category_threshold


def collect_similar_pairs(
    entries: list[dict[str, Any]],
    field: str,
    same_category_threshold: float,
    cross_category_threshold: float,
) -> set[tuple[int, int]]:
    """Find index pairs inside one list whose given field is similar enough."""
    trigram_sets = [build_trigrams(entry[field]) for entry in entries]
    postings = build_postings(trigram_sets)

    pairs: set[tuple[int, int]] = set()
    for first in range(len(entries)):
        shared_counts: Counter[int] = Counter()
        for trigram in trigram_sets[first]:
            for second in postings[trigram]:
                if second > first:
                    shared_counts[second] += 1

        for second, shared in shared_counts.items():
            threshold = threshold_for(
                entries[first],
                entries[second],
                same_category_threshold,
                cross_category_threshold,
            )
            similarity = jaccard(
                len(trigram_sets[first]), len(trigram_sets[second]), shared
            )
            if similarity >= threshold:
                pairs.add((first, second))

    return pairs


def collect_similar_matches(
    new_entries: list[dict[str, Any]],
    base_entries: list[dict[str, Any]],
    field: str,
    same_category_threshold: float,
    cross_category_threshold: float,
) -> set[tuple[int, int]]:
    """Find (new index, base index) pairs whose given field is similar enough."""
    base_trigram_sets = [build_trigrams(entry[field]) for entry in base_entries]
    postings = build_postings(base_trigram_sets)

    matches: set[tuple[int, int]] = set()
    for new_index, new_entry in enumerate(new_entries):
        new_trigrams = build_trigrams(new_entry[field])
        shared_counts: Counter[int] = Counter()
        for trigram in new_trigrams:
            for base_index in postings.get(trigram, ()):
                shared_counts[base_index] += 1

        for base_index, shared in shared_counts.items():
            threshold = threshold_for(
                new_entry,
                base_entries[base_index],
                same_category_threshold,
                cross_category_threshold,
            )
            similarity = jaccard(
                len(new_trigrams), len(base_trigram_sets[base_index]), shared
            )
            if similarity >= threshold:
                matches.add((new_index, base_index))

    return matches


def find_candidate_pairs(entries: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Find index pairs inside one batch worth checking with the model."""
    by_question = collect_similar_pairs(
        entries,
        "question",
        config.CANDIDATE_THRESHOLD,
        config.CANDIDATE_CROSS_CATEGORY_THRESHOLD,
    )
    by_answer = collect_similar_pairs(
        entries,
        "answer",
        config.ANSWER_CANDIDATE_THRESHOLD,
        config.ANSWER_CANDIDATE_THRESHOLD,
    )
    return sorted(by_question | by_answer)


def find_candidate_matches(
    new_entries: list[dict[str, Any]], base_entries: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Find (new index, base index) pairs worth checking against the base."""
    by_question = collect_similar_matches(
        new_entries,
        base_entries,
        "question",
        config.MATCH_THRESHOLD,
        config.MATCH_CROSS_CATEGORY_THRESHOLD,
    )
    by_answer = collect_similar_matches(
        new_entries,
        base_entries,
        "answer",
        config.ANSWER_MATCH_THRESHOLD,
        config.ANSWER_MATCH_THRESHOLD,
    )
    return sorted(by_question | by_answer)
