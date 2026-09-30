"""Local similarity search used to pick candidate pairs before calling the model.

Three signals feed the candidate list, united. Cosine similarity of question
embeddings is the one that sees paraphrases: "не приходит пароль" and "как
получить пароль после регистрации" share few letters but one meaning. Trigram
similarity of questions is cheap and still catches pairs with rare shared
terms. Nearly identical answers are the third: support often pastes the same
instruction for what users describe as different problems, and without this
signal those entries never meet each other.

Embeddings are cached on disk by the hash of the question text, so the base is
embedded once and every later run only pays for the new questions. With
config.EMBEDDING_BACKEND set to "none" the trigram signals are the only ones.

The category of an entry is produced by the model, so it is treated as a hint
rather than a hard filter: entries of different categories are still compared
by trigrams, but they have to clear a higher threshold. Embeddings ignore the
category altogether.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

import common
import config
from common import logger

NON_WORD_PATTERN = re.compile(r"[^а-яёa-z0-9 ]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


# ----- Trigrams ------------------------------------------------------------


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


# ----- Embeddings ----------------------------------------------------------


def load_embedding_cache() -> dict[str, np.ndarray]:
    """Read the embeddings cache: question hash -> vector."""
    if not config.EMBEDDINGS_CACHE.exists():
        return {}
    with np.load(config.EMBEDDINGS_CACHE) as archive:
        return {key: archive[key] for key in archive.files}


def save_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    """Write the embeddings cache atomically, like common.save_json."""
    path = config.EMBEDDINGS_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.npz")
    np.savez(temporary_path, **cache)
    temporary_path.replace(path)


def embed_questions(entries: list[dict[str, Any]], label: str) -> np.ndarray | None:
    """Return one normalized question embedding per entry, or None without a backend.

    Only questions missing from the cache are sent to the model; the cache is
    written after every batch, so an interrupted run resumes where it stopped.
    """
    if config.EMBEDDING_BACKEND == "none" or not entries:
        return None

    cache = load_embedding_cache()
    hashes = [common.hash_text(entry["question"]) for entry in entries]
    pending = {
        key: entry["question"]
        for entry, key in zip(entries, hashes)
        if key not in cache
    }

    if pending:
        embedder = common.build_embedder()
        keys = list(pending)
        batches = [
            keys[start : start + config.EMBEDDING_BATCH_SIZE]
            for start in range(0, len(keys), config.EMBEDDING_BATCH_SIZE)
        ]
        logger.info(
            "%s: embedding %d questions in %d batches (%d already cached)",
            label,
            len(keys),
            len(batches),
            len(entries) - len(keys),
        )
        bar = common.ProgressBar(len(batches), f"{label}: embedding questions")
        for batch in batches:
            vectors = common.embed_texts(embedder, [pending[key] for key in batch])
            for key, vector in zip(batch, vectors):
                cache[key] = np.asarray(vector, dtype=np.float32)
            save_embedding_cache(cache)
            bar.advance(embedded=len(batch))
        bar.finish()

    matrix = np.stack([cache[key] for key in hashes]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def nearest_neighbours(similarity: np.ndarray) -> set[tuple[int, int]]:
    """Return (row, column) pairs of the closest columns of every row.

    At most config.EMBEDDING_TOP_K neighbours per row, each at least
    config.EMBEDDING_CANDIDATE_THRESHOLD similar. The cap keeps a generic
    question from pulling half of the base into the candidate list.
    """
    pairs: set[tuple[int, int]] = set()
    top_k = min(config.EMBEDDING_TOP_K, similarity.shape[1])
    if top_k <= 0:
        return pairs

    for row_index in range(similarity.shape[0]):
        row = similarity[row_index]
        nearest = np.argpartition(-row, top_k - 1)[:top_k]
        for column_index in nearest:
            if row[column_index] >= config.EMBEDDING_CANDIDATE_THRESHOLD:
                pairs.add((row_index, int(column_index)))
    return pairs


def collect_embedding_pairs(matrix: np.ndarray | None) -> set[tuple[int, int]]:
    """Find index pairs inside one list whose question embeddings are close."""
    if matrix is None or matrix.shape[0] < 2:
        return set()
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    return {
        (min(first, second), max(first, second))
        for first, second in nearest_neighbours(similarity)
    }


def collect_embedding_matches(
    new_matrix: np.ndarray | None, base_matrix: np.ndarray | None
) -> set[tuple[int, int]]:
    """Find (new index, base index) pairs whose question embeddings are close."""
    if new_matrix is None or base_matrix is None:
        return set()
    return nearest_neighbours(new_matrix @ base_matrix.T)


# ----- Candidates ----------------------------------------------------------


def find_candidate_pairs(entries: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Find index pairs inside one batch worth checking with the model."""
    by_embedding = collect_embedding_pairs(embed_questions(entries, "Dedup"))
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
    candidates = by_embedding | by_question | by_answer
    logger.info(
        "Candidates: %d by embeddings, %d by question trigrams, "
        "%d by answer trigrams, %d in total",
        len(by_embedding),
        len(by_question),
        len(by_answer),
        len(candidates),
    )
    return sorted(candidates)


def find_candidate_matches(
    new_entries: list[dict[str, Any]], base_entries: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Find (new index, base index) pairs worth checking against the base."""
    by_embedding = collect_embedding_matches(
        embed_questions(new_entries, "Merge"), embed_questions(base_entries, "Merge")
    )
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
    candidates = by_embedding | by_question | by_answer
    logger.info(
        "Candidates: %d by embeddings, %d by question trigrams, "
        "%d by answer trigrams, %d in total",
        len(by_embedding),
        len(by_question),
        len(by_answer),
        len(candidates),
    )
    return sorted(candidates)
