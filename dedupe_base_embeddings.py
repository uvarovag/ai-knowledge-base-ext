"""Дедупликация живой базы по эмбеддингам вопросов.

Замена dedupe_base.py. Тот ищет кандидатов по символьным триграммам, и это
главная причина оставшихся дублей: «Почему не приходит первоначальный пароль
на почту?» и «Как получить пароль для нового пользователя, если он не был
отправлен после регистрации?» почти не делят общих букв, хотя это один вопрос.
Здесь кандидаты ищутся по косинусной близости эмбеддингов GigaChat, категория
записи не учитывается вовсе (её ставит модель, и один вопрос легко попадает в
две категории), а триграммы остаются лишь дополнительным дешёвым сигналом.

Дальше всё как в deduplicate.py: по каждой паре кандидатов модель решает, один
ли это вопрос и как соотносятся ответы (same / alternatives / contradiction),
группы строятся вокруг представителя без транзитивности.

Слияние группы идёт в два шага, чтобы в итоговом ответе не было повторов:
1. Записи, чьи ответы говорят одно и то же (same), сливаются в одну полную.
2. Если после этого в группе осталось несколько разных ответов (alternatives),
   из них делается одна запись со списком возможных причин.
Пары с противоречащими ответами не сливаются никогда: верен только один из
них, и складывать их в список «возможных причин» значит выдать пользователю
устаревшую инструкцию как рабочую. Такие пары пишутся в лог.

Кэши лежат в data/staging/base_dedupe_embeddings/ и привязаны к содержимому
записей, а не к их позициям в базе: эмбеддинги — к тексту вопроса, вердикты —
к паре хешей «вопрос + ответ». Поэтому повторный запуск после того, как база
изменилась, не подхватит чужие вердикты (в dedupe_base.py кэш по индексам
после первого же прогона указывает не на те пары).

База копируется в data/backups/ до записи.

Запуск:
    python dedupe_base_embeddings.py
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from langchain_gigachat.embeddings import GigaChatEmbeddings

import common
import config
import deduplicate
import export
import matching
from common import logger
from deduplicate import RELATION_SAME, Group

STAGING_DIR = config.STAGING_DIR / "base_dedupe_embeddings"
EMBEDDINGS_CACHE = STAGING_DIR / "embeddings.npz"
VERDICTS_CACHE = STAGING_DIR / "verdicts.json"


# ----- Хеши содержимого ------------------------------------------------------


def text_hash(text: str) -> str:
    """Хеш текста; ключ кэшей, не зависящий от позиции записи в базе."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def entry_hash(entry: dict[str, Any]) -> str:
    """Хеш записи целиком: вердикт зависит и от вопроса, и от ответа."""
    return text_hash(entry["question"] + "\n" + entry["answer"])


# ----- Эмбеддинги ------------------------------------------------------------


def build_embedder() -> GigaChatEmbeddings:
    """Клиент эмбеддингов с теми же сертификатами и адресом, что у чат-модели."""
    return GigaChatEmbeddings(
        model=config.GIGACHAT_EMBEDDINGS_MODEL,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(config.CERT_FILE),
        key_file=str(config.KEY_FILE),
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
    )


def embed_batch(embedder: GigaChatEmbeddings, texts: list[str]) -> list[list[float]]:
    """Один вызов эмбеддингов с повторами, как invoke_json для чат-модели."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        common.wait_for_network()
        try:
            vectors = embedder.embed_documents(texts)
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Expected {len(texts)} vectors, got {len(vectors)}"
                )
            return vectors
        except Exception as error:
            logger.warning(
                "Embeddings batch of %d: attempt %d/%d failed: %s",
                len(texts),
                attempt,
                config.MAX_RETRIES,
                error,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError("Embeddings request failed after every retry")


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    """Прочитать кэш эмбеддингов: хеш вопроса -> вектор."""
    if not path.exists():
        return {}
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def save_embedding_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    """Записать кэш атомарно: во временный файл, затем переименовать."""
    temp_path = path.with_suffix(".tmp.npz")
    np.savez(temp_path, **cache)
    temp_path.replace(path)


def embed_questions(entries: list[dict[str, Any]]) -> np.ndarray:
    """Вернуть матрицу нормированных эмбеддингов вопросов, по строке на запись.

    Считаются только вопросы, которых нет в кэше; кэш дописывается после
    каждой пачки, так что прерванный прогон продолжится с места остановки.
    """
    cache = load_embedding_cache(EMBEDDINGS_CACHE)
    hashes = [text_hash(entry["question"]) for entry in entries]

    pending: dict[str, str] = {}
    for entry, key in zip(entries, hashes):
        if key not in cache:
            pending[key] = entry["question"]

    if pending:
        embedder = build_embedder()
        keys = list(pending)
        batches = [
            keys[start : start + config.EMBEDDING_BATCH_SIZE]
            for start in range(0, len(keys), config.EMBEDDING_BATCH_SIZE)
        ]
        logger.info(
            "Embedding %d questions in %d batches (%d already cached)",
            len(keys),
            len(batches),
            len(entries) - len(keys),
        )
        bar = common.ProgressBar(len(batches), "Base dedup: embedding questions")
        for batch in batches:
            vectors = embed_batch(embedder, [pending[key] for key in batch])
            for key, vector in zip(batch, vectors):
                cache[key] = np.asarray(vector, dtype=np.float32)
            save_embedding_cache(EMBEDDINGS_CACHE, cache)
            bar.advance(embedded=len(batch))
        bar.finish()
    else:
        logger.info("All %d question embeddings taken from cache", len(entries))

    matrix = np.stack([cache[key] for key in hashes]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ----- Кандидаты -------------------------------------------------------------


def find_embedding_candidates(matrix: np.ndarray) -> set[tuple[int, int]]:
    """Пары записей, чьи вопросы близки по косинусу.

    Для каждой записи берутся не более EMBEDDING_TOP_K ближайших соседей с
    близостью не ниже EMBEDDING_CANDIDATE_THRESHOLD. Ограничение сверху не даёт
    общим вопросам вроде «Почему не открывается договор?» притянуть к себе
    половину базы.
    """
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)

    pairs: set[tuple[int, int]] = set()
    top_k = min(config.EMBEDDING_TOP_K, matrix.shape[0] - 1)
    if top_k <= 0:
        return pairs

    for first in range(matrix.shape[0]):
        row = similarity[first]
        nearest = np.argpartition(-row, top_k - 1)[:top_k]
        for second in nearest:
            if row[second] >= config.EMBEDDING_CANDIDATE_THRESHOLD:
                pairs.add((min(first, int(second)), max(first, int(second))))
    return pairs


def find_candidates(entries: list[dict[str, Any]], matrix: np.ndarray) -> list[tuple[int, int]]:
    """Объединить кандидатов по эмбеддингам и по триграммам вопроса.

    Триграммы дешёвые и иногда ловят пары с редкими терминами, которые
    эмбеддинги считают далёкими. Категории не учитываются: порог для разных
    категорий здесь равен обычному.
    """
    by_embedding = find_embedding_candidates(matrix)
    by_trigram = matching.collect_similar_pairs(
        entries,
        "question",
        config.CANDIDATE_THRESHOLD,
        config.CANDIDATE_THRESHOLD,
    )
    logger.info(
        "Candidates: %d by embeddings, %d by trigrams, %d in total",
        len(by_embedding),
        len(by_trigram),
        len(by_embedding | by_trigram),
    )
    return sorted(by_embedding | by_trigram)


# ----- Кэш вердиктов по содержимому -----------------------------------------


class ContentVerdictCache:
    """Кэш вердиктов, привязанный к содержимому пары, а не к индексам.

    Интерфейс совпадает с deduplicate.VerdictCache, так что judge_pairs
    работает с ним без изменений. Ключ — упорядоченная пара хешей записей.
    """

    UNIQUE = deduplicate.VerdictCache.UNIQUE

    def __init__(self, path: Path, entries: list[dict[str, Any]]) -> None:
        self.path = path
        self.hashes = [entry_hash(entry) for entry in entries]
        self.verdicts: dict[tuple[str, str], str] = {}
        for record in common.load_json(path):
            first, second = record["key"]
            self.verdicts[(first, second)] = record["relation"]

    def key_for(self, pair: tuple[int, int]) -> tuple[str, str]:
        first, second = self.hashes[pair[0]], self.hashes[pair[1]]
        return (first, second) if first <= second else (second, first)

    def get(self, pair: tuple[int, int]) -> str | None:
        return self.verdicts.get(self.key_for(pair))

    def add(self, pair: tuple[int, int], relation: str) -> None:
        self.verdicts[self.key_for(pair)] = relation

    def save(self) -> None:
        common.save_json(
            self.path,
            [
                {"key": list(key), "relation": relation}
                for key, relation in sorted(self.verdicts.items())
            ],
        )


# ----- Слияние группы --------------------------------------------------------


def split_same_clusters(
    group: Group, relations: dict[tuple[int, int], str]
) -> list[list[int]]:
    """Разбить группу на кластеры записей с одинаковыми по смыслу ответами.

    Две записи попадают в один кластер, если модель назвала их ответы same,
    напрямую или через цепочку таких же вердиктов внутри группы. Записи,
    подтверждённые как alternatives, остаются отдельными кластерами.
    """
    parent = {index: index for index in group.members}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    members = set(group.members)
    for (first, second), relation in relations.items():
        if relation == RELATION_SAME and first in members and second in members:
            parent[find(first)] = find(second)

    clusters: dict[int, list[int]] = {}
    for index in group.members:
        clusters.setdefault(find(index), []).append(index)

    # Представитель группы — самая полная запись, его кластер идёт первым.
    ordered = sorted(
        clusters.values(),
        key=lambda cluster: (group.representative not in cluster, min(cluster)),
    )
    return ordered


def longest_entry(group_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Самая подробная запись кластера; запасной вариант, если слияние не удалось."""
    return max(group_entries, key=lambda entry: len(entry["answer"]))


def merge_same(
    llm: Any, cluster_entries: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    """Слить записи с одинаковым смыслом в одну полную.

    Если модель не справилась или результат не прошёл проверку, остаётся самая
    подробная запись кластера: для одинаковых ответов это ничего не теряет.
    """
    if len(cluster_entries) == 1:
        return cluster_entries[0]
    merged = deduplicate.merge_entries(llm, cluster_entries, False, label)
    if merged is None:
        logger.warning("%s: keeping the most detailed entry instead", label)
        return longest_entry(cluster_entries)
    return merged


def merge_variants(
    llm: Any, cluster_entries: list[dict[str, Any]], label: str
) -> dict[str, Any] | None:
    """Сложить разные ответы на один вопрос в список возможных причин.

    Лимит длины растёт с числом причин: список из шести причин не может
    уложиться в лимит, рассчитанный на две-три.
    """
    merged = common.invoke_json(
        llm,
        deduplicate.MERGE_VARIANTS_SYSTEM_PROMPT,
        deduplicate.MERGE_USER_PROMPT.format(
            entries=deduplicate.format_entries(cluster_entries)
        ),
        required_keys=("category", "question", "answer"),
        label=label,
    )
    if merged is None:
        logger.warning("%s: variants merge failed", label)
        return None

    answer_limit = max(
        config.MAX_VARIANTS_ANSWER_WORDS,
        len(cluster_entries) * config.MAX_ANSWER_WORDS_PER_CAUSE,
    )
    source_text = " ".join(entry["answer"] for entry in cluster_entries)
    rejection_reason = common.validate_entry(merged, source_text, answer_limit)
    if rejection_reason is None and len(merged["answer"]) < max(
        len(entry["answer"]) for entry in cluster_entries
    ):
        rejection_reason = "variants_answer_lost_details"
    if rejection_reason is not None:
        logger.warning("%s: variants merge rejected (%s)", label, rejection_reason)
        return None
    return merged


def finalize(merged: dict[str, Any], group_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Собрать итоговую запись с провенансом всех исходных записей."""
    source_file, source_rows = common.merge_sources(group_entries)
    return {
        "category": merged["category"],
        "question": merged["question"],
        "answer": merged["answer"],
        "source_file": source_file,
        "source_rows": source_rows,
        "source_columns": common.merge_source_columns(group_entries),
        "updated_at": common.today(),
    }


def merge_group(
    llm: Any,
    entries: list[dict[str, Any]],
    group: Group,
    relations: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    """Слить одну группу в одну запись, в два шага.

    Сначала одинаковые по смыслу ответы схлопываются в один полный. Если после
    этого остался один ответ — он и есть результат. Если несколько — модель
    складывает их в список возможных причин. Ничего не теряется: при неудаче
    второго шага возвращаются уже схлопнутые кластеры по отдельности, при
    неудаче первого — самая подробная запись кластера.
    """
    clusters = split_same_clusters(group, relations)
    _, rows = common.merge_sources([entries[index] for index in group.members])
    label = f"merge rows {rows}"

    collapsed: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for position, cluster in enumerate(clusters, start=1):
        cluster_entries = [entries[index] for index in cluster]
        cluster_label = label if len(clusters) == 1 else f"{label} cause {position}"
        collapsed.append((merge_same(llm, cluster_entries, cluster_label), cluster_entries))

    if len(collapsed) == 1:
        merged, cluster_entries = collapsed[0]
        return [finalize(merged, cluster_entries)]

    if len(collapsed) > config.MAX_CAUSES_PER_ENTRY:
        logger.warning(
            "Rows %s: %d different causes exceed MAX_CAUSES_PER_ENTRY=%d, "
            "keeping them as separate entries",
            rows,
            len(collapsed),
            config.MAX_CAUSES_PER_ENTRY,
        )
        return [finalize(merged, cluster_entries) for merged, cluster_entries in collapsed]

    variants = merge_variants(
        llm, [merged for merged, _ in collapsed], f"{label} variants"
    )
    if variants is None:
        logger.warning(
            "Rows %s: keeping %d collapsed entries separate to avoid losing a cause",
            rows,
            len(collapsed),
        )
        return [finalize(merged, cluster_entries) for merged, cluster_entries in collapsed]

    group_entries = [entries[index] for index in group.members]
    return [finalize(variants, group_entries)]


# ----- Точка входа -----------------------------------------------------------


def run() -> None:
    common.configure_logging()

    entries = common.load_json(config.KNOWLEDGE_BASE_JSON)
    if not entries:
        logger.info("Base is empty, nothing to deduplicate")
        return

    backup_path = common.backup_file(config.KNOWLEDGE_BASE_JSON)
    logger.info("Backed up base to %s", backup_path)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    if config.FORCE_REPROCESS:
        for path in (EMBEDDINGS_CACHE, VERDICTS_CACHE):
            if path.exists():
                path.unlink()

    logger.info("Deduplicating %d base entries by question embeddings", len(entries))
    matrix = embed_questions(entries)
    candidates = find_candidates(entries, matrix)

    llm = common.build_llm()
    cache = ContentVerdictCache(VERDICTS_CACHE, entries)
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

    merge_llm = llm.model_copy(update={"max_tokens": config.BASE_MERGE_MAX_TOKENS})
    merged_by_group: dict[int, list[dict[str, Any]]] = {}
    if duplicate_groups:
        bar = common.ProgressBar(len(duplicate_groups), "Base dedup: merging groups")
        with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
            futures = {
                executor.submit(merge_group, merge_llm, entries, group, relations): group
                for group in duplicate_groups
            }
            for future in as_completed(futures):
                group = futures[future]
                result = future.result()
                merged_by_group[group.members[0]] = result
                bar.advance(merged=1, entries_out=len(result))
        bar.finish()

    result: list[dict[str, Any]] = []
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
