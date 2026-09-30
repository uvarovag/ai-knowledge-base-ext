"""Merge one deduplicated batch into the living knowledge base.

For every new entry the script looks for the same question in the current base.
What happens to a match depends on config.MERGE_STRATEGY and on how the two
answers relate:

- answers say the same thing        -> the fresh entry replaces the old one;
- answers contradict each other     -> the fresh entry replaces the old one,
                                       the fresh dump reflects the current state;
- answers name different causes     -> "accumulate": both causes are kept in one
                                       entry listing the possible causes;
                                       "replace": the fresh entry wins.

Matching is one to one: a new entry updates at most one entry of the base, and
an entry of the base is updated by at most one new entry. Entries with no match
are appended.

Verdicts are cached in the staging directory of the dump, so an interrupted
merge does not ask the model about the same pairs again.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import common
import config
import deduplicate
import matching
from common import logger

# ----- Prompt --------------------------------------------------------------

MATCH_SYSTEM_PROMPT_TEMPLATE = """Ты сопоставляешь новую запись со старой записью базы знаний
службы поддержки <<DOMAIN>>.
Каждая запись — вопрос пользователя и ответ на него.
Определи два поля.

ПОЛЕ 1. same_question — строго true или false.
Обе записи отвечают на один и тот же вопрос пользователя.
Ставь true, даже если:
- формулировки вопросов разные;
- ответы разной длины и подробности;
- ответы говорят разное — это нормально, новая запись свежее.

ВАЖНО: одна и та же проблема часто описывается через разные документы и
разделы. Если причина и решение одинаковые, это один вопрос, даже когда в
записях названы разные документы.

Ставь false, если выполнено хотя бы одно:
- разные действия: «создать» и «отменить», «загрузить» и «скачать»;
- разные объекты И разные решения;
- разные этапы процесса: «как отправить» и «что делать после отклонения»;
- разные роли пользователей;
- вопросы про соседние, но не совпадающие темы.
Похожие слова сами по себе не делают записи одинаковыми.
Если сомневаешься — ставь false: лишняя запись в базе безопаснее, чем
затёртая нужная.

ПОЛЕ 2. answers_relation — ровно одно значение из трёх: "same",
"alternatives", "contradiction". Если same_question = false, ставь "same".

"same" — ответы говорят одно и то же, отличаются только словами или
подробностью.

"alternatives" — ответы называют РАЗНЫЕ причины одной проблемы или разные
случаи, и оба могут быть верны одновременно. Признаки: причины не исключают
друг друга; каждая относится к своей ситуации; в одном ответе есть условие
«если …», которого нет в другом.

"contradiction" — ответы несовместимы, верным может быть только один:
разные значения одного и того же параметра; разные названия одного и того же
раздела или кнопки; взаимоисключающие указания.

Если сомневаешься между "alternatives" и "contradiction" — ставь
"contradiction".

ФОРМАТ ОТВЕТА — ровно одна строка, без markdown, без текста до и после:
{"same_question": true, "answers_relation": "same", "reason": "до 10 слов"}

ПРИМЕР 1
Старая. Вопрос: Какой максимальный размер вложения? Ответ: До 20 МБ.
Новая. Вопрос: Какое ограничение по размеру файла? Ответ: Не более 50 МБ.
{"same_question": true, "answers_relation": "contradiction", "reason": "лимит изменился"}

ПРИМЕР 2
Старая. Вопрос: Почему не прикрепляется файл? Ответ: Размер файла не должен превышать 20 МБ.
Новая. Вопрос: Почему не загружается документ? Ответ: Поддерживается только формат PDF.
{"same_question": true, "answers_relation": "alternatives", "reason": "разные причины одной проблемы"}

ПРИМЕР 3
Старая. Вопрос: Как создать заявку? Ответ: Нажмите «Создать заявку».
Новая. Вопрос: Как согласовать заявку? Ответ: Откройте заявку и нажмите «Согласовать».
{"same_question": false, "answers_relation": "same", "reason": "разные действия"}
"""

MATCH_USER_PROMPT = """Старая запись.
Вопрос: {base_question}
Ответ: {base_answer}

Новая запись.
Вопрос: {new_question}
Ответ: {new_answer}
"""

MATCH_SYSTEM_PROMPT = common.render_prompt(
    MATCH_SYSTEM_PROMPT_TEMPLATE, domain=config.DOMAIN_NAME
)

# ----- Matching ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Match:
    """A new entry that answers the same question as an entry of the base."""

    new_index: int
    base_index: int
    relation: str


def check_match(
    llm: Any,
    new_entries: list[dict[str, Any]],
    base_entries: list[dict[str, Any]],
    pair: tuple[int, int],
) -> tuple[tuple[int, int], str | None]:
    """Ask the model whether a new entry answers the same question as a base one.

    Returns the pair with the relation of the answers, VerdictCache.UNIQUE when
    the questions differ, or None when the model gave no usable verdict.
    """
    new_index, base_index = pair
    label = f"match new {new_index} / base {base_index}"
    verdict = common.invoke_json(
        llm,
        MATCH_SYSTEM_PROMPT,
        MATCH_USER_PROMPT.format(
            base_question=base_entries[base_index]["question"],
            base_answer=base_entries[base_index]["answer"],
            new_question=new_entries[new_index]["question"],
            new_answer=new_entries[new_index]["answer"],
        ),
        required_keys=("same_question", "answers_relation"),
        label=label,
    )
    if verdict is None:
        logger.warning("%s: no verdict, treating the new entry as new", label)
        return pair, None
    if verdict.get("same_question") is not True:
        return pair, deduplicate.VerdictCache.UNIQUE
    return pair, deduplicate.read_relation(verdict)


def judge_matches(
    new_entries: list[dict[str, Any]],
    base_entries: list[dict[str, Any]],
    candidates: list[tuple[int, int]],
    cache: deduplicate.VerdictCache,
) -> list[Match]:
    """Ask the model about every candidate pair, reusing cached verdicts."""
    confirmed: list[Match] = []

    def register(pair: tuple[int, int], relation: str) -> None:
        if relation != deduplicate.VerdictCache.UNIQUE:
            confirmed.append(Match(pair[0], pair[1], relation))

    pending: list[tuple[int, int]] = []
    for pair in candidates:
        cached = cache.get(pair)
        if cached is None:
            pending.append(pair)
        else:
            register(pair, cached)

    if len(pending) < len(candidates):
        logger.info(
            "Merge: %d of %d pairs taken from cache",
            len(candidates) - len(pending),
            len(candidates),
        )
    if not pending:
        return confirmed

    llm = common.build_llm()
    completed = 0
    bar = common.ProgressBar(len(pending), "Merge: matching against base")
    with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
        futures = [
            executor.submit(check_match, llm, new_entries, base_entries, pair)
            for pair in pending
        ]
        for future in as_completed(futures):
            pair, relation = future.result()
            if relation is None:
                bar.advance(failed=1)
                continue
            cache.add(pair, relation)
            register(pair, relation)
            if relation == deduplicate.VerdictCache.UNIQUE:
                bar.advance(new=1)
            else:
                bar.advance(matched=1)
            completed += 1
            if completed % config.SAVE_EVERY == 0:
                cache.save()
    bar.finish()
    cache.save()
    return confirmed


def select_matches(confirmed: list[Match]) -> dict[int, Match]:
    """Reduce the confirmed matches to a one to one mapping, keyed by base index.

    A new entry similar to two base entries would otherwise overwrite both with
    the same text, and two new entries pointing at one base entry would let the
    thread that happens to finish last decide. Sorting first makes the choice
    deterministic instead of dependent on completion order.
    """
    selected: dict[int, Match] = {}
    taken_new_indices: set[int] = set()

    for match in sorted(confirmed, key=lambda item: (item.new_index, item.base_index)):
        if match.base_index in selected or match.new_index in taken_new_indices:
            logger.warning(
                "Skipping extra match new %d / base %d",
                match.new_index,
                match.base_index,
            )
            continue
        selected[match.base_index] = match
        taken_new_indices.add(match.new_index)

    return selected


# ----- Update --------------------------------------------------------------


def should_accumulate(match: Match) -> bool:
    """Tell whether the matched entries are folded into a list of causes."""
    return (
        config.MERGE_STRATEGY == "accumulate"
        and match.relation == deduplicate.RELATION_ALTERNATIVES
    )


def build_updated_entry(
    llm: Any,
    match: Match,
    new_entries: list[dict[str, Any]],
    base_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce the entry that replaces a matched entry of the base.

    Under the accumulate strategy an entry whose answer names another cause is
    folded together with the old one; otherwise the fresh entry simply wins.
    """
    new_entry = new_entries[match.new_index]
    if not should_accumulate(match):
        return new_entry

    base_entry = base_entries[match.base_index]
    group_entries = [base_entry, new_entry]
    label = f"accumulate base {match.base_index} / new {match.new_index}"

    merged = deduplicate.merge_entries(
        llm, group_entries, as_variants=True, label=label
    )
    if merged is None:
        logger.warning("%s: keeping the fresh entry as is", label)
        return new_entry
    return deduplicate.finalize(merged, group_entries)


# ----- Entry point ---------------------------------------------------------


def run(deduped_path: Path, staging_dir: Path) -> tuple[int, int]:
    """Merge one batch into the living base.

    Returns the number of updated and of added entries.
    """
    new_entries = common.load_json(deduped_path)
    if not new_entries:
        logger.warning("Nothing to merge from %s", deduped_path)
        return 0, 0

    base_entries = common.load_json(config.KNOWLEDGE_BASE_JSON)
    logger.info(
        "Merging %d new entries into a base of %d, strategy: %s",
        len(new_entries),
        len(base_entries),
        config.MERGE_STRATEGY,
    )

    confirmed: list[Match] = []
    if base_entries:
        candidates = matching.find_candidate_matches(new_entries, base_entries)
        cache_path = staging_dir / "match_verdicts.json"
        if config.FORCE_REPROCESS and cache_path.exists():
            cache_path.unlink()
        cache = deduplicate.VerdictCache(cache_path, new_entries, base_entries)
        confirmed = judge_matches(new_entries, base_entries, candidates, cache)

    matches = select_matches(confirmed)
    accumulated_count = sum(should_accumulate(match) for match in matches.values())
    logger.info(
        "Confirmed %d matches, %d of them merged as lists of possible causes",
        len(matches),
        accumulated_count,
    )

    updated_by_base_index: dict[int, dict[str, Any]] = {}
    if matches:
        llm = common.build_llm(max_tokens=config.MERGE_MAX_TOKENS)
        bar = common.ProgressBar(len(matches), "Merge: updating entries")
        with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
            futures = {
                executor.submit(
                    build_updated_entry, llm, match, new_entries, base_entries
                ): match.base_index
                for match in matches.values()
            }
            for future in as_completed(futures):
                updated_by_base_index[futures[future]] = future.result()
                bar.advance(updated=1)
        bar.finish()

    result = [
        updated_by_base_index.get(base_index, entry)
        for base_index, entry in enumerate(base_entries)
    ]
    matched_new_indices = {match.new_index for match in matches.values()}
    added = [
        entry
        for index, entry in enumerate(new_entries)
        if index not in matched_new_indices
    ]
    result.extend(added)

    backup_path = common.backup_file(config.KNOWLEDGE_BASE_JSON)
    if backup_path:
        logger.info("Previous base backed up to %s", backup_path)

    common.save_json(config.KNOWLEDGE_BASE_JSON, result)
    logger.info(
        "Base updated: %d updated, %d added, %d entries total",
        len(matches),
        len(added),
        len(result),
    )
    return len(matches), len(added)


if __name__ == "__main__":
    common.configure_logging()
    for path in config.DUMP_PATHS:
        staging = common.staging_dir_for(path)
        run(staging / "deduped.json", staging)
