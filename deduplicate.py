"""Collapse duplicates inside one extracted batch.

1. Candidates: local trigram similarity of questions and of answers, no model
   calls.
2. Adjudication: one model call per candidate pair. Besides deciding whether
   the two entries answer the same question, the model classifies how their
   answers relate: same, alternatives or contradiction. Verdicts are cached on
   disk, so an interrupted run does not pay for them twice.
3. Grouping: entries are attached to a representative, and every member of a
   group is confirmed against that representative directly. Duplicate is not a
   transitive relation — A~B and B~C does not make A~C — so a group is never
   built by chaining confirmations together. When an entry is confirmed against
   a member but was never compared to the representative, that missing pair is
   sent to the model instead of being silently dropped.
4. Merge: a group is rewritten into one entry. When the answers name different
   causes of the same problem, the entry lists them as numbered possible causes
   instead of picking one.

Nothing is lost when a merge fails. A group of complementary answers is left
unmerged rather than collapsed into one of its members, and a group of
identical answers keeps only the provenance of the entry that survives.

Entries whose answers contradict each other are never merged inside one batch:
one of them is wrong, and folding them into a list would produce a wrong entry.

This module also exposes the merge helpers used by merge.py when it folds a
fresh entry into an existing one of the living base.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import common
import config
import matching
from common import logger

RELATION_SAME = "same"
RELATION_ALTERNATIVES = "alternatives"
RELATION_CONTRADICTION = "contradiction"
RELATIONS = (RELATION_SAME, RELATION_ALTERNATIVES, RELATION_CONTRADICTION)

# ----- Prompts -------------------------------------------------------------

DUPLICATE_SYSTEM_PROMPT_TEMPLATE = """Ты сравниваешь две записи базы знаний службы поддержки <<DOMAIN>>.
Каждая запись — вопрос пользователя и ответ на него.
Определи два поля.

ПОЛЕ 1. duplicate — строго true или false.
Обе записи отвечают на один и тот же вопрос пользователя.

ГЛАВНОЕ ПРАВИЛО: смотри на ответы, а не на вопросы.
Если оба ответа называют ОДНУ И ТУ ЖЕ причину и ОДНО И ТО ЖЕ действие —
это дубликат. Пользователи описывают одну проблему через разные симптомы,
разные документы и разные разделы, но знание при этом одно.

Ставь true, если пользователь в обоих случаях хочет узнать одно и то же,
даже если формулировки разные и один ответ подробнее другого.
Ставь true, если в вопросах названы разные документы или разделы, а причина
и решение совпадают.

Ставь false, если выполнено хотя бы одно:
- разные действия: «создать» и «отменить», «загрузить» и «скачать»;
- разные объекты И разные решения;
- разные этапы процесса: «как отправить» и «что делать после отклонения»;
- разные роли пользователей.
Похожие слова сами по себе не делают записи дубликатами.
Если сомневаешься — ставь false.

ПОЛЕ 2. answers_relation — ровно одно значение из трёх: "same",
"alternatives", "contradiction". Если duplicate = false, ставь "same".

"same" — ответы говорят одно и то же, отличаются только словами или
подробностью. Один ответ можно выбросить без потери смысла.
Разные примеры одного действия (один ответ про Яндекс.Браузер, другой про
Яндекс.Браузер и Chrome) — это "same", а не "alternatives".

"alternatives" — ответы дополняют друг друга и оба могут быть верны
одновременно. Сюда относятся: разные причины одной проблемы; разные случаи,
каждый со своим условием («если статус Черновик …», «если заявка уже на
согласовании …»); разные способы сделать одно и то же.

"contradiction" — ответы несовместимы, верным может быть только один:
разные значения одного и того же параметра (лимит 20 МБ и 50 МБ, срок 3 дня
и 10 дней); разные названия одного и того же раздела или кнопки;
взаимоисключающие указания («можно» и «нельзя»).

ГЛАВНОЕ РАЗЛИЧИЕ между "alternatives" и "contradiction":
альтернативы дополняют друг друга и могут быть верны обе;
противоречие означает, что один из ответов устарел или ошибочен.
Если сомневаешься между ними — ставь "contradiction".

ФОРМАТ ОТВЕТА — ровно одна строка, без markdown, без текста до и после:
{"duplicate": true, "answers_relation": "same", "reason": "до 10 слов"}

ПРИМЕР 1
A. Вопрос: Где посмотреть статус заявки? Ответ: В разделе «Мои заявки», колонка «Статус».
B. Вопрос: Как узнать, на каком этапе моя заявка? Ответ: Откройте раздел «Мои заявки» и посмотрите колонку «Статус».
{"duplicate": true, "answers_relation": "same", "reason": "один вопрос, один ответ"}

ПРИМЕР 2
A. Вопрос: Как создать заявку? Ответ: Нажмите «Создать заявку» в разделе «Мои заявки».
B. Вопрос: Как отменить заявку? Ответ: Откройте заявку и нажмите «Отозвать».
{"duplicate": false, "answers_relation": "same", "reason": "разные действия"}

ПРИМЕР 3
A. Вопрос: Почему при подписании УПД портал выдает ошибку? Ответ: Откройте «Настройки» > «Сайты» > «Расширенные настройки сайтов», включите cookie-файлы, перезапустите браузер.
B. Вопрос: Что делать при ошибке «Request not valid» во время подписания документов? Ответ: Ошибка возникает из-за отключенных cookie. Для Яндекс.Браузера: «Настройки» > «Сайты» > «Расширенные настройки сайтов», включите «Разрешены». Для Chrome: «Настройки» > «Конфиденциальность и безопасность» > «Файлы cookie».
{"duplicate": true, "answers_relation": "same", "reason": "одна причина, одно решение"}

ПРИМЕР 4
A. Вопрос: Почему не видны заявки по договору? Ответ: Требуется учётная запись типа «Диспетчер» (логин V_...). Создайте её в кабинете администратора.
B. Вопрос: Почему в меню отсутствует раздел «Договоры»? Ответ: Разделы с договорами видны только Диспетчеру. Создайте учётную запись Диспетчера через кабинет администратора.
{"duplicate": true, "answers_relation": "same", "reason": "одна причина, одно решение"}

ПРИМЕР 5
A. Вопрос: Почему модуль «Акты» загружается 5–10 минут? Ответ: Очистите кэш, cookie и историю браузера, используйте Яндекс.Браузер.
B. Вопрос: Почему модуль «Акты» загружается очень долго, от 3 минут? Ответ: Очистите кэш, cookie-файлы и историю браузера за всё время. Используйте Яндекс.Браузер.
{"duplicate": true, "answers_relation": "same", "reason": "одна проблема, одно решение"}

ПРИМЕР 6
A. Вопрос: Почему не прикрепляется файл? Ответ: Размер файла не должен превышать 20 МБ.
B. Вопрос: Почему не загружается документ? Ответ: Поддерживается только формат PDF, файлы .doc не принимаются.
{"duplicate": true, "answers_relation": "alternatives", "reason": "разные причины одной проблемы"}

ПРИМЕР 7
A. Вопрос: Какой максимальный размер вложения? Ответ: До 20 МБ.
B. Вопрос: Какое ограничение по размеру файла? Ответ: Не более 50 МБ.
{"duplicate": true, "answers_relation": "contradiction", "reason": "разные лимиты размера"}

ПРИМЕР 8
A. Вопрос: Как изменить заявку? Ответ: В статусе «Черновик» откройте заявку и нажмите «Редактировать».
B. Вопрос: Как внести правки в заявку? Ответ: Если заявка уже на согласовании, сначала нажмите «Отозвать».
{"duplicate": true, "answers_relation": "alternatives", "reason": "разные случаи с условиями"}
"""

DUPLICATE_USER_PROMPT = """Запись A.
Вопрос: {first_question}
Ответ: {first_answer}

Запись B.
Вопрос: {second_question}
Ответ: {second_answer}
"""

MERGE_SYSTEM_PROMPT_TEMPLATE = """Ты объединяешь несколько записей базы знаний службы поддержки <<DOMAIN>>.
Все записи отвечают на один и тот же вопрос и говорят одно и то же, но с
разной полнотой. Сделай из них ровно одну запись.

КАТЕГОРИЯ — ровно одно значение из списка, без изменений написания:
<<CATEGORY_NAMES>>
Выбери ту, что подходит объединённой записи.

ВОПРОС
1. Одно предложение, от лица пользователя, с вопросительным знаком.
2. Возьми самую общую формулировку, покрывающую все записи. Если записи
   про разные документы, а решение одно, сформулируй вопрос обобщённо:
   «при подписании документа», а не «при подписании УПД».
3. Без номеров, ФИО и дат. Не более <<QUESTION_WORDS>> слов.

ОТВЕТ
1. Перенеси все содержательные детали из всех записей: шаги, условия,
   ограничения, числа, названия разделов и полей.
2. Одинаковое пиши один раз, не повторяй одно и то же разными словами.
3. Если действий несколько — пронумерованные шаги, одно действие в шаге.
4. Названия разделов, вкладок, кнопок и полей копируй дословно.
5. Ориентир по длине — <<ANSWER_WORDS>> слов.

ЕСЛИ ВСЁ НЕ ПОМЕЩАЕТСЯ В ОРИЕНТИР
Не сокращай и не выбрасывай детали. Вместо этого возьми ответ самой полной
записи и верни его без изменений. Полнота важнее длины.

ЗАПРЕЩЕНО
- Добавлять факты, шаги, условия, сроки, лимиты и числа, которых нет ни в одной
  из исходных записей.
- Заменять или округлять числа.
- Писать пояснения о том, что записи были объединены.

ФОРМАТ ОТВЕТА — ровно одна строка, без markdown, без текста до и после:
{"category": "строка из списка", "question": "строка", "answer": "строка"}

ПРИМЕР
Запись 1. Вопрос: Где посмотреть статус заявки? Ответ: В разделе «Мои заявки», колонка «Статус».
Запись 2. Вопрос: Как узнать этап заявки? Ответ: Откройте «Мои заявки», колонка «Статус», там же видно текущего исполнителя.
{"category": "how_to", "question": "Где посмотреть статус заявки?", "answer": "Откройте раздел «Мои заявки» и посмотрите колонку «Статус». В ней отображается этап заявки и текущий исполнитель."}
"""

MERGE_VARIANTS_SYSTEM_PROMPT_TEMPLATE = """Ты объединяешь несколько записей базы знаний службы поддержки <<DOMAIN>>.
Все записи отвечают на один и тот же вопрос, но описывают РАЗНЫЕ причины
проблемы или разные случаи. Все они верны — выбирать между ними нельзя.
Сделай из них ровно одну запись, где перечислены все варианты.

КАТЕГОРИЯ — ровно одно значение из списка, без изменений написания:
<<CATEGORY_NAMES>>
Выбери ту, что подходит объединённой записи.

ВОПРОС
1. Одно предложение, от лица пользователя, с вопросительным знаком.
2. Возьми самую общую формулировку, покрывающую все записи.
3. Без номеров, ФИО и дат. Не более <<QUESTION_WORDS>> слов.

ОТВЕТ — строго по этой структуре:
1. Первая строка: одно предложение о том, что вариантов несколько.
   Например: «Причин может быть несколько.» или «Порядок действий зависит от
   ситуации.»
2. Дальше пронумерованный список. Один пункт — одна запись из входных данных.
3. Каждый пункт начинай с причины или условия, потом что делать.
   Образец: «1. Файл слишком большой. Размер не должен превышать 20 МБ,
   пересохраните файл и загрузите повторно.»
4. Порядок пунктов — как идут записи на входе.
5. Количество пунктов равно количеству записей на входе. Не больше и не меньше.
6. Названия разделов, вкладок, кнопок и полей копируй дословно.
7. Ориентир по длине — <<ANSWER_WORDS>> слов.

ЕСЛИ ВСЁ НЕ ПОМЕЩАЕТСЯ В ОРИЕНТИР
Не выбрасывай пункты и не сокращай их до неузнаваемости. Оставь все пункты,
убрав только повторы между ними. Полнота важнее длины.

ЗАПРЕЩЕНО
- Придумывать причины и условия, которых нет ни в одной записи.
- Объединять две записи в один пункт.
- Выбирать один вариант как главный или писать «скорее всего».
- Добавлять факты, условия, сроки, лимиты и числа, которых нет в исходных записях.
- Заменять или округлять числа.

ФОРМАТ ОТВЕТА — ровно одна строка, без markdown, без текста до и после:
{"category": "строка из списка", "question": "строка", "answer": "строка"}

ПРИМЕР
Запись 1. Вопрос: Почему не прикрепляется файл? Ответ: Размер файла не должен превышать 20 МБ.
Запись 2. Вопрос: Почему не загружается документ? Ответ: Поддерживается только формат PDF, файлы .doc не принимаются.
{"category": "documents", "question": "Почему не прикрепляется файл?", "answer": "Причин может быть несколько. 1. Файл слишком большой. Размер не должен превышать 20 МБ. 2. Неподдерживаемый формат. Принимается только PDF, файлы .doc не загружаются."}
"""

MERGE_USER_PROMPT = """Объедини эти записи в одну:

{entries}
"""

DUPLICATE_SYSTEM_PROMPT = common.render_prompt(
    DUPLICATE_SYSTEM_PROMPT_TEMPLATE, domain=config.DOMAIN_NAME
)

MERGE_SYSTEM_PROMPT = common.render_prompt(
    MERGE_SYSTEM_PROMPT_TEMPLATE,
    domain=config.DOMAIN_NAME,
    category_names=common.format_category_names(),
    question_words=config.QUESTION_WORDS_TARGET,
    answer_words=config.ANSWER_WORDS_TARGET,
)

MERGE_VARIANTS_SYSTEM_PROMPT = common.render_prompt(
    MERGE_VARIANTS_SYSTEM_PROMPT_TEMPLATE,
    domain=config.DOMAIN_NAME,
    category_names=common.format_category_names(),
    question_words=config.QUESTION_WORDS_TARGET,
    answer_words=config.VARIANTS_ANSWER_WORDS_TARGET,
)

# ----- Verdict cache -------------------------------------------------------


class VerdictCache:
    """Verdicts already received from the model, kept on disk between runs.

    Pair indices refer to positions in extracted.json, which is written sorted
    and does not change between runs, so a cached verdict stays valid. Pairs the
    model called unique are cached too: not re-asking about them is most of the
    saving.
    """

    UNIQUE = "unique"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.verdicts: dict[tuple[int, int], str] = {}
        for record in common.load_json(path):
            self.verdicts[(record["pair"][0], record["pair"][1])] = record["relation"]

    def get(self, pair: tuple[int, int]) -> str | None:
        """Return the cached relation of a pair, or None if it is not cached."""
        return self.verdicts.get(pair)

    def add(self, pair: tuple[int, int], relation: str) -> None:
        """Remember the relation of a pair."""
        self.verdicts[pair] = relation

    def save(self) -> None:
        """Write the cache so an interrupted run can pick up here."""
        common.save_json(
            self.path,
            [
                {"pair": list(pair), "relation": relation}
                for pair, relation in sorted(self.verdicts.items())
            ],
        )


# ----- Adjudication --------------------------------------------------------


def check_pair(
    llm: Any, entries: list[dict[str, Any]], pair: tuple[int, int]
) -> tuple[tuple[int, int], dict[str, Any] | None]:
    """Ask the model whether the two entries answer the same question."""
    first, second = pair
    verdict = common.invoke_json(
        llm,
        DUPLICATE_SYSTEM_PROMPT,
        DUPLICATE_USER_PROMPT.format(
            first_question=entries[first]["question"],
            first_answer=entries[first]["answer"],
            second_question=entries[second]["question"],
            second_answer=entries[second]["answer"],
        ),
        required_keys=("duplicate", "answers_relation"),
        label=f"pair {first}/{second}",
    )
    return pair, verdict


def read_relation(verdict: dict[str, Any]) -> str:
    """Return the relation of a verdict, falling back to the safest value."""
    relation = verdict.get("answers_relation")
    if relation in RELATIONS:
        return str(relation)
    logger.warning("Unknown answers_relation %r, treating as contradiction", relation)
    return RELATION_CONTRADICTION


def judge_pairs(
    llm: Any,
    entries: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    cache: VerdictCache,
    label: str,
) -> tuple[dict[tuple[int, int], str], int, int]:
    """Ask the model about every given pair, reusing cached verdicts.

    Returns the confirmed relations, the number of complementary pairs and the
    number of contradicting ones. Contradicting pairs are deliberately left out
    of the relations: they must not be merged.
    """
    relations: dict[tuple[int, int], str] = {}
    alternatives_count = 0
    contradiction_count = 0

    def register(pair: tuple[int, int], relation: str) -> None:
        """Fold one verdict into the results, whether cached or just received."""
        nonlocal alternatives_count, contradiction_count
        if relation == VerdictCache.UNIQUE:
            return
        if relation == RELATION_CONTRADICTION:
            contradiction_count += 1
            return
        relations[pair] = relation
        if relation == RELATION_ALTERNATIVES:
            alternatives_count += 1

    pending: list[tuple[int, int]] = []
    for pair in pairs:
        cached = cache.get(pair)
        if cached is None:
            pending.append(pair)
        else:
            register(pair, cached)

    if len(pending) < len(pairs):
        logger.info(
            "%s: %d of %d pairs taken from cache",
            label,
            len(pairs) - len(pending),
            len(pairs),
        )
    if not pending:
        return relations, alternatives_count, contradiction_count

    save_lock = threading.Lock()
    completed = 0
    bar = common.ProgressBar(len(pending), label)

    with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
        futures = [executor.submit(check_pair, llm, entries, pair) for pair in pending]
        for future in as_completed(futures):
            pair, verdict = future.result()
            if verdict is None:
                logger.warning("Pair %s: no verdict, keeping both entries", pair)
                bar.advance(failed=1)
                continue

            if verdict.get("duplicate") is not True:
                relation = VerdictCache.UNIQUE
                bar.advance(unique=1)
            else:
                relation = read_relation(verdict)
                if relation == RELATION_CONTRADICTION:
                    logger.warning(
                        "Pair %s: answers contradict each other, keeping both (%s)",
                        pair,
                        verdict.get("reason"),
                    )
                    bar.advance(conflicts=1)
                else:
                    bar.advance(duplicates=1)

            with save_lock:
                cache.add(pair, relation)
                register(pair, relation)
                completed += 1
                if completed % config.SAVE_EVERY == 0:
                    cache.save()

    bar.finish()
    cache.save()

    return relations, alternatives_count, contradiction_count


# ----- Grouping ------------------------------------------------------------


@dataclass(slots=True)
class Group:
    """A representative entry and the entries confirmed to duplicate it."""

    representative: int
    members: list[int]
    relations: list[str] = field(default_factory=list)

    @property
    def has_alternatives(self) -> bool:
        return RELATION_ALTERNATIVES in self.relations

    @property
    def is_mixed(self) -> bool:
        return RELATION_SAME in self.relations and self.has_alternatives


def sort_key(pair: tuple[int, int]) -> tuple[int, int]:
    """Return a pair with its indices in ascending order, as relations are keyed."""
    first, second = pair
    return (min(first, second), max(first, second))


def entry_order(entries: list[dict[str, Any]]) -> list[int]:
    """Order entries from the most detailed to the least detailed."""
    return sorted(
        range(len(entries)),
        key=lambda index: (-len(entries[index]["answer"]), index),
    )


def find_missing_pairs(
    entries: list[dict[str, Any]], relations: dict[tuple[int, int], str]
) -> list[tuple[int, int]]:
    """Find entry/representative pairs that were confirmed only indirectly.

    An entry confirmed against a member of a group, but never compared to that
    group's representative, would be left out of the group. Such a pair was not
    a local candidate, so the model has not seen it — return it so it can be
    judged before the groups are built.
    """
    representatives: list[int] = []
    assigned: dict[int, int] = {}
    missing: list[tuple[int, int]] = []

    for index in entry_order(entries):
        for representative in representatives:
            if sort_key((index, representative)) in relations:
                assigned[index] = representative
                break
        else:
            representatives.append(index)
            assigned[index] = index

    for first, second in relations:
        for entry_index, other_index in ((first, second), (second, first)):
            representative = assigned.get(other_index)
            if representative is None or representative == entry_index:
                continue
            if assigned.get(entry_index) == representative:
                continue
            pair = sort_key((entry_index, representative))
            if pair not in relations and pair not in missing:
                missing.append(pair)

    return missing


def build_groups(
    entries: list[dict[str, Any]], relations: dict[tuple[int, int], str]
) -> list[Group]:
    """Group entries around representatives, without assuming transitivity.

    Entries are visited from the most detailed to the least detailed, so the
    fullest entry of a group becomes its representative. An entry joins a group
    only if the model confirmed it against that group's representative
    directly; a chain of confirmations never creates a group on its own.
    """
    groups: list[Group] = []
    for index in entry_order(entries):
        for group in groups:
            relation = relations.get(sort_key((index, group.representative)))
            if relation is None:
                continue
            group.members.append(index)
            group.relations.append(relation)
            break
        else:
            groups.append(Group(representative=index, members=[index]))

    for group in groups:
        group.members.sort()

    groups.sort(key=lambda group: group.members[0])
    return groups


# ----- Merge helpers -------------------------------------------------------


def format_entries(group_entries: list[dict[str, Any]]) -> str:
    """Render a group of entries as the numbered input of a merge prompt."""
    return "\n\n".join(
        f"Запись {i + 1}.\nВопрос: {entry['question']}\nОтвет: {entry['answer']}"
        for i, entry in enumerate(group_entries)
    )


def validate_merged(
    merged: dict[str, Any], group_entries: list[dict[str, Any]], as_variants: bool
) -> str | None:
    """Check a merged entry. Returns a rejection reason, or None if valid."""
    source_text = " ".join(entry["answer"] for entry in group_entries)
    answer_limit = (
        config.MAX_VARIANTS_ANSWER_WORDS if as_variants else config.MAX_ANSWER_WORDS
    )

    rejection_reason = common.validate_entry(merged, source_text, answer_limit)
    if rejection_reason is not None:
        return rejection_reason

    # A list of variants must be at least as long as the longest source answer,
    # and a plain merge must not be shorter than the shortest one.
    answer_length = len(merged["answer"])
    source_lengths = [len(entry["answer"]) for entry in group_entries]
    if as_variants:
        if answer_length < max(source_lengths):
            return "variants_answer_lost_details"
    elif answer_length < min(source_lengths):
        return "merged_answer_lost_details"

    return None


def merge_entries(
    llm: Any, group_entries: list[dict[str, Any]], as_variants: bool, label: str
) -> dict[str, Any] | None:
    """Rewrite several entries into one. Returns None if the result is unusable.

    With as_variants the answers are folded into a numbered list of variants;
    otherwise they are merged into a single answer.
    """
    merged = common.invoke_json(
        llm,
        MERGE_VARIANTS_SYSTEM_PROMPT if as_variants else MERGE_SYSTEM_PROMPT,
        MERGE_USER_PROMPT.format(entries=format_entries(group_entries)),
        required_keys=("category", "question", "answer"),
        label=label,
    )
    if merged is None:
        logger.warning("%s: merge failed", label)
        return None

    rejection_reason = validate_merged(merged, group_entries, as_variants)
    if rejection_reason is not None:
        logger.warning("%s: merge rejected (%s)", label, rejection_reason)
        return None

    return merged


def merge_group(
    llm: Any, entries: list[dict[str, Any]], group: Group
) -> list[dict[str, Any]]:
    """Merge one group into entries for the result.

    Returns one entry on success. On failure nothing is lost: a group of
    complementary answers is returned unmerged, and a group of identical
    answers keeps its representative alone, with only its own provenance.
    """
    group_entries = [entries[index] for index in group.members]
    source_file, source_rows = common.merge_sources(group_entries)
    source_columns = common.merge_source_columns(group_entries)

    # Folding into a list of variants only makes sense while the list stays
    # readable; a longer group is merged the usual way.
    as_variants = (
        group.has_alternatives and len(group.members) <= config.MAX_VARIANTS_PER_ENTRY
    )
    if group.is_mixed:
        logger.info(
            "Rows %s: group mixes identical and complementary answers, "
            "merging as a list of variants",
            source_rows,
        )

    merged = merge_entries(
        llm, group_entries, as_variants, label=f"merge rows {source_rows}"
    )

    if merged is None:
        if group.has_alternatives:
            logger.warning(
                "Rows %s: merge failed, keeping %d entries separate to avoid "
                "losing a cause",
                source_rows,
                len(group_entries),
            )
            return group_entries
        logger.warning(
            "Rows %s: merge failed, keeping the representative entry alone",
            source_rows,
        )
        return [entries[group.representative]]

    return [
        {
            "category": merged["category"],
            "question": merged["question"],
            "answer": merged["answer"],
            "source_file": source_file,
            "source_rows": source_rows,
            "source_columns": source_columns,
            "updated_at": common.today(),
        }
    ]


# ----- Entry point ---------------------------------------------------------


def run(extracted_path: Path, staging_dir: Path) -> Path:
    """Collapse duplicates inside one batch. Returns the path of the result."""
    entries = common.load_json(extracted_path)
    deduped_path = staging_dir / "deduped.json"
    cache_path = staging_dir / "verdicts.json"

    if not entries:
        common.save_json(deduped_path, [])
        return deduped_path

    if config.FORCE_REPROCESS and cache_path.exists():
        cache_path.unlink()
    cache = VerdictCache(cache_path)

    logger.info("Deduplicating %d extracted entries", len(entries))
    candidates = matching.find_candidate_pairs(entries)
    logger.info("Found %d candidate pairs to check with the model", len(candidates))

    llm = common.build_llm()
    relations: dict[tuple[int, int], str] = {}
    alternatives_count = 0
    contradiction_count = 0

    if candidates:
        relations, alternatives_count, contradiction_count = judge_pairs(
            llm, entries, candidates, cache, "Dedup: comparing pairs"
        )

        # An entry confirmed against a group member but never compared to that
        # group's representative would drop out of the group. Judge those pairs
        # too instead of losing the duplicate.
        missing = find_missing_pairs(entries, relations)
        if missing:
            logger.info("Checking %d indirectly confirmed pairs", len(missing))
            extra_relations, extra_alternatives, extra_contradictions = judge_pairs(
                llm, entries, missing, cache, "Dedup: checking indirect pairs"
            )
            relations.update(extra_relations)
            alternatives_count += extra_alternatives
            contradiction_count += extra_contradictions

    logger.info(
        "Model confirmed %d duplicate pairs (%d complementary), "
        "%d contradicting pairs",
        len(relations),
        alternatives_count,
        contradiction_count,
    )

    groups = build_groups(entries, relations)
    duplicate_groups = [group for group in groups if len(group.members) > 1]
    logger.info("Merging %d groups", len(duplicate_groups))

    merged_by_group: dict[int, list[dict[str, Any]]] = {}
    if duplicate_groups:
        bar = common.ProgressBar(len(duplicate_groups), "Dedup: merging groups")
        with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
            futures = {
                executor.submit(merge_group, llm, entries, group): group.members[0]
                for group in duplicate_groups
            }
            for future in as_completed(futures):
                merged_by_group[futures[future]] = future.result()
                bar.advance(merged=1)
        bar.finish()

    result: list[dict[str, Any]] = []
    for group in groups:
        if len(group.members) == 1:
            result.append(entries[group.members[0]])
        else:
            result.extend(merged_by_group[group.members[0]])

    common.save_json(deduped_path, result)
    logger.info("Collapsed %d entries into %d", len(entries), len(result))
    return deduped_path


if __name__ == "__main__":
    common.configure_logging()
    for path in config.DUMP_PATHS:
        staging = common.staging_dir_for(path)
        run(staging / "extracted.json", staging)
