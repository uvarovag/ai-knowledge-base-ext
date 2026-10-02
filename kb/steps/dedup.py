"""Collapse duplicates inside one list of entries.

1. Candidates: found by the matching step and passed in, no model calls.
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
4. Merge: a group is rewritten into one entry in two steps, so the result has
   no repetitions. Entries whose answers say the same thing collapse into one
   full answer first; if several different answers remain, they become one
   entry listing the possible causes.

Nothing is lost when a merge fails. A group of different causes is left as
separate entries rather than collapsed into one of them; a cluster of
identical answers keeps its most detailed entry, with the provenance of all.

Entries whose answers contradict each other are never merged: one of them is
wrong, and folding them into a list would hand the user a stale instruction
as a working one. Such pairs are logged.

Verdicts are cached by the content of the two entries, not by their positions,
so a cache survives the list changing underneath it. Every scenario runs this
flow through collapse_duplicates, and the merging step uses the merge helpers
when it folds a fresh entry into an existing one of the base.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

import config
from kb.utils import gigachat, links, logs, parallel, prompts, storage
from kb.utils.entries import entry_schema, merge_sources, merged_entry, validate_entry
from kb.utils.logs import logger
from kb.utils.settings import Domain

RELATION_SAME = "same"
RELATION_ALTERNATIVES = "alternatives"
RELATION_CONTRADICTION = "contradiction"
# The schema enum makes the model return one of these, nothing else.
AnswersRelation = Literal["same", "alternatives", "contradiction"]


class DuplicateVerdict(BaseModel):
    """Решение, отвечают ли две записи на один вопрос, и как соотносятся ответы."""

    duplicate: bool = Field(description="Обе записи отвечают на один и тот же вопрос")
    answers_relation: AnswersRelation = Field(
        description="Как соотносятся ответы; при duplicate = false — same"
    )
    reason: str = Field(description="Причина решения, до 10 слов")


# ----- Prompts -------------------------------------------------------------

DUPLICATE_SYSTEM_PROMPT_TEMPLATE = """Ты — ревизор базы знаний службы поддержки <<DOMAIN>>.
Ты находишь записи-дубликаты: сравниваешь две записи и решаешь, об одном ли они.
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

ФОРМАТ ОТВЕТА — вызов функции с такими аргументами:
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

ПРИМЕР 9 — разные предметы, но одно решение
A. Вопрос: Как заказать мебель в офис? Ответ: Обратитесь к сервис-менеджеру вашего подразделения, контакты — на досках в лифтовой зоне.
B. Вопрос: Как заказать кулер с помпой? Ответ: Для заказа не ИТ-оборудования обратитесь к сервис-менеджеру подразделения. Его контакты на информационных досках этажа.
{"duplicate": true, "answers_relation": "same", "reason": "один способ заказа не ИТ-оборудования"}

ПРИМЕР 10 — похожая тема, разные справочные значения
A. Вопрос: По какой статье оплатить приобретение POS-терминалов? Ответ: Статья 205.16.
B. Вопрос: По какой статье оплатить приобретение SmartPOS-терминалов и расходных материалов? Ответ: Статья 185.15.
{"duplicate": false, "answers_relation": "same", "reason": "разные предметы, разные статьи"}
"""

DUPLICATE_USER_PROMPT = """Запись A.
Вопрос: {first_question}
Ответ: {first_answer}

Запись B.
Вопрос: {second_question}
Ответ: {second_answer}
"""

MERGE_SYSTEM_PROMPT_TEMPLATE = """Ты — редактор базы знаний службы поддержки <<DOMAIN>>.
Ты сводишь несколько записей об одном и том же в одну полную запись.
Все записи отвечают на один и тот же вопрос и говорят одно и то же, но с
разной полнотой. Сделай из них ровно одну запись.

КАТЕГОРИЯ — ровно одно значение из списка, без изменений написания:
<<CATEGORIES>>
Выбери ту, что подходит объединённой записи.

ВОПРОС
1. Одно предложение, от лица пользователя, с вопросительным знаком.
2. Возьми самую общую формулировку, покрывающую все записи. Если записи
   про разные документы, а решение одно, сформулируй вопрос обобщённо:
   «при подписании документа», а не «при подписании УПД».
3. Без номеров документов, ФИО и дат; код ошибки и номер роли сохрани.
   Не более <<QUESTION_WORDS>> слов.

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

<<LINKS_RULE>>

<<NUMBERS_RULE>>

<<WRITING_STYLE>>

ЗАПРЕЩЕНО
- Добавлять факты, шаги, условия, сроки, лимиты и числа, которых нет ни в одной
  из исходных записей.
- Писать пояснения о том, что записи были объединены.

ФОРМАТ ОТВЕТА — вызов функции с такими аргументами:
{"category": "строка из списка", "question": "строка", "answer": "строка"}

ПРИМЕР 1
Запись 1. Вопрос: Где посмотреть статус заявки? Ответ: В разделе «Мои заявки», колонка «Статус».
Запись 2. Вопрос: Как узнать этап заявки? Ответ: Откройте «Мои заявки», колонка «Статус», там же видно текущего исполнителя.
{"category": "how_to", "question": "Где посмотреть статус заявки?", "answer": "Откройте раздел «Мои заявки» и посмотрите колонку «Статус». В ней отображается этап заявки и текущий исполнитель."}

ПРИМЕР 2 — вопросы про разные предметы, решение одно: вопрос обобщён
Запись 1. Вопрос: Как заказать мебель в офис? Ответ: Обратитесь к сервис-менеджеру вашего подразделения.
Запись 2. Вопрос: Как заказать кулер с помпой? Ответ: Для заказа не ИТ-оборудования обратитесь к сервис-менеджеру подразделения, его контакты размещены на информационных досках в лифтовой зоне.
{"category": "how_to", "question": "Как заказать мебель или бытовую технику в офис?", "answer": "Для заказа мебели, бытовой техники и другого не ИТ-оборудования обратитесь к сервис-менеджеру вашего подразделения. Его контакты размещены на информационных досках в лифтовой зоне."}

ПРИМЕР 3 — подробность из второй записи (шаблон доступа) не потеряна
Запись 1. Вопрос: Как узнать складские запасы? Ответ: Откройте в SAP УВХД транзакцию MB52, заполните параметры и сформируйте отчёт.
Запись 2. Вопрос: Какие запасы есть на складе? Ответ: В SAP УВХД транзакция MB52. Если нет доступа, запросите его по шаблону «Доступ к автоматизированным системам Банка».
{"category": "data", "question": "Как посмотреть складские запасы?", "answer": "1. Откройте в SAP УВХД транзакцию MB52. Если доступа нет, запросите его по шаблону «Доступ к автоматизированным системам Банка». 2. Заполните параметры на селекционном экране. 3. Сформируйте отчёт."}
"""

MERGE_VARIANTS_SYSTEM_PROMPT_TEMPLATE = """Ты — редактор базы знаний службы поддержки <<DOMAIN>>.
Ты сводишь записи, где у одной проблемы разные причины или разные случаи, в одну
запись со списком вариантов.
Все записи отвечают на один и тот же вопрос, но описывают РАЗНЫЕ причины
проблемы или разные случаи. Все они верны — выбирать между ними нельзя.
Сделай из них ровно одну запись, где перечислены все варианты.

КАТЕГОРИЯ — ровно одно значение из списка, без изменений написания:
<<CATEGORIES>>
Выбери ту, что подходит объединённой записи.

ВОПРОС
1. Одно предложение, от лица пользователя, с вопросительным знаком.
2. Возьми самую общую формулировку, покрывающую все записи.
3. Без номеров документов, ФИО и дат; код ошибки и номер роли сохрани.
   Не более <<QUESTION_WORDS>> слов.

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

<<LINKS_RULE>>

<<NUMBERS_RULE>>

<<WRITING_STYLE>>

ЗАПРЕЩЕНО
- Придумывать причины и условия, которых нет ни в одной записи.
- Объединять две записи в один пункт.
- Выбирать один вариант как главный или писать «скорее всего».
- Добавлять факты, условия, сроки, лимиты и числа, которых нет в исходных записях.

ФОРМАТ ОТВЕТА — вызов функции с такими аргументами:
{"category": "строка из списка", "question": "строка", "answer": "строка"}

ПРИМЕР 1
Запись 1. Вопрос: Почему не прикрепляется файл? Ответ: Размер файла не должен превышать 20 МБ.
Запись 2. Вопрос: Почему не загружается документ? Ответ: Поддерживается только формат PDF, файлы .doc не принимаются.
{"category": "documents", "question": "Почему не прикрепляется файл?", "answer": "Причин может быть несколько. 1. Файл слишком большой. Размер не должен превышать 20 МБ. 2. Неподдерживаемый формат. Принимается только PDF, файлы .doc не загружаются."}

ПРИМЕР 2 — разные случаи, каждый со своим условием
Запись 1. Вопрос: Как заказать питьевую воду для кулера? Ответ: Для сети ВСП воспользуйтесь в ДРУГ шаблоном «Заказ питьевой воды».
Запись 2. Вопрос: Нужна вода для кулера в офисе. Ответ: Подразделения ССП обеспечиваются кулерами и пурифайерами; для заказа бутилированной воды обратитесь к менеджеру логистики ТБ.
{"category": "how_to", "question": "Как заказать питьевую воду для кулера?", "answer": "Порядок зависит от подразделения. 1. Сеть ВСП. Воспользуйтесь в ДРУГ шаблоном «Заказ питьевой воды». 2. Подразделения ССП. Они обеспечиваются кулерами и пурифайерами; для заказа бутилированной воды обратитесь к менеджеру логистики ТБ."}

ПРИМЕР 3 — разные способы сделать одно и то же
Запись 1. Вопрос: Где подписать накладную на перемещение? Ответ: В АС SberCost, пункт меню «Мои согласования».
Запись 2. Вопрос: Как подписать документ на перемещение имущества? Ответ: Во входящей почте АС УВХД SBWP: «Входящая почта» — «Поток операций».
{"category": "documents", "question": "Где подписать документ на перемещение имущества?", "answer": "Подписать можно двумя способами. 1. В АС SberCost. Откройте пункт меню «Мои согласования». 2. В АС УВХД SBWP. Откройте «Входящая почта» — «Поток операций»."}
"""

MERGE_USER_PROMPT = """Объедини эти записи в одну:

{entries}
"""


def duplicate_system_prompt(domain: Domain) -> str:
    return prompts.render_prompt(DUPLICATE_SYSTEM_PROMPT_TEMPLATE, domain=domain.name)


def merge_system_prompt(domain: Domain, as_variants: bool) -> str:
    """The prompt of a plain merge, or with as_variants of a list of causes."""
    return prompts.render_prompt(
        MERGE_VARIANTS_SYSTEM_PROMPT_TEMPLATE if as_variants else MERGE_SYSTEM_PROMPT_TEMPLATE,
        domain=domain.name,
        categories=prompts.format_categories(domain),
        question_words=config.QUESTION_WORDS_TARGET,
        answer_words=(
            config.VARIANTS_ANSWER_WORDS_TARGET if as_variants else config.ANSWER_WORDS_TARGET
        ),
        writing_style=prompts.WRITING_STYLE,
        links_rule=prompts.LINKS_RULE,
        numbers_rule=prompts.NUMBERS_RULE,
    )


# ----- Verdict cache -------------------------------------------------------


class VerdictCache:
    """Verdicts already received from the model, kept on disk between runs.

    A verdict is keyed by the hashes of the two entries' question and answer,
    so it stays valid when entries are added, removed or reordered between
    runs. Pairs the model called unique are cached too: not re-asking about
    them is most of the saving.

    The pair indices given to get and add refer to the entry lists passed to
    the constructor; a lookup across two lists (merge.py) gives the second list
    too, and the pair is then (index in first, index in second).
    """

    UNIQUE = "unique"

    def __init__(
        self,
        path: Path,
        entries: list[dict[str, Any]],
        other_entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self.hashes = [storage.hash_entry(entry) for entry in entries]
        self.other_hashes = (
            self.hashes
            if other_entries is None
            else [storage.hash_entry(entry) for entry in other_entries]
        )
        self.verdicts: dict[tuple[str, str], str] = {}

        records = storage.load_json(path)
        if records and "key" not in records[0]:
            logger.warning("%s: verdict cache in an old format, starting over", path)
            records = []
        for record in records:
            first, second = record["key"]
            self.verdicts[(first, second)] = record["relation"]

    def key_for(self, pair: tuple[int, int]) -> tuple[str, str]:
        first, second = self.hashes[pair[0]], self.other_hashes[pair[1]]
        return (first, second) if first <= second else (second, first)

    def get(self, pair: tuple[int, int]) -> str | None:
        """Return the cached relation of a pair, or None if it is not cached."""
        return self.verdicts.get(self.key_for(pair))

    def add(self, pair: tuple[int, int], relation: str) -> None:
        """Remember the relation of a pair."""
        self.verdicts[self.key_for(pair)] = relation

    def save(self) -> None:
        """Write the cache so an interrupted run can pick up here."""
        storage.save_json(
            self.path,
            [
                {"key": list(key), "relation": relation}
                for key, relation in sorted(self.verdicts.items())
            ],
        )


# ----- Adjudication --------------------------------------------------------


def check_pair(
    llm: Any, domain: Domain, entries: list[dict[str, Any]], pair: tuple[int, int]
) -> tuple[tuple[int, int], dict[str, Any] | None]:
    """Ask the model whether the two entries answer the same question."""
    first, second = pair
    verdict = gigachat.invoke_structured(
        llm,
        duplicate_system_prompt(domain),
        DUPLICATE_USER_PROMPT.format(
            first_question=entries[first]["question"],
            first_answer=entries[first]["answer"],
            second_question=entries[second]["question"],
            second_answer=entries[second]["answer"],
        ),
        schema=DuplicateVerdict,
        label=f"pair {first}/{second}",
        caller="duplicate check",
    )
    return pair, verdict


def judge_pairs(
    llm: Any,
    domain: Domain,
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

    completed = 0
    bar = logs.ProgressBar(len(pending), label)

    # Only the workers run in parallel; every result is folded in here, on the
    # main thread, so the cache needs no lock. It is saved on the way out too,
    # so an interrupted run keeps every verdict received.
    try:
        with parallel.workers() as executor:
            futures = [
                executor.submit(check_pair, llm, domain, entries, pair)
                for pair in pending
            ]
            for future in as_completed(futures):
                pair, verdict = future.result()
                title = logs.describe_pair(entries[pair[0]], entries[pair[1]])
                if verdict is None:
                    logger.warning("Pair %s: no verdict, keeping both entries", pair)
                    bar.print(
                        logs.render_item(title, "no verdict, both kept", [], True)
                    )
                    bar.advance(failed=1)
                    continue

                if verdict.get("duplicate") is not True:
                    relation = VerdictCache.UNIQUE
                    bar.advance(unique=1)
                else:
                    relation = verdict["answers_relation"]
                    if relation == RELATION_CONTRADICTION:
                        logger.warning(
                            "Pair %s: answers contradict each other, keeping both (%s)",
                            pair,
                            verdict.get("reason"),
                        )
                        reason = (
                            f"answers contradict, both kept: {verdict.get('reason')}"
                        )
                        bar.print(logs.render_item(title, reason, [], False))
                        bar.advance(conflicts=1)
                    else:
                        bar.advance(duplicates=1)

                cache.add(pair, relation)
                register(pair, relation)
                completed += 1
                if completed % config.SAVE_EVERY == 0:
                    cache.save()
    finally:
        cache.save()
    bar.finish()

    return relations, alternatives_count, contradiction_count


# ----- Grouping ------------------------------------------------------------


@dataclass(slots=True)
class Group:
    """A representative entry and the entries confirmed to duplicate it."""

    representative: int
    members: list[int]


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
            if sort_key((index, group.representative)) in relations:
                group.members.append(index)
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
    merged: dict[str, Any],
    group_entries: list[dict[str, Any]],
    as_variants: bool,
    domain: Domain,
) -> str | None:
    """Check a merged entry. Returns a rejection reason, or None if valid."""
    source_text = " ".join(entry["answer"] for entry in group_entries)
    if as_variants:
        # A list of causes grows with the number of causes: one of six cannot
        # fit the limit meant for two or three.
        answer_limit = max(
            config.MAX_VARIANTS_ANSWER_WORDS,
            len(group_entries) * config.MAX_ANSWER_WORDS_PER_CAUSE,
        )
    else:
        answer_limit = config.MAX_ANSWER_WORDS

    rejection_reason = validate_entry(merged, source_text, domain, answer_limit)
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
    llm: Any,
    domain: Domain,
    group_entries: list[dict[str, Any]],
    as_variants: bool,
    label: str,
) -> dict[str, Any] | None:
    """Rewrite several entries into one. Returns None if the result is unusable.

    With as_variants the answers are folded into a numbered list of variants;
    otherwise they are merged into a single answer.
    """
    masked, link_map = links.mask_links(
        [text for entry in group_entries for text in (entry["question"], entry["answer"])]
    )
    masked_entries = [
        {"question": question, "answer": answer}
        for question, answer in zip(masked[::2], masked[1::2])
    ]
    system_prompt = merge_system_prompt(domain, as_variants)
    user_prompt = MERGE_USER_PROMPT.format(entries=format_entries(masked_entries))
    merged = links.ask_with_links(
        lambda hint: gigachat.invoke_structured(
            llm,
            system_prompt,
            user_prompt + hint,
            schema=entry_schema(domain),
            label=label,
            caller="merge into list" if as_variants else "merge",
        ),
        link_map,
        label,
    )
    if merged is None:
        logger.warning("%s: merge failed", label)
        return None

    rejection_reason = validate_merged(merged, group_entries, as_variants, domain)
    if rejection_reason is not None:
        logger.warning("%s: merge rejected (%s)", label, rejection_reason)
        return None

    return merged


def split_same_clusters(
    group: Group, relations: dict[tuple[int, int], str]
) -> list[list[int]]:
    """Split a group into clusters of entries whose answers say the same thing.

    Two entries share a cluster when the model called their answers same,
    directly or through a chain of such verdicts inside the group. Entries
    confirmed as alternatives stay in clusters of their own. The cluster of
    the representative, the most detailed entry, comes first.
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

    return sorted(
        clusters.values(),
        key=lambda cluster: (group.representative not in cluster, min(cluster)),
    )


def merge_same(
    llm: Any, domain: Domain, cluster_entries: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    """Collapse entries that say the same thing into one full entry.

    When the model fails or the result is rejected, the most detailed entry of
    the cluster stands in: for answers that say the same thing, nothing is lost.
    """
    if len(cluster_entries) == 1:
        return cluster_entries[0]
    merged = merge_entries(llm, domain, cluster_entries, as_variants=False, label=label)
    if merged is None:
        logger.warning("%s: keeping the most detailed entry instead", label)
        return max(cluster_entries, key=lambda entry: len(entry["answer"]))
    return merged


def merge_group(
    llm: Any,
    domain: Domain,
    entries: list[dict[str, Any]],
    group: Group,
    relations: dict[tuple[int, int], str],
) -> list[dict[str, Any]]:
    """Merge one group into one entry, in two steps.

    Answers that say the same thing collapse into one full answer first. If one
    answer remains, it is the result; if several do, the model folds them into
    a list of possible causes. Nothing is lost: when the second step fails the
    collapsed clusters are returned as separate entries, and when the first
    fails the most detailed entry of the cluster stands in.
    """
    clusters = split_same_clusters(group, relations)
    _, rows = merge_sources([entries[index] for index in group.members])
    label = f"merge rows {rows}"

    collapsed: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for position, cluster in enumerate(clusters, start=1):
        cluster_entries = [entries[index] for index in cluster]
        cluster_label = label if len(clusters) == 1 else f"{label} cause {position}"
        collapsed.append(
            (merge_same(llm, domain, cluster_entries, cluster_label), cluster_entries)
        )

    if len(collapsed) == 1:
        merged, cluster_entries = collapsed[0]
        return [merged_entry(merged, cluster_entries)]

    if len(collapsed) > config.MAX_CAUSES_PER_ENTRY:
        logger.warning(
            "Rows %s: %d different causes exceed MAX_CAUSES_PER_ENTRY=%d, "
            "keeping them as separate entries",
            rows,
            len(collapsed),
            config.MAX_CAUSES_PER_ENTRY,
        )
        return [merged_entry(merged, cluster_entries) for merged, cluster_entries in collapsed]

    variants = merge_entries(
        llm,
        domain,
        [merged for merged, _ in collapsed],
        as_variants=True,
        label=f"{label} variants",
    )
    if variants is None:
        logger.warning(
            "Rows %s: keeping %d collapsed entries separate to avoid losing a cause",
            rows,
            len(collapsed),
        )
        return [merged_entry(merged, cluster_entries) for merged, cluster_entries in collapsed]

    return [merged_entry(variants, [entries[index] for index in group.members])]


# ----- Collapsing a list ---------------------------------------------------


def collapse_duplicates(
    domain: Domain,
    entries: list[dict[str, Any]],
    candidates: list[tuple[int, int]],
    cache: VerdictCache,
    label: str,
) -> list[dict[str, Any]]:
    """Judge the candidate pairs, group the confirmed ones and merge every group.

    Returns the collapsed list in the order of the original entries. Shared by
    the in-batch deduplication and by dedupe_base.py, which only differ in
    where the entries and the candidates come from.
    """
    relations: dict[tuple[int, int], str] = {}
    alternatives_count = 0
    contradiction_count = 0

    if candidates:
        llm = gigachat.build_llm()
        relations, alternatives_count, contradiction_count = judge_pairs(
            llm, domain, entries, candidates, cache, f"{label}: comparing pairs"
        )

        # An entry confirmed against a group member but never compared to that
        # group's representative would drop out of the group. Judge those pairs
        # too instead of losing the duplicate.
        missing = find_missing_pairs(entries, relations)
        if missing:
            logger.info("Checking %d indirectly confirmed pairs", len(missing))
            extra_relations, extra_alternatives, extra_contradictions = judge_pairs(
                llm, domain, entries, missing, cache, f"{label}: checking indirect pairs"
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
        merge_llm = gigachat.build_llm(max_tokens=config.MERGE_MAX_TOKENS)
        bar = logs.ProgressBar(len(duplicate_groups), f"{label}: merging groups")
        with parallel.workers() as executor:
            futures = {
                executor.submit(
                    merge_group, merge_llm, domain, entries, group, relations
                ): group.members[0]
                for group in duplicate_groups
            }
            for future in as_completed(futures):
                result = future.result()
                merged_by_group[futures[future]] = result
                bar.advance(merged=1, entries_out=len(result))
        bar.finish()

    result: list[dict[str, Any]] = []
    for group in groups:
        if len(group.members) == 1:
            result.append(entries[group.members[0]])
        else:
            result.extend(merged_by_group[group.members[0]])

    logger.info("Collapsed %d entries into %d", len(entries), len(result))
    return result
