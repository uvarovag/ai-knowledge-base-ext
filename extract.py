"""Extract knowledge base candidates from one support dump.

Two model calls per pair, executed in parallel:
1. Filter: decide whether the pair generalizes beyond a single ticket.
2. Transform: classify and rewrite the pair into canonical form.

The filter rejects a pair only for reasons rewriting cannot fix. Private data
is not one of them: the transform stage strips greetings, names, logins and
document numbers, so losing a good instruction over a polite salutation would
be the worse error. An answer whose substance is one person's name — "обратитесь
к Иванову" — still fails, but on general_answer, because it does not generalize.

Every entry also carries the columns listed in config.SOURCE_EXTRA_COLUMNS,
copied verbatim from the dump, so a reviewer can trace it back to the ticket.

An interrupted run resumes: rows already written into the staging directory are
skipped, so a crash near the end of a large dump does not cost the whole run.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

import common
import config
from common import logger

# ----- Prompts -------------------------------------------------------------

FILTER_SYSTEM_PROMPT_TEMPLATE = """Ты — классификатор обращений в службу поддержки <<DOMAIN>>.
На вход дана одна пара: вопрос пользователя и ответ поддержки.
Оцени пару по четырём независимым признакам. Каждый — строго true или false.

ПРИЗНАК 1. reusable_question
Вопрос может возникнуть у другого пользователя в другой день.
Ставь false, если вопрос про конкретный объект и без него бессмысленен:
«где моя заявка 4500123456», «проверьте мой документ», «почему мне не пришло».
Ставь false, если вопрос нельзя понять без переписки: «а сейчас?»,
«поправьте, пожалуйста», «см. скриншот».
Ставь false, если в вопросе есть номер конкретного документа и вопрос
именно про него: «где документ №180063649», «что со статусом акта 0000197785».

ПРИЗНАК 2. general_answer
Ответ содержит правило, инструкцию или факт, применимый к любому пользователю
с таким же вопросом.

Проверь, кто выполняет действие:
- Действие выполняет ПОЛЬЗОВАТЕЛЬ: «перейдите», «включите», «нажмите»,
  «проверьте», «пересохраните», «необходимо сделать» — это инструкция, true.
- Действие выполнил СОТРУДНИК ПОДДЕРЖКИ: «поправил», «перевыставил»,
  «завёл вручную», «переназначил», «внёс изменения», «обновил у вас» — false.

Ставь false, если выполнено хотя бы одно:
- ответ описывает разовое действие сотрудника поддержки;
- ответ описывает разовую ситуацию: «был сбой», «шли работы»,
  «сотрудник был в отпуске», «попробуйте позже», «сейчас недоступно»;
- ответ — обещание сделать что-то позже: «сделаем», «поправим»,
  «вернёмся с ответом»;
- ответ сообщает состояние одного объекта или одной организации:
  «документ находится в статусе», «сейчас у вас указано», «на данный момент
  выставляется», «уведомления поступили, но ещё не зарегистрированы»,
  «ваш номер относится к допсоглашению»;
- ответ отсылает к КОНКРЕТНОМУ человеку по имени: «обратитесь к Иванову И.И.»,
  «пишите Петровой на почту» — такой ответ бесполезен другому пользователю;
- ответ содержит реплику оператора из переписки: «пришлите скриншот»,
  «предоставьте скриншот», «приложите файл», «уточните и сообщите»;
- ответ ссылается на данные этого обращения: «в предоставленном файле»,
  «в приложенном списке», «во вложении»;
- ответ содержит расчёт по конкретным датам и адресам этого обращения;
- проблему решает только поддержка, пользователь сам сделать ничего не может.

Слова «сейчас», «на данный момент», «в настоящее время» — сильный признак
того, что ответ описывает состояние конкретного обращения, а не правило.
Исключение: «на данный момент функционал не реализован» — это факт о системе,
он подходит для базы знаний.

Отсылка к РОЛИ, а не к человеку — это нормально, ставь true:
«обратитесь к администратору вашей компании», «ответственный за договор
формирует запрос», «сотрудник банка отклоняет документ».

Если в ответе есть и действие сотрудника, и инструкция — реши, что главное.
Если без вмешательства поддержки задача не решается, ставь false.

ПРИЗНАК 3. complete_answer
Ответа достаточно, чтобы пользователь решил задачу сам.
Ставь false, если ответ отсылает вовне без сути: «направил коллегам»,
«обратитесь к куратору», «позвоните», «см. вложение», «уточните у коллег».
Ставь false, если ответ только подтверждает получение: «принято», «в работе»,
«зарегистрировано», «выполнено», а также если ответа по существу нет.

ПРИЗНАК 4. no_private_data
В содержательной части ответа нет данных, которые нельзя показывать другим.

ВАЖНО: приветствие, обращение по имени и подпись — это НЕ приватные данные.
«Уважаемый Иван Иванович», «Здравствуйте, Мария», «С уважением, Петров»,
«отвечая на ваш вопрос» — вежливая часть письма, она будет удалена на
следующем шаге. Такие ФИО игнорируй, они на этот признак не влияют.

Ставь false, только если приватные данные внутри СУТИ ответа:
- ФИО третьего лица, к которому отсылают: «обратитесь к Иванову И.И.»;
- телефон, личная почта, логин конкретной учётной записи («VA_0005497»);
- номер заявки, договора, счёта, документа, ИНН, сумма в рублях;
- внутренняя ссылка на конкретный объект.

Названия систем, разделов, вкладок, кнопок, пути в настройках, номер роли,
код ошибки, номера пунктов инструкций и шаблон логина без цифр («логин
начинается на V_») — НЕ приватные данные.

Этот признак используется для статистики и не отклоняет пару сам по себе.
Оценивай его честно, не подгоняя под остальные признаки.

ОБЩИЕ ПРАВИЛА
- Оценивай только то, что написано. Ничего не додумывай.
- Если данных для признака не хватает — ставь false.
- topic — тема пары в 3–8 словах, в именительном падеже, без номеров и имён.
- Не отвечай на вопрос пользователя и не объясняй решение — только оцени пару.

ФОРМАТ ОТВЕТА — вызов функции с такими аргументами:
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": true, "topic": "строка"}

ПРИМЕР 1
Вопрос: Где посмотреть статус моей заявки?
Ответ: Статус виден в разделе «Мои заявки», колонка «Статус».
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": true, "topic": "статус заявки"}

ПРИМЕР 2 — обращение по имени не делает ответ приватным
Вопрос: Как отключить уведомления на электронную почту?
Ответ: Уважаемый Иван Иванович! Отвечая на ваш вопрос: настройка уведомлений выполняется в Личном кабинете, раздел «Уведомления». С уважением, служба поддержки.
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": true, "topic": "отключение уведомлений на почту"}

ПРИМЕР 3 — отсылка к конкретному человеку, ответ не обобщается
Вопрос: К кому обратиться по вопросу оцифровки договора?
Ответ: Добрый день! По этому договору обратитесь к Иванову Ивану Ивановичу, он ответственный.
{"reusable_question": true, "general_answer": false, "complete_answer": false, "no_private_data": false, "topic": "ответственный за оцифровку договора"}

ПРИМЕР 4 — отсылка к роли, а не к человеку
Вопрос: Почему у контрагента отсутствует кнопка создания акта?
Ответ: У пользователя нет роли для работы с актами. Обратитесь к сотруднику организации с учетной записью администратора, он назначит недостающие роли.
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": true, "topic": "отсутствует кнопка создания акта"}

ПРИМЕР 5
Вопрос: При входе появляется ошибка Session expired. Как её исправить?
Ответ: Очистите cookie для домена системы, перезапустите браузер и войдите заново.
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": true, "topic": "ошибка Session expired при входе"}

ПРИМЕР 6
Вопрос: Заявка 4500987123 висит на согласовании вторую неделю.
Ответ: Согласующий Иванов И.И. был в отпуске, переназначил на Петрова, заявка ушла дальше.
{"reusable_question": false, "general_answer": false, "complete_answer": true, "no_private_data": false, "topic": "заявка застряла на согласовании"}

ПРИМЕР 7
Вопрос: Какой срок обработки заявки?
Ответ: Направил ваш вопрос коллегам, они ответят дополнительно.
{"reusable_question": true, "general_answer": false, "complete_answer": false, "no_private_data": true, "topic": "срок обработки заявки"}

ПРИМЕР 8
Вопрос: Где находится документ расхождений №180063649 и что делать дальше?
Ответ: Документ №180063649 находится в статусе «На согласовании». После согласования сформируйте доставку на остаток.
{"reusable_question": false, "general_answer": false, "complete_answer": true, "no_private_data": false, "topic": "статус документа расхождений"}

ПРИМЕР 9
Вопрос: Как получить пароль для нового пользователя, если он не пришёл?
Ответ: Зайдите в УЗ VA_0005497, откройте карточку пользователя и нажмите «сгенерировать новый пароль».
{"reusable_question": true, "general_answer": true, "complete_answer": true, "no_private_data": false, "topic": "получение пароля нового пользователя"}

ПРИМЕР 10
Вопрос: Как сформировать УПД из архива актов?
Ответ: Если акты в Архиве, по ним уже сформирован УПД. Номера УПД указаны в предоставленном файле.
{"reusable_question": true, "general_answer": false, "complete_answer": false, "no_private_data": true, "topic": "формирование УПД из архива актов"}

ПРИМЕР 11
Вопрос: Как добавить ставку НДС 22% при формировании счета на аванс?
Ответ: Зайдите в кабинет администратора, заполните систему налогообложения, сохраните, пришлите скриншот, перезайдите на портал.
{"reusable_question": true, "general_answer": false, "complete_answer": true, "no_private_data": true, "topic": "ставка НДС в счете на аванс"}

ПРИМЕР 12
Вопрос: Не отображается строка в отчёте, хотя я её добавил.
Ответ: Проверил, у вас данные не доехали из-за ошибки синхронизации. Поправил вручную, обновите страницу.
{"reusable_question": true, "general_answer": false, "complete_answer": true, "no_private_data": true, "topic": "строка не отображается в отчёте"}
"""

FILTER_USER_PROMPT = """Оцени пару. Не отвечай на вопрос пользователя — только оцени.

Вопрос: {question}
Ответ: {answer}
"""

# Flags that gate a pair. no_private_data is collected for statistics but never
# rejects: the transform stage strips greetings, names, logins and document
# numbers anyway. An answer whose substance is one person's name fails on
# general_answer instead, which is the right reason.
FILTER_FLAGS: tuple[str, ...] = (
    "reusable_question",
    "general_answer",
    "complete_answer",
)


class FilterVerdict(BaseModel):
    """Оценка пары «вопрос — ответ» по четырём признакам и её тема."""

    reusable_question: bool = Field(
        description="Признак 1: вопрос может возникнуть у другого пользователя"
    )
    general_answer: bool = Field(
        description="Признак 2: ответ — правило или инструкция для любого пользователя"
    )
    complete_answer: bool = Field(
        description="Признак 3: ответа достаточно, чтобы решить задачу самому"
    )
    no_private_data: bool = Field(
        description="Признак 4: в сути ответа нет приватных данных"
    )
    topic: str = Field(description="Тема пары в 3–8 словах, без номеров и имён")


TRANSFORM_SYSTEM_PROMPT_TEMPLATE = """Ты готовишь одну запись базы знаний ассистента службы поддержки <<DOMAIN>>.
Тебе дают вопрос пользователя и ответ поддержки из реальной переписки.
Перепиши их в канонический вид и выбери категорию.

ГЛАВНОЕ ПРАВИЛО ВСЕЙ ЗАДАЧИ
Запись читают другие пользователи, у которых свои заявки и свои документы.
Поэтому в записи НЕ ДОЛЖНО остаться ни личных данных, ни номеров конкретных
объектов: ФИО, телефонов, почты, логинов, номеров заявок, договоров,
допсоглашений, УПД, актов, счетов, ИНН, сумм, версий ПО и дат.
Вместо номера пиши общее слово: «по заявке», «по договору», «в УПД».
Это правило важнее всех остальных. Ниже оно повторяется ещё несколько раз —
это не ошибка, а напоминание.

КАТЕГОРИЯ — ровно одно значение из списка, без изменений написания:
<<CATEGORIES>>

КАК ПЕРЕПИСАТЬ ВОПРОС
1. Одно предложение, от лица пользователя, с вопросительным знаком.
   Если исходный вопрос был утверждением («не можем найти договор»),
   переделай его в вопрос («как найти договор?»).
2. Убери приветствие, благодарность, извинения, эмоции, слова «пожалуйста», «срочно».
3. УДАЛИ ИЗ ВОПРОСА ВСЕ НОМЕРА: заявок, договоров, допсоглашений, УПД, актов,
   счетов, ИНН, суммы, версии ПО и даты. Замени их общим словом.
   Было: «Как изменить наименование услуги в УПД 0000197785 от 26.06.26?»
   Стало: «Как изменить наименование услуги в УПД?»
   Было: «Как изменить версию ПО на v3.11.00.26А?»
   Стало: «Как изменить актуальную версию ПО для оборудования?»
4. Сохрани код или текст ошибки и номер роли — по ним запись будут искать.
   «Request not valid», «Session expired», «роль 723» — это оставляем.
5. Не более <<QUESTION_WORDS>> слов.

ЕЩЁ РАЗ ПРО НОМЕРА В ВОПРОСЕ
В вопросе не должно быть длинных цифровых номеров и версий. Если ты видишь
в своём вопросе последовательность из трёх и более цифр подряд — это почти
всегда номер объекта, и его надо убрать. Оставить можно только номер роли и
код ошибки. Номер УПД, договора, акта, заявки, версию ПО — убрать всегда.

КАК ПЕРЕПИСАТЬ ОТВЕТ
1. УБЕРИ ПРИВЕТСТВИЕ И ОБРАЩЕНИЕ ПО ИМЕНИ ЦЕЛИКОМ: «Уважаемый Иван Иванович»,
   «Здравствуйте, Мария», «Добрый день», «отвечая на ваш вопрос», а также
   подпись в конце: «С уважением, служба поддержки», «С уважением, Петров».
   Начинай ответ сразу с сути.
2. Убери слова о том, что сделал оператор: «я проверил», «мы поправили».
3. Убери реплики из переписки: «пришлите скриншот», «сообщите результат».
4. Пиши нейтральную инструкцию или факт.
5. Если действий несколько — пронумерованные шаги, одно действие в шаге.
6. Названия разделов, вкладок, кнопок, полей и пути в настройках копируй
   дословно, включая кавычки и символ «>».
7. Сохрани все условия и оговорки: «только в статусе Черновик»,
   «для этого браузера».
8. УДАЛИ ИЗ ОТВЕТА ФИО, телефоны, почту, логины конкретных учётных записей,
   номера заявок, договоров, УПД и актов. Если ответ отсылает к конкретному
   человеку по имени, замени имя на роль: «обратитесь к Иванову И.И.» →
   «обратитесь к ответственному за договор».
   Номера ролей, коды ошибок, лимиты («20 МБ»), номера пунктов инструкций и
   шаблон логина без цифр («логин начинается на V_») оставь — они относятся
   к правилу, а не к одному пользователю.
9. Не более <<ANSWER_WORDS>> слов.

ЗАПРЕЩЕНО
- Оставлять в вопросе или ответе ФИО, телефон, почту, логин конкретной
  учётной записи, номер заявки, договора, допсоглашения, УПД, акта, счёта,
  ИНН, версию ПО, сумму или дату обращения.
- Добавлять факты, шаги, условия, сроки, лимиты и числа, которых нет в исходном ответе.
- Заменять число другим числом или округлять его.
- Дописывать вежливые фразы и советы от себя.
- Удлинять короткий ответ. Если в исходном ответе одно предложение — в итоговом
  тоже одно предложение.

ПРОВЕРЬ СЕБЯ ПЕРЕД ТЕМ, КАК ОТВЕТИТЬ
1. Прочитай свой вопрос. Есть ли в нём цифры? Если это не номер роли и не код
   ошибки — убери их и переформулируй.
2. Прочитай свой ответ. Начинается ли он с приветствия или имени? Убери.
   Есть ли в нём ФИО, номер документа или обращение к пользователю из
   переписки? Убери.
3. Все ли числа твоего ответа встречаются в исходном ответе? Если нет — убери
   те, которых там не было.

ФОРМАТ ОТВЕТА — вызов функции с такими аргументами:
{"category": "строка из списка", "question": "строка", "answer": "строка"}

ПРИМЕР 1
Вопрос: Добрый день! Уже третий раз загружаю файл, а он не прикрепляется, что не так?? Заявка 78123.
Ответ: Здравствуйте, Мария! Посмотрел — у вас файл 34 Мб, а лимит 20. И формат .doc не поддерживается. Пересохраните в pdf и загрузите заново.
{"category": "documents", "question": "Почему не прикрепляется файл к заявке?", "answer": "Проверьте файл: размер не должен превышать 20 МБ, поддерживается формат PDF. Файлы .doc не принимаются. Пересохраните документ в PDF и загрузите повторно."}

ПРИМЕР 2 — обрати внимание, как убраны обращение и подпись
Вопрос: Уважаемая поддержка! Подскажите, пожалуйста, как отключить уведомления на почту? Спасибо!
Ответ: Уважаемый Иван Иванович! Отвечая на ваш вопрос: настройка уведомлений выполняется в Личном кабинете Портала, раздел «Уведомления». Порядок описан в инструкции «П.ПП.052.001 Настройка уведомлений». С уважением, служба поддержки.
{"category": "how_to", "question": "Как отключить уведомления на электронную почту?", "answer": "Настройка уведомлений выполняется в Личном кабинете Портала, раздел «Уведомления». Порядок описан в инструкции «П.ПП.052.001 Настройка уведомлений»."}

ПРИМЕР 3
Вопрос: Добрый день. При входе в систему появляется надпись Session expired. Как мне вылечить данную ошибку?
Ответ: Добрый день. Если при входе система выдает сообщение "Session expired", то необходимо: Перейти в Настройки > Сайты > Расширенные настройки сайтов > В блоке Cookie-файлы включите опцию Разрешены (рекомендуется). После выполнения настроек необходимо перезапустить браузер и повторить вход.
{"category": "troubleshooting", "question": "Что делать при ошибке «Session expired» во время входа?", "answer": "Ошибка появляется, если в браузере запрещены cookie-файлы. Откройте «Настройки» > «Сайты» > «Расширенные настройки сайтов», в блоке «Cookie-файлы» включите опцию «Разрешены (рекомендуется)», перезапустите браузер и повторите вход."}

ПРИМЕР 4 — обрати внимание, как убран номер документа и оставлен номер роли
Вопрос: Здравствуйте! Не получается согласовать акт 0000197785 от 26.06.26, кнопка неактивна.
Ответ: Добрый день! Для согласования актов нужна роль 723 «Диспетчер». Оформите запрос на неё в личном кабинете администратора, после назначения роли кнопка станет активной.
{"category": "access", "question": "Почему неактивна кнопка согласования акта?", "answer": "Для согласования актов требуется роль 723 «Диспетчер». Оформите запрос на роль в личном кабинете администратора — после её назначения кнопка станет активной."}
"""

TRANSFORM_USER_PROMPT = """Перепиши эту пару.
Напоминание: убери приветствие, обращение по имени и подпись; в вопросе не
должно остаться номеров заявок, договоров, УПД, актов, версий ПО и дат.
Номер роли и код ошибки — оставь.

Вопрос: {question}
Ответ: {answer}
"""

FILTER_SYSTEM_PROMPT = common.render_prompt(
    FILTER_SYSTEM_PROMPT_TEMPLATE, domain=config.DOMAIN_NAME
)

TRANSFORM_SYSTEM_PROMPT = common.render_prompt(
    TRANSFORM_SYSTEM_PROMPT_TEMPLATE,
    domain=config.DOMAIN_NAME,
    categories=common.format_categories(),
    question_words=config.QUESTION_WORDS_TARGET,
    answer_words=config.ANSWER_WORDS_TARGET,
)

# ----- Data model ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportPair:
    """One raw question/answer pair taken from the source spreadsheet."""

    row_number: int
    question: str
    answer: str
    source_columns: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PairOutcome:
    """Result of processing a single pair through both stages."""

    pair: SupportPair
    entry: dict[str, Any] | None = None
    rejection_reason: str | None = None
    topic: str | None = None
    had_private_data: bool = False


@dataclass(slots=True)
class Statistics:
    """Counters collected over one dump."""

    accepted: int = 0
    rejected_by_filter: int = 0
    rejected_by_validation: int = 0
    failed: int = 0
    private_data_seen: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def register(self, reason: str) -> None:
        """Count one rejection, both in the totals and per reason."""
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
        if reason.startswith("filter:"):
            self.rejected_by_filter += 1
        elif is_failure(reason):
            self.failed += 1
        else:
            self.rejected_by_validation += 1


def is_failure(reason: str) -> bool:
    """Tell a pair the pipeline could not process from one the model rejected."""
    return reason.endswith("_failed") or reason == "unhandled_error"


# ----- Loading -------------------------------------------------------------


def join_columns(row: pd.Series, columns: Sequence[str]) -> str:
    """Join the non-empty values of the given columns into a single text."""
    parts = [
        str(row[column]).strip()
        for column in columns
        if pd.notna(row[column]) and str(row[column]).strip()
    ]
    return config.COLUMN_SEPARATOR.join(parts)


def available_extra_columns(dataframe: pd.DataFrame, dump_name: str) -> list[str]:
    """Return the configured extra columns present in the dump.

    A missing extra column only costs traceability, so it is a warning: an older
    dump may simply not have it.
    """
    available = [
        column for column in config.SOURCE_EXTRA_COLUMNS if column in dataframe.columns
    ]
    for column in config.SOURCE_EXTRA_COLUMNS:
        if column not in available:
            logger.warning("Column '%s' not found in %s, skipped", column, dump_name)
    return available


def read_extra_columns(row: pd.Series, columns: Sequence[str]) -> dict[str, str]:
    """Return the non-empty values of the given columns of one row."""
    return {
        column: str(row[column]).strip()
        for column in columns
        if pd.notna(row[column]) and str(row[column]).strip()
    }


def load_pairs(path: Path) -> list[SupportPair]:
    """Read a dump and return rows that have both a question and an answer."""
    if not path.exists():
        raise FileNotFoundError(f"Dump not found: {path}")

    logger.info("Reading dump: %s", path)
    dataframe = pd.read_excel(path, sheet_name=0)
    logger.info("Loaded %d rows, %d columns", len(dataframe), len(dataframe.columns))

    for column in (*config.QUESTION_COLUMNS, *config.ANSWER_COLUMNS):
        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' not found in {path.name}. "
                f"Available: {dataframe.columns.tolist()}"
            )

    extra_columns = available_extra_columns(dataframe, path.name)

    pairs: list[SupportPair] = []
    for index, row in dataframe.iterrows():
        question = join_columns(row, config.QUESTION_COLUMNS)
        answer = join_columns(row, config.ANSWER_COLUMNS)
        if question and answer:
            pairs.append(
                SupportPair(
                    row_number=int(index) + config.FIRST_DATA_ROW,
                    question=question,
                    answer=answer,
                    source_columns=read_extra_columns(row, extra_columns),
                )
            )

    logger.info("Found %d rows with both question and answer", len(pairs))
    return pairs


def load_completed(
    extracted_path: Path, rejected_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int], Statistics]:
    """Read what a previous, interrupted run already produced for this dump.

    Returns the entries and rejections written so far, the source rows they
    cover, and the statistics rebuilt from them, so the final summary describes
    the whole dump rather than only the resumed part. Rows that failed rather
    than got rejected (the model never gave a usable reply) are dropped from
    the rejections and processed again: a failure is not a verdict.
    """
    entries = common.load_json(extracted_path)
    rejected = [
        record
        for record in common.load_json(rejected_path)
        if not is_failure(record.get("reason", "unknown"))
    ]
    statistics = Statistics()
    processed_rows: set[int] = set()

    for entry in entries:
        processed_rows.update(entry.get("source_rows", []))
        statistics.accepted += 1

    for record in rejected:
        processed_rows.add(record["source_row"])
        statistics.register(record.get("reason", "unknown"))

    return entries, rejected, processed_rows, statistics


# ----- Stages --------------------------------------------------------------


def run_filter(llm: Any, pair: SupportPair) -> dict[str, Any] | None:
    """Score a pair against the reusability flags."""
    return common.invoke_structured(
        llm,
        FILTER_SYSTEM_PROMPT,
        FILTER_USER_PROMPT.format(question=pair.question, answer=pair.answer),
        schema=FilterVerdict,
        label=f"row {pair.row_number} filter",
    )


def run_transform(llm: Any, pair: SupportPair) -> dict[str, Any] | None:
    """Rewrite a pair into a canonical knowledge base entry."""
    return common.invoke_structured(
        llm,
        TRANSFORM_SYSTEM_PROMPT,
        TRANSFORM_USER_PROMPT.format(question=pair.question, answer=pair.answer),
        schema=common.Entry,
        label=f"row {pair.row_number} transform",
    )


def process_pair(llm: Any, pair: SupportPair, source_file: str) -> PairOutcome:
    """Run both stages for a single pair and validate the result."""
    verdict = run_filter(llm, pair)
    if verdict is None:
        return PairOutcome(pair=pair, rejection_reason="filter_failed")

    topic = verdict.get("topic") if isinstance(verdict.get("topic"), str) else None
    had_private_data = verdict.get("no_private_data") is not True

    # Anything other than an explicit true rejects the pair: a missing or
    # malformed flag means the model did not confirm it. no_private_data is not
    # in FILTER_FLAGS — the transform stage cleans that up.
    failed_flags = [flag for flag in FILTER_FLAGS if verdict.get(flag) is not True]
    if failed_flags:
        return PairOutcome(
            pair=pair,
            rejection_reason=f"filter:{failed_flags[0]}",
            topic=topic,
            had_private_data=had_private_data,
        )

    entry = run_transform(llm, pair)
    if entry is None:
        return PairOutcome(
            pair=pair,
            rejection_reason="transform_failed",
            topic=topic,
            had_private_data=had_private_data,
        )

    validation_error = common.validate_entry(entry, pair.answer)
    if validation_error:
        return PairOutcome(
            pair=pair,
            rejection_reason=validation_error,
            topic=topic,
            had_private_data=had_private_data,
        )

    return PairOutcome(
        pair=pair,
        entry={
            "category": entry["category"],
            "question": entry["question"],
            "answer": entry["answer"],
            "source_file": source_file,
            "source_rows": [pair.row_number],
            "source_columns": pair.source_columns,
            "updated_at": common.today(),
        },
        topic=topic,
        had_private_data=had_private_data,
    )


def log_summary(statistics: Statistics, total: int) -> None:
    """Print the final counters and the breakdown of rejection reasons."""
    logger.info(
        "Extracted: %d accepted, %d rejected by filter, %d rejected by validation, "
        "%d failed, out of %d pairs",
        statistics.accepted,
        statistics.rejected_by_filter,
        statistics.rejected_by_validation,
        statistics.failed,
        total,
    )
    if statistics.private_data_seen:
        logger.info(
            "  %d accepted answers contained private data before rewriting",
            statistics.private_data_seen,
        )
    for reason, count in sorted(
        statistics.rejection_reasons.items(), key=lambda item: -item[1]
    ):
        logger.info("  %-32s %d", reason, count)


# ----- Entry point ---------------------------------------------------------


def run(dump_path: Path, staging_dir: Path) -> Path:
    """Extract candidates from one dump into the staging directory.

    Rows already processed by an earlier, interrupted run are skipped. Returns
    the path of the file holding the extracted entries.
    """
    pairs = load_pairs(dump_path)
    extracted_path = staging_dir / "extracted.json"
    rejected_path = staging_dir / "rejected.json"

    if not pairs:
        logger.warning("No usable rows in %s", dump_path.name)
        common.save_json(extracted_path, [])
        common.save_json(rejected_path, [])
        return extracted_path

    if config.FORCE_REPROCESS:
        entries: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        processed_rows: set[int] = set()
        statistics = Statistics()
    else:
        entries, rejected, processed_rows, statistics = load_completed(
            extracted_path, rejected_path
        )

    pending = [pair for pair in pairs if pair.row_number not in processed_rows]
    if processed_rows:
        logger.info(
            "Resuming: %d rows already processed, %d left",
            len(processed_rows),
            len(pending),
        )
    if not pending:
        logger.info("Nothing left to extract from %s", dump_path.name)
        log_summary(statistics, len(pairs))
        return extracted_path

    source_file = dump_path.name
    llm = common.build_llm()
    completed = 0
    bar = common.ProgressBar(len(pending), f"Extract {dump_path.name}")

    def save_progress() -> None:
        """Write both output files so an interrupted run can pick up here."""
        common.save_json(extracted_path, entries)
        common.save_json(rejected_path, rejected)

    # Only the workers run in parallel; every result is folded in here, on the
    # main thread, so the lists need no lock.
    with ThreadPoolExecutor(max_workers=config.WORKER_COUNT) as executor:
        futures = {
            executor.submit(process_pair, llm, pair, source_file): pair
            for pair in pending
        }

        for future in as_completed(futures):
            pair = futures[future]
            try:
                outcome = future.result()
            except Exception as error:
                logger.error("Row %d: unhandled error: %s", pair.row_number, error)
                outcome = PairOutcome(pair=pair, rejection_reason="unhandled_error")

            if outcome.entry is not None:
                entries.append(outcome.entry)
                statistics.accepted += 1
                if outcome.had_private_data:
                    statistics.private_data_seen += 1
                bar.advance(accepted=1)
            else:
                reason = outcome.rejection_reason or "unknown"
                rejected.append(
                    {
                        "source_row": pair.row_number,
                        "reason": reason,
                        "topic": outcome.topic,
                        "question": pair.question,
                        "answer": pair.answer,
                    }
                )
                statistics.register(reason)
                bar.advance(rejected=1)

            completed += 1
            if completed % config.SAVE_EVERY == 0:
                save_progress()

    bar.finish()

    entries.sort(key=lambda entry: entry["source_rows"])
    rejected.sort(key=lambda record: record["source_row"])
    save_progress()

    log_summary(statistics, len(pairs))
    return extracted_path


if __name__ == "__main__":
    common.configure_logging()
    for path in config.DUMP_PATHS:
        run(path, common.staging_dir_for(path))
