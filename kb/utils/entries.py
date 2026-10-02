"""Knowledge base entries: the schema the model fills, validation and provenance."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

import config
from kb.utils import links, storage
from kb.utils.settings import Domain

# Schemas the model fills through function calling: their docstrings and field
# descriptions are sent to the model, so they are written in Russian. The
# categories come from the TOML config, so a schema is built per domain.


def entry_schema(domain: Domain) -> type[BaseModel]:
    """Return the schema of an entry whose category is one of the domain's."""

    class Entry(BaseModel):
        """Запись базы знаний: категория, вопрос пользователя и ответ на него."""

        category: Literal[tuple(domain.categories)] = Field(
            description="Категория — ровно одно значение из списка"
        )
        question: str = Field(description="Вопрос от лица пользователя")
        answer: str = Field(description="Ответ")

    return Entry


# ----- Validation ----------------------------------------------------------

NUMBER_PATTERN = re.compile(r"\d+")
# A list marker is a short number followed by "." or ")" at the start of a line
# or after a sentence end, a comma, a dash or an arrow: answers listing steps
# are often written on one line ("несколько. 1. Файл слишком большой. 2. …",
# "по шаблону [ссылка-1], 1. Тип заявки → Открытие").
LIST_MARKER_PATTERN = re.compile(
    r"(?:^|(?<=[.!?:;,»)—–→-])\s+)\d{1,2}[.)]\s+", flags=re.MULTILINE
)


# Number words a rewrite may turn into digits ("через десять дней" → "через 10
# дней"): the digits count as present in the source when the word is there.
NUMBER_WORDS: dict[str, str] = {
    "ноль": "0", "нуля": "0", "нулю": "0", "нулём": "0", "нулем": "0",
    "один": "1", "одна": "1", "одно": "1", "одного": "1", "одной": "1",
    "одну": "1",
    "два": "2", "две": "2", "двух": "2", "двум": "2",
    "три": "3", "трёх": "3", "трех": "3", "трём": "3", "трем": "3",
    "четыре": "4", "четырёх": "4", "четырех": "4",
    "пять": "5", "пяти": "5", "шесть": "6", "шести": "6", "семь": "7", "семи": "7",
    "восемь": "8", "восьми": "8", "девять": "9", "девяти": "9",
    "десять": "10", "десяти": "10", "пятнадцать": "15", "пятнадцати": "15",
    "двадцать": "20", "двадцати": "20", "тридцать": "30", "тридцати": "30",
    "сорок": "40", "сорока": "40", "пятьдесят": "50", "пятидесяти": "50",
    "сто": "100", "ста": "100",
}
WORD_PATTERN = re.compile(r"[а-яё]+")


def find_invented_links(rewritten_answer: str, source_text: str) -> list[str]:
    """Return links of the rewritten answer that are not verbatim in the source.

    A model "fixing" a typo in an encoded link, or copying a link from a
    prompt example, produces a link that leads nowhere.
    """
    return [
        link for link in links.find_links(rewritten_answer) if link not in source_text
    ]


def find_invented_numbers(rewritten_answer: str, source_text: str) -> list[str]:
    """Return numbers present in the rewritten answer but absent from the source.

    Links and step numbering are stripped first, so neither is flagged.
    """
    # Links are checked whole by find_invented_links: the digits of an encoded
    # link are not facts. A link becomes a period, so a list marker right after
    # it still follows a sentence end.
    without_links = links.URL_PATTERN.sub(".", rewritten_answer)
    without_markers = LIST_MARKER_PATTERN.sub("", without_links)
    # "03 квартал" and "3 квартал" are one number: compare without leading zeros.
    source_numbers = {
        number.lstrip("0") or "0" for number in NUMBER_PATTERN.findall(source_text)
    }
    source_numbers |= {
        NUMBER_WORDS[word]
        for word in WORD_PATTERN.findall(source_text.lower())
        if word in NUMBER_WORDS
    }
    return [
        number
        for number in NUMBER_PATTERN.findall(without_markers)
        if len(number) >= config.MIN_CHECKED_NUMBER_LENGTH
        and (number.lstrip("0") or "0") not in source_numbers
        and not is_expanded_year(number, source_numbers)
    ]


def is_expanded_year(number: str, source_numbers: set[str]) -> bool:
    """Tell a two-digit year written out in full: "26 года" → "2026 года"."""
    return (
        len(number) == 4
        and number.startswith("20")
        and (number[2:].lstrip("0") or "0") in source_numbers
    )


def validate_entry(
    entry: dict[str, Any],
    source_text: str,
    domain: Domain,
    max_answer_words: int | None = None,
) -> str | None:
    """Check a generated entry against the source text.

    Returns a rejection reason, or None if the entry is valid. Trims the
    question and the answer in place when they pass. Pass max_answer_words to
    override the default ceiling, as an entry listing several possible causes
    needs more room than a single-cause one.
    """
    answer_limit = (
        config.MAX_ANSWER_WORDS if max_answer_words is None else max_answer_words
    )

    category = entry.get("category")
    if category not in domain.categories:
        return f"unknown_category:{category}"

    question = entry.get("question")
    answer = entry.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        return "non_string_fields"

    question = question.strip()
    answer = answer.strip()
    if not question or not answer:
        return "empty_fields"

    if len(question.split()) > config.MAX_QUESTION_WORDS:
        return "question_too_long"
    if len(answer.split()) > answer_limit:
        return "answer_too_long"

    invented_links = find_invented_links(answer, source_text)
    if invented_links:
        return f"invented_links:{','.join(invented_links)}"
    if links.PLACEHOLDER_PATTERN.search(question + " " + answer):
        return "unknown_link_placeholder"

    invented_numbers = find_invented_numbers(answer, source_text)
    if invented_numbers:
        return f"invented_numbers:{','.join(invented_numbers)}"

    entry["question"] = question
    entry["answer"] = answer
    return None


# ----- Building entries ----------------------------------------------------


def merge_sources(entries: list[dict[str, Any]]) -> tuple[str, list[int]]:
    """Combine the source file names and row numbers of several entries."""
    files: list[str] = []
    rows: set[int] = set()
    for entry in entries:
        source_file = entry.get("source_file", "")
        if source_file and source_file not in files:
            files.append(source_file)
        rows.update(entry.get("source_rows", []))
    return "; ".join(files), sorted(rows)


def merge_source_columns(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Combine the extra source columns of several entries.

    Values of one column are joined in the order the entries are given, with
    duplicates dropped: several tickets about one problem often share a date but
    never a ticket number.
    """
    merged: dict[str, list[str]] = {}
    for entry in entries:
        for column, value in entry.get("source_columns", {}).items():
            values = merged.setdefault(column, [])
            for part in value.split(config.SOURCE_VALUE_SEPARATOR):
                part = part.strip()
                if part and part not in values:
                    values.append(part)

    return {
        column: config.SOURCE_VALUE_SEPARATOR.join(values)
        for column, values in merged.items()
    }


def new_entry(
    fields: dict[str, Any],
    source_file: str,
    source_row: int,
    source_columns: dict[str, str],
) -> dict[str, Any]:
    """Build an entry from model output for one source row."""
    return {
        "category": fields["category"],
        "question": fields["question"],
        "answer": fields["answer"],
        "source_file": source_file,
        "source_rows": [source_row],
        "source_columns": source_columns,
        "updated_at": storage.today(),
    }


def merged_entry(
    fields: dict[str, Any], group_entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the entry that replaces several, carrying the provenance of all."""
    source_file, source_rows = merge_sources(group_entries)
    return {
        "category": fields["category"],
        "question": fields["question"],
        "answer": fields["answer"],
        "source_file": source_file,
        "source_rows": source_rows,
        "source_columns": merge_source_columns(group_entries),
        "updated_at": storage.today(),
    }
