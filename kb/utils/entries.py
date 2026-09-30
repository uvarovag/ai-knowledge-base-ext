"""Knowledge base entries: the schema the model fills, validation and provenance."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

import config
from kb.utils import storage

# Schemas the model fills through function calling: their docstrings and field
# descriptions are sent to the model, so they are written in Russian.


class Entry(BaseModel):
    """Запись базы знаний: категория, вопрос пользователя и ответ на него."""

    category: Literal[tuple(config.CATEGORIES)] = Field(
        description="Категория — ровно одно значение из списка"
    )
    question: str = Field(description="Вопрос от лица пользователя")
    answer: str = Field(description="Ответ")


# ----- Validation ----------------------------------------------------------

NUMBER_PATTERN = re.compile(r"\d+")
# A list marker is a short number followed by "." or ")" at the start of a line
# or right after a sentence end: answers listing several causes are written on
# one line ("несколько. 1. Файл слишком большой. 2. Неподдерживаемый формат").
LIST_MARKER_PATTERN = re.compile(
    r"(?:^|(?<=[.!?:;»)])\s+)\d{1,2}[.)]\s+", flags=re.MULTILINE
)


def find_invented_numbers(rewritten_answer: str, source_text: str) -> list[str]:
    """Return numbers present in the rewritten answer but absent from the source.

    Step numbering is stripped first, so generated lists are not flagged.
    """
    without_markers = LIST_MARKER_PATTERN.sub("", rewritten_answer)
    source_numbers = set(NUMBER_PATTERN.findall(source_text))
    return [
        number
        for number in NUMBER_PATTERN.findall(without_markers)
        if len(number) >= config.MIN_CHECKED_NUMBER_LENGTH
        and number not in source_numbers
    ]


def validate_entry(
    entry: dict[str, Any],
    source_text: str,
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
    if category not in config.CATEGORIES:
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
