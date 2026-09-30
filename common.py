"""Shared infrastructure: logging, progress, JSON storage, network and model calls."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import urlparse

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from pydantic import BaseModel, Field

import config

logger = logging.getLogger("kb")


# ----- Progress ------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Render a duration as 1h 05m, 5m 20s or 12s."""
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h {total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total}s"


class ProgressBar:
    """A single-line progress bar with named counters and an ETA.

    Renders to stderr while it is a terminal; otherwise falls back to periodic
    log lines, so redirected output stays readable. Only one bar is active at a
    time — the logging handler uses it to keep log lines from breaking the bar.
    """

    WIDTH = 24
    FALLBACK_STEP_PERCENT = 10

    active: ClassVar[ProgressBar | None] = None

    def __init__(self, total: int, label: str) -> None:
        self.total = max(total, 1)
        self.label = label
        self.completed = 0
        self.counters: dict[str, int] = {}
        self.started_at = time.monotonic()
        self.is_interactive = sys.stderr.isatty()
        self._last_reported_step = -1
        self._line_length = 0

        ProgressBar.active = self
        self._render()

    # ----- Rendering -------------------------------------------------------

    def _counters_text(self) -> str:
        return ", ".join(f"{name} {count}" for name, count in self.counters.items())

    def _eta_text(self) -> str:
        elapsed = time.monotonic() - self.started_at
        if self.completed == 0 or elapsed < 1:
            return "ETA --"
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate if rate else 0
        return f"{rate:.1f}/s, ETA {format_duration(remaining)}"

    def _render(self) -> None:
        if not self.is_interactive:
            return
        percent = self.completed * 100 // self.total
        filled = self.completed * self.WIDTH // self.total
        bar = "█" * filled + "░" * (self.WIDTH - filled)

        parts = [f"{self.label} [{bar}] {self.completed}/{self.total} {percent:3d}%"]
        counters = self._counters_text()
        if counters:
            parts.append(counters)
        parts.append(self._eta_text())

        line = " | ".join(parts)
        self._line_length = len(line)
        sys.stderr.write("\r" + line)
        sys.stderr.flush()

    def _report_to_log(self) -> None:
        """Print a progress line when stderr is not a terminal."""
        percent = self.completed * 100 // self.total
        step = percent // self.FALLBACK_STEP_PERCENT
        if step == self._last_reported_step and self.completed != self.total:
            return
        self._last_reported_step = step

        counters = self._counters_text()
        logger.info(
            "%s: %d/%d (%d%%)%s",
            self.label,
            self.completed,
            self.total,
            percent,
            f" | {counters}" if counters else "",
        )

    def clear(self) -> None:
        """Erase the bar so a log line can be printed over it."""
        if self.is_interactive and self._line_length:
            sys.stderr.write("\r" + " " * self._line_length + "\r")
            sys.stderr.flush()

    def redraw(self) -> None:
        """Draw the bar again after a log line was printed."""
        self._render()

    # ----- Updating --------------------------------------------------------

    def advance(self, **counters: int) -> None:
        """Count one finished item, adding the given named counters."""
        self.completed += 1
        for name, value in counters.items():
            self.counters[name] = self.counters.get(name, 0) + value

        if self.is_interactive:
            self._render()
        else:
            self._report_to_log()

    def finish(self) -> None:
        """Close the bar and print the final summary line."""
        elapsed = time.monotonic() - self.started_at
        self.clear()
        ProgressBar.active = None

        counters = self._counters_text()
        logger.info(
            "%s: done, %d items in %s%s",
            self.label,
            self.completed,
            format_duration(elapsed),
            f" | {counters}" if counters else "",
        )


class ProgressAwareHandler(logging.StreamHandler):
    """Stream handler that keeps log lines from breaking an active progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        bar = ProgressBar.active
        if bar is not None:
            bar.clear()
        super().emit(record)
        if bar is not None:
            bar.redraw()


def configure_logging() -> None:
    """Set up the single logging format used by every step of the pipeline.

    Warnings and errors also go to config.ERROR_LOG, appended across runs: a
    failed model call scrolls away among thousands of progress lines, and the
    file is what tells why a row ended up as filter_failed.
    """
    console_handler = ProgressAwareHandler(sys.stderr)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    config.ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(config.ERROR_LOG, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(error_handler)
    root.setLevel(logging.INFO)


# ----- Prompt building -----------------------------------------------------


def render_prompt(template: str, **values: str | int) -> str:
    """Substitute <<KEY>> placeholders in a prompt template.

    Plain str.format cannot be used here: prompt templates contain the curly
    braces of the JSON output examples.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"<<{key.upper()}>>", str(value))
    return rendered


def format_categories() -> str:
    """Render the configured categories as a bullet list for a prompt."""
    return "\n".join(
        f"- {name} — {description}" for name, description in config.CATEGORIES.items()
    )


def format_category_names() -> str:
    """Render the configured category keys as a comma separated list."""
    return ", ".join(config.CATEGORIES)


# ----- Storage -------------------------------------------------------------


def load_json(path: Path) -> list[dict[str, Any]]:
    """Read a list of records from a UTF-8 JSON file. Missing file means empty list."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write a list of records to disk as UTF-8 JSON, atomically.

    The knowledge base is the source of truth and the staging files are what an
    interrupted run resumes from, so a crash in the middle of a write must not
    leave a truncated file behind: the data goes to a temporary file first and
    replaces the target in one step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def backup_file(path: Path) -> Path | None:
    """Copy a file into the backup directory with a timestamp in its name."""
    if not path.exists():
        return None
    config.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = config.BACKUP_DIR / f"{path.stem}-{stamp}{path.suffix}"
    shutil.copy2(path, destination)
    return destination


def today() -> str:
    """Return the current date as an ISO string, stored on every entry."""
    return datetime.now().date().isoformat()


# ----- Staging -------------------------------------------------------------


def hash_file(path: Path) -> str:
    """Return the SHA-256 of a file, used to detect a re-uploaded dump."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(text: str) -> str:
    """Return the SHA-256 of a text: a cache key independent of an entry's position."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def hash_entry(entry: dict[str, Any]) -> str:
    """Hash an entry by its question and answer: a verdict depends on both."""
    return hash_text(entry["question"] + "\n" + entry["answer"])


def staging_dir_for(dump_path: Path) -> Path:
    """Return the staging directory of a dump, creating it if needed.

    The hash is part of the name so that two dumps with the same file name, or
    a corrected re-upload of one dump, do not share intermediate files. Every
    entry point derives the directory through this function, so running a
    module on its own resumes the same run the pipeline started.
    """
    directory = config.STAGING_DIR / f"{dump_path.stem}-{hash_file(dump_path)[:8]}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ----- Network availability ------------------------------------------------


def is_network_alive() -> bool:
    """Quick TCP check that the GigaChat host is reachable on port 443."""
    parsed = urlparse(config.GIGACHAT_BASE_URL)
    host = parsed.hostname
    port = parsed.port or 443
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def wait_for_network() -> None:
    """Block until the GigaChat host becomes reachable again."""
    if is_network_alive():
        return
    logger.warning("Network unreachable, waiting for it to come back")
    waited_seconds = 0
    while not is_network_alive():
        time.sleep(config.NETWORK_CHECK_INTERVAL_SECONDS)
        waited_seconds += config.NETWORK_CHECK_INTERVAL_SECONDS
        if waited_seconds % 300 == 0:
            logger.warning("Still no network after %d seconds", waited_seconds)
    logger.info("Network is back after %d seconds", waited_seconds)


# ----- Model ---------------------------------------------------------------


def build_llm(max_tokens: int = config.GIGACHAT_MAX_TOKENS) -> GigaChat:
    """Instantiate the GigaChat client with project defaults.

    Merging several entries into one needs a bigger output budget than the
    default, so the merge steps pass config.MERGE_MAX_TOKENS.
    """
    return GigaChat(
        model=config.GIGACHAT_MODEL_NAME,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(config.CERT_FILE),
        key_file=str(config.KEY_FILE),
        profanity_check=False,
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
        top_p=config.GIGACHAT_TOP_P,
        temperature=config.GIGACHAT_TEMPERATURE,
        max_tokens=max_tokens,
    )


def build_embedder() -> GigaChatEmbeddings:
    """Instantiate the embeddings client with the same endpoint and certificates."""
    return GigaChatEmbeddings(
        model=config.GIGACHAT_EMBEDDINGS_MODEL,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(config.CERT_FILE),
        key_file=str(config.KEY_FILE),
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
    )


def embed_texts(embedder: GigaChatEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed one batch of texts, retrying like invoke_structured does for the chat model.

    Raises:
        RuntimeError: if every attempt fails. Unlike a failed chat call, which
        costs one entry, missing vectors would silently hide duplicates, so the
        run stops instead.
    """
    for attempt in range(1, config.MAX_RETRIES + 1):
        wait_for_network()
        try:
            vectors = embedder.embed_documents(texts)
            if len(vectors) != len(texts):
                raise ValueError(f"Expected {len(texts)} vectors, got {len(vectors)}")
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
    raise RuntimeError(
        "Embeddings request failed after every retry; set "
        'EMBEDDING_BACKEND = "none" in config.py to run on trigrams only'
    )


# Schemas the model fills through function calling: their docstrings and field
# descriptions are sent to the model, so they are written in Russian.


class Entry(BaseModel):
    """Запись базы знаний: категория, вопрос пользователя и ответ на него."""

    category: Literal[tuple(config.CATEGORIES)] = Field(
        description="Категория — ровно одно значение из списка"
    )
    question: str = Field(description="Вопрос от лица пользователя")
    answer: str = Field(description="Ответ")


def describe_reply(reply: Any) -> str:
    """Summarize a raw model reply for the error log: finish reason, calls, text."""
    metadata = getattr(reply, "response_metadata", {}) or {}
    content = str(getattr(reply, "content", ""))
    return (
        f"finish_reason={metadata.get('finish_reason')}, "
        f"tool_calls={getattr(reply, 'tool_calls', None)}, "
        f"content={content[:500]!r}"
    )


def invoke_structured(
    llm: GigaChat,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    label: str,
) -> dict[str, Any] | None:
    """Call the model and return its answer as a dict of the schema, retrying on failure.

    The model is forced to call a function whose arguments are the schema
    (structured output, method="function_calling"), so its reply is never free
    text to be fished for JSON: a short reply like "Хочу новый стул" used to be
    answered in prose instead of being classified. Pydantic validates the
    arguments; a reply that fails validation is retried like a network error.

    Returns None if every attempt fails. The label identifies the item in logs.
    """
    # include_raw keeps the model's reply next to the parsed result, so a
    # failure is logged with what the model actually said.
    structured_llm = llm.with_structured_output(
        schema, method="function_calling", include_raw=True
    )
    messages = [("system", system_prompt), ("user", user_prompt)]

    for attempt in range(1, config.MAX_RETRIES + 1):
        wait_for_network()
        try:
            result = structured_llm.invoke(messages)
            if result["parsed"] is None:
                parsing_error = result["parsing_error"] or "none made"
                raise ValueError(
                    f"no valid function call ({parsing_error}); "
                    f"{describe_reply(result['raw'])}"
                )
            return result["parsed"].model_dump()
        except Exception as error:
            logger.warning(
                "%s: attempt %d/%d failed: %s: %s",
                label,
                attempt,
                config.MAX_RETRIES,
                type(error).__name__,
                error,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)

    return None


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


# ----- Entry helpers -------------------------------------------------------


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
