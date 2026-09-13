"""Single settings module for the knowledge base pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# ----- Dumps ---------------------------------------------------------------

# Full paths to the support dumps to process. Every dump must have the same
# column structure. The order matters: dumps are applied one after another and
# a later dump updates matching entries of an earlier one, so list them from
# the oldest to the freshest.
DUMP_PATHS: tuple[Path, ...] = (
    Path("/Users/19480633/Desktop/Обращения ПП.xlsx"),
)

# Reprocess every dump on each run, ignoring the ledger. For debugging.
FORCE_REPROCESS = False

# ----- Paths ---------------------------------------------------------------

DATA_DIR = Path.cwd() / "data"

STAGING_DIR = DATA_DIR / "staging"
BACKUP_DIR = DATA_DIR / "backups"

KNOWLEDGE_BASE_JSON = DATA_DIR / "knowledge_base.json"
KNOWLEDGE_BASE_XLSX = DATA_DIR / "knowledge_base.xlsx"
PROCESSED_DUMPS_JSON = DATA_DIR / "processed_dumps.json"

# ----- Source columns ------------------------------------------------------

# Columns of a dump joined into the question and the answer of one pair.
QUESTION_COLUMNS: tuple[str, ...] = ("Описания обращения",)
ANSWER_COLUMNS: tuple[str, ...] = ("Решение",)
COLUMN_SEPARATOR = "\n"

# Columns copied verbatim from the dump into every entry, for traceability.
# When several tickets collapse into one entry, their values are listed
# comma-separated in the same order as source_rows. Names must match the dump
# headers exactly — run inspect_dump.py to check them. A column missing from a
# dump is skipped with a warning, not an error.
SOURCE_EXTRA_COLUMNS: tuple[str, ...] = (
    "Обозначение ВидДокум",
    "Направление закупки",
    "Направление",
    "Категория",
    "Тема",
)

# Separator between the values of one column when an entry has several sources.
SOURCE_VALUE_SEPARATOR = "| "

# Row 1 is the header, so the first data row of the sheet is row 2.
FIRST_DATA_ROW = 2

# ----- Domain --------------------------------------------------------------

# Named in every prompt so the model knows what it is reading. Keep it short
# and put the product or department name here.
DOMAIN_NAME = "внутренних систем компании"

# Categories offered to the model. The key is stored in the entry, the value is
# shown to the model as the description of that key. Replace them with the
# categories of your own domain; the prompts are built from this mapping.
CATEGORIES: dict[str, str] = {
    "how_to": "как выполнить действие: где найти, как создать, изменить, настроить",
    "troubleshooting": "ошибки и сбои: что-то не работает, не открывается, не сохраняется",
    "access": "доступы, роли, права, регистрация, вход, пароли",
    "documents": "документы и файлы: загрузка, формирование, подписание, форматы",
    "data": "данные и отчёты: выгрузки, поиск, проверка значений",
    "policy": "правила, регламенты, сроки, зоны ответственности",
    "other": "не подходит ни одна категория выше",
}

# ----- GigaChat ------------------------------------------------------------

CERT_FILE = Path.cwd() / ".gigachat" / "client-cert.pem"
KEY_FILE = Path.cwd() / ".gigachat" / "client-cert.key"
GIGACHAT_MODEL_NAME = "GigaChat-3-Ultra"
GIGACHAT_BASE_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1"
GIGACHAT_VERIFY_SSL_CERTS = False
GIGACHAT_TIMEOUT_SECONDS = 180
GIGACHAT_TEMPERATURE = 0.0
GIGACHAT_TOP_P = 0.1
GIGACHAT_MAX_TOKENS = 1500

# ----- Runtime -------------------------------------------------------------

WORKER_COUNT = 5
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5
NETWORK_CHECK_INTERVAL_SECONDS = 30
SAVE_EVERY = 25

# ----- Content limits ------------------------------------------------------

# TARGET values are asked for in the prompts, MAX values reject an entry during
# validation. The gap absorbs the model overshooting a little without losing a
# good entry.
QUESTION_WORDS_TARGET = 20
MAX_QUESTION_WORDS = 150

ANSWER_WORDS_TARGET = 120
MAX_ANSWER_WORDS = 300

# An entry that lists several possible causes holds two or three answers at
# once, so it gets its own, higher limits.
VARIANTS_ANSWER_WORDS_TARGET = 250
MAX_VARIANTS_ANSWER_WORDS = 500

# Every number in a rewritten answer is checked against the source. Set to 2 to
# ignore single digits if generated step numbering starts producing false hits.
MIN_CHECKED_NUMBER_LENGTH = 1

# ----- Duplicate search ----------------------------------------------------

TRIGRAM_SIZE = 3

# A pair goes to the model when its QUESTIONS are similar enough. Support
# users phrase the same problem very differently ("не видны заявки" vs
# "отсутствует раздел Договоры"), so these thresholds are deliberately low:
# a false candidate only costs one model call, a missed one leaves a duplicate
# in the base forever.
CANDIDATE_THRESHOLD = 0.32
MATCH_THRESHOLD = 0.28

# A pair also goes to the model when its ANSWERS are nearly identical, even if
# the questions are not. This catches the common case where support pastes the
# same instruction for what users describe as different problems. Kept high:
# a shared boilerplate ("clear the cache") does not make two problems the same,
# and a low threshold here collapses unrelated tickets into one entry.
ANSWER_CANDIDATE_THRESHOLD = 0.78
ANSWER_MATCH_THRESHOLD = 0.75

# The category is generated by the model, so it is a strong hint, not a fact:
# the same knowledge can land in two categories and would then never be
# compared. Entries of different categories are therefore still compared, but
# only when they are much more similar. Set these to 1.1 to disable
# cross-category comparison.
CANDIDATE_CROSS_CATEGORY_THRESHOLD = 0.55
MATCH_CROSS_CATEGORY_THRESHOLD = 0.50

# How many entries at most are folded into one list of possible causes.
# Beyond this the group is merged as usual: a list of six causes is unreadable
# and usually means the candidate thresholds are too low.
MAX_VARIANTS_PER_ENTRY = 4

# ----- Merge strategy ------------------------------------------------------

# What happens when an entry of the new dump answers a question that is already
# in the knowledge base and the two answers name DIFFERENT causes:
#
# "accumulate" — both causes are kept and the entry is rewritten as a numbered
#   list of possible causes. Use when a problem genuinely has several causes
#   and support answers cover them one at a time.
#
# "replace" — the fresh answer wins and the old one is dropped. Use when the
#   process changes often and an old answer is more likely outdated than
#   complementary.
#
# Answers that simply say the same thing, and answers that contradict each
# other, are always replaced by the fresh one under either strategy.
MERGE_STRATEGY: Literal["accumulate", "replace"] = "accumulate"

# ----- Excel export --------------------------------------------------------

SHEET_TITLE = "База знаний"

# Base columns; one more column is appended per entry in SOURCE_EXTRA_COLUMNS.
HEADERS: tuple[str, ...] = (
    "Вопрос",
    "Ответ",
    "Категория",
    "Файл источника",
    "Строки источника",
    "Обновлено",
)
COLUMN_WIDTHS: tuple[int, ...] = (50, 90, 22, 24, 18, 14)

# Width used for every column added from SOURCE_EXTRA_COLUMNS.
EXTRA_COLUMN_WIDTH = 22

HEADER_FILL_COLOR = "D9E1F2"

# ----- Base deduplication by embeddings (dedupe_base_embeddings.py) --------

# Embeddings model served by the same GigaChat endpoint. "EmbeddingsGigaR" is
# the stronger one; fall back to "Embeddings" if the endpoint has no GigaR.
GIGACHAT_EMBEDDINGS_MODEL = "EmbeddingsGigaR"

# Questions per embeddings request.
EMBEDDING_BATCH_SIZE = 50

# A pair goes to the model when the cosine similarity of the question
# embeddings reaches this value. Kept moderate: a false candidate costs one
# model call, a missed one leaves a duplicate in the base.
EMBEDDING_CANDIDATE_THRESHOLD = 0.80

# At most this many nearest neighbours per entry are considered, so a generic
# question does not pull half of the base into the candidate list.
EMBEDDING_TOP_K = 8

# Output budget for the merge model: a list of several causes is longer than
# the chat default in GIGACHAT_MAX_TOKENS allows.
BASE_MERGE_MAX_TOKENS = 4000

# Word limit of a list of causes grows with the number of causes: each cause
# may take this many words on top of the MAX_VARIANTS_ANSWER_WORDS floor.
MAX_ANSWER_WORDS_PER_CAUSE = 150

# A group with more different causes than this is left as separate entries:
# such a list is unreadable and usually means the candidate threshold is too low.
MAX_CAUSES_PER_ENTRY = 8
