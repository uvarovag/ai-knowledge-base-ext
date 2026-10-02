"""Technical settings shared by every run.

What a run works on — the base, its source and output files, columns, domain,
categories and merge strategy — is in the TOML config passed on the command
line (kb/utils/settings.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

# Reprocess everything on each run, ignoring the dump ledger, the staging
# files and the verdict caches. For debugging, or after editing the prompts.
FORCE_REPROCESS = False

# ----- Technical storage ---------------------------------------------------

# Everything a run keeps for itself; the files people read are written where
# the TOML config of the run says.
DATA_DIR = Path.cwd() / "data"

# One directory per base, named in its TOML config: the base JSON (the source
# of truth), the ledger of processed dumps, staging, backups.
BASES_DIR = DATA_DIR / "bases"
# One directory per repair, named in its TOML config: staging and the JSON of
# the repaired base.
REPAIRS_DIR = DATA_DIR / "repairs"

# Question embeddings, keyed by the hash of the question text, shared by every
# base and every step that searches for duplicates. Never needs clearing: the
# same text always gets the same vector.
EMBEDDINGS_CACHE = DATA_DIR / "embeddings.npz"

# Every run writes its warnings and errors — why a model call failed, with the
# model's raw reply — to <work dir of the base>/logs/<command>_<time>.log.
# make inspect and make models, which have no base, write here.
TOOLS_LOG_DIR = DATA_DIR / "logs"

# ----- Source sheets -------------------------------------------------------

# Joins the columns of a row given in the TOML config into one question or
# one answer.
COLUMN_SEPARATOR = "\n"

# Separator between the values of one column when an entry has several sources.
SOURCE_VALUE_SEPARATOR = "| "

# Row 1 is the header, so the first data row of the sheet is row 2.
FIRST_DATA_ROW = 2

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

# Output budget of the model when it merges entries: a list of several causes
# is longer than the chat default above allows.
MERGE_MAX_TOKENS = 4000

# Embeddings model served by the same endpoint. "EmbeddingsGigaR" is the
# stronger one; fall back to "Embeddings" if the endpoint has no GigaR.
GIGACHAT_EMBEDDINGS_MODEL = "EmbeddingsGigaR"

# ----- Runtime -------------------------------------------------------------

WORKER_COUNT = 5
# Attempts of every GigaChat call, chat and embeddings: a failed call loses a
# row, while a retry only costs one more call and a wait.
MAX_RETRIES = 10

# A 429 "Too many requests" does not use up an attempt: every call of the
# process pauses together, first for RATE_LIMIT_BASE_SECONDS, doubling while
# 429s keep coming, up to RATE_LIMIT_MAX_SECONDS. A call gives up after
# RATE_LIMIT_MAX_WAITS such pauses. Parallel runs share the server's limit:
# lower WORKER_COUNT when two terminals keep hitting it.
RATE_LIMIT_BASE_SECONDS = 10
RATE_LIMIT_MAX_SECONDS = 120
RATE_LIMIT_MAX_WAITS = 30

# Links reach the model as placeholders («[ссылка-1]», kb/utils/links.py). A
# reply with a placeholder the source did not have is asked again, told which
# ones exist, up to this many calls in all.
LINK_PLACEHOLDER_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5
NETWORK_CHECK_INTERVAL_SECONDS = 30
# A model call this slow is marked yellow in the terminal, twice that red.
SLOW_CALL_SECONDS = 30
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

# Candidate pairs for the model come from three cheap local signals, united:
# cosine similarity of question embeddings, trigram similarity of questions
# and trigram similarity of answers. Embeddings are the only signal that sees
# a paraphrase ("не приходит пароль" vs "как получить пароль после
# регистрации"); trigrams still catch pairs with rare shared terms.

# Where question embeddings come from:
#   "gigachat" — the embeddings model of the GigaChat endpoint;
#   "none"     — no embeddings: candidates come from trigrams only. Use while
#                access to the embeddings model is not granted yet.
EMBEDDING_BACKEND: Literal["gigachat", "none"] = "gigachat"

# Questions per embeddings request.
EMBEDDING_BATCH_SIZE = 50

# A pair goes to the model when the cosine similarity of the question
# embeddings reaches this value. Kept moderate: a false candidate costs one
# model call, a missed one leaves a duplicate in the base.
EMBEDDING_CANDIDATE_THRESHOLD = 0.80

# At most this many nearest neighbours per entry are considered, so a generic
# question does not pull half of the base into the candidate list.
EMBEDDING_TOP_K = 8

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

# Word limit of a list of causes grows with the number of causes: each cause
# may take this many words on top of the MAX_VARIANTS_ANSWER_WORDS floor.
MAX_ANSWER_WORDS_PER_CAUSE = 150

# A group with more different causes than this is left as separate entries:
# such a list is unreadable and usually means the candidate thresholds are
# too low.
MAX_CAUSES_PER_ENTRY = 8

# ----- Repairing a base (scenario 2, make repair-base) ---------------------

# A base holds long regulatory answers that a repair can shorten only so far,
# so its answers get a higher ceiling than MAX_ANSWER_WORDS, and the model a
# larger output budget than GIGACHAT_MAX_TOKENS to write them: a reply cut off
# by the budget is a broken function call, retried and finally lost.
REPAIR_MAX_ANSWER_WORDS = 500
REPAIR_MAX_TOKENS = 4000

# ----- Excel export --------------------------------------------------------

SHEET_TITLE = "База знаний"

# Base columns; one more column is appended per extra column of the source.
HEADERS: tuple[str, ...] = (
    "Вопрос",
    "Ответ",
    "Категория",
    "Файл источника",
    "Строки источника",
    "Обновлено",
)
COLUMN_WIDTHS: tuple[int, ...] = (50, 90, 22, 24, 18, 14)

# Width used for every extra column of the source.
EXTRA_COLUMN_WIDTH = 22

# Sheet of the rows left out of a repaired base.
# The model's columns are filled when the code validation rejected its version.
REJECTED_HEADERS: tuple[str, ...] = (
    "Строка источника",
    "Причина",
    "Вопрос",
    "Ответ",
    "Вопрос модели",
    "Ответ модели",
)
REJECTED_COLUMN_WIDTHS: tuple[int, ...] = (16, 40, 50, 90, 50, 90)

HEADER_FILL_COLOR = "D9E1F2"

# ----- Base deduplication (make dedupe-base) -------------------------------

# Trigram threshold of the base deduplication when EMBEDDING_BACKEND is
# "none". Much lower than CANDIDATE_THRESHOLD: paraphrased duplicates share
# few letters, and without embeddings this is the only signal. Expect several
# thousand model calls on a base of a few hundred entries; a missed pair is a
# duplicate kept forever. The pipeline keeps its usual thresholds: a batch
# of thousands of tickets would not survive this one.
TRIGRAM_ONLY_CANDIDATE_THRESHOLD = 0.20
