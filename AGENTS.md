# AGENTS.md

Guidance for AI coding agents (Claude Code, GigaCode, any other) working in this repository.

## How this file works

- **Do not add, change or remove anything in this file without the user's explicit confirmation.**
  Propose the change, wait for a "yes", then apply it.
- Code wins over any document. Derive behaviour from `config.py`, the prompts and the code; treat
  stale text in a doc (this file, `README.md`) as a bug and fix the doc.
- This repository is self-contained: don't reference files of sibling repositories or of any
  workspace that happens to hold them.

## Overview

A batch pipeline that builds and maintains a support knowledge base from Excel exports of support
tickets (question/answer pairs). Python 3.13, GigaChat through `langchain-gigachat`, pandas/openpyxl
for Excel. Input: Excel with columns configured in `config.py`. Output: `data/knowledge_base.json`
(source of truth) and `data/knowledge_base.xlsx` (review-friendly view).

## Commands

```bash
make setup          # venv + uv + requirements.txt (PYTHON and SBEROSC_TOKEN from .env, see .env.example)
source activate.sh  # activate the venv with the same environment as the Makefile
make inspect        # check a new dump's columns against config.py and that GigaChat and embeddings answer
make run            # full pipeline run under caffeinate (network calls die when the Mac sleeps)
make dedupe-base    # deduplicate the living base by question embeddings and trigrams
make help           # every target
```

There is no test suite, linter or type-checker in this repo — don't invent commands for them.
`requirements.txt` is pinned by hand; `make setup` installs it.

A run can be interrupted with Ctrl+C and resumed by running it again; see "Resumability". To
reprocess everything from scratch (e.g. after editing prompts), set `FORCE_REPROCESS = True` in
`config.py`: it ignores the dump ledger, the staging files and the verdict caches.

## Conventions

### Principles

- **Simplicity and reliability** decide between implementations: the one with fewer moving
  parts and fewer ways to fail wins over the more general, faster or more elegant one.
- **DRY** — extract only what duplicates the same _meaning_, not the same shape.
- **KISS** — the simplest solution that satisfies the actual requirement.
- **YAGNI** — build only what is requested: no speculative parameters, flags or configurability.
- **Low coupling** — narrow interfaces, one-directional imports: every step module may use
  `common`, `config` and `matching`, never the other way round.
- **High cohesion** — one module owns one pipeline step.
- When DRY and low coupling conflict, prefer duplication and say so in a comment.

### Naming

Google Python Style Guide. No abbreviations (`msg`, `req`, `cfg`) — full words. Acronyms are
words: `HtmlParser`, `parse_html`, never `HTMLParser`. Settings in `config.py` are
`UPPER_SNAKE_CASE`.

### Language

**English**: logs, exception messages, comments and `TODO`s, docstrings, identifiers.

**Russian** (the model and the reviewers read it — don't "fix" it): every prompt and every text
the model reads (`CATEGORIES` descriptions, `DOMAIN_NAME`, docstrings and field descriptions of
the structured output schemas), and `README.md`, which is for people. Every prompt opens with the
model's role («Ты — сортировщик …», «Ты — редактор …») and carries few-shot examples, taken from
real tickets where possible.

### Comments and docstrings

Only where the code alone doesn't tell _why_ (a non-obvious decision, an invariant, a trap), and
then brief and dense — one or two sentences. No comment that narrates the next line; delete a
comment the code has outgrown rather than leave it stale. A module-level docstring stating a design
constraint is the norm here; a restatement of the signature is noise.

### Block comment separators

Only between large logical blocks, never between every function; one line, dashes to about
column 79, no `=`, no boxes:

```python
# ----- Prompt ----------------------------------------------------------------
```

## Pipeline

`pipeline.py` is the entry point. For every dump listed in `config.DUMP_PATHS` (processed in the
given order, oldest to freshest — later dumps update matching entries from earlier ones) that isn't
already in `data/processed_dumps.json` (matched by file SHA-256), it runs three steps, then rebuilds
the Excel export once at the end:

1. **`extract.py`** — two LLM calls per ticket pair, run in parallel: a _filter_ call (does the pair
   generalize beyond one ticket — reusable question, instructional not one-off, complete, no private
   data) and a _transform_ call (rewrite into canonical form with a category from
   `config.CATEGORIES`). A pair failing the filter is written to `rejected.json` with a reason
   instead of being dropped silently.
2. **`deduplicate.py`** — collapses duplicates within one batch. `matching.py` finds _candidate_
   pairs cheaply from three local signals, united: cosine similarity of GigaChat question
   embeddings (the only signal that sees a paraphrase; `EMBEDDING_BACKEND = "none"` turns it
   off), trigram/Jaccard similarity over questions and, separately, over answers (support often
   pastes the same instruction under different reported symptoms, so answer similarity is its own
   signal) — no LLM call is made for pairs that don't clear this bar. Each candidate pair is then
   judged by the LLM as `same` (merge), `alternatives` (kept as a list of possible causes), or
   `contradiction` (kept separate, logged as a warning). Groups form around a representative
   entry — a pair only joins a group if the LLM confirmed it against the group's representative
   specifically; similarity is **not transitive**. A group is merged in two steps
   (`deduplicate.merge_group`): `same` answers collapse into one full answer first, then the
   different causes that remain become one numbered list.
3. **`merge.py`** — merges the deduplicated batch into the living base (one-to-one matching, by
   the same three signals). On a matched entry with differing answers `config.MERGE_STRATEGY`
   decides: `"accumulate"` appends the new cause, `"replace"` lets the fresh answer win. Answers
   that say the same thing, or contradict each other, are always replaced by the fresh one.
4. **`export.py`** — rebuilds `data/knowledge_base.xlsx` from the JSON base (frozen header row,
   autofilter, one column per `config.SOURCE_EXTRA_COLUMNS` entry).

**`dedupe_base.py`** (`make dedupe-base`) is run by hand, outside the pipeline: it compares the
entries already in the base against each other, which the pipeline never does (and one-to-one
matching leaves a second duplicate in the base untouched). Candidates come from question
embeddings plus question trigrams, categories ignored; without embeddings the trigram threshold
drops to `TRIGRAM_ONLY_CANDIDATE_THRESHOLD`. The rest is `deduplicate.collapse_duplicates`, the
same judge-group-merge flow the pipeline runs on a batch. Its verdict cache lives in
`data/staging/base_dedupe/`.

### Guardrails on LLM output

Every generated entry is validated in code, not trusted (`common.validate_entry`): category must be
one of `config.CATEGORIES`, question/answer must respect word-count limits, and — the main check —
every number in the rewritten answer must appear in the source text
(`common.find_invented_numbers`). A merged "list of causes" answer must also be at least as long as
the longest source answer it merges (`deduplicate.validate_merged`); its word limit grows with the
number of causes (`MAX_ANSWER_WORDS_PER_CAUSE`). A failed check never destroys data: a group with
different causes is left as separate entries, a cluster of identical answers keeps its most
detailed entry with the provenance of all of them.

### Resumability

Per-dump state lives in `data/staging/<dump-stem>-<hash-prefix>/` (`common.staging_dir_for`, keyed
by content hash so a corrected dump doesn't collide with the original): `extract.py` skips rows
already in that dump's `extracted.json` (rows that _failed_ rather than got rejected are retried —
a failure is not a verdict); `deduplicate.py` reuses verdicts from `verdicts.json` and `merge.py`
from `match_verdicts.json` (`deduplicate.VerdictCache`, keyed by the content hashes of the two
entries, never by their positions). Question embeddings are cached once for every step in
`data/staging/embeddings.npz`, keyed by question text. The base is copied to `data/backups/`
before every write; writes are atomic (temp file + `os.replace`, `common.save_json`).

### Domain configuration

Retargeting the pipeline at another support domain means editing `config.py`, not the code:
`DUMP_PATHS`, `DOMAIN_NAME` (substituted into every prompt), `CATEGORIES` (key → description shown
to the model), `QUESTION_COLUMNS` / `ANSWER_COLUMNS` (joined with `COLUMN_SEPARATOR`),
`SOURCE_EXTRA_COLUMNS` (copied per entry; a missing column is a warning — check with
`make inspect`). Deduplication thresholds: question ones (`CANDIDATE_THRESHOLD`, `MATCH_THRESHOLD`)
are deliberately low — a false candidate costs one LLM call, a missed one leaves a permanent
duplicate; the `ANSWER_*` ones are deliberately high — shared boilerplate must not merge unrelated
problems; `*_CROSS_CATEGORY_THRESHOLD` gates comparisons across model-assigned categories;
`EMBEDDING_CANDIDATE_THRESHOLD` / `EMBEDDING_TOP_K` bound the embedding signal, and
`EMBEDDING_BACKEND = "none"` runs on trigrams alone while the embeddings model is unavailable.

### Shared infrastructure (`common.py`)

Atomic JSON load/save; `ProgressBar`/`ProgressAwareHandler` (one-line bar on a tty, periodic log
lines otherwise); `render_prompt` does literal `<<KEY>>` substitution — not `str.format`, because
templates contain literal `{` `}` from JSON examples; `invoke_structured` calls the LLM with
structured output (function calling against a Pydantic schema: `common.Entry`, and a verdict schema
next to each prompt; docstrings and field descriptions are Russian, the model reads them), retries
any failure — a GigaChat error, a failed validation, a reply without the function call — with
backoff and waits on `wait_for_network` when the GigaChat host drops; a failed attempt is logged
with the model's raw reply (`describe_reply`); `configure_logging` also appends every warning and
error to `data/errors.log`; `build_llm(max_tokens)` / `build_embedder` / `embed_texts` are the only places that
construct GigaChat clients (merges pass `MERGE_MAX_TOKENS`); `hash_text` / `hash_entry` are the
content keys of every cache; `merge_sources` / `merge_source_columns` keep provenance when tickets
collapse into one entry.

## Data layout (git-ignored)

`data/`: `knowledge_base.json` / `.xlsx` (the live base and its view), `processed_dumps.json`
(ledger of processed dump hashes), `errors.log` (warnings and errors of every run),
`staging/` (resumable intermediate state; `staging/<dump>/rejected.json` holds the filtered-out
pairs with reasons), `backups/` (pre-write snapshots of the base). The dumps
contain real tickets: never commit them or anything under `data/`. GigaChat mTLS certificates live
in `.gigachat/` (`client-cert.pem`, `client-cert.key`).
