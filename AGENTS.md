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
make inspect        # check a new dump's column names/samples against config.py
make run            # full pipeline run under caffeinate (network calls die when the Mac sleeps)
make dedupe-base    # deduplicate the living base by question embeddings
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
the model reads (`CATEGORIES` descriptions, `DOMAIN_NAME`), and `README.md`, which is for people.

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
   pairs cheaply via local trigram/Jaccard similarity over questions and, separately, over answers
   (support often pastes the same instruction under different reported symptoms, so answer
   similarity is its own signal) — no LLM call is made for pairs that don't clear this bar. Each
   candidate pair is then judged by the LLM as `same` (merge), `alternatives` (kept as a list of
   possible causes), or `contradiction` (kept separate, logged as a warning). Groups form around a
   representative entry — a pair only joins a group if the LLM confirmed it against the group's
   representative specifically; similarity is **not transitive**.
3. **`merge.py`** — merges the deduplicated batch into the living base (one-to-one matching, again
   by trigrams). On a matched entry with differing answers `config.MERGE_STRATEGY` decides:
   `"accumulate"` appends the new cause, `"replace"` lets the fresh answer win. Answers that say the
   same thing, or contradict each other, are always replaced by the fresh one.
4. **`export.py`** — rebuilds `data/knowledge_base.xlsx` from the JSON base (frozen header row,
   autofilter, one column per `config.SOURCE_EXTRA_COLUMNS` entry).

**`dedupe_base_embeddings.py`** (`make dedupe-base`) is run by hand, outside the pipeline: it
compares the entries already in the base against each other, which the pipeline never does.
Candidates come from cosine similarity of GigaChat question embeddings plus question trigrams,
categories ignored; `EMBEDDING_BACKEND = "none"` falls back to trigrams only. A group is merged in
two steps: `same` answers collapse into one, then different causes become one list. Its caches in
`data/staging/base_dedupe_embeddings/` are keyed by content, not by position. `dedupe_base.py` is
its trigram-only predecessor.

### Guardrails on LLM output

Every generated entry is validated in code, not trusted (`common.validate_entry`): category must be
one of `config.CATEGORIES`, question/answer must respect word-count limits, and — the main check —
every number in the rewritten answer must appear in the source text
(`common.find_invented_numbers`). A merged "list of causes" answer must also be at least as long as
the longest source answer it merges (`deduplicate.validate_merged`). A failed check never destroys
data: a group with different causes is left unmerged, a group of identical answers keeps the
representative with its own sources.

### Resumability

Per-dump state lives in `data/staging/<dump-stem>-<hash-prefix>/` (`common.staging_dir_for`, keyed
by content hash so a corrected dump doesn't collide with the original): `extract.py` skips rows
already in that dump's `extracted.json`; `deduplicate.py` reuses verdicts from `verdicts.json`
(`deduplicate.VerdictCache`). The base is copied to `data/backups/` before every write; writes are
atomic (temp file + `os.replace`, `common.save_json`).

### Domain configuration

Retargeting the pipeline at another support domain means editing `config.py`, not the code:
`DUMP_PATHS`, `DOMAIN_NAME` (substituted into every prompt), `CATEGORIES` (key → description shown
to the model), `QUESTION_COLUMNS` / `ANSWER_COLUMNS` (joined with `COLUMN_SEPARATOR`),
`SOURCE_EXTRA_COLUMNS` (copied per entry; a missing column is a warning — check with
`make inspect`). Deduplication thresholds: question ones (`CANDIDATE_THRESHOLD`, `MATCH_THRESHOLD`)
are deliberately low — a false candidate costs one LLM call, a missed one leaves a permanent
duplicate; the `ANSWER_*` ones are deliberately high — shared boilerplate must not merge unrelated
problems; `*_CROSS_CATEGORY_THRESHOLD` gates comparisons across model-assigned categories.

### Shared infrastructure (`common.py`)

Atomic JSON load/save; `ProgressBar`/`ProgressAwareHandler` (one-line bar on a tty, periodic log
lines otherwise); `render_prompt` does literal `<<KEY>>` substitution — not `str.format`, because
templates contain literal `{` `}` from JSON examples; `invoke_json` calls the LLM, parses a JSON
object from the reply, retries with backoff and waits on `wait_for_network` when the GigaChat host
drops; `merge_sources` / `merge_source_columns` keep provenance when tickets collapse into one
entry.

## Data layout (git-ignored)

`data/`: `knowledge_base.json` / `.xlsx` (the live base and its view), `processed_dumps.json`
(ledger of processed dump hashes), `rejected.json` (filtered-out pairs with reasons),
`staging/` (resumable intermediate state), `backups/` (pre-write snapshots of the base). The dumps
contain real tickets: never commit them or anything under `data/`. GigaChat mTLS certificates live
in `.gigachat/` (`client-cert.pem`, `client-cert.key`).
