# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Comments, docstrings, and the README in this repo are written in Russian; code identifiers are in English.

## What this is

A batch pipeline that builds and maintains a support knowledge base from Excel exports of support tickets (question/answer pairs). Input: Excel with columns configured in `config.py`. Output: `data/knowledge_base.json` (source of truth) and `data/knowledge_base.xlsx` (review-friendly view). Uses GigaChat (via `langchain-gigachat`) as the LLM.

## Commands

```bash
source activate.sh          # creates/activates .venv (uv if available, else venv), installs requirements.txt if changed
python inspect_dump.py      # one-off: check a new dump's column names/samples against config
caffeinate -i python pipeline.py   # full run; -i keeps the Mac awake (network calls die on sleep); add -s on battery
```

There is no test suite, linter, or type-checker configured in this repo — don't invent commands for them.

A run can be interrupted with Ctrl+C and resumed by re-running `pipeline.py`; see "Resumability" below.

To reprocess everything from scratch (e.g. after editing prompts), set `FORCE_REPROCESS = True` in `config.py`. This ignores the dump ledger, staging files, and the verdict cache.

`make dump [path]` dumps all `.py`/`.md` files under `path` (default `.`) into `.code_dump.txt`, respecting `.gitignore`. `dump.js` (Node) is a more general-purpose version of the same idea for other project types, unrelated to the knowledge-base pipeline itself.

## Pipeline architecture

`pipeline.py` is the entry point. For every dump listed in `config.DUMP_PATHS` (processed in the given order, oldest to freshest — later dumps update matching entries from earlier ones) that isn't already in `data/processed_dumps.json` (matched by file SHA-256), it runs three steps, then rebuilds the Excel export once at the end:

1. **`extract.py`** — two LLM calls per ticket pair, run in parallel: a *filter* call (does the pair generalize beyond one ticket — reusable question, instructional not one-off, complete, no private data) and a *transform* call (rewrite into canonical form with a category from `config.CATEGORIES`). A pair failing the filter is written to `rejected.json` with a reason instead of being dropped silently.

2. **`deduplicate.py`** — collapses duplicates within one batch. `matching.py` finds *candidate* pairs cheaply via local trigram/Jaccard similarity over questions and, separately, over answers (support often pastes the same instruction under different reported symptoms, so answer similarity is its own signal) — no LLM call is made for pairs that don't clear this bar. Each candidate pair is then judged by the LLM as `same` (merge), `alternatives` (kept as a list of possible causes), or `contradiction` (kept separate, logged as a warning). Groups form around a representative entry — a pair only joins a group if the LLM confirmed it against the group's representative specifically; similarity is **not transitive** (A~B and B~C does not imply a group of three).

3. **`merge.py`** — merges the deduplicated batch into the living base (matching is one-to-one: each new entry matches at most one base entry and vice versa). Behavior on a matched entry with differing answers is controlled by `config.MERGE_STRATEGY`: `"accumulate"` appends the new cause to a running list, `"replace"` lets the fresh answer win. Answers that say the same thing, or that contradict each other, are always replaced by the fresh one regardless of strategy. The previous base is copied to `data/backups/` before every merge; writes are atomic (temp file + `os.replace`, see `common.save_json`).

4. **`export.py`** — rebuilds `data/knowledge_base.xlsx` from the JSON base (frozen header row, autofilter, one column per `config.SOURCE_EXTRA_COLUMNS` entry).

### Guardrails on LLM output

Every generated entry is validated in code, not trusted blindly (`common.validate_entry`): category must be one of `config.CATEGORIES`, question/answer must respect word-count limits, and — the main check — every number in the rewritten answer must appear somewhere in the source text (`common.find_invented_numbers`), to catch fabricated figures/steps. A merged "list of causes" answer must additionally be at least as long as the longest source answer it merges (`deduplicate.validate_merged`). A failed check never destroys data: a group with genuinely different causes is left unmerged, and a group of identical answers keeps the representative with its original sources.

### Resumability

Per-dump intermediate state lives in `data/staging/<dump-stem>-<hash-prefix>/` (`common.staging_dir_for`, keyed by content hash so a re-uploaded/corrected dump doesn't collide with the original): `extract.py` skips rows already present in that dump's `extracted.json`; `deduplicate.py` reuses cached verdicts from `verdicts.json` (`deduplicate.VerdictCache`) instead of re-asking the LLM for pairs already judged.

### Domain configuration (`config.py`)

Retargeting this pipeline at a different support domain means editing `config.py`, not the pipeline code:
- `DUMP_PATHS` — ordered list of Excel files to process; all must share the same column layout.
- `DOMAIN_NAME` — substituted into every prompt.
- `CATEGORIES` — dict of category key → description shown to the model; prompts are built from this mapping.
- `QUESTION_COLUMNS` / `ANSWER_COLUMNS` — one or more source columns joined (with `COLUMN_SEPARATOR`) into the question/answer text; `SOURCE_EXTRA_COLUMNS` are copied verbatim per-entry for traceability (missing column → warning, not an error; run `inspect_dump.py` to verify names against a new dump).
- Deduplication thresholds: `CANDIDATE_THRESHOLD`/`MATCH_THRESHOLD` (question similarity, deliberately low — a false-positive candidate only costs one LLM call, a missed one leaves a permanent duplicate) and the parallel `ANSWER_*` thresholds (deliberately high — shared boilerplate phrasing must not merge unrelated problems). `*_CROSS_CATEGORY_THRESHOLD` values gate comparisons across different (model-assigned, thus not fully trusted) categories.
- `MERGE_STRATEGY` — `"accumulate"` vs `"replace"`, see above.
- GigaChat connection settings (model name, base URL, cert paths under `.gigachat/`, timeouts) and runtime knobs (`WORKER_COUNT`, `MAX_RETRIES`, `SAVE_EVERY`, etc.).

### Shared infrastructure (`common.py`)

JSON load/save (atomic writes, see above), the `ProgressBar`/`ProgressAwareHandler` pair (single-line progress bar on a tty, periodic log lines otherwise, coordinated so log output doesn't tear the bar), prompt templating (`render_prompt` does literal `<<KEY>>` substitution — not `str.format`, because prompt templates contain literal `{` `}` from JSON examples), `invoke_json` (calls the LLM, extracts/parses a JSON object from the reply, retries with backoff, waits on `wait_for_network` if the GigaChat host becomes unreachable mid-run), and the entry-merging helpers (`merge_sources`, `merge_source_columns`) used when several tickets collapse into one entry.

## Data layout (gitignored)

`data/` is not committed. Its layout: `knowledge_base.json` / `.xlsx` (the live base and its view), `processed_dumps.json` (ledger of processed dump hashes), `rejected.json` (filtered-out pairs with reasons, per run), `staging/<dump>-<hash>/` (per-dump resumable intermediate state), `backups/` (pre-merge snapshots of the base). GigaChat client certs live in `.gigachat/` (also gitignored).
