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

Builds and maintains a support knowledge base with GigaChat. Python 3.13, `langchain-gigachat`,
pandas/openpyxl for Excel. Each scenario is one make target:

1. **Tickets to base** (`make run`) — creates a living base from an Excel dump of support
   tickets, or updates it from the next dump.
2. **Merge a base** (`make merge-base`) — merges a good base (an Excel this project wrote) into
   a living base.
3. **Repair a base** (`make repair-base`) — turns a poor base (an Excel sheet of questions and
   answers) into a good, separate one, plus a sheet of the rows left out and why. A poor base is
   never merged as is: repair it, then merge the result.

People review and edit the Excel files of a base; the JSON in the technical storage follows them
(`living_base.sync_from_excel`, see "Living base files").

## Configuration

Two levels, never mixed:

- **`config.py`** — technical settings shared by every run: GigaChat, retries, workers,
  thresholds, word limits, Excel styling, and the storage: `KNOWLEDGE_BASES_DIR`
  (`/Users/19480633/Desktop/Базы знаний`, the working folder: «Обращения» — the dumps; «Готовые базы» — the bases, every Excel file the
  runs write, `output_dir` of every config) and in it `DATA_DIR`
  («Технические данные»: `BASES_DIR`, `REPAIRS_DIR`, `EMBEDDINGS_CACHE`, `TOOLS_LOG_DIR`).
- **A TOML config per run** (`configs/`), the only command-line argument of every script
  (`make <target> CONFIG=...`). It holds only what that script needs, and its files are Excel
  only — people hand in and get back `.xlsx`, JSON is technical state. Read by
  `kb/utils/settings.py` into pydantic models with `extra="forbid"` (a typo is an error):
  - every config: `name` (the base; a directory name and the start of the output file
    names), `output_dir`, `[domain]` with `name` (substituted into every prompt) and
    `[domain.categories]` (key stored in the entry → description shown to the model);
  - `TicketsSettings` (`make run`): `merge_strategy` and `[dump]` — one dump: `path`,
    `question_columns`, `answer_columns`, `extra_columns`;
  - `MergeBaseSettings` (`make merge-base`): `merge_strategy` and `[source]` — `path` of the
    good base's Excel; its columns are this project's own (`excel.read_knowledge_base`);
  - `DedupeSettings` (`make dedupe-base`): nothing more;
  - `RepairSettings` (`make repair-base`): `[input]` — the base to repair, same keys as
    `[dump]`.

An output file is `<output_dir>/<name>_<YYYY-MM-DD>.xlsx` (a repair adds
`<name>_<date>_rejected.xlsx`), so every run leaves its own dated version. The base is
identified by `name` only: any dump in `[dump].path` updates the base of that name. Steps get
the `settings.Domain` passed explicitly; prompts and the category enum of the schemas are
built from it per call.

Configs in the repository, all with `output_dir` set to `KNOWLEDGE_BASES_DIR/Готовые базы`: for the base
«Портал поставщика SAP» `configs/supplier-portal-sap.toml` (`make run`, dump
`Обращения/Обращения ПП.xlsx`), `configs/supplier-portal-sap-merge.toml` (`make merge-base`,
the source path an example), `configs/supplier-portal-sap-dedupe.toml` (`make dedupe-base`). A
base made elsewhere becomes a living base once its file is named `<name>_<YYYY-MM-DD>.xlsx` and
has this project's columns (`excel.read_knowledge_base`); otherwise repair it first.

## Commands

```bash
make setup          # venv + uv + requirements.txt (PYTHON and SBEROSC_TOKEN from .env, see .env.example)
source activate.sh  # activate the venv with the same environment as the Makefile
make inspect CONFIG=...      # check the source file of a config, both certificates, and that the chat and embeddings models answer
make models                  # list the models each certificate (chat, embeddings) is granted, marking the configured ones
make run CONFIG=...          # scenario 1 under caffeinate (network calls die when the Mac sleeps)
make merge-base CONFIG=...   # merge a good base into a base
make dedupe-base CONFIG=...  # scenario 1 maintenance: deduplicate a base against itself
make repair-base CONFIG=...  # scenario 2
make help                    # every target
```

Every command is a module run with `python -m` from the repository root (see the Makefile).
There is no test suite, linter or type-checker in this repo — don't invent commands for them.
`requirements.txt` is pinned by hand; `make setup` installs it.

A run can be interrupted with Ctrl+C and resumed by running it again; see "Resumability". Ctrl+C
cancels the queued model calls and waits only for the running ones (`parallel.workers`). To
reprocess everything from scratch (e.g. after editing prompts), set `FORCE_REPROCESS = True` in
`config.py`: it ignores the dump ledger, the staging files and the verdict caches.

## Conventions

### Principles

- **Simplicity and reliability** decide between implementations: the one with fewer moving
  parts and fewer ways to fail wins over the more general, faster or more elegant one.
- **DRY** — extract only what duplicates the same _meaning_, not the same shape.
- **KISS** — the simplest solution that satisfies the actual requirement.
- **YAGNI** — build only what is requested: no speculative parameters, flags or configurability.
- **Low coupling** — narrow interfaces, one-directional imports: `kb/scenarios` and `kb/tools`
  → `kb/steps` → `kb/utils` → `config`. A step may build on another step (`merging` on `dedup`
  and `matching`), never on a scenario; a utility knows nothing of steps.
- **High cohesion** — one module owns one step or one kind of infrastructure; a scenario only
  composes steps and holds no logic a second scenario would need.
- When DRY and low coupling conflict, prefer duplication and say so in a comment.

### Naming

Google Python Style Guide. No abbreviations (`msg`, `req`, `cfg`) — full words. Acronyms are
words: `HtmlParser`, `parse_html`, never `HTMLParser`. Settings in `config.py` are
`UPPER_SNAKE_CASE`.

### Language

**English**: logs, exception messages, comments and `TODO`s, docstrings, identifiers.

**Russian** (the model and the reviewers read it — don't "fix" it): every prompt and every text
the model reads (category descriptions and the domain name of the TOML configs, docstrings and field descriptions of
the structured output schemas), and `README.md`, which is for people. Every prompt opens with the
model's role («Ты — сортировщик …», «Ты — редактор …») and carries few-shot examples, taken from
real tickets where possible. Every prompt that writes a question or an answer of the base includes
the one shared style block `prompts.WRITING_STYLE` (Ilyakhov's «Пиши, сокращай») through the
`<<WRITING_STYLE>>` placeholder — change the style there, never per prompt. The model never
writes a link: GigaChat cannot copy a long percent-encoded one. Before every call that writes text
(`rewriting`, `repairing`, `dedup.merge_entries`) the links of the source become placeholders
(`links.mask_links`, «[ссылка-1]»), the prompt tells the model to move them (`prompts.LINKS_RULE`),
and `links.unmask_links` puts the exact source links back; prompt examples show placeholders too.
A placeholder the model declines («по [ссылке-1]») is still recognised; one it made up is asked
again with a hint naming the placeholders that exist (`links.ask_with_links`, up to
`LINK_PLACEHOLDER_ATTEMPTS` calls in all) before the validation rejects the entry. Numbers follow
the one shared block `prompts.NUMBERS_RULE` (copied as they are; a year not expanded, a number word
may become digits — what `entries.find_invented_numbers` lets through). Every prompt that writes an
entry shows the categories with their descriptions (`prompts.format_categories`), and none removes an
error code or a role number from a question.

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

## Layout

`config.py` (repository root) holds the technical settings, `configs/` the TOML configs of
the runs. Everything else lives in `kb/`:

- **`kb/utils/`** — infrastructure with no knowledge base logic: `settings` (TOML configs),
  `logs` (logger, rich `ProgressBar`, model call
  records, `logs.run`), `storage` (atomic JSON, backups, content hashes, `staging_dir_for`),
  `gigachat` (clients, `invoke_structured`, embeddings, network wait), `prompts`
  (`render_prompt`, the shared `WRITING_STYLE` and `LINKS_RULE` blocks), `links` (link
  placeholders), `entries` (the `Entry` schema, `validate_entry`, provenance, `new_entry` /
  `merged_entry`), `excel` (`read_pairs` from any sheet by configured columns,
  `write_knowledge_base` and its inverse `read_knowledge_base`, `write_rejected`), `living_base`
  (the files of a living base: sync from its newest Excel, save with a backup, export, the
  ledger of merged files), `batch` (`process_rows`: resumable parallel row processing into entries and
  rejections), `parallel` (`workers`: the thread pool of every parallel step, whose queue an
  exception cancels).
- **`kb/steps/`** — one transformation each, on lists of entries, with its prompt and schema:
  `filtering`, `rewriting`, `repairing`, `matching`, `dedup`, `merging`.
- **`kb/scenarios/`** — entry points that compose the steps: `tickets_to_base`, `merge_base`,
  `dedupe_base`, `repair_base`. **`kb/tools/`** — `inspect_dump`, `list_models`.

## Scenario 1: tickets to base (`kb/scenarios/tickets_to_base.py`)

Syncs the base `name` from its newest Excel, merges the dump of `[dump].path` into it — the
first dump creates the base — unless the dump is already in the base's `processed_dumps.json`
(matched by file SHA-256), then writes today's Excel. Dumps are fed one run at a time, oldest to freshest — a later dump
updates matching entries from earlier ones:

1. **Filter and rewrite** (`filtering`, `rewriting`, run through `batch.process_rows`) — two LLM
   calls per ticket: a _filter_ (does the pair generalize beyond one ticket — reusable question,
   instructional not one-off, complete) and a _transform_ (rewrite into canonical form with a
   category from `[domain.categories]`; names become roles, a vague question is rebuilt from the
   answer). The model scores the flags; the code decides
   (`filtering.failed_flags`): a pair failing at most `MAX_DOUBTFUL_FLAGS` (1) is kept as
   _doubtful_ — the entry stores `doubtful`, the reason for a reviewer (`filtering.doubt_reason`),
   and its row in the Excel is yellow with the reason in «Почему спорная»; a pair failing more is
   written to `rejected.json` with the reason (`filter:<flags>`). An entry merged from several
   (`entries.merged_entry`), or replacing one with the same answer (`merging`), stays doubtful only
   when every source was: another ticket with the same question confirms it.
2. **Deduplicate the batch** (`matching`, `dedup`) — `matching` finds _candidate_ pairs cheaply
   from three local signals, united: cosine similarity of GigaChat question embeddings (the only
   signal that sees a paraphrase; `EMBEDDING_BACKEND = "none"` turns it off), trigram/Jaccard
   similarity over questions and, separately, over answers (support often pastes the same
   instruction under different reported symptoms) — no LLM call is made for pairs that don't clear
   this bar. `dedup.collapse_duplicates` has the LLM judge each candidate as `same`,
   `alternatives` or `contradiction` (never merged, logged). Groups form around a representative
   entry — a pair only joins a group if the LLM confirmed it against the representative itself;
   similarity is **not transitive**. A group is merged in two steps (`dedup.merge_group`): `same`
   answers collapse into one full answer first, then the different causes become one list.
3. **Merge into the base** (`merging.merge_into_base`) — one-to-one matching by the same three
   signals. `merging` works on lists; the scenario syncs, saves and exports the base (`living_base`). On a matched
   entry with differing answers `merge_strategy` of the config decides:
   `"accumulate"` appends the new cause, `"replace"` lets the fresh answer win. Answers that say
   the same thing, or contradict each other, are always replaced by the fresh one. A cause the
   model fails to append is not lost: the old entry stays and the new one is added next to it.

**`dedupe_base`** (`make dedupe-base`) is run by hand: it compares the entries already in the base
against each other, which scenario 1 never does (and one-to-one matching leaves a second duplicate
untouched). Candidates come from question embeddings plus question trigrams, categories ignored;
without embeddings the trigram threshold drops to `TRIGRAM_ONLY_CANDIDATE_THRESHOLD`. The rest is
`dedup.collapse_duplicates`. It syncs from the newest Excel first, like every scenario on a
living base. Its verdict cache lives in `<base work dir>/staging/base_dedupe/`.

**`merge_base`** (`make merge-base`) reads the good base with `excel.read_knowledge_base` — its
entries are canonical already, so no filter, rewriting or in-batch dedup — marks doubtful an
entry whose category the base does not have, and runs `merging.merge_into_base` into the synced
base, as step 3 above. The source file goes into the same ledger as dumps; match verdicts are
cached in `staging/<source stem>-<hash>/`.

## Living base files (`kb/utils/living_base.py`)

The Excel of a base holds every field of an entry (question, answer, category, doubt reason,
source file and rows, update date, extra source columns), so `excel.read_knowledge_base` reads
it back losslessly. Every scenario on a living base (`run`, `merge-base`, `dedupe-base`) starts
with `sync_from_excel`: the newest `<output_dir>/<name>_<YYYY-MM-DD>.xlsx`, by the date in its
name, becomes the base JSON when it was saved after the JSON (an older one was not edited since, and reading it back would undo a run that stopped before its export); the JSON is backed up first and written only when it differs, order ignored. So a
row the reviewer deleted does not come back with the next dump, an edit stays, a cleared doubt
reason confirms the entry, and an emptied question or answer drops the row.

## Scenario 2: repair a base (`kb/scenarios/repair_base.py`)

Reads `[input]` of the repair config (columns «Вопрос» / «Ответ» read an Excel this project
wrote). There is **no** "belongs in a base" or
"one-off answer" check — somebody already put the entry there. Per row, `repairing` makes one LLM
call that repairs rather than rejects: the answer holds the knowledge, so a vague question, a bare
request or a question the answer does not quite answer is rewritten to fit the answer, a partial
answer is completed from the question, both are brought to the canonical format — keeping the
names and contacts of people in charge, unlike scenario 1. Only a row whose
answer holds nothing to keep (empty, cut off, a reply from the conversation, an answer that
explains nothing) is left out, with the model's reason. Long regulatory answers get their own
ceiling (`REPAIR_MAX_ANSWER_WORDS`) and output budget (`REPAIR_MAX_TOKENS`). A row without an
answer is left out without a call. Then
`dedup.collapse_duplicates` over the whole result, candidates as in scenario 1. Outputs go to
`output_dir`: `<name>_<date>.xlsx` and `<name>_<date>_rejected.xlsx` (source row, reason,
question, answer, and the model's question and answer when the code validation rejected them —
`batch.Outcome.model_fields`, in both scenarios' `rejected.json` too). Staging lives in
`data/repairs/<name>/`, apart from the living bases, which are never touched.

Runs with different configs may go in parallel: staging is per base name, and the files
several runs share are written through a per-process temporary file (`storage.save_json`,
`matching.save_embedding_cache`; two runs adding embeddings at once may drop each other's new
vectors, which only costs re-embedding) Two runs on the same base are not supported.

### Guardrails on LLM output

Every model reply comes through structured output (`gigachat.invoke_structured`: function calling
against a Pydantic schema; the schema lives next to its prompt, `entries.entry_schema(domain)` for an entry).
Every generated entry is validated in code, not trusted (`entries.validate_entry`): category must be one of `[domain.categories]`, question/answer must respect word-count limits, and — the main check —
every link in the rewritten answer must be verbatim in the source (`entries.find_invented_links`: a
"fixed" encoded link or one copied from a prompt example leads nowhere), and every number must
appear in the source question or answer (`entries.find_invented_numbers`; links and list markers are
left out, a number word in the source such as «десять» counts as its digits, leading zeros are
ignored, a two-digit year written out as «2026» counts as «26»); a placeholder left over from an
unknown link rejects the entry. A merged "list of causes" answer must also be at least as long
as the longest source answer it merges (`dedup.validate_merged`); its word limit grows with the
number of causes (`MAX_ANSWER_WORDS_PER_CAUSE`). A failed check never destroys data: a group with
different causes is left as separate entries, a cluster of identical answers keeps its most
detailed entry with the provenance of all of them.

### Resumability

Per-source state lives in `<base work dir>/staging/<stem>-<hash-prefix>/`
(`storage.staging_dir_for`, keyed by content hash so a corrected file doesn't collide with the
original): `batch.process_rows` skips rows already in the accepted or rejected file (rows
that _failed_ rather than got rejected are retried — a failure is not a verdict); `dedup` reuses
verdicts from `verdicts.json` and `merging` from `match_verdicts.json` (`dedup.VerdictCache`, keyed
by the content hashes of the two entries, never by their positions). Rows and verdicts are saved
every `SAVE_EVERY` items and on the way out, an interrupted run included. Question embeddings are cached
once for every step and every base in `data/embeddings.npz`, keyed by question text. The base
is copied to `<base work dir>/backups/` before every write; writes are atomic (temp file + `os.replace`,
`storage.save_json`).

### Domain configuration

Retargeting at another support domain means a new TOML config, not code: `[domain]` and the
columns of the source (question and answer columns joined with `config.COLUMN_SEPARATOR`;
`extra_columns` copied per entry, a missing one is skipped with a note — check with `make inspect`).
Deduplication thresholds, in `config.py`: question ones (`CANDIDATE_THRESHOLD`,
`MATCH_THRESHOLD`) are deliberately low — a false candidate costs one LLM call, a missed one leaves
a permanent duplicate; the `ANSWER_*` ones are deliberately high — shared boilerplate must not merge
unrelated problems; `*_CROSS_CATEGORY_THRESHOLD` gates comparisons across model-assigned
categories; `EMBEDDING_CANDIDATE_THRESHOLD` / `EMBEDDING_TOP_K` bound the embedding signal, and
`EMBEDDING_BACKEND = "none"` runs on trigrams alone while the embeddings model is unavailable.

### Shared infrastructure notes

`prompts.render_prompt` does literal `<<KEY>>` substitution — not `str.format`, because templates
contain literal `{` `}` from JSON examples. Every GigaChat call, chat and embeddings, goes through
`gigachat.call_with_retries`: a failure — a GigaChat error, a failed validation, a reply without the
function call — uses one of `MAX_RETRIES` (10) attempts with a growing pause and is logged with the
model's raw reply (`describe_reply`); a 429 uses none — every thread of the process waits out one
shared cooldown that doubles while 429s keep coming (`RATE_LIMIT_*`), so the process backs off as a
whole; the host dropping is waited out by `wait_for_network`. The cooldown is per process: parallel
runs share the server's limit; `WORKER_COUNT` is 1 by default, raise it for speed while no 429s
come. The library's own 429 warnings are silenced, the pause is logged once.

Logging (`kb/utils/logs.py`): every entry point runs inside
`logs.run(log_dir, command)`. The terminal gets this project's INFO lines (progress of the steps,
pauses for the network and 429), a rich progress bar per step, and above it every item that went
wrong — a row rejected or failed, with each of its model calls (step, seconds, `ok` / `429` /
`error: <class>`; slow ones yellow from `SLOW_CALL_SECONDS`, red from twice that), a pair without a
verdict or with contradicting answers. Warnings and errors of every logger, a crash's traceback
included, go to the run's own file `<work dir of the base>/logs/<command>_<time>.log`
(`config.TOOLS_LOG_DIR` for `inspect` and `models`), created only when something is written. When
the run ends, failed or interrupted too, it prints the model calls summed up by step and where the
log file is. `call_with_retries(call, label, caller)` notes every attempt under `caller`
(`logs.note_call`); `batch.process_rows` collects a row's calls with `logs.recording_calls`. Keep
per-item problems out of the terminal's INFO: they are printed above the bar or logged as warnings. `gigachat.build_llm(max_tokens)` / `build_embedder` are the only places that construct GigaChat clients,
each with its own certificate (`CERT_FILE` / `EMBEDDINGS_CERT_FILE`), and stop with the missing file
named when it is not there (`require_certificate`) (merges pass `MERGE_MAX_TOKENS`); `storage.hash_text` / `hash_entry` are
the content keys of every cache.

## Data layout (outside the repository)

`KNOWLEDGE_BASES_DIR/Готовые базы` holds the Excel files of every base (`<name>_<date>.xlsx`, a repair's
`<name>_<date>_rejected.xlsx`) and `DATA_DIR` («Технические данные», technical state only):
`bases/<name>/` — `knowledge_base.json` (follows the newest Excel of the base), `processed_dumps.json` (ledger of the hashes of merged dumps and bases), `staging/`
(resumable intermediate state; `staging/<source>/rejected.json` holds the left-out rows with
reasons), `backups/` (pre-write snapshots of the base); `repairs/<name>/staging/` (scenario 2);
`logs/` under every base and repair (warnings and errors of each run, one file per run);
`embeddings.npz`; `logs/` for `inspect` and `models`. All of it is outside the repository. The
input files contain real tickets: never commit them, a base, or technical state. GigaChat mTLS certificates live
in `.certs/`: `glm.pem` / `glm.key` for the chat model (`CERT_FILE`, `KEY_FILE`), `gigachat.pem` /
`gigachat.key` for the embeddings model (`EMBEDDINGS_CERT_FILE`, `EMBEDDINGS_KEY_FILE`).
