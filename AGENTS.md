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
pandas/openpyxl for Excel. Two scenarios, each one make target:

1. **Tickets to base** (`make run`) — grows a living base from an Excel dump of support
   tickets.
2. **Repair a base** (`make repair-base`) — turns a poor base (an Excel sheet of questions and
   answers) into a good one, plus a sheet of the rows left out and why.

## Configuration

Two levels, never mixed:

- **`config.py`** — technical settings shared by every run: GigaChat, retries, workers,
  thresholds, word limits, Excel styling, and the technical storage (`DATA_DIR`,
  `BASES_DIR`, `REPAIRS_DIR`, `EMBEDDINGS_CACHE`, `ERROR_LOG`).
- **A TOML config per run** (`configs/`), the only command-line argument of every script
  (`make <target> CONFIG=...`). It holds only what that script needs, and its files are Excel
  only — people hand in and get back `.xlsx`, JSON is technical state. Read by
  `kb/utils/settings.py` into pydantic models with `extra="forbid"` (a typo is an error):
  - every config: `name` (the base; a directory name and the start of the output file
    names), `output_dir`, `[domain]` with `name` (substituted into every prompt) and
    `[domain.categories]` (key stored in the entry → description shown to the model);
  - `TicketsSettings` (`make run`): `merge_strategy` and `[dump]` — one dump: `path`,
    `question_columns`, `answer_columns`, `extra_columns`;
  - `DedupeSettings` (`make dedupe-base`): nothing more;
  - `RepairSettings` (`make repair-base`): `[input]` — the base to repair, same keys as
    `[dump]`.

An output file is `<output_dir>/<name>_<YYYY-MM-DD>.xlsx` (a repair adds
`<name>_<date>_rejected.xlsx`), so every run leaves its own dated version. The base is
identified by `name` only: any dump in `[dump].path` updates the base of that name. Steps get
the `settings.Domain` passed explicitly; prompts and the category enum of the schemas are
built from it per call.

Configs in the repository: `configs/supplier-portal-sap.toml` (`make run`) and
`configs/supplier-portal-sap-dedupe.toml` (`make dedupe-base`) for the base «Портал
поставщика SAP».

## Commands

```bash
make setup          # venv + uv + requirements.txt (PYTHON and SBEROSC_TOKEN from .env, see .env.example)
source activate.sh  # activate the venv with the same environment as the Makefile
make inspect CONFIG=...      # check the source file of a run or repair config and that GigaChat and embeddings answer
make models                  # list the models GigaChat makes available to the certificate
make run CONFIG=...          # scenario 1 under caffeinate (network calls die when the Mac sleeps)
make dedupe-base CONFIG=...  # scenario 1 maintenance: deduplicate a base against itself
make repair-base CONFIG=...  # scenario 2
make help                    # every target
```

Every command is a module run with `python -m` from the repository root (see the Makefile).
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
`LINK_PLACEHOLDER_ATTEMPTS` calls in all) before the validation rejects the entry.

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
  `logs` (logger, `ProgressBar`,
  `configure_logging`), `storage` (atomic JSON, backups, content hashes, `staging_dir_for`),
  `gigachat` (clients, `invoke_structured`, embeddings, network wait), `prompts`
  (`render_prompt`, the shared `WRITING_STYLE` and `LINKS_RULE` blocks), `links` (link
  placeholders), `entries` (the `Entry` schema, `validate_entry`, provenance, `new_entry` /
  `merged_entry`), `excel` (`read_pairs` from any sheet by configured columns, `write_knowledge_base`,
  `write_rejected`), `batch` (`process_rows`: resumable parallel row processing into entries and
  rejections).
- **`kb/steps/`** — one transformation each, on lists of entries, with its prompt and schema:
  `filtering`, `rewriting`, `repairing`, `matching`, `dedup`, `merging`.
- **`kb/scenarios/`** — entry points that compose the steps: `tickets_to_base`, `dedupe_base`,
  `repair_base`. **`kb/tools/`** — `inspect_dump`, `list_models`.

## Scenario 1: tickets to base (`kb/scenarios/tickets_to_base.py`)

Merges the dump of `[dump].path` into the base `name` (`data/bases/<name>/knowledge_base.json`),
unless it is already in that base's `processed_dumps.json` (matched by file SHA-256), then
writes the Excel view. Dumps are fed one run at a time, oldest to freshest — a later dump
updates matching entries from earlier ones:

1. **Filter and rewrite** (`filtering`, `rewriting`, run through `batch.process_rows`) — two LLM
   calls per ticket: a _filter_ (does the pair generalize beyond one ticket — reusable question,
   instructional not one-off, complete) and a _transform_ (rewrite into canonical form with a
   category from `[domain.categories]`; names become roles, a vague question is rebuilt from the
   answer). A pair failing the filter is written to `rejected.json` with a reason instead of being
   dropped silently.
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
   signals. `merging` works on lists; the scenario loads, backs up and saves the base. On a matched
   entry with differing answers `merge_strategy` of the config decides:
   `"accumulate"` appends the new cause, `"replace"` lets the fresh answer win. Answers that say
   the same thing, or contradict each other, are always replaced by the fresh one.

**`dedupe_base`** (`make dedupe-base`) is run by hand: it compares the entries already in the base
against each other, which scenario 1 never does (and one-to-one matching leaves a second duplicate
untouched). Candidates come from question embeddings plus question trigrams, categories ignored;
without embeddings the trigram threshold drops to `TRIGRAM_ONLY_CANDIDATE_THRESHOLD`. The rest is
`dedup.collapse_duplicates`. Its verdict cache lives in `data/bases/<name>/staging/base_dedupe/`.

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
vectors, which only costs re-embedding) or appended (`errors.log`). Two runs on the same base are not supported.

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
by the content hashes of the two entries, never by their positions). Question embeddings are cached
once for every step and every base in `data/embeddings.npz`, keyed by question text. The base
is copied to `data/bases/<name>/backups/` before every write; writes are atomic (temp file + `os.replace`,
`storage.save_json`).

### Domain configuration

Retargeting at another support domain means a new TOML config, not code: `[domain]` and the
columns of the source (question and answer columns joined with `config.COLUMN_SEPARATOR`;
`extra_columns` copied per entry, a missing one is a warning — check with `make inspect`).
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
runs share the server's limit, so lower `WORKER_COUNT` when they keep hitting it. The library's own
429 warnings are silenced in `configure_logging`, the pause is logged once. `logs.configure_logging` also appends every warning and error to
`data/errors.log`. `gigachat.build_llm(max_tokens)` / `build_embedder` are the only places that
construct GigaChat clients (merges pass `MERGE_MAX_TOKENS`); `storage.hash_text` / `hash_entry` are
the content keys of every cache.

## Data layout (git-ignored)

`data/` (`config.DATA_DIR`, technical state only): `bases/<name>/` — `knowledge_base.json` (the
source of truth of a base), `processed_dumps.json` (ledger of processed dump hashes), `staging/`
(resumable intermediate state; `staging/<source>/rejected.json` holds the left-out rows with
reasons), `backups/` (pre-write snapshots of the base); `repairs/<name>/staging/` (scenario 2);
`embeddings.npz`; `errors.log` (warnings and errors of every run). `output/` — the Excel files
of the committed configs. The input files contain real tickets: never commit them or anything
under `data/` or `output/`. GigaChat mTLS certificates live
in `.gigachat/` (`client-cert.pem`, `client-cert.key`).
