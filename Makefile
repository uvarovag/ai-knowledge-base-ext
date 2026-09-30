SHELL := /bin/bash
-include .env
export

PYTHON ?= python3.13

define SETUP_ENV
	source .venv/bin/activate && \
	set -a && { [ ! -f .env ] || source .env; } && set +a && \
	if [ -n "$${SBEROSC_TOKEN}" ]; then \
		export PIP_INDEX_URL="https://token:$${SBEROSC_TOKEN}@sberosc.sigma.sbrf.ru/repo/pypi/simple" && \
		export UV_DEFAULT_INDEX="$${PIP_INDEX_URL}"; \
	fi && \
	export UV_HTTP_TIMEOUT=90 && \
	export UV_CACHE_DIR=.uv-cache && \
	export UV_NATIVE_TLS=1 && \
	export UV_INSECURE_HOST="sberosc.sigma.sbrf.ru"
endef

# UV_NATIVE_TLS makes uv trust the system certificate store instead of its own bundle: the
# corporate SSL_CERT_FILE is a GOST .cer that uv cannot parse, so with the bundle every mirror
# request fails with "invalid peer certificate: UnknownIssuer". UV_INSECURE_HOST covers the
# mirror when the store lacks the root as well. Same as [tool.uv] system-certs and
# allow-insecure-host of a pyproject.toml, which this repo has none.

.PHONY: help setup run inspect mark-vague dedupe-base clean dump dump-diff

help: ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.*Makefile://' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv, install uv and requirements.txt
	$(PYTHON) -m venv .venv
	@$(SETUP_ENV) && pip install --upgrade pip && pip install uv && uv pip install -r requirements.txt
	@echo "✅ Venv created, dependencies installed."

inspect: ## Check the dumps against config.py and that GigaChat and embeddings answer
	@$(SETUP_ENV) && python inspect_dump.py

mark-vague: ## Fill vague questions of the dumps yellow, in place, for support to rewrite
	@$(SETUP_ENV) && caffeinate -is python mark_vague_questions.py

# caffeinate keeps the Mac awake for hours-long runs: network calls die when it sleeps.
run: ## Process new dumps from config.DUMP_PATHS and rebuild the Excel export
	@$(SETUP_ENV) && caffeinate -is python pipeline.py

dedupe-base: ## Deduplicate the living base by question embeddings and trigrams
	@$(SETUP_ENV) && caffeinate -is python dedupe_base.py

clean: ## Remove the venv and caches (never touches data/)
	@rm -rf .venv/ .uv-cache/ __pycache__/
	@echo "✅ Cleaned."

dump: ## Dump all .py, .md files to .code_dump.txt. Usage: make dump folder/path
	@$(eval ARGS := $(filter-out $@,$(MAKECMDGOALS)))
	@$(eval TARGET_DIR := $(if $(ARGS),$(ARGS),.))
	@echo "--- DUMP OF CODE FILES IN: $(TARGET_DIR) ---" > .code_dump.txt
	@find $(TARGET_DIR) \
		\( -name "*.py" -o -name "*.md" \) \
		-not -path "*/.*" -not -path "*/.venv/*" -not -path "*/__pycache__/*" -not -path "*/node_modules/*" | while read -r file; do \
		echo -e "\n# --- FILE: $$file ---\n" >> .code_dump.txt; \
		cat "$$file" >> .code_dump.txt; \
	done
	@echo "✅ All selected files from '$(TARGET_DIR)' saved to .code_dump.txt"

dump-diff: ## Dump uncommitted .py, .md files to .code_dump.txt
	@echo "--- DUMP OF UNCOMMITTED FILES ---" > .code_dump.txt
	@{ git diff --name-only; \
	   git diff --name-only --cached; \
	   git ls-files --others --exclude-standard; } \
		| sort -u \
		| grep -E '\.(py|md)$$' \
		| while read -r file; do \
		[ -f "$$file" ] || continue; \
		echo -e "\n# --- FILE: $$file ---\n" >> .code_dump.txt; \
		cat "$$file" >> .code_dump.txt; \
	done
	@echo "✅ Uncommitted files saved to .code_dump.txt"

# Swallows the positional argument of a target (`make dump <path>`), which make would
# otherwise treat as a target to build.
%:
	@:
