SHELL := /bin/bash
-include .env
export

PYTHON ?= python3.13

help: ## Show this help menu
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/^.*Makefile://' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

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
	
%:
	@:
