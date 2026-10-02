"""Merge a good base into a living base.

The source is the Excel file of a base this project wrote — another living
base, or a poor base repaired by make repair-base (a poor base is repaired
first, never merged as is). Its entries are already canonical, so there is
no filter and no rewriting: they are matched against the base and merged
into it as the entries of a dump are (merging, by merge_strategy).

The newest Excel file of the base, with the reviewer's edits, becomes the base
first (living_base). A source already in the ledger of the base (matched by
file content) is skipped. Match verdicts are cached in the staging directory
of the source, so an interrupted run resumes.

Usage:
    make merge-base CONFIG=configs/<base>-merge.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import config
from kb.steps import merging
from kb.utils import excel, gigachat, living_base, logs, settings, storage
from kb.utils.logs import logger
from kb.utils.settings import Domain


def mark_foreign_categories(
    domain: Domain, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Mark doubtful an entry whose category the base does not have.

    A base of another domain may use other categories: the entry is kept for a
    reviewer to place, not dropped.
    """
    for entry in entries:
        if entry["category"] not in domain.categories and not entry.get("doubtful"):
            entry["doubtful"] = f"категория «{entry['category']}» не из списка базы"
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a good base into a base.")
    parser.add_argument("config", type=Path, help="TOML config of the merge")
    base = settings.load(settings.MergeBaseSettings, parser.parse_args().config)
    source = base.source.path

    with logs.run(base.log_dir, "merge-base"):
        gigachat.require_certificates()
        if not source.exists():
            raise SystemExit(f"Base to merge not found: {source}")

        entries_of_base = living_base.sync_from_excel(base)
        file_hash = storage.hash_file(source)
        if living_base.is_merged(base, file_hash) and not config.FORCE_REPROCESS:
            logger.info(
                "%s is already in base %s, nothing to merge", source.name, base.name
            )
        else:
            logger.info("=== %s -> base %s ===", source.name, base.name)
            incoming = mark_foreign_categories(
                base.domain, excel.read_knowledge_base(source)
            )
            logger.info("Read %d entries from %s", len(incoming), source.name)
            entries_of_base, updated, added = merging.merge_into_base(
                base.domain,
                base.merge_strategy,
                entries_of_base,
                incoming,
                storage.staging_dir_for(source, base.staging_dir),
            )
            living_base.save(base, entries_of_base)
            living_base.register(base, source, file_hash, updated, added)
        living_base.export(base, entries_of_base)


if __name__ == "__main__":
    main()
