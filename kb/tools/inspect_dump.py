"""Inspect the source Excel file of a config and check that the models answer.

Run this when a new file arrives: it lists every sheet and column with a
sample, so the question and answer columns of the TOML config can be checked
against the file. It then sends the first pair to the chat model the way the
scenario of the config does — a filter call for a dump, a repair call for a
base to repair — compares it with itself on the judging model, and sends one
batch to the embeddings model, so a missing certificate or model access
shows up here, not hours into a run.

Usage:
    make inspect CONFIG=configs/<name>.toml
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

import config
from kb.steps import dedup, filtering, repairing
from kb.utils import excel, gigachat, logs, settings
from kb.utils.excel import SourcePair
from kb.utils.settings import Domain, Source

ChatCheck = Callable[[Any, SourcePair], dict[str, Any] | None]

def inspect(path: Path) -> None:
    """Print sheet names, then column index, name, dtype and a sample per sheet."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    workbook = pd.ExcelFile(path)
    print(f"File: {path}")
    print(f"Sheets ({len(workbook.sheet_names)}): {workbook.sheet_names}")
    print()

    for sheet in workbook.sheet_names:
        dataframe = pd.read_excel(path, sheet_name=sheet)
        print(
            f"=== Sheet: {sheet!r} | rows: {len(dataframe)} "
            f"| cols: {len(dataframe.columns)} ==="
        )
        for index, column in enumerate(dataframe.columns):
            values = dataframe[column].dropna()
            sample = "" if values.empty else str(values.iloc[0])
            sample = sample.replace("\n", " ").replace("\r", " ")
            if len(sample) > 80:
                sample = sample[:77] + "..."
            print(
                f"  [{index:>3}] {column!r:<50} "
                f"dtype={dataframe[column].dtype}  e.g. {sample!r}"
            )
        print()


def check_models(source: Source, domain: Domain, chat_check: ChatCheck) -> None:
    """Send the first pair of the source to the chat, judging and embeddings models."""
    print("=== Model check ===")
    certificates = {
        "glm": config.GLM_CERTIFICATE,
        "gigachat": config.GIGACHAT_CERTIFICATE,
    }
    missing = {
        name: gigachat.missing_files(*files) for name, files in certificates.items()
    }
    for name, (cert_file, _) in certificates.items():
        state = f"MISSING {', '.join(missing[name])}" if missing[name] else "found"
        print(f"  Certificate {cert_file.name}: {state}")
    if not gigachat.is_network_alive():
        print(f"  GigaChat host unreachable: {config.GIGACHAT_BASE_URL}")
        print()
        return

    pairs = excel.read_pairs(
        source.path, source.question_columns, source.answer_columns, source.extra_columns
    )
    pair = next((pair for pair in pairs if pair.question and pair.answer), None)
    if pair is None:
        print("  No row with both a question and an answer to send")
        print()
        return
    print(f"  Row {pair.row_number}: {pair.question[:77]!r}")

    def report(
        role: str, model: str, missing_files: list[str], check: Callable[[], Any]
    ) -> None:
        if missing_files:
            print(f"  {role} {model}: skipped, no certificate")
        elif (reply := check()) is None:
            print(f"  {role} {model}: FAILED, see the log file")
        else:
            print(f"  {role} {model}: OK, {reply}")

    report(
        "Chat model",
        config.MODEL_NAME,
        missing["glm"],
        lambda: chat_check(gigachat.build_llm(), pair),
    )
    entry = {"question": pair.question, "answer": pair.answer}
    report(
        "Judging model",
        config.JUDGE_MODEL_NAME,
        missing["gigachat"],
        lambda: dedup.check_pair(gigachat.build_judge_llm(), domain, [entry], (0, 0))[1],
    )

    if config.EMBEDDING_BACKEND == "none":
        print('  Embeddings: skipped, EMBEDDING_BACKEND = "none"')
    elif missing["gigachat"]:
        print(f"  Embeddings {config.EMBEDDINGS_MODEL_NAME}: skipped, no certificate")
    else:
        try:
            vectors = gigachat.embed_texts(gigachat.build_embedder(), [pair.question])
        except RuntimeError as error:
            print(f"  Embeddings {config.EMBEDDINGS_MODEL_NAME}: FAILED, {error}")
        else:
            print(
                f"  Embeddings {config.EMBEDDINGS_MODEL_NAME}: OK, "
                f"vector of {len(vectors[0])} floats"
            )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the source file of a config.")
    parser.add_argument("config", type=Path, help="TOML config of make run or repair-base")
    config_path: Path = parser.parse_args().config

    with logs.run(config.TOOLS_LOG_DIR, "inspect"):
        # A repair config has an [input] table, a dump config a [dump] one; the
        # clean-up config has no source file to inspect.
        if "input" in settings.read(config_path):
            repair = settings.load(settings.RepairSettings, config_path)
            source, domain = repair.input, repair.domain

            def chat_check(llm: Any, pair: SourcePair) -> dict[str, Any] | None:
                return repairing.run_repair(llm, repair.domain, pair)

        else:
            base = settings.load(settings.TicketsSettings, config_path)
            source, domain = base.dump, base.domain

            def chat_check(llm: Any, pair: SourcePair) -> dict[str, Any] | None:
                return filtering.run_filter(llm, base.domain, pair)

        inspect(source.path)
        check_models(source, domain, chat_check)


if __name__ == "__main__":
    main()
