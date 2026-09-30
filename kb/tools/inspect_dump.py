"""Inspect the configured input files and check that the models answer.

Run this when a new file arrives: it lists every sheet and column with a
sample, so the question and answer columns in config can be checked against
the structure of the dumps and of the base to repair. For a dump it then
sends the first pair to the chat model and to the embeddings model exactly the
way scenario 1 does — one filter call and one embeddings batch — so a missing
certificate or model access shows up here, not hours into a run.

Usage:
    make inspect
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from kb.steps import filtering
from kb.utils import excel, gigachat, logs


def inspect(path: Path) -> None:
    """Print sheet names, then column index, name, dtype and a sample per sheet."""
    if not path.exists():
        raise FileNotFoundError(f"Dump not found: {path}")

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


def check_models(path: Path) -> None:
    """Send the first pair of the dump to the chat and embeddings models."""
    print("=== Model check ===")
    if not gigachat.is_network_alive():
        print(f"  GigaChat host unreachable: {config.GIGACHAT_BASE_URL}")
        print()
        return

    pairs = excel.read_pairs(
        path, config.QUESTION_COLUMNS, config.ANSWER_COLUMNS, config.SOURCE_EXTRA_COLUMNS
    )
    pair = next((pair for pair in pairs if pair.question and pair.answer), None)
    if pair is None:
        print("  No row with both a question and an answer to send")
        print()
        return
    print(f"  Row {pair.row_number}: {pair.question[:77]!r}")

    verdict = filtering.run_filter(gigachat.build_llm(), pair)
    if verdict is None:
        print(f"  Chat model {config.GIGACHAT_MODEL_NAME}: FAILED, see warnings above")
    else:
        print(f"  Chat model {config.GIGACHAT_MODEL_NAME}: OK, filter verdict {verdict}")

    if config.EMBEDDING_BACKEND == "none":
        print('  Embeddings: skipped, EMBEDDING_BACKEND = "none"')
    else:
        try:
            vectors = gigachat.embed_texts(gigachat.build_embedder(), [pair.question])
        except RuntimeError as error:
            print(f"  Embeddings {config.GIGACHAT_EMBEDDINGS_MODEL}: FAILED, {error}")
        else:
            print(
                f"  Embeddings {config.GIGACHAT_EMBEDDINGS_MODEL}: OK, "
                f"vector of {len(vectors[0])} floats"
            )
    print()


if __name__ == "__main__":
    logs.configure_logging()
    for dump_path in config.DUMP_PATHS:
        inspect(dump_path)
        check_models(dump_path)
    if config.REPAIR_INPUT_PATH.exists():
        inspect(config.REPAIR_INPUT_PATH)
