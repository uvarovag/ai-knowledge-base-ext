"""Inspect the configured dumps: list sheets and print every column with a sample.

Run this when a new dump arrives to check that QUESTION_COLUMNS and
ANSWER_COLUMNS in config still match its structure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


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


if __name__ == "__main__":
    for dump_path in config.DUMP_PATHS:
        inspect(dump_path)
