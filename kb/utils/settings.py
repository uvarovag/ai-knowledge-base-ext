"""Settings of one run, read from the TOML config given on the command line.

config.py holds the technical settings shared by every run, the technical
storage among them. A TOML config holds only what one script needs: the base
it works on, the Excel file it reads, the folder it writes Excel files to and
the domain the model is told about. An unknown key is an error, so a typo
cannot silently fall back to a default.

The technical state of a base lives under a directory named after the base,
never next to the output files, so the base keeps its identity whatever file
feeds it. An output file is named after the base and the date of the run that
wrote it ("Портал поставщика SAP_2026-10-02.xlsx").

Relative paths are relative to the repository root, where every command runs.
"""

from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

import config


def check_excel(path: Path) -> Path:
    # People hand in and get back Excel files only; JSON is technical state.
    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ValueError("must be an Excel file (.xlsx)")
    return path


UserPath = Annotated[Path, AfterValidator(Path.expanduser)]
ExcelPath = Annotated[UserPath, AfterValidator(check_excel)]
Columns = Annotated[tuple[str, ...], Field(min_length=1)]
MergeStrategy = Literal["accumulate", "replace"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Domain(StrictModel):
    """What the model is told: whose support it is and the categories."""

    # Substituted into every prompt: «служба поддержки <name>».
    name: str
    # Key stored in the entry -> description shown to the model.
    categories: dict[str, str] = Field(min_length=1)


class Source(StrictModel):
    """An Excel file and the columns joined into a question and an answer."""

    path: ExcelPath
    question_columns: Columns
    answer_columns: Columns
    # Copied verbatim into every entry, for traceability.
    extra_columns: tuple[str, ...] = ()


class RunSettings(StrictModel):
    """What every config has. The base lives in config.BASES_DIR/<name>/."""

    # A directory name and the start of the output file names: no path
    # separators, no leading dot.
    name: str = Field(pattern=r"^\w[\w .-]*$")
    output_dir: UserPath
    domain: Domain

    def output_file(self, suffix: str = "") -> Path:
        """An Excel file of this run, named after the base and today's date."""
        return self.output_dir / f"{self.name}_{date.today().isoformat()}{suffix}.xlsx"

    @property
    def work_dir(self) -> Path:
        return config.BASES_DIR / self.name

    @property
    def base_json(self) -> Path:
        """The source of truth of the base; the output workbooks are its views."""
        return self.work_dir / "knowledge_base.json"

    @property
    def processed_dumps_json(self) -> Path:
        return self.work_dir / "processed_dumps.json"

    @property
    def staging_dir(self) -> Path:
        return self.work_dir / "staging"

    @property
    def backup_dir(self) -> Path:
        return self.work_dir / "backups"


class TicketsSettings(RunSettings):
    """make run: update the base from one dump of tickets."""

    merge_strategy: MergeStrategy
    dump: Source


class DedupeSettings(RunSettings):
    """make dedupe-base: deduplicate the base against itself."""


class RepairSettings(RunSettings):
    """make repair-base: repair a poor base into a new one.

    Its staging is kept in config.REPAIRS_DIR, apart from the living bases, so
    a repair named like one of them never touches it.
    """

    input: Source

    @property
    def work_dir(self) -> Path:
        return config.REPAIRS_DIR / self.name


Settings = TypeVar("Settings", bound=RunSettings)


def read(path: Path) -> dict[str, Any]:
    """Read a TOML config as is."""
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    with path.open("rb") as stream:
        return tomllib.load(stream)


def load(kind: type[Settings], path: Path) -> Settings:
    """Read and check a TOML config; exit with the reason if it is wrong."""
    try:
        return kind.model_validate(read(path))
    except ValidationError as error:
        raise SystemExit(f"Wrong config {path}:\n{error}") from None
