"""Terminal and log file of a run.

The terminal shows what a person watches: the progress of every step, the
rows that went wrong above the bar, and in the end the model calls summed up
by step. Warnings and errors — a failed attempt with the model's raw reply, a
rejected merge — go to the run's own log file, written only when there is
something to write: thousands of rows would bury them in the terminal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.console import Console, RenderableType
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

import config

logger = logging.getLogger("kb")

# No highlighting: questions and answers are printed as they are.
console = Console(stderr=True, highlight=False)


def format_duration(seconds: float) -> str:
    """Render a duration as 1h 05m, 5m 20s or 12s."""
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h {total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total}s"


# ----- Progress ------------------------------------------------------------


class ProgressBar:
    """A bar at the bottom of the terminal: done/total, time spent and left, counters.

    print puts a line above the bar, where it stays; the bar moves on. One bar
    at a time: the steps run one after another.
    """

    # The running bar, stopped by run() when a step fails or is interrupted:
    # a bar left running keeps the terminal's cursor hidden.
    active: ClassVar[ProgressBar | None] = None

    def __init__(self, total: int, label: str) -> None:
        self.label = label
        self.counters: dict[str, int] = defaultdict(int)
        self.started_at = time.monotonic()
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("left"),
            TimeRemainingColumn(),
            TextColumn("{task.fields[counters]}", style="dim"),
            console=console,
            # Gone when done: finish logs one summary line in its place.
            transient=True,
        )
        self.task = self.progress.add_task(label, total=total, counters="")
        self.progress.start()
        ProgressBar.active = self

    def counters_text(self) -> str:
        return ", ".join(f"{name} {count}" for name, count in self.counters.items())

    def advance(self, **counters: int) -> None:
        """Count one finished item, adding the given named counters."""
        for name, value in counters.items():
            self.counters[name] += value
        self.progress.update(self.task, advance=1, counters=self.counters_text())

    def print(self, renderable: RenderableType) -> None:
        """Print above the bar."""
        self.progress.console.print(renderable)

    def finish(self) -> None:
        """Remove the bar and log one summary line in its place."""
        self.progress.stop()
        ProgressBar.active = None
        completed = int(self.progress.tasks[0].completed)
        counters = self.counters_text()
        logger.info(
            "%s: done, %d items in %s%s",
            self.label,
            completed,
            format_duration(time.monotonic() - self.started_at),
            f" | {counters}" if counters else "",
        )


# ----- Model calls ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One attempt of a model call: the step that made it, how long, what came back."""

    caller: str
    seconds: float
    # "ok", "429" or "error: <exception class>".
    result: str


# Every attempt of the process, for the summary of the run; and, per thread,
# the attempts of the item it is processing, for the line of a row that went
# wrong (recording_calls).
_calls_lock = threading.Lock()
_run_calls: list[ModelCall] = []
_item_calls = threading.local()


def note_call(caller: str, started: float, result: str) -> None:
    """Note one attempt that began at started (time.monotonic())."""
    call = ModelCall(caller, round(time.monotonic() - started, 1), result)
    with _calls_lock:
        _run_calls.append(call)
    item_calls = getattr(_item_calls, "calls", None)
    if item_calls is not None:
        item_calls.append(call)


@contextmanager
def recording_calls() -> Iterator[list[ModelCall]]:
    """Collect the model calls this thread makes inside the block."""
    calls: list[ModelCall] = []
    _item_calls.calls = calls
    try:
        yield calls
    finally:
        _item_calls.calls = None


def speed_style(seconds: float) -> str:
    """Yellow from config.SLOW_CALL_SECONDS, red from twice that: a slow GigaChat."""
    if seconds >= 2 * config.SLOW_CALL_SECONDS:
        return "red"
    return "yellow" if seconds >= config.SLOW_CALL_SECONDS else "green"


def render_item(title: str, reason: str, calls: Sequence[ModelCall], failed: bool) -> Tree:
    """Draw one item that went wrong: its reason, and under it every model call."""
    mark = Text("✗ ", style="bold red") if failed else Text("– ", style="yellow")
    tree = Tree(
        Text.assemble(mark, (title, "bold"), "  ", (reason, "red" if failed else "yellow"))
    )
    width = max((len(call.caller) for call in calls), default=0)
    for call in calls:
        tree.add(
            Text.assemble(
                (f"{call.caller:<{width}}  ", "cyan"),
                (f"{call.seconds:>5.1f} s", speed_style(call.seconds)),
                "  ",
                (call.result, "" if call.result == "ok" else "red"),
            )
        )
    return tree


def describe_pair(first: dict[str, Any], second: dict[str, Any]) -> str:
    """Name a pair of entries by their questions, cut to fit a line."""
    return " / ".join(f"«{entry['question'][:60]}»" for entry in (first, second))


def print_model_calls(calls: Sequence[ModelCall]) -> None:
    """Sum the model calls up by step: how many, how many failed or hit 429, how long."""
    if not calls:
        return
    by_caller: dict[str, list[ModelCall]] = defaultdict(list)
    for call in calls:
        by_caller[call.caller].append(call)
    table = Table(title="Model calls by step", title_justify="left")
    table.add_column("step", style="cyan")
    for column in ("calls", "failed", "429", "avg s", "max s", "total"):
        table.add_column(column, justify="right")
    for caller, caller_calls in sorted(
        by_caller.items(), key=lambda item: -sum(call.seconds for call in item[1])
    ):
        seconds = [call.seconds for call in caller_calls]
        failed = sum(call.result.startswith("error") for call in caller_calls)
        rate_limited = sum(call.result == "429" for call in caller_calls)
        table.add_row(
            caller,
            str(len(seconds)),
            Text(str(failed), style="red" if failed else ""),
            Text(str(rate_limited), style="yellow" if rate_limited else ""),
            f"{sum(seconds) / len(seconds):.1f}",
            Text(f"{max(seconds):.1f}", style=speed_style(max(seconds))),
            format_duration(sum(seconds)),
        )
    console.print()
    console.print(table)


# ----- Logging -------------------------------------------------------------


class ConsoleHandler(logging.Handler):
    """Print through the shared console, so a line never breaks a running bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            console.print(self.format(record), markup=False)
        except Exception:
            self.handleError(record)


def configure_logging(log_path: Path) -> None:
    """Send the progress of the run to the terminal and its warnings to log_path.

    The terminal gets the INFO lines of this project only: libraries log every
    HTTP request at INFO. The file gets warnings and errors of every logger,
    and is created only by the first of them.
    """
    console_handler = ConsoleHandler()
    console_handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    console_handler.addFilter(lambda record: record.levelno == logging.INFO)
    console_handler.addFilter(logging.Filter("kb"))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root.setLevel(logging.INFO)
    # The GigaChat library warns on every 429 by itself; call_with_retries
    # already reports the pause, once for all workers.
    logging.getLogger("gigachat").setLevel(logging.ERROR)


@contextmanager
def run(log_dir: Path, command: str) -> Iterator[None]:
    """Log a run to <log_dir>/<command>_<YYYY-MM-DD_HH-MM-SS>.log and sum it up.

    The summary — model calls by step, and where the log file is — is printed
    when the run ends, interrupted or failed too; a crash goes to the file.
    """
    log_path = log_dir / f"{command}_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    configure_logging(log_path)
    try:
        yield
    except KeyboardInterrupt:
        logger.info("Interrupted: run the same command again to resume")
        raise
    except Exception:
        # The traceback goes to the log file too: the terminal scrolls away.
        logger.exception("Run failed")
        raise
    finally:
        if ProgressBar.active is not None:
            ProgressBar.active.progress.stop()
            ProgressBar.active = None
        with _calls_lock:
            calls = list(_run_calls)
        print_model_calls(calls)
        if log_path.exists():
            logger.info("Warnings and errors of this run: %s", log_path)
        else:
            logger.info("No warnings or errors in this run")
