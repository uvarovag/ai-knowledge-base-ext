"""Logging and progress reporting shared by every command."""

from __future__ import annotations

import logging
import sys
import time
from typing import ClassVar

import config

logger = logging.getLogger("kb")


def format_duration(seconds: float) -> str:
    """Render a duration as 1h 05m, 5m 20s or 12s."""
    total = int(seconds)
    if total >= 3600:
        return f"{total // 3600}h {total % 3600 // 60:02d}m"
    if total >= 60:
        return f"{total // 60}m {total % 60:02d}s"
    return f"{total}s"


class ProgressBar:
    """A single-line progress bar with named counters and an ETA.

    Renders to stderr while it is a terminal; otherwise falls back to periodic
    log lines, so redirected output stays readable. Only one bar is active at a
    time — the logging handler uses it to keep log lines from breaking the bar.
    """

    WIDTH = 24
    FALLBACK_STEP_PERCENT = 10

    active: ClassVar[ProgressBar | None] = None

    def __init__(self, total: int, label: str) -> None:
        self.total = max(total, 1)
        self.label = label
        self.completed = 0
        self.counters: dict[str, int] = {}
        self.started_at = time.monotonic()
        self.is_interactive = sys.stderr.isatty()
        self._last_reported_step = -1
        self._line_length = 0

        ProgressBar.active = self
        self._render()

    # ----- Rendering -------------------------------------------------------

    def _counters_text(self) -> str:
        return ", ".join(f"{name} {count}" for name, count in self.counters.items())

    def _eta_text(self) -> str:
        elapsed = time.monotonic() - self.started_at
        if self.completed == 0 or elapsed < 1:
            return "ETA --"
        rate = self.completed / elapsed
        remaining = (self.total - self.completed) / rate if rate else 0
        return f"{rate:.1f}/s, ETA {format_duration(remaining)}"

    def _render(self) -> None:
        if not self.is_interactive:
            return
        percent = self.completed * 100 // self.total
        filled = self.completed * self.WIDTH // self.total
        bar = "█" * filled + "░" * (self.WIDTH - filled)

        parts = [f"{self.label} [{bar}] {self.completed}/{self.total} {percent:3d}%"]
        counters = self._counters_text()
        if counters:
            parts.append(counters)
        parts.append(self._eta_text())

        line = " | ".join(parts)
        self._line_length = len(line)
        sys.stderr.write("\r" + line)
        sys.stderr.flush()

    def _report_to_log(self) -> None:
        """Print a progress line when stderr is not a terminal."""
        percent = self.completed * 100 // self.total
        step = percent // self.FALLBACK_STEP_PERCENT
        if step == self._last_reported_step and self.completed != self.total:
            return
        self._last_reported_step = step

        counters = self._counters_text()
        logger.info(
            "%s: %d/%d (%d%%)%s",
            self.label,
            self.completed,
            self.total,
            percent,
            f" | {counters}" if counters else "",
        )

    def clear(self) -> None:
        """Erase the bar so a log line can be printed over it."""
        if self.is_interactive and self._line_length:
            sys.stderr.write("\r" + " " * self._line_length + "\r")
            sys.stderr.flush()

    def redraw(self) -> None:
        """Draw the bar again after a log line was printed."""
        self._render()

    # ----- Updating --------------------------------------------------------

    def advance(self, **counters: int) -> None:
        """Count one finished item, adding the given named counters."""
        self.completed += 1
        for name, value in counters.items():
            self.counters[name] = self.counters.get(name, 0) + value

        if self.is_interactive:
            self._render()
        else:
            self._report_to_log()

    def finish(self) -> None:
        """Close the bar and print the final summary line."""
        elapsed = time.monotonic() - self.started_at
        self.clear()
        ProgressBar.active = None

        counters = self._counters_text()
        logger.info(
            "%s: done, %d items in %s%s",
            self.label,
            self.completed,
            format_duration(elapsed),
            f" | {counters}" if counters else "",
        )


class ProgressAwareHandler(logging.StreamHandler):
    """Stream handler that keeps log lines from breaking an active progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        bar = ProgressBar.active
        if bar is not None:
            bar.clear()
        super().emit(record)
        if bar is not None:
            bar.redraw()


def configure_logging() -> None:
    """Set up the single logging format used by every step of the pipeline.

    Warnings and errors also go to config.ERROR_LOG, appended across runs: a
    failed model call scrolls away among thousands of progress lines, and the
    file is what tells why a row ended up as filter_failed.
    """
    console_handler = ProgressAwareHandler(sys.stderr)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    config.ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    error_handler = logging.FileHandler(config.ERROR_LOG, encoding="utf-8")
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(
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
    root.addHandler(error_handler)
    root.setLevel(logging.INFO)
