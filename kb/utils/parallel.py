"""The pool of worker threads every parallel step runs its model calls on.

A ThreadPoolExecutor left through an exception waits for every task still
queued: Ctrl+C in the middle of a dump would then finish the dump before the
process stops. The pool here cancels the queued tasks instead, so only the
calls already running are waited for.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import config


@contextmanager
def workers(count: int | None = None) -> Iterator[ThreadPoolExecutor]:
    """A pool of count threads, config.WORKER_COUNT by default, whose queue an
    exception empties. A step on the judging model passes
    config.JUDGE_WORKER_COUNT, the embeddings config.EMBEDDING_WORKER_COUNT."""
    with ThreadPoolExecutor(max_workers=count or config.WORKER_COUNT) as executor:
        try:
            yield executor
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
