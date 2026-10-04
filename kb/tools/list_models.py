"""List each certificate's models; measure how many calls the configured ones hold.

The chat models and the embeddings model have certificates of their own, and
each one is granted its own models: run this to check that MODEL_NAME,
JUDGE_MODEL_NAME and EMBEDDINGS_MODEL_NAME of config are among the models
of their certificate, and to choose WORKER_COUNT, JUDGE_WORKER_COUNT and
EMBEDDING_WORKER_COUNT.

For every configured model it sends 1, 2, … up to PROBE_MAX_CONCURRENCY
short requests at the same instant, and stops at the first wave that gets a
429 or an error: the model holds the last wave that went through. The
requests bypass the retries of call_with_retries, so a 429 shows instead of
being waited out. The limit seen is the one of this moment, shared with every
other client of the certificate. The other models are only listed.

Usage:
    make models
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gigachat.exceptions import RateLimitError

import config
from kb.utils import gigachat, logs
from kb.utils.logs import logger

PROBE_MAX_CONCURRENCY = 10
# A pause between waves, so the next one is not refused for the last one.
PROBE_PAUSE_SECONDS = 3
# A reasoning model may spend the budget on its thinking: the reply is cut
# off then, which still counts as a call that went through.
PROBE_MAX_TOKENS = 50


def run_wave(call: Callable[[], Any], size: int, caller: str) -> str:
    """Send size calls at once; return "ok", "429" or "error: <class>"."""
    barrier = threading.Barrier(size)

    def one() -> str:
        barrier.wait()
        started = time.monotonic()
        try:
            call()
        except RateLimitError:
            result = "429"
        except Exception as error:  # noqa: BLE001 — any failure ends the probe
            logger.warning("%s: %s: %s", caller, type(error).__name__, error)
            result = f"error: {type(error).__name__}"
        else:
            result = "ok"
        logs.note_call(caller, started, result)
        return result

    with ThreadPoolExecutor(max_workers=size) as executor:
        results = list(executor.map(lambda _: one(), range(size)))
    failures = [result for result in results if result != "ok"]
    return "429" if "429" in failures else (failures[0] if failures else "ok")


def probe(model: str, call: Callable[[], Any]) -> str:
    """Find how many calls at once the model takes; describe it in one line."""
    held = 0
    for size in range(1, PROBE_MAX_CONCURRENCY + 1):
        if size > 1:
            time.sleep(PROBE_PAUSE_SECONDS)
        result = run_wave(call, size, f"probe {model}")
        logger.info("%s: %d at once — %s", model, size, result)
        if result != "ok":
            if held == 0:
                return f"no call went through ({result})"
            return f"holds {held} at once ({result} at {size})"
        held = size
    return f"holds {held} at once or more (not tried above)"


def print_models(
    name: str,
    configured: tuple[str, ...],
    certificate: tuple[Path, Path],
    get_models: Callable[[], Any],
    make_call: Callable[[str], Callable[[], Any]],
) -> None:
    """Print the models one certificate is granted, the configured ones marked
    and measured for how many calls at once they hold."""
    print(
        f"{name} certificate {certificate[0].name}, configured {', '.join(configured)}:"
    )
    missing = gigachat.missing_files(*certificate)
    if missing:
        print(f"  certificate not found: {', '.join(missing)}\n")
        return
    models = gigachat.call_with_retries(get_models, f"List models: {name}", "models")
    if models is None:
        print("  FAILED, see the log file\n")
        return
    names = sorted(entry.id_ for entry in models.data)
    width = max(map(len, names), default=0)
    for entry in names:
        if entry in configured:
            print(f"  * {entry:<{width}}  {probe(entry, make_call(entry))}", flush=True)
        else:
            print(f"    {entry}", flush=True)
    for model in configured:
        if model not in names:
            print(f"  {model} is NOT available to this certificate")
    print()


def chat_call(model: str) -> Callable[[], Any]:
    llm = gigachat.build_llm(max_tokens=PROBE_MAX_TOKENS, model=model)
    return lambda: llm.invoke("Ответь одним словом: готово")


def embeddings_call(model: str) -> Callable[[], Any]:
    embedder = gigachat.build_embedder(model=model)
    return lambda: embedder.embed_documents(["проверка"])


def main() -> None:
    with logs.run(config.TOOLS_LOG_DIR, "models"):
        print(f"Models at {config.GIGACHAT_BASE_URL}\n", flush=True)
        print_models(
            "Chat",
            (config.MODEL_NAME, config.JUDGE_MODEL_NAME),
            (config.CERT_FILE, config.KEY_FILE),
            lambda: gigachat.build_llm().get_models(),
            chat_call,
        )
        # The embeddings client has no get_models of its own; its underlying
        # gigachat client, built with the embeddings certificate, does.
        print_models(
            "Embeddings",
            (config.EMBEDDINGS_MODEL_NAME,),
            (config.EMBEDDINGS_CERT_FILE, config.EMBEDDINGS_KEY_FILE),
            lambda: gigachat.build_embedder()._client.get_models(),
            embeddings_call,
        )


if __name__ == "__main__":
    main()
