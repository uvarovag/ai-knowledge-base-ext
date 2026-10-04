"""GigaChat clients and calls: network wait, retries, structured output, embeddings."""

from __future__ import annotations

import random
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from gigachat.exceptions import RateLimitError
from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from pydantic import BaseModel

import config
from kb.utils.logs import logger, note_call

# ----- Network availability ------------------------------------------------


def is_network_alive() -> bool:
    """Quick TCP check that the GigaChat host is reachable on port 443."""
    parsed = urlparse(config.GIGACHAT_BASE_URL)
    host = parsed.hostname
    port = parsed.port or 443
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=5):
            return True
    except OSError:
        return False


def wait_for_network() -> None:
    """Block until the GigaChat host becomes reachable again."""
    if is_network_alive():
        return
    # INFO, not a warning: a pause, not a failure, and the terminal shows it.
    logger.info("Network unreachable, waiting for it to come back")
    waited_seconds = 0
    while not is_network_alive():
        time.sleep(config.NETWORK_CHECK_INTERVAL_SECONDS)
        waited_seconds += config.NETWORK_CHECK_INTERVAL_SECONDS
        if waited_seconds % 300 == 0:
            logger.info("Still no network after %d seconds", waited_seconds)
    logger.info("Network is back after %d seconds", waited_seconds)


# ----- Retries and the rate limit ------------------------------------------

# A 429 means "slow down", not "this request is bad": it does not use up an
# attempt. Instead every thread of the process pauses until a shared cooldown
# ends, and the cooldown doubles while 429s keep coming, so the process as a
# whole backs off instead of each worker hammering the server on its own.
_cooldown_lock = threading.Lock()
_cooldown_until = 0.0
_rate_limit_streak = 0

Result = TypeVar("Result")


class FinalReply(Exception):
    """A reply that asking again would not change: no more attempts."""

    # How the attempt is noted (note_call) and what the log says to do.
    result = "error: final reply"
    advice = ""


class Blocked(FinalReply):
    """GigaChat's own content filter refused the text (finish_reason=blacklist)."""

    result = "blacklist"
    advice = "GigaChat's content filter refused the topic"


class CutOff(FinalReply):
    """The output budget ran out before the function call (finish_reason=length)."""

    result = "error: cut off"
    advice = "the reply hit max_tokens; raise MAX_TOKENS and run again"


def wait_for_cooldown() -> None:
    """Sleep while a rate-limit cooldown of this process is running."""
    while (delay := _cooldown_until - time.monotonic()) > 0:
        time.sleep(delay)


def start_cooldown(error: RateLimitError) -> float | None:
    """Extend the shared cooldown after a 429; return its length if it grew.

    The delay doubles with every 429 in a row, from RATE_LIMIT_BASE_SECONDS up
    to RATE_LIMIT_MAX_SECONDS, with jitter so the workers do not come back at
    the same instant; a Retry-After header, when the server sends one, wins.
    """
    global _cooldown_until, _rate_limit_streak
    with _cooldown_lock:
        now = time.monotonic()
        # A 429 during a cooldown answers a request sent before it began: the
        # same wave, not a reason to wait longer.
        if now < _cooldown_until:
            return None
        _rate_limit_streak += 1
        delay = min(
            config.RATE_LIMIT_MAX_SECONDS,
            config.RATE_LIMIT_BASE_SECONDS * 2 ** (_rate_limit_streak - 1),
        )
        delay = max(delay, error.retry_after) * random.uniform(1.0, 1.5)
        _cooldown_until = now + delay
        return delay


def end_rate_limit_streak() -> None:
    """A call went through: the next 429 starts from the base delay again."""
    global _rate_limit_streak
    with _cooldown_lock:
        _rate_limit_streak = 0


def call_with_retries(
    call: Callable[[], Result], label: str, caller: str
) -> Result | None:
    """Run a GigaChat call, retrying failures; None if they never stop.

    A failure uses one of config.MAX_RETRIES attempts, with a growing pause.
    A 429 uses none: it waits out the shared cooldown, up to
    config.RATE_LIMIT_MAX_WAITS times per call. A FinalReply — the content
    filter, a reply cut off by the budget — ends the call at once: the same
    request gets the same reply. Every attempt is noted under caller, the
    step making the call (note_call). The label names the item in the log.
    """
    attempt = 0
    rate_limit_waits = 0
    while True:
        wait_for_network()
        wait_for_cooldown()
        started = time.monotonic()
        try:
            result = call()
        except RateLimitError as error:
            note_call(caller, started, "429")
            rate_limit_waits += 1
            if rate_limit_waits > config.RATE_LIMIT_MAX_WAITS:
                logger.warning(
                    "%s: still rate limited (429) after %d waits, giving up",
                    label,
                    config.RATE_LIMIT_MAX_WAITS,
                )
                return None
            delay = start_cooldown(error)
            if delay is not None:
                # INFO, not a warning: a pause, not a failure, and the terminal shows it.
                logger.info(
                    "Rate limited (429) at %s: pausing every call for %.0f s",
                    label,
                    delay,
                )
            continue
        except FinalReply as error:
            note_call(caller, started, error.result)
            logger.warning("%s: %s, not asked again: %s", label, error.advice, error)
            return None
        except Exception as error:
            note_call(caller, started, f"error: {type(error).__name__}")
            attempt += 1
            logger.warning(
                "%s: attempt %d/%d failed: %s: %s",
                label,
                attempt,
                config.MAX_RETRIES,
                type(error).__name__,
                error,
            )
            if attempt >= config.MAX_RETRIES:
                return None
            time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
            continue
        note_call(caller, started, "ok")
        end_rate_limit_streak()
        return result


# ----- Clients and calls ---------------------------------------------------


def missing_files(*paths: Path) -> list[str]:
    return [str(path) for path in paths if not path.is_file()]


Certificate = tuple[Path, Path]


def require_certificate(certificate: Certificate, models: str) -> None:
    """Stop with the missing file named, not with an SSL error at the first call."""
    missing = missing_files(*certificate)
    if missing:
        raise SystemExit(f"Certificate of {models} not found: {', '.join(missing)}")


def require_certificates() -> None:
    """Check both certificates before a run, so a missing one stops it now and
    not at the first call of its model, which may come hours in."""
    require_certificate(config.GLM_CERTIFICATE, config.MODEL_NAME)
    require_certificate(
        config.GIGACHAT_CERTIFICATE,
        f"{config.JUDGE_MODEL_NAME} and {config.EMBEDDINGS_MODEL_NAME}",
    )


def build_llm(
    max_tokens: int = config.MAX_TOKENS,
    model: str = config.MODEL_NAME,
    certificate: Certificate = config.GLM_CERTIFICATE,
) -> GigaChat:
    """Instantiate the client of the main chat model with project defaults.

    Merging several entries into one needs a bigger output budget than the
    default, so the merge steps pass config.MERGE_MAX_TOKENS; build_judge_llm
    and make models pass another model with its certificate.
    """
    require_certificate(certificate, model)
    cert_file, key_file = certificate
    return GigaChat(
        model=model,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(cert_file),
        key_file=str(key_file),
        profanity_check=False,
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
        top_p=config.GIGACHAT_TOP_P,
        temperature=config.GIGACHAT_TEMPERATURE,
        max_tokens=max_tokens,
    )


def build_judge_llm() -> GigaChat:
    """Instantiate the client of the judging model: the duplicate checks and
    the matching against the base, which only compare two entries."""
    return build_llm(
        model=config.JUDGE_MODEL_NAME, certificate=config.GIGACHAT_CERTIFICATE
    )


def build_embedder(model: str = config.EMBEDDINGS_MODEL_NAME) -> GigaChatEmbeddings:
    """Instantiate the embeddings client: the same endpoint, the gigachat
    certificate."""
    require_certificate(config.GIGACHAT_CERTIFICATE, model)
    cert_file, key_file = config.GIGACHAT_CERTIFICATE
    return GigaChatEmbeddings(
        model=model,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(cert_file),
        key_file=str(key_file),
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
    )


def embed_texts(embedder: GigaChatEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed one batch of texts, retrying like invoke_structured does for the chat model.

    Raises:
        RuntimeError: if every attempt fails. Unlike a failed chat call, which
        costs one entry, missing vectors would silently hide duplicates, so the
        run stops instead.
    """

    def embed() -> list[list[float]]:
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError(f"Expected {len(texts)} vectors, got {len(vectors)}")
        return vectors

    vectors = call_with_retries(embed, f"Embeddings batch of {len(texts)}", "embeddings")
    if vectors is not None:
        return vectors
    raise RuntimeError(
        "Embeddings request failed after every retry; set "
        'EMBEDDING_BACKEND = "none" in config.py to run on trigrams only'
    )


def describe_reply(reply: Any) -> str:
    """Summarize a raw model reply for the error log: finish reason, calls, text."""
    metadata = getattr(reply, "response_metadata", {}) or {}
    content = str(getattr(reply, "content", ""))
    return (
        f"finish_reason={metadata.get('finish_reason')}, "
        f"tool_calls={getattr(reply, 'tool_calls', None)}, "
        f"content={content[:500]!r}"
    )


def invoke_structured(
    llm: GigaChat,
    system_prompt: str,
    user_prompt: str,
    schema: type[BaseModel],
    label: str,
    caller: str,
) -> dict[str, Any] | None:
    """Call the model and return its answer as a dict of the schema, retrying on failure.

    The model is forced to call a function whose arguments are the schema
    (structured output, method="function_calling"), so its reply is never free
    text to be fished for JSON: a short reply like "Хочу новый стул" used to be
    answered in prose instead of being classified. Pydantic validates the
    arguments; a reply that fails validation is retried like a network error,
    a 429 waits out the rate-limit cooldown (call_with_retries).

    Returns None if every attempt fails. The label identifies the item in the
    log, the caller the step in the summary of the run.
    """
    # include_raw keeps the model's reply next to the parsed result, so a
    # failure is logged with what the model actually said.
    structured_llm = llm.with_structured_output(
        schema, method="function_calling", include_raw=True
    )
    messages = [("system", system_prompt), ("user", user_prompt)]

    def invoke() -> dict[str, Any]:
        result = structured_llm.invoke(messages)
        if result["parsed"] is None:
            metadata = getattr(result["raw"], "response_metadata", {}) or {}
            finish_reason = metadata.get("finish_reason")
            if finish_reason == "blacklist":
                raise Blocked(describe_reply(result["raw"]))
            if finish_reason == "length":
                raise CutOff(describe_reply(result["raw"]))
            parsing_error = result["parsing_error"] or "none made"
            raise ValueError(
                f"no valid function call ({parsing_error}); "
                f"{describe_reply(result['raw'])}"
            )
        return result["parsed"].model_dump()

    return call_with_retries(invoke, label, caller)
