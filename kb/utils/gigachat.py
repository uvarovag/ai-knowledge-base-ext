"""GigaChat clients and calls: network wait, retries, structured output, embeddings."""

from __future__ import annotations

import socket
import time
from typing import Any
from urllib.parse import urlparse

from langchain_gigachat.chat_models import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from pydantic import BaseModel

import config
from kb.utils.logs import logger

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
    logger.warning("Network unreachable, waiting for it to come back")
    waited_seconds = 0
    while not is_network_alive():
        time.sleep(config.NETWORK_CHECK_INTERVAL_SECONDS)
        waited_seconds += config.NETWORK_CHECK_INTERVAL_SECONDS
        if waited_seconds % 300 == 0:
            logger.warning("Still no network after %d seconds", waited_seconds)
    logger.info("Network is back after %d seconds", waited_seconds)


# ----- Clients and calls ---------------------------------------------------


def build_llm(max_tokens: int = config.GIGACHAT_MAX_TOKENS) -> GigaChat:
    """Instantiate the GigaChat client with project defaults.

    Merging several entries into one needs a bigger output budget than the
    default, so the merge steps pass config.MERGE_MAX_TOKENS.
    """
    return GigaChat(
        model=config.GIGACHAT_MODEL_NAME,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(config.CERT_FILE),
        key_file=str(config.KEY_FILE),
        profanity_check=False,
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
        top_p=config.GIGACHAT_TOP_P,
        temperature=config.GIGACHAT_TEMPERATURE,
        max_tokens=max_tokens,
    )


def build_embedder() -> GigaChatEmbeddings:
    """Instantiate the embeddings client with the same endpoint and certificates."""
    return GigaChatEmbeddings(
        model=config.GIGACHAT_EMBEDDINGS_MODEL,
        base_url=config.GIGACHAT_BASE_URL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL_CERTS,
        cert_file=str(config.CERT_FILE),
        key_file=str(config.KEY_FILE),
        timeout=config.GIGACHAT_TIMEOUT_SECONDS,
    )


def embed_texts(embedder: GigaChatEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed one batch of texts, retrying like invoke_structured does for the chat model.

    Raises:
        RuntimeError: if every attempt fails. Unlike a failed chat call, which
        costs one entry, missing vectors would silently hide duplicates, so the
        run stops instead.
    """
    for attempt in range(1, config.MAX_RETRIES + 1):
        wait_for_network()
        try:
            vectors = embedder.embed_documents(texts)
            if len(vectors) != len(texts):
                raise ValueError(f"Expected {len(texts)} vectors, got {len(vectors)}")
            return vectors
        except Exception as error:
            logger.warning(
                "Embeddings batch of %d: attempt %d/%d failed: %s",
                len(texts),
                attempt,
                config.MAX_RETRIES,
                error,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)
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
) -> dict[str, Any] | None:
    """Call the model and return its answer as a dict of the schema, retrying on failure.

    The model is forced to call a function whose arguments are the schema
    (structured output, method="function_calling"), so its reply is never free
    text to be fished for JSON: a short reply like "Хочу новый стул" used to be
    answered in prose instead of being classified. Pydantic validates the
    arguments; a reply that fails validation is retried like a network error.

    Returns None if every attempt fails. The label identifies the item in logs.
    """
    # include_raw keeps the model's reply next to the parsed result, so a
    # failure is logged with what the model actually said.
    structured_llm = llm.with_structured_output(
        schema, method="function_calling", include_raw=True
    )
    messages = [("system", system_prompt), ("user", user_prompt)]

    for attempt in range(1, config.MAX_RETRIES + 1):
        wait_for_network()
        try:
            result = structured_llm.invoke(messages)
            if result["parsed"] is None:
                parsing_error = result["parsing_error"] or "none made"
                raise ValueError(
                    f"no valid function call ({parsing_error}); "
                    f"{describe_reply(result['raw'])}"
                )
            return result["parsed"].model_dump()
        except Exception as error:
            logger.warning(
                "%s: attempt %d/%d failed: %s: %s",
                label,
                attempt,
                config.MAX_RETRIES,
                type(error).__name__,
                error,
            )
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)

    return None
