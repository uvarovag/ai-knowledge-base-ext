"""List the models GigaChat makes available to each certificate (GET /models).

The chat model and the embeddings model have certificates of their own, and
each one is granted its own models: run this to check that GIGACHAT_MODEL_NAME
and GIGACHAT_EMBEDDINGS_MODEL of config are among the models of their
certificate.

Usage:
    make models
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import config
from kb.utils import gigachat, logs


def print_models(
    name: str, model: str, certificate: tuple[Path, Path], get_models: Callable[[], Any]
) -> None:
    """Print the models one certificate is granted, marking the configured one."""
    print(f"{name} certificate {certificate[0].name}, configured model {model}:")
    missing = gigachat.missing_files(*certificate)
    if missing:
        print(f"  certificate not found: {', '.join(missing)}\n")
        return
    models = gigachat.call_with_retries(get_models, f"List models: {name}", "models")
    if models is None:
        print("  FAILED, see the log file")
        return
    names = sorted(entry.id_ for entry in models.data)
    for entry in names:
        print(f"  {'*' if entry == model else ' '} {entry}")
    if model not in names:
        print(f"  {model} is NOT available to this certificate")
    print()


def main() -> None:
    with logs.run(config.TOOLS_LOG_DIR, "models"):
        print(f"Models at {config.GIGACHAT_BASE_URL}\n")
        print_models(
            "Chat",
            config.GIGACHAT_MODEL_NAME,
            (config.CERT_FILE, config.KEY_FILE),
            lambda: gigachat.build_llm().get_models(),
        )
        # The embeddings client has no get_models of its own; its underlying
        # gigachat client, built with the embeddings certificate, does.
        print_models(
            "Embeddings",
            config.GIGACHAT_EMBEDDINGS_MODEL,
            (config.EMBEDDINGS_CERT_FILE, config.EMBEDDINGS_KEY_FILE),
            lambda: gigachat.build_embedder()._client.get_models(),
        )


if __name__ == "__main__":
    main()
