"""List the models GigaChat makes available to our certificate (GET /models).

Run this to pick GIGACHAT_MODEL_NAME and GIGACHAT_EMBEDDINGS_MODEL in config:
the list depends on the endpoint and on the access granted to the certificate.

Usage:
    make models
"""

from __future__ import annotations

import config
from kb.utils import gigachat, logs

if __name__ == "__main__":
    with logs.run(config.TOOLS_LOG_DIR, "models"):
        models = gigachat.call_with_retries(
            gigachat.build_llm().get_models, "List models", "models"
        )
        if models is None:
            raise SystemExit(f"Could not list models at {config.GIGACHAT_BASE_URL}")
        print(f"Models at {config.GIGACHAT_BASE_URL}:")
        for model in sorted(models.data, key=lambda model: model.id_):
            print(f"  {model.id_:<40} owned by {model.owned_by}")
