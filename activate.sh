#!/bin/bash
# Usage: source activate.sh
# After this you can run python pipeline.py, uv pip install, etc. directly.
# Exports the same variables as SETUP_ENV in the Makefile; keep the two in step.

if [ ! -f .venv/bin/activate ]; then
    echo "❌ .venv not found — run \`make setup\` first"
    return 1 2>/dev/null || exit 1
fi

source .venv/bin/activate

set -a
[ -f .env ] && source .env
set +a

if [ -n "${SBEROSC_TOKEN}" ]; then
    export PIP_INDEX_URL="https://token:${SBEROSC_TOKEN}@sberosc.sigma.sbrf.ru/repo/pypi/simple"
    export UV_DEFAULT_INDEX="${PIP_INDEX_URL}"
fi
export UV_HTTP_TIMEOUT=90
export UV_CACHE_DIR=.uv-cache
# System certificate store instead of uv's bundle, see the note in the Makefile.
export UV_NATIVE_TLS=1
export UV_INSECURE_HOST="sberosc.sigma.sbrf.ru"

echo "✅ venv activated, uv environment variables set"
