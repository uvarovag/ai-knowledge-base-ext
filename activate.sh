#!/bin/bash
# Использование: source activate.sh

set -a
[ -f .env ] && source .env
set +a

REQUIREMENTS="requirements.txt"
STAMP_FILE=".venv/.requirements-hash"

# 1. Проверяем, существует ли .venv
if [ ! -d ".venv" ]; then
    echo "⚙️  Venv not found. Creating..."
    # Пробуем uv, если его нет — обычный python
    if command -v uv &> /dev/null; then
        uv venv
    else
        python3.13 -m venv .venv
    fi
fi

# 2. Активируем (важно: через source в текущем шелле)
source .venv/bin/activate

export PIP_CONFIG_FILE=./.pip/pip.conf

# 3. Ставим зависимости, если requirements.txt изменился с прошлой установки
if [ ! -f "$REQUIREMENTS" ]; then
    echo "⚠️  $REQUIREMENTS not found, skipping dependency install"
else
    CURRENT_HASH=$(shasum -a 256 "$REQUIREMENTS" | cut -d' ' -f1)
    SAVED_HASH=""
    [ -f "$STAMP_FILE" ] && SAVED_HASH=$(cat "$STAMP_FILE")

    if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
        echo "📦 Installing dependencies..."
        if command -v uv &> /dev/null; then
            uv pip install -r "$REQUIREMENTS"
        else
            pip install -r "$REQUIREMENTS"
        fi

        if [ $? -eq 0 ]; then
            echo "$CURRENT_HASH" > "$STAMP_FILE"
            echo "✅ dependencies installed"
        else
            echo "❌ Dependency installation failed"
            echo "   Try: pip install -r $REQUIREMENTS"
        fi
    fi
fi

echo "✅ venv activated"
