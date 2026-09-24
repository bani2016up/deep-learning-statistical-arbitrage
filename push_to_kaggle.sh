#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_DIR="$REPO_ROOT/kaggle"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
KAGGLE_JSON="$KAGGLE_CONFIG_DIR/kaggle.json"

mkdir -p "$KAGGLE_CONFIG_DIR"

if [ -f "$KAGGLE_JSON" ]; then
    echo "Using existing Kaggle credentials at $KAGGLE_JSON"
elif [ -f "$REPO_ROOT/kaggle.json" ]; then
    echo "Installing $REPO_ROOT/kaggle.json -> $KAGGLE_JSON"
    cp "$REPO_ROOT/kaggle.json" "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
elif [ -f "$REPO_ROOT/.env" ] && grep -q '^KAGGLE_USERNAME=' "$REPO_ROOT/.env" && grep -q '^KAGGLE_KEY=' "$REPO_ROOT/.env"; then
    echo "Synthesizing $KAGGLE_JSON from $REPO_ROOT/.env"
    KAGGLE_USERNAME="$(grep '^KAGGLE_USERNAME=' "$REPO_ROOT/.env" | cut -d= -f2-)"
    KAGGLE_KEY="$(grep '^KAGGLE_KEY=' "$REPO_ROOT/.env" | cut -d= -f2-)"
    printf '{"username":"%s","key":"%s"}\n' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > "$KAGGLE_JSON"
    chmod 600 "$KAGGLE_JSON"
else
    echo "No Kaggle credentials found (checked $KAGGLE_JSON, $REPO_ROOT/kaggle.json, $REPO_ROOT/.env)." >&2
    exit 1
fi

chmod 600 "$KAGGLE_JSON"

echo "Executing Kaggle push script..."
uv run --with kaggle python "$KAGGLE_DIR/kaggle_push.py"
