#!/bin/bash
# =============================================================================
# UMRS Client Entrypoint Script
# =============================================================================
# Installs UMRS Python requirements at startup (from bind-mounted source),
# then executes the container CMD.
# =============================================================================

set -e

# Install UMRS Python requirements if source is mounted
if [[ -f /opt/umrs/requirements.txt ]]; then
    echo "Installing UMRS Python requirements..."
    pip install --no-cache-dir -r /opt/umrs/requirements.txt || true

    # Download spaCy model if not already available
    if ! python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
        echo "Downloading spaCy model en_core_web_sm..."
        python -m spacy download en_core_web_sm || true
    fi
    echo "UMRS requirements installed."
fi

exec "$@"
