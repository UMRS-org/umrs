#!/bin/bash
# =============================================================================
# UMRS Client Entrypoint Script
# =============================================================================
# Python requirements and spaCy models are pre-installed at image build time.
# This entrypoint only re-installs if the bind-mounted requirements.txt has
# changed since the image was built.
# =============================================================================

set -e

# Re-install only if bind-mounted requirements differ from the baked copy
if [[ -f /opt/umrs/requirements.txt ]]; then
    if diff -q /opt/umrs/requirements.txt /opt/umrs-requirements.txt >/dev/null 2>&1; then
        echo "Python requirements already installed (matching image build)."
    else
        echo "Requirements changed since image build, updating..."
        pip install --no-cache-dir -r /opt/umrs/requirements.txt || true
        cp /opt/umrs/requirements.txt /opt/umrs-requirements.txt 2>/dev/null || true
    fi

    # Ensure spaCy model is available
    if ! python -c "import spacy; spacy.load('en_core_web_sm')" >/dev/null 2>&1; then
        echo "Downloading spaCy model en_core_web_sm..."
        python -m spacy download en_core_web_sm || true
    fi
fi

exec "$@"
