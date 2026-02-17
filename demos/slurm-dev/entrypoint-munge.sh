#!/bin/bash
# =============================================================================
# Munge Entrypoint Script
# =============================================================================
# Generates munge key if not exists, then starts munged.
# =============================================================================

set -e

MUNGE_KEY=/etc/munge/munge.key

# Generate munge key if not present
if [[ ! -f "$MUNGE_KEY" ]]; then
    echo "Generating new munge key..."
    dd if=/dev/urandom bs=1 count=1024 > "$MUNGE_KEY" 2>/dev/null
    chmod 400 "$MUNGE_KEY"
    echo "Munge key generated."
else
    echo "Using existing munge key."
fi

# Fix permissions
chmod 400 "$MUNGE_KEY" 2>/dev/null || true
chmod 755 /run/munge 2>/dev/null || true

echo "Starting munged..."
exec "$@"
