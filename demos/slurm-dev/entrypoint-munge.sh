#!/bin/bash
# =============================================================================
# Munge Entrypoint Script
# =============================================================================
# Generates munge key if not exists, fixes permissions on the bind-mounted
# munge-key directory, then starts munged as the munge user.
#
# Runs as root (Dockerfile does NOT set USER munge) so we can chown the
# host-mounted /etc/munge directory.
# =============================================================================

set -e

MUNGE_KEY=/etc/munge/munge.key

# Fix ownership/permissions on /etc/munge (bind-mounted from host)
chown munge:munge /etc/munge
chmod 0700 /etc/munge

# Generate munge key if not present
if [[ ! -f "$MUNGE_KEY" ]]; then
    echo "Generating new munge key..."
    dd if=/dev/urandom bs=1 count=1024 > "$MUNGE_KEY" 2>/dev/null
    chown munge:munge "$MUNGE_KEY"
    chmod 400 "$MUNGE_KEY"
    echo "Munge key generated."
else
    echo "Using existing munge key."
    chown munge:munge "$MUNGE_KEY"
    chmod 400 "$MUNGE_KEY"
fi

# Ensure /run/munge and log dir are correct
chown munge:munge /run/munge /var/log/munge
chmod 755 /run/munge

echo "Starting munged as munge user..."
exec runuser -s /bin/bash munge -- "$@"
