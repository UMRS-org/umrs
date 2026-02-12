#!/bin/bash
# =============================================================================
# Slurmctld Entrypoint Script
# =============================================================================
# Waits for munge key, then starts slurmctld.
# =============================================================================

set -e

MUNGE_KEY=/etc/munge/munge.key

echo "Waiting for munge key..."
while [[ ! -f "$MUNGE_KEY" ]]; do
    sleep 1
done
echo "Munge key found."

# Start munge client (connects to munge server via socket)
# Note: In this setup, each container runs its own munged with the shared key
echo "Starting local munged..."
chown -R munge:munge /etc/munge /var/log/munge 2>/dev/null || true
mkdir -p /run/munge
chown munge:munge /run/munge
chmod 755 /run/munge

# Start munged in background
su -s /bin/bash munge -c "munged" || {
    echo "Warning: Could not start munged as munge user, trying as root..."
    munged --force
}

# Wait for munged to be ready
sleep 2

# Verify munge is working
if ! munge -n > /dev/null 2>&1; then
    echo "ERROR: Munge is not working"
    exit 1
fi
echo "Munge is ready."

# Ensure state directories exist with correct permissions
mkdir -p /var/spool/slurmctld
chown -R slurm:slurm /var/spool/slurmctld

# Ensure job info directory exists
mkdir -p /var/run/rivault
chmod 777 /var/run/rivault

echo "Starting slurmctld..."
exec "$@"
