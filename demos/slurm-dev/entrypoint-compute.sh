#!/bin/bash
# =============================================================================
# Compute (compute-01) Entrypoint Script
# =============================================================================
# Waits for munge key, then starts slurmd.
# =============================================================================

set -e

MUNGE_KEY=/etc/munge/munge.key

echo "Waiting for munge key..."
while [[ ! -f "$MUNGE_KEY" ]]; do
    sleep 1
done
echo "Munge key found."

# Start munge client
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

# Configure podman storage for nested containers (podman-in-podman)
if command -v podman >/dev/null 2>&1; then
    echo "Configuring podman for nested container operation..."
    mkdir -p "${HOME}/.config/containers/"
    cat > "${HOME}/.config/containers/storage.conf" << 'STOREOF'
[storage]
driver="overlay"
[storage.options.overlay]
force_mask="700"
STOREOF

    echo "Starting podman system service..."
    podman system service -t 0 &>/var/log/podman-service.log &
    sleep 1 || true
    echo "podman service started (if supported). Logs: /var/log/podman-service.log"
fi

# Ensure state directories exist with correct permissions
mkdir -p /var/spool/slurmd
chown -R slurm:slurm /var/spool/slurmd

# Ensure job info directory exists
mkdir -p /var/run/rivault
chmod 777 /var/run/rivault

# Start SSH for debugging (optional)
/usr/sbin/sshd 2>/dev/null || true

# Wait for controller to be ready
echo "Waiting for slurmctld..."
MAX_WAIT=60
WAITED=0
while ! sinfo -N > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo "WARNING: slurmctld not responding after ${MAX_WAIT}s, starting anyway..."
        break
    fi
done

echo "Starting slurmd on $(hostname)..."
exec "$@"
