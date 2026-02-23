#!/bin/bash
# =============================================================================
# AI-LOGIN (controller) Entrypoint Script
# =============================================================================
# Waits for munge key, then starts slurmctld and UMRS if available.
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

# Start SSH daemon (required for rivault.py SSH tunneling to compute nodes)
if command -v sshd >/dev/null 2>&1; then
    echo "Starting SSH daemon..."
    /usr/sbin/sshd 2>/dev/null || true
fi

# Start Podman system service if available
if command -v podman >/dev/null 2>&1; then
    echo "Starting podman system service..."
    # run service in background; keep running (-t 0)
    podman system service -t 0 &>/var/log/podman-service.log &
    sleep 1 || true
    echo "podman service started (if supported). Logs: /var/log/podman-service.log"
fi

# Ensure state directories exist with correct permissions
mkdir -p /var/spool/slurmctld
chown -R slurm:slurm /var/spool/slurmctld

# Ensure job info directory exists
mkdir -p /var/run/rivault
chmod 777 /var/run/rivault

# If UMRS source is mounted, install requirements and prepare the server.
# Note: UMRS is NOT started automatically because the Rivault stack must be
# running first (the rivault:// model type triggers Slurm job submission).
# Use ./setup.sh rivault to start everything in the correct order.
if [[ -d /opt/umrs ]]; then
    echo "UMRS sources detected at /opt/umrs"
    if command -v pip3 >/dev/null 2>&1; then
        echo "Installing UMRS Python requirements (may take a while)..."
        pip3 install --no-cache-dir -r /opt/umrs/requirements.txt || true
        echo "UMRS requirements installed. Server will be started shortly."

        # Ensure UMRS config directory structure exists
        UMRS_CFG_DIR="${HOME}/.config/umrs"
        mkdir -p "${UMRS_CFG_DIR}/users" "${UMRS_CFG_DIR}/rivault_data" "${UMRS_CFG_DIR}/tmp"

        # start the UMRS server in the background (it will wait for the database to be ready)
        echo "Starting UMRS server in the background..."
        # Note: In production, the server would be started separately from the controller (e.g. in a separate container or process) and the worker would communicate with it via the network. Here
        # we start it in the controller for simplicity, and the worker can access it via localhost.
        python3 /opt/umrs/server_umrs.py -c /root/conf_umrs.json &>/var/log/umrs-server.log &
        echo "UMRS server started. Logs: /var/log/umrs-server.log"

    else
        echo "pip3 not available; cannot install UMRS requirements."
    fi
fi

echo "Starting slurmctld..."
exec "$@"
