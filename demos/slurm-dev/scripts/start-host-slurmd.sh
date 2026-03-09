#!/bin/bash
# =============================================================================
# Start Host slurmd + munged
# =============================================================================
#
# This script prepares and starts slurmd and munged directly on the host
# machine, so the host acts as the Slurm compute node "compute-01".
#
# The control plane (slurmctld, UMRS, postgres, minio) runs in containers
# via docker-compose.yaml.  The munge container generates the shared key
# into ./munge-key/ which this script copies to the host's /etc/munge/.
#
# Prerequisites:
#   - sudo apt-get install -y slurmd munge libmunge-dev
#   - Container control plane started: podman-compose up -d
#     (munge container must have generated ./munge-key/munge.key)
#   - nvidia-container-toolkit installed (for CDI GPU support)
#
# Usage:
#   cd demos/slurm-dev
#   sudo bash scripts/start-host-slurmd.sh
#
# =============================================================================

set -e

# Resolve paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

# Source .env if it exists (for COMPUTE_HOST_FQDN, MODEL_PATH, etc.)
if [[ -f "$DEMO_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$DEMO_DIR/.env"
    set +a
fi

MUNGE_KEY_SRC="$DEMO_DIR/munge-key/munge.key"
MUNGE_KEY_DST="/etc/munge/munge.key"
SLURM_CONF_SRC="$DEMO_DIR/slurm.conf"
SLURM_CONF_DST="/etc/slurm/slurm.conf"
GRES_CONF_SRC="$DEMO_DIR/gres.conf"
GRES_CONF_DST="/etc/slurm/gres.conf"
CGROUP_CONF_SRC="$DEMO_DIR/cgroup.conf"
CGROUP_CONF_DST="/etc/slurm/cgroup.conf"

# The node name that slurmd registers as (must match NodeName in slurm.conf)
SLURM_NODE_NAME="${SLURM_NODE_NAME:-compute-01}"

# Rivault directory (used by mini-rivault-job.sh / serve-model.slurm)
RIVAULT_DIR="${RIVAULT_DIR:-$DEMO_DIR/rivault}"

# Job info directory (shared with ai-login container via ./job-info mount)
JOB_INFO_DIR="${JOB_INFO_DIR:-$DEMO_DIR/job-info}"

echo "============================================================"
echo "  Starting host slurmd (compute node: $SLURM_NODE_NAME)"
echo "============================================================"
echo "  Demo dir:     $DEMO_DIR"
echo "  Rivault dir:  $RIVAULT_DIR"
echo "  Job info dir: $JOB_INFO_DIR"
echo ""

# ---- Check prerequisites ----
if [[ $EUID -ne 0 ]]; then
    echo "ERROR: This script must be run as root (sudo)."
    exit 1
fi

if ! command -v slurmd >/dev/null 2>&1; then
    echo "ERROR: slurmd not found. Install with: sudo apt-get install -y slurmd"
    exit 1
fi

if ! command -v munged >/dev/null 2>&1; then
    echo "ERROR: munged not found. Install with: sudo apt-get install -y munge"
    exit 1
fi

# ---- Step 1: Copy munge key from container-shared directory ----
echo "[1/8] Setting up munge key..."
if [[ ! -f "$MUNGE_KEY_SRC" ]]; then
    echo "  ERROR: Munge key not found at $MUNGE_KEY_SRC"
    echo "  Make sure the control plane is started: podman-compose up -d"
    echo "  The munge container generates the key into ./munge-key/"
    exit 1
fi

mkdir -p /etc/munge
cp "$MUNGE_KEY_SRC" "$MUNGE_KEY_DST"
chown munge:munge "$MUNGE_KEY_DST"
chmod 400 "$MUNGE_KEY_DST"
echo "  Munge key installed at $MUNGE_KEY_DST"

# ---- Step 2: Copy slurm.conf and gres.conf ----
echo "[2/8] Installing Slurm configuration..."
mkdir -p /etc/slurm
cp "$SLURM_CONF_SRC" "$SLURM_CONF_DST"
# Patch the host copy so slurmd connects to slurmctld via 127.0.0.1 (the
# container publishes port 6817 on the host).  We must NOT add "ai-login" to
# the host's /etc/hosts because podman rootless containers inherit that file,
# which would override DNS and break inter-container resolution.
sed -i 's|^SlurmctldHost=ai-login$|SlurmctldHost=ai-login(127.0.0.1)|' "$SLURM_CONF_DST"
echo "  Copied slurm.conf to $SLURM_CONF_DST (patched SlurmctldHost with 127.0.0.1)"

if [[ -f "$GRES_CONF_SRC" ]]; then
    cp "$GRES_CONF_SRC" "$GRES_CONF_DST"
    echo "  Copied gres.conf to $GRES_CONF_DST"
fi

if [[ -f "$CGROUP_CONF_SRC" ]]; then
    cp "$CGROUP_CONF_SRC" "$CGROUP_CONF_DST"
    echo "  Copied cgroup.conf to $CGROUP_CONF_DST"
fi

# ---- Step 3: Clean stale /etc/hosts entries ----
echo "[3/8] Cleaning hostname resolution..."
# Remove any old "ai-login" entries that were previously added to /etc/hosts.
# Podman rootless containers inherit the host's /etc/hosts, so if "ai-login"
# maps to 127.0.0.1 here, all containers will resolve it to loopback instead
# of the container's real IP on the bridge network.
if grep -q "ai-login" /etc/hosts; then
    sed -i '/ai-login/d' /etc/hosts
    echo "  Removed stale ai-login entry from /etc/hosts"
else
    echo "  No ai-login entry in /etc/hosts (OK)"
fi

# ---- Step 4: Create required directories ----
echo "[4/8] Creating required directories..."
mkdir -p /var/spool/slurmd
mkdir -p /var/log/slurm
mkdir -p "$JOB_INFO_DIR"
chown slurm:slurm /var/spool/slurmd /var/log/slurm 2>/dev/null || true
chmod 755 "$JOB_INFO_DIR"
echo "  /var/spool/slurmd, /var/log/slurm, $JOB_INFO_DIR"

# ---- Step 5: Symlink UMRS config dir for Slurm output files ----
# slurmctld (in ai-login) passes --output=/root/.config/umrs/tmp/slurm-%j.out
# to sbatch.  The container maps /root/.config/umrs → ./umrs-config via a
# bind mount.  slurmd on the host needs to write output files to the same
# location, so we symlink the host's /root/.config/umrs → ./umrs-config
# (unless it already points there or is the same path).
UMRS_CFG_HOST="/root/.config/umrs"
UMRS_CFG_TARGET="$DEMO_DIR/umrs-config"
if [[ -L "$UMRS_CFG_HOST" ]]; then
    echo "  $UMRS_CFG_HOST already symlinked → $(readlink "$UMRS_CFG_HOST")"
elif [[ -d "$UMRS_CFG_HOST" ]]; then
    echo "  WARNING: $UMRS_CFG_HOST exists as a directory."
    echo "  Backing up to ${UMRS_CFG_HOST}.bak and replacing with symlink."
    mv "$UMRS_CFG_HOST" "${UMRS_CFG_HOST}.bak"
    mkdir -p "$(dirname "$UMRS_CFG_HOST")"
    ln -sf "$UMRS_CFG_TARGET" "$UMRS_CFG_HOST"
    echo "  Symlinked $UMRS_CFG_HOST → $UMRS_CFG_TARGET"
else
    mkdir -p "$(dirname "$UMRS_CFG_HOST")"
    ln -sf "$UMRS_CFG_TARGET" "$UMRS_CFG_HOST"
    echo "  Symlinked $UMRS_CFG_HOST → $UMRS_CFG_TARGET"
fi

# ---- Step 5: Generate CDI spec for NVIDIA GPU (if nvidia-ctk available) ----
echo "[6/8] Checking NVIDIA CDI configuration..."
if command -v nvidia-ctk >/dev/null 2>&1; then
    if [[ ! -f /etc/cdi/nvidia.yaml ]]; then
        echo "  Generating CDI spec..."
        mkdir -p /etc/cdi
        nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        echo "  CDI spec generated at /etc/cdi/nvidia.yaml"
    else
        echo "  CDI spec already exists at /etc/cdi/nvidia.yaml"
    fi
else
    echo "  WARNING: nvidia-ctk not found. GPU support via CDI will not work."
    echo "  Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# ---- Step 7: Stop any existing munged/slurmd ----
echo "[7/8] Stopping any existing munge/slurmd processes..."
if systemctl stop munge 2>/dev/null; then
    echo "  Stopped munge via systemctl."
else
    # Kill only the host munged by PID (avoids killing munged inside containers)
    MUNGE_PIDFILE="${MUNGE_PIDFILE:-/var/run/munge/munged.pid}"
    if [[ -f "$MUNGE_PIDFILE" ]]; then
        MUNGE_PID=$(cat "$MUNGE_PIDFILE")
        if [[ -n "$MUNGE_PID" ]] && kill -0 "$MUNGE_PID" 2>/dev/null; then
            kill "$MUNGE_PID" 2>/dev/null || true
            echo "  Stopped munged (PID $MUNGE_PID) via pidfile."
        fi
    else
        # Fallback: find host munged by executable path to avoid hitting containers
        mapfile -t HOST_MUNGE_PIDS < <(pgrep -x munged 2>/dev/null || true)
        if [[ ${#HOST_MUNGE_PIDS[@]} -gt 0 ]]; then
            kill "${HOST_MUNGE_PIDS[@]}" 2>/dev/null || true
            echo "  Stopped munged (PIDs ${HOST_MUNGE_PIDS[*]}) via pgrep."
        fi
    fi
fi
if systemctl stop slurmd 2>/dev/null; then
    echo "  Stopped slurmd via systemctl."
else
    mapfile -t HOST_SLURMD_PIDS < <(pgrep -x slurmd 2>/dev/null || true)
    if [[ ${#HOST_SLURMD_PIDS[@]} -gt 0 ]]; then
        kill "${HOST_SLURMD_PIDS[@]}" 2>/dev/null || true
        echo "  Stopped slurmd (PIDs ${HOST_SLURMD_PIDS[*]}) via pgrep."
    fi
fi
sleep 1

# ---- Step 8: Start munged and slurmd ----
echo "[8/8] Starting munged and slurmd..."

# Start munged
mkdir -p /run/munge
chown munge:munge /run/munge
chmod 755 /run/munge
su -s /bin/bash munge -c "munged" || munged --force
sleep 1

# Verify munge works
if ! munge -n >/dev/null 2>&1; then
    echo "ERROR: Munge authentication test failed."
    echo "Check: ls -la /etc/munge/munge.key"
    echo "       journalctl -u munge  or  cat /var/log/munge/munged.log"
    exit 1
fi
echo "  munged is running and verified."

# Start slurmd with hostname override and environment variables
# -N compute-01  : register as "compute-01" regardless of actual hostname
# -D             : run in foreground (so this script blocks and can be Ctrl-C'd)
echo ""
echo "  Starting slurmd as '$SLURM_NODE_NAME' (foreground)..."
echo "  Press Ctrl-C to stop."
echo ""

# ---- Create /opt/rivault symlink ----
# The Slurm job script defaults to RIVAULT_DIR=/opt/rivault.
# Since batch jobs don't inherit slurmd's environment, we create a symlink so
# the default path works on the host.
echo "  Setting up /opt/rivault symlink..."
if [[ -L /opt/rivault ]]; then
    echo "  /opt/rivault already symlinked → $(readlink /opt/rivault)"
elif [[ -d /opt/rivault ]]; then
    echo "  /opt/rivault already exists as a directory"
else
    ln -sf "$RIVAULT_DIR" /opt/rivault
    echo "  Symlinked /opt/rivault → $RIVAULT_DIR"
fi

# ---- Install podman-compose (needed by serve-model.slurm) ----
if ! command -v podman-compose >/dev/null 2>&1; then
    echo "  Installing podman-compose..."
    pip3 install --quiet 'podman-compose>=1.2.0'
else
    echo "  podman-compose already installed."
fi

# ---- Authorize ai-login's SSH key for root on this host ----
# rivault.py creates an SSH tunnel by SSH-ing directly from ai-login into the
# compute node (this host) to forward ai-login:LOCAL_PORT → 127.0.0.1:11434
# (rootless podman only binds published ports on 127.0.0.1, not 0.0.0.0).
# The ai-login container's root SSH public key must be in /root/.ssh/authorized_keys.
AI_LOGIN_PUBKEY=$(podman exec ai-login cat /root/.ssh/id_ed25519.pub 2>/dev/null)
if [[ -n "$AI_LOGIN_PUBKEY" ]]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    if ! grep -qF "$AI_LOGIN_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null; then
        echo "$AI_LOGIN_PUBKEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        echo "  Authorized ai-login's SSH public key for root on this host."
    else
        echo "  ai-login's SSH public key already in /root/.ssh/authorized_keys."
    fi
else
    echo "  WARNING: Could not read ai-login's SSH public key (container not running?)."
fi

export RIVAULT_DIR
export JOB_INFO_DIR
export VLLM_PATH_TO_HUGGINGFACE="${VLLM_PATH_TO_HUGGINGFACE:-$RIVAULT_DIR/models/vllm/huggingface}"
export NGINX_LITELLM_DOCKER_PORT="${NGINX_LITELLM_DOCKER_PORT:-11434}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-demo-dev}"

if [[ "${1:-}" == "--bg" ]]; then
    echo "  Starting slurmd in BACKGROUND mode..."
    slurmd -N "$SLURM_NODE_NAME" -f "$SLURM_CONF_DST" -vv
    sleep 2
    if pgrep -x slurmd >/dev/null; then
        echo "  slurmd is running (PID $(pgrep -x slurmd))."
        echo "  Log: /var/log/slurm/slurmd.log"
    else
        echo "  ERROR: slurmd failed to start. Check /var/log/slurm/slurmd.log"
        exit 1
    fi
else
    exec slurmd -D -N "$SLURM_NODE_NAME" \
        -f "$SLURM_CONF_DST" \
        -vv
fi
