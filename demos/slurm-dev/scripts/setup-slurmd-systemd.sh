#!/bin/bash
# =============================================================================
# Setup and start slurmd via systemd (proper cgroup v2 delegation)
# =============================================================================
# Usage:  cd demos/slurm-dev && sudo bash scripts/setup-slurmd-systemd.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

# Source .env
if [[ -f "$DEMO_DIR/.env" ]]; then
    set -a; source "$DEMO_DIR/.env"; set +a
fi

SLURM_NODE_NAME="${SLURM_NODE_NAME:-compute-01}"
# Rivault scripts moved from demos/slurm-dev/rivault/ to src/umrs/mini-rivault-llm/
RIVAULT_DIR="${RIVAULT_DIR:-$DEMO_DIR/../../src/umrs/mini-rivault-llm}"
JOB_INFO_DIR="${JOB_INFO_DIR:-$DEMO_DIR/job-info}"

if [[ $EUID -ne 0 ]]; then echo "ERROR: run with sudo"; exit 1; fi

echo "============================================================"
echo "  Setting up slurmd via systemd (node: $SLURM_NODE_NAME)"
echo "============================================================"

# ---- 1. Munge key ----
echo "[1/9] Munge key..."
MUNGE_KEY_SRC="$DEMO_DIR/munge-key/munge.key"
if [[ ! -f "$MUNGE_KEY_SRC" ]]; then
    echo "  ERROR: $MUNGE_KEY_SRC not found. Start containers first."
    exit 1
fi
mkdir -p /etc/munge
cp "$MUNGE_KEY_SRC" /etc/munge/munge.key
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key
echo "  OK"

# ---- 2. Slurm config files ----
echo "[2/9] Slurm configs..."
mkdir -p /etc/slurm
cp "$DEMO_DIR/slurm.conf" /etc/slurm/slurm.conf
sed -i 's|^SlurmctldHost=ai-login$|SlurmctldHost=ai-login(127.0.0.1)|' /etc/slurm/slurm.conf
echo "  Patched SlurmctldHost with 127.0.0.1"

[[ -f "$DEMO_DIR/gres.conf" ]] && cp "$DEMO_DIR/gres.conf" /etc/slurm/gres.conf && echo "  gres.conf OK"
[[ -f "$DEMO_DIR/cgroup.conf" ]] && cp "$DEMO_DIR/cgroup.conf" /etc/slurm/cgroup.conf && echo "  cgroup.conf OK"

# ---- 3. Clean /etc/hosts ----
echo "[3/9] /etc/hosts cleanup..."
if grep -q "ai-login" /etc/hosts; then
    sed -i '/ai-login/d' /etc/hosts
    echo "  Removed stale ai-login entry"
else
    echo "  OK (no ai-login entry)"
fi

# ---- 4. Directories ----
echo "[4/9] Directories..."
mkdir -p /var/spool/slurmd /var/log/slurm "$JOB_INFO_DIR"
chown slurm:slurm /var/spool/slurmd /var/log/slurm 2>/dev/null || true
chmod 755 "$JOB_INFO_DIR"
echo "  OK"

# ---- 5. UMRS config symlink ----
echo "[5/9] UMRS config symlink..."
UMRS_CFG_HOST="/root/.config/umrs"
UMRS_CFG_TARGET="$DEMO_DIR/umrs-config"
mkdir -p "$UMRS_CFG_TARGET/tmp"
if [[ -L "$UMRS_CFG_HOST" ]]; then
    echo "  Already: $UMRS_CFG_HOST → $(readlink "$UMRS_CFG_HOST")"
elif [[ -d "$UMRS_CFG_HOST" ]]; then
    mv "$UMRS_CFG_HOST" "${UMRS_CFG_HOST}.bak"
    mkdir -p "$(dirname "$UMRS_CFG_HOST")"
    ln -sf "$UMRS_CFG_TARGET" "$UMRS_CFG_HOST"
    echo "  Replaced dir with symlink"
else
    mkdir -p "$(dirname "$UMRS_CFG_HOST")"
    ln -sf "$UMRS_CFG_TARGET" "$UMRS_CFG_HOST"
    echo "  Created symlink"
fi

# ---- 6. CDI ----
echo "[6/9] NVIDIA CDI..."
if command -v nvidia-ctk >/dev/null 2>&1; then
    if [[ ! -f /etc/cdi/nvidia.yaml ]]; then
        mkdir -p /etc/cdi
        nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        echo "  Generated"
    else
        echo "  Already exists"
    fi
else
    echo "  WARNING: nvidia-ctk not found"
fi

# ---- 7. /opt/rivault symlink ----
echo "[7/9] /opt/rivault symlink..."
if [[ -L /opt/rivault ]]; then
    ln -sfn "$RIVAULT_DIR" /opt/rivault
    echo "  Updated: /opt/rivault → $RIVAULT_DIR"
elif [[ ! -d /opt/rivault ]]; then
    ln -sf "$RIVAULT_DIR" /opt/rivault
    echo "  Created"
else
    echo "  Directory already exists"
fi

# ---- 8. SSH key authorization ----
echo "[8/9] SSH key for ai-login..."
AI_LOGIN_PUBKEY=$(podman exec ai-login cat /root/.ssh/id_ed25519.pub 2>/dev/null || true)
if [[ -n "$AI_LOGIN_PUBKEY" ]]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    if ! grep -qF "$AI_LOGIN_PUBKEY" /root/.ssh/authorized_keys 2>/dev/null; then
        echo "$AI_LOGIN_PUBKEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        echo "  Authorized"
    else
        echo "  Already authorized"
    fi
else
    echo "  WARNING: Could not read ai-login SSH key"
fi

# ---- 9. Start services via systemd ----
echo "[9/9] Starting munged + slurmd via systemd..."

# Configure slurmd to use our node name and pass environment
mkdir -p /etc/systemd/system/slurmd.service.d
cat > /etc/systemd/system/slurmd.service.d/override.conf << EOF
[Service]
Environment="SLURMD_OPTIONS=-N $SLURM_NODE_NAME -f /etc/slurm/slurm.conf -vv"
Environment="RIVAULT_DIR=$RIVAULT_DIR"
Environment="JOB_INFO_DIR=$JOB_INFO_DIR"
Environment="VLLM_PATH_TO_HUGGINGFACE=${VLLM_PATH_TO_HUGGINGFACE:-$RIVAULT_DIR/models/vllm/huggingface}"
Environment="NGINX_LITELLM_DOCKER_PORT=${NGINX_LITELLM_DOCKER_PORT:-11434}"
Environment="LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY:-sk-demo-dev}"
EOF
echo "  Created systemd override for slurmd"

systemctl daemon-reload

# Restart munge
systemctl restart munge
sleep 1
if munge -n >/dev/null 2>&1; then
    echo "  munged: running and verified"
else
    echo "  ERROR: munge verification failed"
    exit 1
fi

# Restart slurmd
systemctl restart slurmd
sleep 2
if systemctl is-active --quiet slurmd; then
    echo "  slurmd: running (PID $(pgrep -x slurmd))"
else
    echo "  ERROR: slurmd failed to start. Check: journalctl -u slurmd --no-pager -n 30"
    journalctl -u slurmd --no-pager -n 20
    exit 1
fi

echo ""
echo "============================================================"
echo "  Done! slurmd is running as '$SLURM_NODE_NAME' via systemd"
echo "  Check:  sinfo -N"
echo "  Logs:   journalctl -u slurmd -f"
echo "============================================================"
