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

# DEMO model pre-downloads
## use scripts/download_rivault_models.sh to pre-download models into the shared rivault/models directory
#hf download openai/clip-vit-base-patch32 #\
#--local-dir clip-vit-base-patch32
#hf download intfloat/e5-base-v2 #\
#--local-dir e5-base-v2


# Create /opt/rivault symlink so sbatch can find mini-rivault-job.sh.
# The default RIVAULT_SLURM_SCRIPT points to /opt/rivault/mini-rivault-job.sh;
# inside this container the rivault scripts live under /opt/umrs/umrs/rivault/.
if [[ -d /opt/umrs/umrs/rivault ]] && [[ ! -e /opt/rivault ]]; then
    ln -s /opt/umrs/umrs/rivault /opt/rivault
    echo "Symlinked /opt/rivault → /opt/umrs/umrs/rivault"
fi

# If UMRS source is mounted, install requirements and prepare the server.
# Note: UMRS is NOT started automatically because the Rivault stack must be
# running first (the rivault:// model type triggers Slurm job submission).
# Use ./setup.sh rivault to start everything in the correct order.
if [[ -d /opt/umrs ]]; then
    echo "UMRS sources detected at /opt/umrs"
    if command -v uv >/dev/null 2>&1; then
        # Requirements are pre-installed at image build time.
        # Only re-install if the bind-mounted requirements.txt differs from the baked copy.
        if [[ -f /opt/umrs/requirements.txt ]]; then
            if diff -q /opt/umrs/requirements.txt /opt/umrs-requirements.txt >/dev/null 2>&1; then
                echo "Python requirements already installed (matching image build)."
            else
                echo "Requirements changed since image build, updating..."
                uv pip install -r /opt/umrs/requirements.txt || true
                cp /opt/umrs/requirements.txt /opt/umrs-requirements.txt 2>/dev/null || true
            fi
        fi

        # Ensure UMRS config directory structure exists
        UMRS_CFG_DIR="${HOME}/.config/umrs"
        mkdir -p "${UMRS_CFG_DIR}/users" "${UMRS_CFG_DIR}/rivault_data" "${UMRS_CFG_DIR}/tmp"

        # start the UMRS server in the background (it will wait for the database to be ready)
        echo "Starting UMRS server in the background..."
        # Note: In production, the server would be started separately from the controller (e.g. in a separate container or process) and the worker would communicate with it via the network. Here
        # we start it in the controller for simplicity, and the worker can access it via localhost.
        # Before starting, ensure any spaCy models referenced by the user config are available.
        USER_CFG="/root/.config/umrs/users/user_rivault.json"
        if [[ -f "$USER_CFG" ]]; then
            echo "Checking for spaCy models referenced in $USER_CFG"
            mapfile -t _SPACY_MODELS < <(python3 - <<PY
import json,sys
try:
    j=json.load(open('$USER_CFG'))
except Exception:
    sys.exit(0)
models=set()
def walk(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k=='spacy_model' and isinstance(v,str):
                models.add(v)
            else:
                walk(v)
    elif isinstance(o,list):
        for i in o:
            walk(i)
walk(j)
print('\n'.join(sorted(models)))
PY
)
            for sm in "${_SPACY_MODELS[@]:-}"; do
                if [[ -z "$sm" ]]; then
                    continue
                fi
                echo "Ensuring spaCy model: $sm"
                if python3 -c "import spacy; spacy.load('$sm')" >/dev/null 2>&1; then
                    echo "  -> $sm already available"
                    continue
                fi
                # Try to install spacy package if missing
                if ! python3 -c 'import spacy' >/dev/null 2>&1; then
                    echo "Installing spaCy runtime via pip"
                    pip3 install -U spacy || true
                fi
                echo "Downloading spaCy model $sm"
                python3 -m spacy download "$sm" || echo "Warning: failed to download spaCy model $sm"
            done
        fi

        #python3 /opt/umrs/server_umrs.py -c /root/conf_umrs.json &>/var/log/umrs-server.log &
        #echo "UMRS server started. Logs: /var/log/umrs-server.log"

    else
        echo "pip3 not available; cannot install UMRS requirements."
    fi
fi

echo "Starting slurmctld in background..."
# If slurmctl command available, use it; otherwise fall back to any provided CMD/args
if command -v slurmctld >/dev/null 2>&1; then
    # Clean incompatible state files from previous Slurm versions to avoid
    # "Can not recover last_conf_lite, incompatible version" fatal errors.
    if [[ -f /var/spool/slurmctld/last_config_lite ]]; then
        echo "  Clearing stale slurmctld state (version upgrade safe)..."
        rm -f /var/spool/slurmctld/last_config_lite \
              /var/spool/slurmctld/last_config_lite.old \
              /var/spool/slurmctld/clustername
        chown -R slurm:slurm /var/spool/slurmctld
    fi
    slurmctld &
else
    if [ "$#" -gt 0 ]; then
        "$@" &
    else
        echo "Warning: no slurmctld command found and no CMD provided; continuing"
    fi
fi

# Start RAG service in background if UMRS_RAG_ENABLE is set
if [[ "${UMRS_RAG_ENABLE:-0}" == "1" ]] && [[ -f /opt/umrs/rag_service.py ]]; then
    echo "Starting UMRS RAG service in background (port ${UMRS_RAG_PORT:-8081})..."
    # Ensure conf_umrs.json is reachable at the UMRS_CONFIG directory.
    # The compose file mounts it to /root/conf_umrs.json; symlink it into the
    # config directory so the RAG service can discover it automatically.
    _rag_cfg_dir="${HOME}/.config/umrs"
    mkdir -p "${_rag_cfg_dir}"
    if [[ -f /root/conf_umrs.json && ! -e "${_rag_cfg_dir}/conf_umrs.json" ]]; then
        ln -s /root/conf_umrs.json "${_rag_cfg_dir}/conf_umrs.json"
    fi
    UMRS_CONFIG="${_rag_cfg_dir}" \
    DB_HOST="${DB_HOST:-postgres}" \
    DB_PORT="${DB_PORT:-5432}" \
    DB_SUPERUSER_PASSWORD="${DB_SUPERUSER_PASSWORD:-}" \
    VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}" \
    python3 -m uvicorn rag_service:app \
        --host 0.0.0.0 --port "${UMRS_RAG_PORT:-8081}" \
        --app-dir /opt/umrs \
        &>/var/log/umrs-rag.log &
    echo "RAG service PID: $!  Logs: /var/log/umrs-rag.log"
fi

echo "Starting UMRS server in foreground (this is the container's main process)"
exec python3 -u /opt/umrs/server_umrs.py -c /root/conf_umrs.json
