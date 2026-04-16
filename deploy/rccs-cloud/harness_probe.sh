#!/bin/bash
# =============================================================================
# harness_probe.sh — lightweight infrastructure probe for each partition
# =============================================================================
# Submit to each available partition. Checks: arch, podman, systemd, network
# creation, and optionally vision image load.
#
# Usage (on login node):
#   bash harness_probe.sh
# =============================================================================
set -euo pipefail

REPO="/lvs0/dne1/rccs-dmp/mnagaso/umrs-presigned-e2e"
RESULTS_DIR="${REPO}/tmp/harness_probe_$(date +%Y%m%d%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Partitions to probe — one job per partition, targeting one node
PARTITIONS=(
    ng-dgx-m0
    ng-dgx-m1
    ng-dgx-m2
    ng-dgx-m3
    a100
    ai-h100l
    ai-h200-brc
    ai-l40s
    qc-a100
    qc-gh200
    qc-mi210
    qc-mi250
    qc-pvc
    fx700
    genoa
    mi100
    fs-mi300a
    fs-mi300x
    r340
)

PROBE_SCRIPT="${RESULTS_DIR}/_probe_job.sh"
cat > "$PROBE_SCRIPT" <<'PROBEOF'
#!/bin/bash
set -x

RESULT_FILE="$1"
PARTITION_NAME="$2"

exec > "$RESULT_FILE" 2>&1

echo "===== HARNESS PROBE ====="
echo "PARTITION=$PARTITION_NAME"
echo "NODE=$(hostname)"
echo "FQDN=$(hostname -f)"
echo "ARCH=$(uname -m)"
echo "KERNEL=$(uname -r)"
echo "DATE=$(date -Iseconds)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"

# 1. Check podman
echo ""
echo "--- PODMAN ---"
if command -v podman >/dev/null 2>&1; then
    echo "PODMAN_FOUND=yes"
    podman --version 2>&1 || echo "PODMAN_VERSION_ERROR=yes"
else
    echo "PODMAN_FOUND=no"
    echo "RESULT=FAIL_NO_PODMAN"
    exit 0
fi

# 2. Check podman info (storage driver, cgroup manager)
echo ""
echo "--- PODMAN INFO ---"
podman info --format '{{.Host.CgroupManager}}' 2>&1 && true
podman info --format '{{.Store.GraphDriverName}}' 2>&1 && true
podman info --format '{{.Host.OCIRuntime.Name}}' 2>&1 && true
echo "OCI_RUNTIME=$(podman info --format '{{.Host.OCIRuntime.Name}}' 2>/dev/null || echo unknown)"
echo "OCI_RUNTIME_PATH=$(podman info --format '{{.Host.OCIRuntime.Path}}' 2>/dev/null || echo unknown)"

# 2b. Network backend detection
echo ""
echo "--- NETWORK BACKEND ---"
_net_backend="unknown"
if [[ -f "/var/tmp/podman_$(whoami)/storage/defaultNetworkBackend" ]]; then
    _net_backend="$(cat "/var/tmp/podman_$(whoami)/storage/defaultNetworkBackend" 2>/dev/null)"
fi
echo "DEFAULT_NETWORK_BACKEND=${_net_backend}"
[[ -x /usr/libexec/podman/netavark ]] && echo "NETAVARK=available" || echo "NETAVARK=not_found"
[[ -x /usr/libexec/cni/bridge ]] && echo "CNI_BRIDGE=available" || echo "CNI_BRIDGE=not_found"
[[ -x /usr/lib/cni/bridge ]] && echo "CNI_BRIDGE_ALT=available" || echo "CNI_BRIDGE_ALT=not_found"

# 3. Check storage setup
echo ""
echo "--- STORAGE SETUP ---"
mkdir -p "${HOME}/.config/containers/"
cat <<SEOF > "${HOME}/.config/containers/storage.conf"
[storage]
driver="overlay"
runroot = "${XDG_RUNTIME_DIR}/containers/storage"
rootless_storage_path="/var/tmp/podman_$(whoami)/storage"
graphroot = "/var/tmp/podman_$(whoami)/storage"
[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
[storage.options.overlay]
mountopt="nodev,metacopy=on"
SEOF
echo "STORAGE_CONF=written"
ls -la /usr/bin/fuse-overlayfs 2>&1 && echo "FUSE_OVERLAYFS=yes" || echo "FUSE_OVERLAYFS=no"

# 4. Cgroup / systemd bootstrap
echo ""
echo "--- CGROUP/SYSTEMD ---"
loginctl enable-linger "$(whoami)" 2>&1 && echo "ENABLE_LINGER=ok" || echo "ENABLE_LINGER=fail"
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
    mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
    echo "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
else
    echo "XDG_RUNTIME_DIR=unset"
fi
if command -v systemctl >/dev/null 2>&1; then
    systemd --user &
    sleep 5
    systemctl --user daemon-reload 2>&1 && echo "SYSTEMD_RELOAD=ok" || echo "SYSTEMD_RELOAD=fail"
    systemctl --user enable --now podman.socket 2>&1 && echo "PODMAN_SOCKET=ok" || echo "PODMAN_SOCKET=fail"
else
    echo "SYSTEMCTL=not_found"
fi

# 5. Network creation
echo ""
echo "--- NETWORK ---"
# Clean any leftovers
podman rm -af 2>/dev/null || true
podman network prune -f 2>/dev/null || true
podman network rm -f genai-private-net 2>/dev/null || true

NET_OK=0
for IP1TO3 in '172.16.4' '10.0.4' '192.168.4'; do
    if podman network create --disable-dns --subnet "${IP1TO3}.0/24" genai-private-net 2>/dev/null; then
        echo "NETWORK_CREATED=yes (${IP1TO3}.0/24)"
        NET_OK=1
        break
    fi
done
if [ "$NET_OK" -ne 1 ]; then
    echo "NETWORK_CREATED=no"
fi

# 6. GPU detection
echo ""
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus 2>&1 | head -5 && echo "GPU_TYPE=nvidia"
    # CDI device count
    if command -v nvidia-ctk >/dev/null 2>&1; then
        _cdi_count=$(nvidia-ctk cdi list 2>/dev/null | grep -c 'nvidia.com/gpu' || echo 0)
        echo "CDI_DEVICES=${_cdi_count}"
        nvidia-ctk cdi list 2>/dev/null | head -10
    else
        echo "CDI_DEVICES=nvidia-ctk_not_found"
    fi
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi -i 2>&1 | head -5 && echo "GPU_TYPE=amd"
    echo "CDI_DEVICES=n/a_amd"
else
    ls /dev/nvidia* 2>/dev/null && echo "GPU_TYPE=nvidia_devonly" || true
    ls /dev/dri/render* 2>/dev/null && echo "GPU_TYPE=possible_amd_or_intel" || true
    echo "GPU_TYPE=none_detected"
    echo "CDI_DEVICES=0"
fi

# 7. Container run test (hello-world equivalent)
echo ""
echo "--- CONTAINER RUN ---"
if podman run --rm --network genai-private-net alpine:latest echo "CONTAINER_RUN=ok" 2>&1; then
    true
else
    echo "CONTAINER_RUN=fail"
fi

# 8. Vision tarball check
echo ""
echo "--- VISION IMAGE ---"
TARBALL="/lvs0/dne1/rccs-dmp/mnagaso/umrs_models/umrs-vision.tar"
if [[ -f "$TARBALL" ]]; then
    echo "VISION_TARBALL=found ($(stat -c%s "$TARBALL" 2>/dev/null || stat -f%z "$TARBALL" 2>/dev/null) bytes)"
    if podman load -i "$TARBALL" 2>&1; then
        LOADED_ARCH=$(podman image inspect umrs-vision:latest --format '{{.Architecture}}' 2>/dev/null || true)
        HOST_ARCH=$(uname -m)
        echo "VISION_IMAGE_ARCH=$LOADED_ARCH"
        echo "HOST_ARCH=$HOST_ARCH"
        case "$HOST_ARCH" in
            aarch64|arm64) _expected="arm64" ;;
            x86_64|amd64)  _expected="amd64" ;;
            *)             _expected="$HOST_ARCH" ;;
        esac
        if [[ "$LOADED_ARCH" == "$_expected" ]]; then
            echo "VISION_ARCH_MATCH=yes"
        else
            echo "VISION_ARCH_MATCH=no (need $_expected, got $LOADED_ARCH)"
        fi
    else
        echo "VISION_LOAD=fail"
    fi
else
    echo "VISION_TARBALL=not_found"
fi

# 9. Lustre access check
echo ""
echo "--- LUSTRE ---"
ls /lvs0/dne1/rccs-dmp/mnagaso/umrs-presigned-e2e/src/umrs/mini-rivault-llm/serve-model.slurm >/dev/null 2>&1 \
    && echo "LUSTRE_REPO=accessible" || echo "LUSTRE_REPO=not_accessible"

# Cleanup
podman rm -af 2>/dev/null || true
podman network rm -f genai-private-net 2>/dev/null || true

echo ""
echo "===== PROBE DONE ====="
PROBEOF
chmod +x "$PROBE_SCRIPT"

echo "Probe script: $PROBE_SCRIPT"
echo "Results dir: $RESULTS_DIR"

# Submit one job per partition
for PART in "${PARTITIONS[@]}"; do
    RESULT_FILE="${RESULTS_DIR}/${PART}.log"
    # Check if partition is available (up) and has at least one non-down node
    _avail=$(sinfo -p "$PART" -h -o "%a" 2>/dev/null | head -1)
    if [[ "$_avail" != "up" ]]; then
        echo "SKIP: $PART (partition not 'up': $_avail)"
        echo "SKIP: partition not available ($_avail)" > "$RESULT_FILE"
        continue
    fi
    _node_states=$(sinfo -p "$PART" -N -h -o "%N %T" 2>/dev/null)
    _has_idle=$(echo "$_node_states" | grep -cE 'idle|mixed' || true)
    if [[ "$_has_idle" -eq 0 ]]; then
        echo "SKIP: $PART (no idle/mixed nodes)"
        echo "SKIP: no idle or mixed nodes available" > "$RESULT_FILE"
        continue
    fi
    echo "SUBMIT: $PART → $RESULT_FILE"
    sbatch --partition="$PART" \
           --nodes=1 \
           --time=00:15:00 \
           --job-name="probe-${PART}" \
           --output=/dev/null \
           --error=/dev/null \
           "$PROBE_SCRIPT" "$RESULT_FILE" "$PART" 2>&1 || \
    echo "SBATCH_FAIL: $PART" | tee -a "$RESULT_FILE"
done

echo ""
echo "All probe jobs submitted. Results will appear in: $RESULTS_DIR"
echo "Monitor with: watch 'ls -la $RESULTS_DIR/*.log | head -30'"
