#!/bin/bash
# =============================================================================
# _cluster_loop_job.sh — research_loop_job driver on the cluster venue
# (ADR-036 D2 second half / WP-05 round 2).
#
# Inside the job-private venue (rivault machinery): run the loop driver in
# the staged image with the umrs source tree mounted read-only. Loop truth
# travels server-side via the UMRS gateway; the gateway handshake secret
# arrives hex-encoded, is materialized into a podman secret per job, and is
# deleted at teardown. Nothing is echo'ed, never in podman env output.
# =============================================================================

set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-manual}"
LOOP_ID="${LOOP_ID:?LOOP_ID required}"
R4_GATEWAY_BASE="${R4_GATEWAY_BASE:-https://data0.ai.r-ccs.riken.jp/rccs}"
R4_LOOP_JOB_ID="${R4_LOOP_JOB_ID:-}"
R4_LOOP_ATTEMPT_ID="${R4_LOOP_ATTEMPT_ID:-}"
R4_LOOP_OWNER="${R4_LOOP_OWNER:-}"
R4_LOOP_PROJECT="${R4_LOOP_PROJECT:-}"
LOOP_IMAGE="${LOOP_JOB_IMAGE:-localhost/r4-mcp:latest}"
RIVAULT_WRAPPER_DIR="${RIVAULT_WRAPPER_DIR:?RIVAULT_WRAPPER_DIR required}"

_venue_sh="${RIVAULT_WRAPPER_DIR}/../../src/umrs/mini-rivault-llm/venue.sh"
[[ -f "${_venue_sh}" ]] || {
  echo "[loop-job] FATAL: venue.sh not found at ${_venue_sh}"
  exit 1
}
# shellcheck disable=SC1090
source "${_venue_sh}"
rivault_venue_prepare
rivault_net_create "genai-loop-net-${JOB_ID}" || exit 1
PRIV_NETWORK="genai-loop-net-${JOB_ID}"

echo "[loop-job] job=${JOB_ID} image=${LOOP_IMAGE} net=${PRIV_NETWORK} gateway=${R4_GATEWAY_BASE}"

# ── Secrets materialization (hex-encoded wire-safe) ─────────────────────────
[[ -n "${UMRS_GATEWAY_HMAC_SECRET_HEX:-}" ]] || {
  echo "[loop-job] FATAL: UMRS_GATEWAY_HMAC_SECRET_HEX required"
  exit 78
}

SEC_DIR="$(mktemp -d "/tmp/loopsecrets-${JOB_ID}.XXXXXX")"
chmod 0700 "${SEC_DIR}"
python3 - <<PYEOF
blob = bytes.fromhex("${UMRS_GATEWAY_HMAC_SECRET_HEX}").decode().strip()
if not blob:
    raise SystemExit(1)
with open("${SEC_DIR}/gateway-hmac", "w") as fh:
    fh.write(blob)
PYEOF
[[ $? -eq 0 ]] || {
  echo "[loop-job] FATAL: HMAC hex decode produced empty material"
  rm -rf "${SEC_DIR}"
  exit 78
}
chmod 0400 "${SEC_DIR}/gateway-hmac"

podman secret create "loop-${JOB_ID}-gateway-hmac" "${SEC_DIR}/gateway-hmac" 2>/dev/null || {
  printf '%s' "$(cat "${SEC_DIR}/gateway-hmac")" | podman secret create "loop-${JOB_ID}-gateway-hmac" - >/dev/null
}

# Egress guard: only the venue host and (optionally) declared provider hosts
# are reachable from the driver network; everything else refuses fail-closed.
ALLOWLIST="data0.ai.r-ccs.riken.jp"
[[ -n "${EGRESS_EXTRA_ALLOWLIST:-}" ]] && ALLOWLIST="${ALLOWLIST},${EGRESS_EXTRA_ALLOWLIST}"
export EGRESS_DOMAIN_ALLOWLIST="${ALLOWLIST}"
export HTTPS_PROXY="http://127.0.0.1:8888" HTTP_PROXY="http://127.0.0.1:8888" \
  NO_PROXY="localhost,127.0.0.1,data0.ai.r-ccs.riken.jp"

cleanup() {
  rm -rf "${SEC_DIR}"
  podman secret rm "loop-${JOB_ID}-gateway-hmac" 2>/dev/null || true
  rivault_net_remove "${PRIV_NETWORK}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Diag (temporary): same mount inside the run, reporting as the image user.
podman run --rm --entrypoint /bin/sh \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --tmpfs /tmp:size=1g \
  --tmpfs /run:size=64m \
  -v "${R4_DIR}:/umrs-repo/src/umrs:ro,Z" \
  "${LOOP_IMAGE}" -c '
    id
    ls -la /umrs-repo/src/umrs | head -8
    ls /umrs-repo/src/umrs/knowledge 2>&1 | head -5
    PYTHONPATH=/umrs-repo/src python -c "import umrs; print(umrs.__file__)" 2>&1 | tail -2
    PYTHONPATH=/umrs-repo/src python -c "import umrs.knowledge; print(\"knowledge OK\")" 2>&1 | tail -2
  ' 2>&1 | sed 's/^/[diag] /'

exec podman run --rm \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --network "${PRIV_NETWORK}" \
  --tmpfs /tmp:size=1g \
  --tmpfs /run:size=64m \
  --secret "loop-${JOB_ID}-gateway-hmac,type=env,target=UMRS_GATEWAY_HMAC_SECRET" \
  -v "${R4_DIR}:/umrs-repo/src/umrs:ro,Z" \
  -e LOOP_ID="${LOOP_ID}" \
  -e R4_GATEWAY_BASE="${R4_GATEWAY_BASE}" \
  -e R4_LOOP_JOB_ID="${R4_LOOP_JOB_ID}" \
  -e R4_LOOP_ATTEMPT_ID="${R4_LOOP_ATTEMPT_ID}" \
  -e R4_LOOP_OWNER="${R4_LOOP_OWNER}" \
  -e R4_LOOP_PROJECT="${R4_LOOP_PROJECT}" \
  -e UMRS_GATEWAY_HMAC_SECRET_FILE="" \
  -e PYTHONPATH="/umrs-repo/src" \
  -e HTTPS_PROXY="${HTTPS_PROXY}" -e HTTP_PROXY="${HTTP_PROXY}" \
  -e NO_PROXY="${NO_PROXY}" \
  "${LOOP_IMAGE}" \
  python -m umrs.knowledge.cluster_loop_entry \
    --loop-id "${LOOP_ID}" \
    --job-id "${R4_LOOP_JOB_ID}" \
    --attempt-id "${R4_LOOP_ATTEMPT_ID}"
