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
# U3 S6d: neutral UMRS_* names first, then the legacy R4_* ones (one release).
R4_LOOP_JOB_ID="${UMRS_LOOP_JOB_ID:-${R4_LOOP_JOB_ID:-}}"
R4_LOOP_ATTEMPT_ID="${UMRS_LOOP_ATTEMPT_ID:-${R4_LOOP_ATTEMPT_ID:-}}"
R4_LOOP_OWNER="${UMRS_LOOP_OWNER:-${R4_LOOP_OWNER:-}}"
R4_LOOP_PROJECT="${UMRS_LOOP_PROJECT:-${R4_LOOP_PROJECT:-}}"
R4_DIR="${UMRS_SRC_DIR:-${R4_DIR:-}}"
RIVAULT_WRAPPER_DIR="${RIVAULT_WRAPPER_DIR:?RIVAULT_WRAPPER_DIR required}"

# BEGIN loop-job declarations (ADR-117, U3 S6e-2)
# The gateway base (site.yaml network.gateway_base) and the driver image
# (site.yaml runtime.loop_job_image) are site-declared; core names neither,
# so an undeclared one refuses here. The gateway's host is the only egress
# destination of the driver network.
loop_gateway_host() {
  # scheme://[user@]host[:port][/path] -> host
  local rest="${1#*://}"
  rest="${rest%%/*}"
  rest="${rest##*@}"
  printf '%s' "${rest%%:*}"
}
R4_GATEWAY_BASE="${UMRS_GATEWAY_BASE:-${R4_GATEWAY_BASE:-}}"
if [[ -z "${R4_GATEWAY_BASE}" ]]; then
  echo "[loop-job] FATAL: this site declares no gateway base (site.yaml network.gateway_base / UMRS_GATEWAY_BASE; ADR-117); refusing to run the loop"
  exit 1
fi
GATEWAY_HOST="$(loop_gateway_host "${R4_GATEWAY_BASE}")"
if [[ -z "${GATEWAY_HOST}" ]]; then
  echo "[loop-job] FATAL: gateway base '${R4_GATEWAY_BASE}' names no host; refusing to run the loop"
  exit 1
fi
LOOP_IMAGE="${LOOP_JOB_IMAGE:-}"
if [[ -z "${LOOP_IMAGE}" ]]; then
  echo "[loop-job] FATAL: this site declares no loop driver image (site.yaml runtime.loop_job_image / LOOP_JOB_IMAGE; ADR-117); refusing to run the loop"
  exit 1
fi
# END loop-job declarations

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
ALLOWLIST="${GATEWAY_HOST}"
[[ -n "${EGRESS_EXTRA_ALLOWLIST:-}" ]] && ALLOWLIST="${ALLOWLIST},${EGRESS_EXTRA_ALLOWLIST}"
export EGRESS_DOMAIN_ALLOWLIST="${ALLOWLIST}"
export HTTPS_PROXY="http://127.0.0.1:8888" HTTP_PROXY="http://127.0.0.1:8888" \
  NO_PROXY="localhost,127.0.0.1,${GATEWAY_HOST}"

cleanup() {
  rm -rf "${SEC_DIR}"
  podman secret rm "loop-${JOB_ID}-gateway-hmac" 2>/dev/null || true
  rivault_net_remove "${PRIV_NETWORK}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# NOTE on --user 0: the image may default to a non-root user. Where the umrs
# source tree lives on Lustre (first seen on rccs), rootless subuid-mapped
# non-root users have no Lustre principal — every read came back EACCES
# (diagnosed live; legs 381037/381042). uid 0 inside the rootless user
# namespace maps to the *submitting* host user, exactly like the venue
# serving pods, so the read-only umrs-src bind mount resolves correctly.
# Containment is unaffected: cap-drop ALL / no-new-privileges / read-only
# rootfs / private network apply to root equally.
exec podman run --rm \
  --user 0 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --read-only \
  --network "${PRIV_NETWORK}" \
  --tmpfs /tmp:size=1g \
  --tmpfs /run:size=64m \
  --secret "loop-${JOB_ID}-gateway-hmac,type=env,target=UMRS_GATEWAY_HMAC_SECRET" \
  -v "${R4_DIR}:/umrs-repo/src/umrs:ro,Z" \
  -e LOOP_ID="${LOOP_ID}" \
  -e UMRS_GATEWAY_BASE="${R4_GATEWAY_BASE}" \
  -e UMRS_LOOP_JOB_ID="${R4_LOOP_JOB_ID}" \
  -e UMRS_LOOP_ATTEMPT_ID="${R4_LOOP_ATTEMPT_ID}" \
  -e UMRS_LOOP_OWNER="${R4_LOOP_OWNER}" \
  -e UMRS_LOOP_PROJECT="${R4_LOOP_PROJECT}" \
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
