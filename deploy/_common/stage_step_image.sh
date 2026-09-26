#!/usr/bin/env bash
# =============================================================================
# stage_step_image.sh — build the platform step image from the pinned umrs
# commit and save it where compute nodes load it (ADR-074 D-7; superproject
# docs/development/rccs-step-lane-spec-2026-09-26.md §9)
# =============================================================================
# Runs on the login node over SSH from the converge, right after the checkout
# is pinned. It never pulls from or pushes to a registry except the digest-
# pinned base image, and never touches the node's default podman store.
#
# Usage:
#   stage_step_image.sh --arch aarch64|x86_64 --archive-dir DIR
#                       [--checkout DIR] [--if-missing] [--authfile FILE]
#     --arch         target machine, as `uname -m` prints it on the nodes
#     --archive-dir  where umrs-step-<sha12>-<arch>.tar (+ .sha256, .id) go
#     --checkout     the umrs checkout to build (default: this script's)
#     --if-missing   exit 0 without building when the archive is present,
#                    matches its .sha256 and holds the image its .id names
#                    (this commit, arch, contract label and local name)
#     --authfile     registry auth file for the base-image pull (an empty {}
#                    file forces an anonymous pull past stale credentials)
#
# Steps: refuse a checkout with tracked changes; `git archive HEAD src` (so
# untracked files never reach the image); build the umrs wheel; install it
# with the hash-pinned docker/step-requirements.txt into a site-packages dir
# for <arch>-manylinux_2_28 (binary wheels only); `podman build` the COPY-only
# docker/Containerfile.step in a throwaway store under /var/tmp; refuse the
# image when a layer it adds holds a file named (case-insensitively) env.*,
# *.env, .env, .env.*, s3cfg, s3cfg_*, .s3cfg, id_*, *.pem, *.key, .netrc,
# credentials[.*] or .git-credentials (the base layers are digest-pinned and carry
# the CA bundle's *.pem); `podman save` to a temp name in the archive dir, then
# rename it into place next to its .sha256 and .id; keep the newest 3
# archives per arch.
#
# stdout: exactly one JSON line (status built|present|failed); logs: stderr.
# Exit: 0 ok, 2 usage, 3 dirty or unusable checkout, 4 layer scan refused,
# 1 anything else.
# =============================================================================
set -euo pipefail
umask 077

readonly KEEP=3
readonly CONTRACT=1
readonly SCAN_PATTERNS='env.* *.env .env .env.* s3cfg s3cfg_* .s3cfg id_* *.pem *.key .netrc credentials credentials.* .git-credentials'

ARCH=""
ARCHIVE_DIR=""
CHECKOUT=""
IF_MISSING=0
AUTHFILE=""
WS=""

log() { printf 'stage_step_image: %s\n' "$*" >&2; }

json_str() {
    local s="${1//[[:cntrl:]]/ }"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '"%s"' "${s}"
}

fail() {
    local code="$1"
    shift
    log "$*"
    printf '{"status":"failed","exit":%d,"reason":%s}\n' "${code}" "$(json_str "$*")"
    exit "${code}"
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail 2 "option $1 needs a value"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --arch) need_value "$@"; ARCH="$2"; shift 2 ;;
            --archive-dir) need_value "$@"; ARCHIVE_DIR="$2"; shift 2 ;;
            --checkout) need_value "$@"; CHECKOUT="$2"; shift 2 ;;
            --authfile) need_value "$@"; AUTHFILE="$2"; shift 2 ;;
            --if-missing) IF_MISSING=1; shift ;;
            -h|--help) sed -n '2,/^set -euo/{/^set -euo/!p}' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
            *) fail 2 "unknown option: $1" ;;
        esac
    done
    case "${ARCH}" in
        aarch64) PLATFORM="linux/arm64"; OCI_ARCH="arm64" ;;
        x86_64) PLATFORM="linux/amd64"; OCI_ARCH="amd64" ;;
        *) fail 2 "--arch must be aarch64 or x86_64 (got '${ARCH}')" ;;
    esac
    [[ -n "${ARCHIVE_DIR}" ]] || fail 2 "--archive-dir is required"
    [[ -z "${AUTHFILE}" || -r "${AUTHFILE}" ]] || fail 2 "authfile not readable: ${AUTHFILE}"
    CHECKOUT="${CHECKOUT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
    UV_PLATFORM="${ARCH}-manylinux_2_28"
}

# The pinned commit is what the image claims to be, so a checkout whose
# tracked files differ from HEAD is refused. Untracked files are ignored:
# `git archive` cannot see them. --no-optional-locks keeps status from
# rewriting the index of the read-only host checkout.
check_checkout() {
    local dirty
    git -C "${CHECKOUT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || fail 3 "not a git checkout: ${CHECKOUT}"
    dirty="$(git -C "${CHECKOUT}" --no-optional-locks status --porcelain --untracked-files=no)" \
        || fail 3 "git status failed in ${CHECKOUT}"
    [[ -z "${dirty}" ]] || fail 3 "checkout ${CHECKOUT} has tracked changes; refusing to build: $(printf '%s' "${dirty}" | head -5 | tr '\n' ' ')"
    local f
    for f in docker/Containerfile.step docker/step-requirements.txt; do
        [[ -f "${CHECKOUT}/${f}" ]] || fail 3 "missing ${f} in ${CHECKOUT}"
    done
    SHA="$(git -C "${CHECKOUT}" rev-parse HEAD)"
    SHA12="${SHA:0:12}"
    EPOCH="$(git -C "${CHECKOUT}" log -1 --format=%ct HEAD)"
    IMAGE="localhost/umrs-step:${SHA12}"
    ARCHIVE="${ARCHIVE_DIR%/}/umrs-step-${SHA12}-${ARCH}.tar"
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# What a job checks after `podman load`: the id (the config digest), the
# arch, the contract label, the revision and the local name the load gives.
ARCHIVE_META_PY='
import hashlib, json, sys, tarfile
with tarfile.open(sys.argv[1]) as outer:
    manifest = json.load(outer.extractfile("manifest.json"))
    if len(manifest) != 1:
        sys.exit("archive holds %d images" % len(manifest))
    config = outer.extractfile(manifest[0]["Config"]).read()
parsed = json.loads(config)
labels = parsed.get("config", {}).get("Labels") or {}
print(hashlib.sha256(config).hexdigest(), parsed.get("architecture"),
      labels.get("org.umrs.step-contract"), labels.get("org.opencontainers.image.revision"),
      ",".join(manifest[0].get("RepoTags") or []))
'

# An archive whose sha256 matches but that jobs would refuse after loading
# it (a wrong .id among them) is not present: it is rebuilt in place.
archive_present() {
    local recorded found
    [[ -f "${ARCHIVE}" && -f "${ARCHIVE}.sha256" && -f "${ARCHIVE}.id" ]] || return 1
    recorded="$(cut -d' ' -f1 "${ARCHIVE}.sha256")"
    [[ -n "${recorded}" && "$(sha256_of "${ARCHIVE}")" == "${recorded}" ]] || return 1
    found="$(python3 -I -c "${ARCHIVE_META_PY}" "${ARCHIVE}" 2>/dev/null)" || return 1
    [[ "${found}" == "$(cat "${ARCHIVE}.id") ${OCI_ARCH} ${CONTRACT} ${SHA} ${IMAGE}" ]]
}

cleanup() {
    [[ -n "${WS}" && -d "${WS}" ]] || return 0
    [[ -n "${TMP_TAR:-}" ]] && rm -f "${TMP_TAR}" "${TMP_TAR}.sha256" "${TMP_TAR}.id"
    # Layers in the rootless store belong to subordinate ids; only the user
    # namespace can remove them.
    podman unshare rm -rf "${WS}/root" >/dev/null 2>&1 || true
    rm -rf "${WS}" 2>/dev/null || true
}

# A private storage.conf / containers.conf, like venue.sh writes for jobs: the
# shared ones in the Lustre HOME are stale and the node's default store is not
# to be touched.
make_workspace() {
    local mount_program=""
    WS="$(mktemp -d /var/tmp/umrs-step-stage.XXXXXXXX)"
    trap cleanup EXIT
    mkdir -p "${WS}/root" "${WS}/run" "${WS}/tree" "${WS}/wheel" "${WS}/ctx/site-packages"
    mount_program="$(command -v fuse-overlayfs 2>/dev/null || true)"
    {
        echo '[storage]'
        echo 'driver = "overlay"'
        echo "runroot = \"${WS}/run\""
        echo "graphroot = \"${WS}/root\""
        if [[ -n "${mount_program}" ]]; then
            echo '[storage.options.overlay]'
            echo "mount_program = \"${mount_program}\""
        fi
    } > "${WS}/storage.conf"
    printf '[engine]\ncgroup_manager = "cgroupfs"\nevents_logger = "none"\n[containers]\nlabel = false\n' \
        > "${WS}/containers.conf"
    export CONTAINERS_STORAGE_CONF="${WS}/storage.conf" CONTAINERS_CONF="${WS}/containers.conf"
    if [[ -n "${XDG_RUNTIME_DIR:-}" && ! -w "${XDG_RUNTIME_DIR}" ]]; then
        unset XDG_RUNTIME_DIR
    fi
}

pm() { podman --root "${WS}/root" --runroot "${WS}/run" "$@"; }

prepare_site_packages() {
    local site="${WS}/ctx/site-packages" wheel
    # Everything from the recorded commit, nothing from the worktree.
    git -C "${CHECKOUT}" archive --format=tar "${SHA}" src docker/Containerfile.step \
        docker/step-requirements.txt | tar -x -C "${WS}/tree"
    cp "${WS}/tree/docker/Containerfile.step" "${WS}/ctx/Containerfile.step"
    cp "${WS}/tree/docker/step-requirements.txt" "${WS}/step-requirements.txt"
    log "building the umrs wheel from ${SHA12}"
    (cd "${WS}" && uv build --no-config --quiet --wheel --python 3.11 --out-dir "${WS}/wheel" "${WS}/tree/src") >&2
    wheel="$(find "${WS}/wheel" -maxdepth 1 -name 'umrs-*.whl' | head -1)"
    [[ -n "${wheel}" ]] || fail 1 "uv build produced no umrs wheel"
    log "installing the step tier for ${UV_PLATFORM}"
    (cd "${WS}" && uv pip install --no-config --quiet --target "${site}" --python 3.11 \
        --python-version 3.11 --python-platform "${UV_PLATFORM}" --only-binary :all: \
        --require-hashes -r "${WS}/step-requirements.txt") >&2
    # The umrs wheel declares server-side dependencies (r4-contracts) the
    # worker never imports; the tier above is the whole runtime.
    (cd "${WS}" && uv pip install --no-config --quiet --target "${site}" --python 3.11 \
        --python-version 3.11 --python-platform "${UV_PLATFORM}" --no-deps "${wheel}") >&2
    # Console scripts carry the build host's interpreter path; the PEP 610
    # origin and uv's cache stamp name the throwaway workspace. Without them a
    # rebuild of one commit yields the same image id.
    rm -rf "${site:?}/bin"
    local info
    for info in "${site}"/umrs-*.dist-info; do
        rm -f "${info}/direct_url.json" "${info}/uv_cache.json"
        sed -i -e '/\/direct_url\.json,/d' -e '/\/uv_cache\.json,/d' "${info}/RECORD"
    done
    # umask 077 above; the job's --user is not the file owner in the image.
    chmod -R u=rwX,go=rX "${site}"
}

build_image() {
    local base_ref base_id
    base_ref="$(sed -n 's/^FROM[[:space:]]\{1,\}\([^[:space:]]\{1,\}\).*/\1/p' "${WS}/ctx/Containerfile.step" | head -1)"
    [[ "${base_ref}" == *@sha256:* ]] || fail 1 "Containerfile.step base is not digest-pinned: ${base_ref}"
    log "pulling ${base_ref} for ${PLATFORM}"
    base_id="$(pm pull -q --platform "${PLATFORM}" ${AUTHFILE:+--authfile "${AUTHFILE}"} "${base_ref}")" \
        || fail 1 "base image pull failed: ${base_ref}"
    BASE_LAYERS="$(pm image inspect --format '{{json .RootFS.Layers}}' "${base_id}")"
    log "building ${IMAGE} (${PLATFORM})"
    pm build --pull=never --platform "${PLATFORM}" --timestamp "${EPOCH}" \
        --identity-label=false --build-arg "UMRS_REVISION=${SHA}" \
        -t "${IMAGE}" -f "${WS}/ctx/Containerfile.step" "${WS}/ctx" >&2 \
        || fail 1 "podman build failed"
}

check_image() {
    local fmt got want
    fmt='{{.Architecture}} {{index .Labels "org.umrs.step-contract"}} {{index .Labels "org.opencontainers.image.revision"}}'
    got="$(pm image inspect --format "${fmt}" "${IMAGE}")"
    want="${OCI_ARCH} ${CONTRACT} ${SHA}"
    [[ "${got}" == "${want}" ]] || fail 1 "built image is '${got}', expected '${want}'"
    IMAGE_ID="$(pm image inspect --format '{{.Id}}' "${IMAGE}")"
}

# Reads the docker-archive podman just wrote: checks that the image starts
# with exactly the base layers, lists every member of each added layer, and
# reports the config digest (the id `podman load` will give the image).
SCAN_PY='
import fnmatch, hashlib, json, os, sys, tarfile
archive, base_layers, patterns = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3].split()
with tarfile.open(archive) as outer:
    manifest = json.load(outer.extractfile("manifest.json"))
    if len(manifest) != 1:
        sys.exit("archive holds %d images" % len(manifest))
    config = outer.extractfile(manifest[0]["Config"]).read()
    diff_ids = json.loads(config)["rootfs"]["diff_ids"]
    layers = manifest[0]["Layers"]
    if diff_ids[:len(base_layers)] != base_layers or len(layers) != len(diff_ids):
        sys.exit("image layers do not start with the base image layers")
    hits, members = [], 0
    for name in layers[len(base_layers):]:
        with tarfile.open(fileobj=outer.extractfile(name), mode="r|*") as layer:
            for member in layer:
                members += 1
                base = os.path.basename(member.name.rstrip("/"))
                if any(fnmatch.fnmatchcase(base.lower(), p) for p in patterns):
                    hits.append(member.name)
for hit in hits[:20]:
    sys.stderr.write("  forbidden: %s\n" % hit)
print(json.dumps({"image_id": hashlib.sha256(config).hexdigest(),
                  "layers_scanned": len(layers) - len(base_layers),
                  "members": members, "hits": len(hits)}))
sys.exit(4 if hits else 0)
'

save_archive() {
    local scan rc=0
    mkdir -p "${ARCHIVE_DIR}"
    TMP_TAR="${ARCHIVE_DIR%/}/.umrs-step-${SHA12}-${ARCH}.tar.$$.tmp"
    log "saving ${IMAGE} to ${TMP_TAR}"
    pm save --format docker-archive -o "${TMP_TAR}" "${IMAGE}" >&2 || fail 1 "podman save failed"
    scan="$(python3 -I -c "${SCAN_PY}" "${TMP_TAR}" "${BASE_LAYERS}" "${SCAN_PATTERNS}")" || rc=$?
    case "${rc}" in
        0) ;;
        4) fail 4 "layer scan refused ${IMAGE}: an added layer holds a forbidden file name (${scan})" ;;
        *) fail 1 "layer scan could not verify ${TMP_TAR}" ;;
    esac
    [[ "${scan}" == *"\"image_id\": \"${IMAGE_ID}\""* ]] || fail 1 "archive config digest differs from image id ${IMAGE_ID}"
    SCAN_JSON="${scan}"
    ARCHIVE_SHA256="$(sha256_of "${TMP_TAR}")"
    printf '%s  %s\n' "${ARCHIVE_SHA256}" "$(basename "${ARCHIVE}")" > "${TMP_TAR}.sha256"
    printf '%s\n' "${IMAGE_ID}" > "${TMP_TAR}.id"
    # Sidecars first: an archive under its final name always has both.
    mv -f "${TMP_TAR}.id" "${ARCHIVE}.id"
    mv -f "${TMP_TAR}.sha256" "${ARCHIVE}.sha256"
    mv -f "${TMP_TAR}" "${ARCHIVE}"
    TMP_TAR=""
}

prune_archives() {
    local old
    PRUNED=()
    while IFS= read -r old; do
        [[ -n "${old}" && "${old}" != "${ARCHIVE}" ]] || continue
        rm -f "${old}" "${old}.sha256" "${old}.id"
        PRUNED+=("$(basename "${old}")")
    done < <(find "${ARCHIVE_DIR}" -maxdepth 1 -type f \
                  -name "umrs-step-????????????-${ARCH}.tar" -printf '%T@ %p\n' \
             | sort -rn | tail -n +$((KEEP + 1)) | cut -d' ' -f2-)
}

emit_result() {
    local status="$1" scan="${2:-null}" seconds="${3:-0}" pruned="" p
    for p in "${PRUNED[@]}"; do
        pruned+="${pruned:+,}$(json_str "${p}")"
    done
    printf '{"status":"%s","image":%s,"image_id":"%s","revision":"%s","arch":"%s","archive":%s,"sha256":"%s","size_bytes":%d,"build_seconds":%d,"scan":%s,"pruned":[%s]}\n' \
        "${status}" "$(json_str "${IMAGE}")" "$(cat "${ARCHIVE}.id")" "${SHA}" "${ARCH}" \
        "$(json_str "${ARCHIVE}")" "$(cut -d' ' -f1 "${ARCHIVE}.sha256")" \
        "$(stat -c %s "${ARCHIVE}")" "${seconds}" "${scan}" "${pruned}"
}

main() {
    parse_args "$@"
    check_checkout
    PRUNED=()
    if [[ "${IF_MISSING}" -eq 1 ]] && archive_present; then
        log "archive present and verified: ${ARCHIVE}"
        # Pruning goes by mtime: a re-pinned archive is the newest in use, so a
        # later build never prunes the image the server still runs.
        touch -c "${ARCHIVE}" "${ARCHIVE}.sha256" "${ARCHIVE}.id"
        emit_result present
        return 0
    fi
    local command
    for command in git uv podman python3 sha256sum; do
        command -v "${command}" >/dev/null 2>&1 || fail 1 "${command} not found on PATH"
    done
    make_workspace
    prepare_site_packages
    build_image
    check_image
    save_archive
    prune_archives
    log "staged ${ARCHIVE} ($(du -h "${ARCHIVE}" | cut -f1)) in ${SECONDS}s"
    emit_result built "${SCAN_JSON}" "${SECONDS}"
}

main "$@"
