#!/usr/bin/env bash
# =============================================================================
# migrate-volumes.sh — relocate umrs-front data volumes onto another filesystem
# =============================================================================
# Moves the persistent podman volumes (Postgres + MinIO) off the rootless
# podman graphroot under $HOME and onto a dedicated filesystem, then re-creates
# them as bind-backed named volumes so docker-compose.yaml keeps working
# unchanged (both volumes are declared `external: true`).
#
# Why a script instead of `cp -a`:
#   The volume contents are owned by a *subuid* (e.g. 363142 = subuid base +
#   container uid), so the invoking user cannot read them. `cp`, `rsync` and
#   `podman volume export` all either fail or silently produce a truncated
#   copy. The copy therefore runs inside a container that mounts both the
#   source volume and the destination directory, so podman performs the
#   user-namespace mapping.
#
# Two phases, on purpose:
#   (default)  copy + verify only — non-destructive, the live volumes are
#              untouched and the stack can still be started as before.
#   --commit   re-verify, then swap the named volumes over to the new
#              location. This removes the original volumes and is NOT
#              reversible; take a database dump first.
#
# Prerequisites (require root, do these first):
#   sudo mkdir -p /share/umrs
#   sudo chown "$USER":"$USER" /share/umrs
#   sudo semanage fcontext -a -e /home /share/umrs   # SELinux: treat like $HOME
#   sudo restorecon -Rv /share/umrs
#
# Usage:
#   ./migrate-volumes.sh --root /share/umrs            # copy + verify
#   ./migrate-volumes.sh --root /share/umrs --commit   # swap over
#
# After a successful --commit, set UMRS_VOLUME_ROOT in .env so a future
# start.sh recreates the volumes in the right place:
#   UMRS_VOLUME_ROOT=/share/umrs
# =============================================================================
set -euo pipefail

VOLUME_ROOT=""
COMMIT=false
VOLUMES=(umrs_pg_data umrs_minio_data)
# Any small image with cp/find/du. Alpine is already present on umrs-front.
MIGRATE_IMAGE="${MIGRATE_IMAGE:-docker.io/library/alpine:latest}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)    VOLUME_ROOT="${2:?--root needs a path}"; shift 2 ;;
        --volumes) read -r -a VOLUMES <<<"${2:?--volumes needs a list}"; shift 2 ;;
        --commit)  COMMIT=true; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

[[ -n "$VOLUME_ROOT" ]] || { echo "ERROR: --root is required" >&2; exit 2; }
VOLUME_ROOT="${VOLUME_ROOT%/}"

if [[ -z "${XDG_RUNTIME_DIR:-}" ]]; then
    XDG_RUNTIME_DIR="/run/user/$(id -u)"
    export XDG_RUNTIME_DIR
fi

# ─── Preflight ───────────────────────────────────────────────────────────────
echo "=== Preflight ==="

[[ -d "$VOLUME_ROOT" ]] || {
    echo "ERROR: '$VOLUME_ROOT' does not exist. Create it as root first (see header)." >&2
    exit 1
}
[[ -w "$VOLUME_ROOT" ]] || {
    echo "ERROR: '$VOLUME_ROOT' is not writable by $(whoami). Fix ownership as root." >&2
    exit 1
}

# Refuse to migrate underneath a running stack: Postgres will not survive
# having its data directory copied out from under it.
running="$(podman ps --format '{{.Names}}' 2>/dev/null || true)"
if [[ -n "$running" ]]; then
    echo "ERROR: containers are running. Stop the stack first (./stop.sh --all):" >&2
    echo "$running" | sed 's/^/         /' >&2
    exit 1
fi
echo "  no containers running"

if command -v getenforce >/dev/null 2>&1 && [[ "$(getenforce)" == "Enforcing" ]]; then
    echo "  SELinux is Enforcing — destination will be relabelled via the :z mount flag"
fi

echo "  destination : $VOLUME_ROOT ($(df -h "$VOLUME_ROOT" | awk 'NR==2{print $4}') available)"
echo "  volumes     : ${VOLUMES[*]}"

# ─── Helpers ─────────────────────────────────────────────────────────────────
# Count files and total bytes inside a mounted tree, from within a container so
# subuid-owned files are readable.
_measure() {
    local mount_spec="$1"
    podman run --rm "$mount_spec" "$MIGRATE_IMAGE" \
        sh -c 'printf "%s %s\n" "$(find /m -mount | wc -l)" "$(du -sb /m | cut -f1)"' 2>/dev/null
}

# ─── Phase 1: copy + verify ──────────────────────────────────────────────────
echo
echo "=== Copy ==="

for vol in "${VOLUMES[@]}"; do
    if ! podman volume inspect "$vol" >/dev/null 2>&1; then
        echo "ERROR: volume '$vol' does not exist" >&2
        exit 1
    fi

    dest="${VOLUME_ROOT}/${vol}"
    echo "-- $vol -> $dest"

    mkdir -p "$dest"

    # Copy inside a container: /from is the live volume, /to is the destination
    # bind mount. `:z` lets podman apply a shared container_file_t label, which
    # SELinux requires before any container may write there. `cp -a` preserves
    # the container-side uid/gid, which podman maps back to the subuid range.
    podman run --rm \
        -v "${vol}:/from:ro" \
        -v "${dest}:/to:z" \
        "$MIGRATE_IMAGE" \
        sh -c 'cp -a /from/. /to/'

    src_stat="$(_measure "-v=${vol}:/m:ro")"
    dst_stat="$(_measure "-v=${dest}:/m:z")"

    echo "     source      : ${src_stat% *} files, ${src_stat#* } bytes"
    echo "     destination : ${dst_stat% *} files, ${dst_stat#* } bytes"

    if [[ "$src_stat" != "$dst_stat" ]]; then
        echo "ERROR: verification failed for '$vol' — file count or size differs." >&2
        echo "       The original volume has NOT been touched. Investigate before retrying." >&2
        exit 1
    fi
    echo "     verified OK"
done

if [[ "$COMMIT" != true ]]; then
    echo
    echo "Copy complete and verified. Nothing has been changed yet."
    echo "The live volumes are untouched and the stack still starts as before."
    echo
    echo "To swap over (destructive, removes the original volumes):"
    echo "  $0 --root $VOLUME_ROOT --commit"
    exit 0
fi

# ─── Phase 2: swap ───────────────────────────────────────────────────────────
echo
echo "=== Commit ==="
echo "This removes the original volumes and re-creates them bind-backed at"
echo "$VOLUME_ROOT. This is not reversible."
read -r -p "Type 'yes' to continue: " confirm
[[ "$confirm" == "yes" ]] || { echo "Aborted."; exit 1; }

for vol in "${VOLUMES[@]}"; do
    dest="${VOLUME_ROOT}/${vol}"
    echo "-- swapping $vol"

    podman volume rm "$vol"
    podman volume create --driver local \
        --opt type=none --opt o=bind --opt device="$dest" "$vol" >/dev/null

    now="$(_measure "-v=${vol}:/m")"
    echo "     now backed by $dest — ${now% *} files, ${now#* } bytes"
done

echo
echo "Done. Add this to .env so start.sh recreates them in the right place:"
echo "  UMRS_VOLUME_ROOT=$VOLUME_ROOT"
