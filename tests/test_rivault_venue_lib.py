"""Behaviour pins for ``mini-rivault-llm/venue.sh`` (ADR-045, WP-01 round 3b).

The library is exercised through bash against a fake ``podman``/``squeue`` pair
whose state lives in files, so every decision the wrapper makes on a compute
node — which stale networks to reap, which /24 to allocate, where an image
comes from, what podman configuration a job reads — is asserted here without
a cluster. The fake mirrors the behaviour observed live on rccs-cloud
(podman 4.9.3, CNI backend, 2026-09-07): "subnet ... is already used on the
host or by another config", dependent containers blocking ``network rm -f``,
and the state database pinning the runroot.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
VENUE_SH = UMRS_ROOT / "src" / "umrs" / "mini-rivault-llm" / "venue.sh"

ARCH = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "amd64": "amd64"}.get(
    platform.machine(), platform.machine()
)

FAKE_PODMAN = r"""#!/usr/bin/env bash
# File-backed fake podman. State dir: $FAKE_STATE
#   networks       name|subnet
#   used_subnets   subnets taken on the "host"
#   images         refs present in the node-local store
#   containers     cid|name|network|status
#   dependents     cid:dependent_cid   (removing cid fails while dependent exists)
#   auto_fail      when present, `network create` without --subnet fails
#   pinned_runroot runroot the state database pins (reported while db.sql exists)
state="${FAKE_STATE:?}"
touch "$state/networks" "$state/used_subnets" "$state/images" "$state/containers"
echo "$*" >> "$state/calls.log"
cmd="$1 $2"
case "$cmd" in
  "network ls") awk -F'|' '{print $1}' "$state/networks" ;;
  "network rm")
      shift 2; [[ "$1" == "-f" ]] && shift; name="$1"
      # dependent containers block the implicit container removal
      while IFS='|' read -r cid cname cnet status; do
          [[ "$cnet" == "$name" ]] || continue
          if grep -q "^$cid:" "$state/dependents" 2>/dev/null; then
              dep=$(grep "^$cid:" "$state/dependents" | cut -d: -f2)
              if grep -q "^$dep|" "$state/containers"; then
                  echo "Error: container $cid has dependent containers which must be removed before it: $dep: container already exists" >&2; exit 125
              fi
          fi
      done < "$state/containers"
      grep -v "|$name|" "$state/containers" > "$state/containers.tmp" || true; mv "$state/containers.tmp" "$state/containers"
      grep -v "^$name|" "$state/networks" > "$state/networks.tmp" || true; mv "$state/networks.tmp" "$state/networks" ;;
  "network create")
      shift 2; subnet=""; name=""
      while [[ $# -gt 0 ]]; do case "$1" in --subnet) subnet="$2"; shift 2;; --disable-dns) shift;; *) name="$1"; shift;; esac; done
      if [[ -z "$subnet" ]]; then
          [[ -f "$state/auto_fail" ]] && { echo "Error: could not find free subnet" >&2; exit 125; }
          subnet="10.89.7.0/24"
      fi
      if grep -qx "$subnet" "$state/used_subnets" || awk -F'|' '{print $2}' "$state/networks" | grep -qx "$subnet"; then
          echo "Error: subnet $subnet is already used on the host or by another config" >&2; exit 125
      fi
      echo "$name|$subnet" >> "$state/networks"; echo "$name" ;;
  "network inspect") grep "^$3|" "$state/networks" | awk -F'|' '{print $2" "}' ;;
  "image exists") grep -qx "$3" "$state/images" ;;
  "load -i")
      [[ "$(basename "$3")" == *broken* ]] && { echo "Error: payload does not match any of the supported image formats" >&2; exit 125; }
      ref="$(cat "$3")"; echo "$ref" >> "$state/images"; echo "Loaded image: $ref" ;;
  "tag "*) echo "$3" >> "$state/images" ;;
  "pull "*) [[ "$2" == *pullable* ]] && { echo "$2" >> "$state/images"; exit 0; }
      echo "Error: initializing source docker://$2: connection refused" >&2; exit 125 ;;
  "ps -a")
      shift 2; f=""; while [[ $# -gt 0 ]]; do [[ "$1" == "--filter" ]] && f="$2"; shift; done
      awk -F'|' -v net="${f#network=}" '$3 == net {print $1}' "$state/containers" ;;
  "ps -q") awk -F'|' '$4 == "Up" {print $1}' "$state/containers" ;;
  "rm -f")
      cid="$3"
      if grep -q "^$cid:" "$state/dependents" 2>/dev/null; then
          dep=$(grep "^$cid:" "$state/dependents" | cut -d: -f2)
          grep -q "^$dep|" "$state/containers" && { echo "Error: container $cid has dependent containers" >&2; exit 125; }
      fi
      grep -v "^$cid|" "$state/containers" > "$state/containers.tmp" || true; mv "$state/containers.tmp" "$state/containers" ;;
  "info --format")
      fmt="$3"
      if [[ -n "${RIVAULT_STORAGE_ROOT:-}" && -f "${RIVAULT_STORAGE_ROOT}/db.sql" && -f "$state/pinned_runroot" ]]; then
          runroot="$(cat "$state/pinned_runroot")"
      else
          runroot="${RIVAULT_RUNROOT:-/fake/run}"
      fi
      fmt="${fmt//\{\{.Store.RunRoot\}\}/$runroot}"; fmt="${fmt//\{\{.Host.NetworkBackend\}\}/cni}"
      fmt="${fmt//\{\{.Store.GraphRoot\}\}/${RIVAULT_STORAGE_ROOT:-/fake/storage}}"; echo "$fmt" ;;
  "--version "*|"--version ") echo "podman version 4.9.3-fake" ;;
  *) echo "fake podman: unhandled: $*" >&2; exit 2 ;;
esac
"""

FAKE_SQUEUE = r"""#!/usr/bin/env bash
state="${FAKE_STATE:?}"
[[ -f "$state/squeue_down" ]] && { echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }
job=""; while [[ $# -gt 0 ]]; do [[ "$1" == "-j" ]] && job="$2"; shift; done
if [[ -n "$job" ]]; then
    grep -qx "$job" "$state/running_jobs" 2>/dev/null && { echo "$job"; exit 0; }
    echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1
fi
cat "$state/running_jobs" 2>/dev/null; exit 0
"""


@pytest.fixture
def venue(tmp_path):
    """A fake bin dir + state dir; returns a runner for bash snippets."""
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    (fakebin / "podman").write_text(FAKE_PODMAN)
    (fakebin / "squeue").write_text(FAKE_SQUEUE)
    for f in fakebin.iterdir():
        f.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()

    def run(snippet: str, *, env: dict | None = None, strict: bool = False):
        full_env = dict(os.environ)
        full_env.update(
            {
                "PATH": f"{fakebin}:{os.environ['PATH']}",
                "FAKE_STATE": str(state),
                "XDG_RUNTIME_DIR": str(tmp_path / "xdg"),
                "RIVAULT_VENUE_ROOT": str(tmp_path / "venue"),
            }
        )
        # The wrapper's inherited environment must not leak into the pins.
        for key in list(full_env):
            if key.startswith(
                ("RIVAULT_STORAGE", "RIVAULT_RUNROOT", "RIVAULT_NETWORK", "RIVAULT_JOB")
            ):
                full_env.pop(key)
        full_env.update(env or {})
        opts = "set -euo pipefail" if strict else "set -uo pipefail"
        return subprocess.run(
            ["bash", "-c", f"{opts}; source '{VENUE_SH}'; {snippet}"],
            capture_output=True,
            text=True,
            env=full_env,
            check=False,
        )

    run.state = state  # type: ignore[attr-defined]
    return run


def _write(state: Path, name: str, lines: list[str]) -> None:
    (state / name).write_text("".join(f"{line}\n" for line in lines))


def _calls(state: Path) -> str:
    return (state / "calls.log").read_text() if (state / "calls.log").exists() else ""


# ---------------------------------------------------------------------------
# D2 — garbage collection
# ---------------------------------------------------------------------------


def test_gc_removes_only_networks_of_dead_jobs(venue):
    _write(
        venue.state,
        "networks",
        [
            "genai-private-net-270437|172.16.4.0/24",
            "genai-private-net-270477|10.0.4.0/24",
            "genai-private-net-400000|172.16.9.0/24",
            "genai-private-net-400001|172.16.10.0/24",
            "podman|10.88.0.0/16",
        ],
    )
    _write(venue.state, "running_jobs", ["400000"])
    r = venue(
        'rivault_net_gc genai-private-net- genai-private-net-400001; echo "R=$RIVAULT_NET_GC_REMOVED K=$RIVAULT_NET_GC_KEPT"'
    )
    assert r.returncode == 0, r.stderr
    names = (venue.state / "networks").read_text()
    assert "genai-private-net-270437" not in names and "genai-private-net-270477" not in names
    assert "genai-private-net-400000" in names  # running job
    assert "genai-private-net-400001" in names  # this job's own name (keep)
    assert "podman|" in names
    assert "R=2 K=1" in r.stdout
    assert "keeping genai-private-net-400000 (job 400000 still running)" in r.stdout


def test_gc_keeps_everything_when_slurm_is_unreachable(venue):
    _write(venue.state, "networks", ["genai-private-net-1|172.16.4.0/24"])
    (venue.state / "squeue_down").touch()
    r = venue('rivault_net_gc genai-private-net- ""; echo "R=$RIVAULT_NET_GC_REMOVED"')
    assert "R=0" in r.stdout
    assert "slurm unavailable" in r.stdout
    assert "genai-private-net-1" in (venue.state / "networks").read_text()


def test_gc_dry_run_reports_without_removing(venue):
    _write(venue.state, "networks", ["genai-private-net-7|172.16.4.0/24"])
    _write(venue.state, "running_jobs", [])
    r = venue('rivault_net_gc genai-private-net- ""', env={"RIVAULT_NET_GC_DRY_RUN": "1"})
    assert "would remove genai-private-net-7" in r.stdout
    assert "genai-private-net-7" in (venue.state / "networks").read_text()
    assert "network rm" not in _calls(venue.state)


def test_gc_force_removes_dependent_leftover_containers(venue):
    # Live finding (job 373989): nginx depends on litellm, so `network rm -f`
    # fails on the first pass; the library force-removes per container, twice.
    _write(venue.state, "networks", ["genai-private-net-373989|192.168.4.0/24"])
    _write(
        venue.state,
        "containers",
        [
            "be53|rivault-373989_litellm_1|genai-private-net-373989|Created",
            "00ca|rivault-373989_nginx_1|genai-private-net-373989|Created",
        ],
    )
    _write(venue.state, "dependents", ["be53:00ca"])
    _write(venue.state, "running_jobs", [])
    r = venue('rivault_net_gc genai-private-net- ""; echo "R=$RIVAULT_NET_GC_REMOVED"')
    assert "removed genai-private-net-373989" in r.stdout, r.stdout
    assert "R=1" in r.stdout
    assert (venue.state / "containers").read_text().strip() == ""


def test_gc_leaves_networks_without_job_id_alone_by_default(venue):
    _write(venue.state, "networks", ["genai-private-net-abc123|172.16.4.0/24"])
    r = venue('rivault_net_gc genai-private-net- ""; echo "R=$RIVAULT_NET_GC_REMOVED"')
    assert "R=0" in r.stdout and "no job id in name" in r.stdout
    r2 = venue(
        'rivault_net_gc genai-private-net- ""; echo "R=$RIVAULT_NET_GC_REMOVED"',
        env={"RIVAULT_NET_GC_WITHOUT_SLURM": "1"},
    )
    assert "R=1" in r2.stdout


# ---------------------------------------------------------------------------
# D2 — pool allocation
# ---------------------------------------------------------------------------


def test_candidates_walk_the_pool_from_a_job_derived_offset(venue):
    r = venue("rivault_net_candidates 172.16.0.0/16 379194 4 | tr '\\n' ' '")
    # 379194 % 254 + 1 == 227
    assert r.stdout.strip() == "172.16.227 172.16.228 172.16.229 172.16.230"
    wrap = venue(
        "rivault_net_candidates 172.16.0.0/16 253 3 | tr '\\n' ' '"
    )  # 253 % 254 + 1 == 254
    assert wrap.stdout.strip() == "172.16.254 172.16.1 172.16.2"
    single = venue("rivault_net_candidates 10.0.4.0/24 999 5 | tr '\\n' ' '")
    assert single.stdout.strip() == "10.0.4"
    preferred = venue("rivault_net_candidates 172.16.0.0/16 379194 2 192.168.4 | tr '\\n' ' '")
    assert preferred.stdout.strip() == "192.168.4 172.16.227 172.16.228"
    bad = venue("rivault_net_candidates 172.16.0.0/20 1 2")
    assert bad.returncode == 1 and "only /16 pools or a single /24" in bad.stdout


def test_create_skips_occupied_subnets_and_reports_the_mode(venue):
    _write(venue.state, "used_subnets", ["172.16.227.0/24", "192.168.4.0/24"])
    r = venue(
        "rivault_net_create genai-private-net-379194 172.16.0.0/16 379194 192.168.4 16; "
        'echo "IP=$RIVAULT_NET_IP1TO3 SUB=$RIVAULT_NET_SUBNET MODE=$RIVAULT_NET_MODE"'
    )
    assert r.returncode == 0, r.stdout
    assert "IP=172.16.228 SUB=172.16.228.0/24 MODE=pool" in r.stdout
    assert "attempt 3" in r.stdout  # preferred, 227, then 228
    ok = venue('rivault_net_create n2 172.16.0.0/16 5 172.16.40 3; echo "MODE=$RIVAULT_NET_MODE"')
    assert "MODE=preferred" in ok.stdout


def test_create_falls_back_to_podman_auto_allocation(venue):
    _write(venue.state, "used_subnets", ["10.0.4.0/24"])
    r = venue(
        'rivault_net_create n1 10.0.4.0/24 1 "" 4; echo "IP=$RIVAULT_NET_IP1TO3 MODE=$RIVAULT_NET_MODE"'
    )
    assert r.returncode == 0, r.stdout
    assert "IP=10.89.7 MODE=auto" in r.stdout
    assert "auto-allocated by podman" in r.stdout


def test_create_failure_lists_every_attempt_and_diagnostics(venue):
    _write(venue.state, "used_subnets", ["10.0.4.0/24"])
    (venue.state / "auto_fail").touch()
    r = venue('rivault_net_create genai-private-net-9 10.0.4.0/24 9 "" 4')
    assert r.returncode == 1
    assert "failed to create genai-private-net-9 after 2 attempts" in r.stdout
    assert (
        "10.0.4.0/24: Error: subnet 10.0.4.0/24 is already used on the host or by another config"
        in r.stdout
    )
    assert "auto: Error: could not find free subnet" in r.stdout
    assert "diagnostics:" in r.stdout and "backend=cni" in r.stdout


# ---------------------------------------------------------------------------
# D3 — image staging
# ---------------------------------------------------------------------------


def test_archive_path_convention(venue):
    r = venue(
        "rivault_image_archive_path /shared/images localhost/r4-mcp:latest arm64; "
        "rivault_image_archive_path /shared/images/ ghcr.io/x/umrs-vision amd64; "
        "rivault_image_archive_path /s localhost/r4-mcp@sha256:abcd arm64"
    )
    assert r.stdout.split() == [
        "/shared/images/r4-mcp-latest-arm64.tar",
        "/shared/images/umrs-vision-latest-amd64.tar",
        "/s/r4-mcp-latest-arm64.tar",
    ]


def test_ensure_image_present_short_circuits(venue):
    _write(venue.state, "images", ["localhost/r4-mcp:latest"])
    r = venue(
        'rivault_ensure_image localhost/r4-mcp:latest /nowhere; echo "SRC=$RIVAULT_IMAGE_SOURCE"'
    )
    assert r.returncode == 0 and "SRC=present" in r.stdout
    assert "load" not in _calls(venue.state) and "pull" not in _calls(venue.state)


def test_ensure_image_loads_the_conventional_archive(venue, tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    (archives / f"r4-mcp-latest-{ARCH}.tar").write_text("localhost/r4-mcp:latest")
    r = venue(
        f'rivault_ensure_image localhost/r4-mcp:latest {archives}; echo "SRC=$RIVAULT_IMAGE_SOURCE"'
    )
    assert r.returncode == 0, r.stdout
    assert "SRC=archive" in r.stdout and "loaded from archive" in r.stdout
    assert "pull" not in _calls(venue.state)


def test_ensure_image_retags_an_archive_saved_under_another_name(venue, tmp_path):
    archive = tmp_path / "vision.tar"
    archive.write_text("localhost/umrs-vision:latest")
    r = venue(
        f'rivault_ensure_image umrs-vision:latest "" {archive}; echo "SRC=$RIVAULT_IMAGE_SOURCE"'
    )
    assert r.returncode == 0, r.stdout
    assert "SRC=archive" in r.stdout
    assert "tag localhost/umrs-vision:latest umrs-vision:latest" in _calls(venue.state)


def test_ensure_image_pulls_when_no_archive_exists(venue, tmp_path):
    r = venue(
        f'rivault_ensure_image registry.example/pullable:1 {tmp_path}; echo "SRC=$RIVAULT_IMAGE_SOURCE"'
    )
    assert r.returncode == 0 and "SRC=pull" in r.stdout


def test_ensure_image_failure_names_the_missing_archive(venue, tmp_path):
    r = venue(f"rivault_ensure_image localhost/r4-mcp:latest {tmp_path}/images")
    assert r.returncode == 1
    assert f"no archive at {tmp_path}/images/r4-mcp-latest-{ARCH}.tar, and pull failed" in r.stdout
    none = venue("rivault_ensure_image localhost/r4-mcp:latest")
    assert "no archive dir declared (site.yaml runtime.image_archive_dir)" in none.stdout


def test_ensure_image_broken_archive_falls_through_to_pull(venue, tmp_path):
    (tmp_path / f"pullable-1-{ARCH}.tar").write_text("x")
    # basename contains "broken"? no — force the fake to fail the load by name
    (tmp_path / f"broken-1-{ARCH}.tar").write_text("localhost/broken:1")
    r = venue(f"rivault_ensure_image localhost/broken:1 {tmp_path}")
    assert r.returncode == 1
    assert "podman load" in r.stdout and "failed" in r.stdout


# ---------------------------------------------------------------------------
# D1 — job-private container configuration + stable runroot
# ---------------------------------------------------------------------------


def _prepared_venue(venue):
    return venue(
        'rivault_venue_prepare 4242 cgroupfs; cat "$CONTAINERS_CONF"; echo ---; cat "$CONTAINERS_STORAGE_CONF"; '
        'echo "JOBCONF=$RIVAULT_JOB_CONF_DIR RUN=$RIVAULT_RUNROOT NET=$RIVAULT_NETWORK_CONFIG_DIR"',
        strict=True,
    )


def test_container_configs_point_podman_at_job_private_node_local_paths(venue, tmp_path):
    r = _prepared_venue(venue)
    assert r.returncode == 0, r.stderr + r.stdout
    venue_root = tmp_path / "venue"
    assert f'network_config_dir = "{venue_root}/networks"' in r.stdout
    assert f'graphroot = "{venue_root}/storage"' in r.stdout
    assert (
        f'runroot = "{tmp_path}/xdg/containers"' in r.stdout
    )  # podman's rootless default under XDG
    assert f"JOBCONF={tmp_path}/xdg/rivault-4242/containers" in r.stdout


def test_containers_conf_carries_engine_defaults_and_no_backend_unless_asked(venue, tmp_path):
    r = _prepared_venue(venue)
    assert 'cgroup_manager = "cgroupfs"' in r.stdout
    assert "label = false" in r.stdout
    assert "network_backend" not in r.stdout  # podman decides unless the profile says so
    fuse_present = shutil.which("fuse-overlayfs") or Path("/usr/bin/fuse-overlayfs").exists()
    assert ("mount_program" in r.stdout) == bool(fuse_present)


def test_partition_profile_switches_the_network_backend(venue, tmp_path):
    r = venue(
        'rivault_venue_prepare 1 cgroupfs; cat "$CONTAINERS_CONF"; cat "$RIVAULT_STORAGE_ROOT/defaultNetworkBackend"',
        env={"RIVAULT_PARTITION_NEEDS_NETAVARK": "1"},
    )
    assert 'network_backend = "netavark"' in r.stdout
    assert r.stdout.rstrip().endswith("netavark")
    explicit = venue("rivault_venue_network_backend", env={"RIVAULT_NETWORK_BACKEND": "cni"})
    assert explicit.stdout.strip() == "cni"


def test_site_overrides_win_over_venue_defaults(venue, tmp_path):
    r = venue(
        'rivault_venue_defaults 1; echo "$RIVAULT_STORAGE_ROOT $RIVAULT_NETWORK_CONFIG_DIR $RIVAULT_RUNROOT"',
        env={
            "RIVAULT_STORAGE_ROOT": "/scratch/podman/storage",
            "RIVAULT_NETWORK_CONFIG_DIR": "/scratch/podman/nets",
        },
    )
    assert r.stdout.split() == [
        "/scratch/podman/storage",
        "/scratch/podman/nets",
        f"{tmp_path}/xdg/containers",
    ]


def _pinned_state(venue, tmp_path):
    """A graphroot whose state database pins a dead job's runroot (live finding)."""
    storage = tmp_path / "venue" / "storage"
    (storage / "libpod").mkdir(parents=True)
    (storage / "overlay-containers" / "deadbeef").mkdir(parents=True)
    (storage / "overlay-containers" / "containers.json").write_text('[{"id":"deadbeef"}]')
    (storage / "overlay-images").mkdir()
    (storage / "db.sql").write_text("sqlite")
    _write(venue.state, "pinned_runroot", ["/tmp/job-373989/xdg/containers/storage"])
    return storage


def test_runroot_stays_pinned_while_containers_run(venue, tmp_path):
    storage = _pinned_state(venue, tmp_path)
    _write(venue.state, "containers", ["c1|rivault-1_vllm_1|genai-private-net-1|Up"])
    busy = venue("rivault_venue_prepare 5 cgroupfs")
    assert "keeping database-pinned runroot /tmp/job-373989/xdg/containers/storage" in busy.stdout
    assert (storage / "db.sql").exists()


def test_runroot_is_re_pinned_when_idle_and_images_survive(venue, tmp_path):
    # Rootless podman keeps the runroot recorded in db.sql and ignores
    # storage.conf (only --runroot errors); with nothing running the container
    # state is dropped so the configured runroot takes effect.
    storage = _pinned_state(venue, tmp_path)
    _write(venue.state, "containers", [])
    idle = venue("rivault_venue_prepare 6 cgroupfs")
    assert "re-pinning runroot /tmp/job-373989/xdg/containers/storage ->" in idle.stdout
    assert not (storage / "db.sql").exists() and not (storage / "libpod").exists()
    assert not (storage / "overlay-containers" / "deadbeef").exists()
    assert (storage / "overlay-containers" / "containers.json").read_text() == "[]"
    assert (storage / "overlay-images").exists()  # images untouched
    assert "WARN runroot is still" not in idle.stdout


def _fake_gpu_host(tmp_path):
    dev = tmp_path / "dev"
    (dev / "nvidia-caps").mkdir(parents=True)
    for n in ("nvidia0", "nvidia1", "nvidiactl", "nvidia-uvm", "nvidia-caps/nvidia-cap1"):
        (dev / n).touch()
    lib = tmp_path / "lib64"
    lib.mkdir()
    (lib / "libcuda.so.1").touch()
    (lib / "libnvidia-ptxjitcompiler.so.1").touch()
    return dev, lib


def _gpu_override(venue, tmp_path):
    dev, lib = _fake_gpu_host(tmp_path)
    out = tmp_path / "ovr.yaml"
    r = venue(f"rivault_write_gpu_override {out} {dev} /nonexistent:{lib}", strict=True)
    assert r.returncode == 0, r.stderr
    return out.read_text(), lib


def test_gpu_override_maps_every_device_node(venue, tmp_path):
    text, _lib = _gpu_override(venue, tmp_path)
    assert "- /dev/nvidia0:/dev/nvidia0:rwm" in text
    assert "- /dev/nvidia1:/dev/nvidia1:rwm" in text
    assert "- /dev/nvidia-caps/nvidia-cap1:/dev/nvidia-caps/nvidia-cap1:rwm" in text
    assert text.count("bentoml-vision:") == 1
    assert "shm_size: ${VLLM_NCCL_MEM_LIMIT:?error}" in text


def test_gpu_override_binds_only_host_libraries_that_exist(venue, tmp_path):
    text, lib = _gpu_override(venue, tmp_path)
    assert f"- {lib}/libcuda.so.1:{lib}/libcuda.so.1:ro" in text
    assert f"- {lib}/libnvidia-ptxjitcompiler.so.1:{lib}/libnvidia-ptxjitcompiler.so:ro" in text
    assert "libnvidia-ml" not in text  # absent on this fake host -> not bound


def test_gpu_override_returns_one_without_a_gpu_node(venue, tmp_path):
    (tmp_path / "dev").mkdir()
    r = venue(f"rivault_write_gpu_override {tmp_path}/o.yaml {tmp_path}/dev")
    assert r.returncode == 1 and not (tmp_path / "o.yaml").exists()


# ---------------------------------------------------------------------------
# hygiene
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_venue_library_is_shellcheck_clean():
    r = subprocess.run(
        ["shellcheck", "-s", "bash", str(VENUE_SH)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stdout


def test_library_is_sourceable_under_strict_mode(venue):
    r = venue("echo sourced", strict=True)
    assert r.returncode == 0 and "sourced" in r.stdout
