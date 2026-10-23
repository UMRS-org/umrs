"""Essential-sidecar health in the serving-stack monitor (2026-10-23).

bdrt jobs 8218/8219 ran 30 h with a healthy vLLM while their LiteLLM
container had exited at start; nginx answered 502 and the monitor in
serve-model.slurm, which watched only the engine, never failed the job. The
monitor now treats LiteLLM and the nginx front as essential: an exited one,
or a front that keeps answering 502, is respawned from the per-job budget
(attempt_budget.sh) and the job exits 125 once that budget is gone.

Two layers, tested the way test_attempt_budget.py tests its library: the
stack_health.sh functions with stub podman/curl on PATH, and the real
monitor section of serve-model.slurm run against the same stubs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

MINI_RIVAULT = Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm"
LIB = MINI_RIVAULT / "stack_health.sh"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
PROJECT = "rivault-77"

_PODMAN = r"""#!/bin/bash
echo "podman $*" >> "$STUB_DIR/calls.log"
state_of() { cat "$STUB_DIR/state.$1" 2>/dev/null; }
case "$1 $2" in
  "ps -q") ls "$STUB_DIR" | grep -c '^state\.' ; exit 0 ;;
  "container exists") [ -f "$STUB_DIR/state.$3" ]; exit $? ;;
  "container inspect")
      [ -f "$STUB_DIR/state.$3" ] || exit 125
      case "$5" in *ExitCode*) echo 1 ;; *) state_of "$3" ;; esac
      exit 0 ;;
esac
exit 0
"""

# Recreate = the container comes back in the state STUB_RECREATE_STATE names.
_COMPOSE = r"""#!/bin/bash
echo "podman-compose $*" >> "$STUB_DIR/calls.log"
svc="${@: -1}"
echo "${STUB_RECREATE_STATE:-exited}" > "$STUB_DIR/state.rivault-77_${svc}_1"
"""

_CURL = r"""#!/bin/bash
printf '%s' "${STUB_HTTP_CODE:-200}"
"""


def _stub_bin(root: Path) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("podman", _PODMAN),
        ("podman-compose", _COMPOSE),
        ("curl", _CURL),
        ("sleep", "#!/bin/bash\nexit 0\n"),
    ):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    return bin_dir


@pytest.fixture()
def stub(tmp_path: Path) -> Path:
    """A stub dir whose state.<container> files are the containers' states."""
    _stub_bin(tmp_path)
    (tmp_path / "info").mkdir()
    for svc, state in (("vllm", "running"), ("litellm", "running"), ("nginx", "running")):
        (tmp_path / f"state.{PROJECT}_{svc}_1").write_text(state + "\n")
    return tmp_path


def _env(stub: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{stub / 'bin'}:{env['PATH']}",
            "STUB_DIR": str(stub),
            "JOB_INFO_DIR": str(stub / "info"),
        }
    )
    env.update(extra)
    return env


def _lib(stub: Path, script: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"source {LIB} && {script}"],
        env=_env(stub, **extra),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _set_state(stub: Path, svc: str, state: str) -> None:
    (stub / f"state.{PROJECT}_{svc}_1").write_text(state + "\n")


def _calls(stub: Path) -> str:
    log = stub / "calls.log"
    return log.read_text() if log.exists() else ""


# ── stack_health.sh ─────────────────────────────────────────────────────────


def test_exited_sidecars_names_only_present_dead_ones(stub: Path) -> None:
    _set_state(stub, "litellm", "exited")
    (stub / f"state.{PROJECT}_nginx_1").unlink()  # profile off: not checked
    r = _lib(stub, f"stack_exited_sidecars {PROJECT}")
    assert r.stdout == "litellm", r.stderr


def test_a_running_stack_names_no_sidecar(stub: Path) -> None:
    assert _lib(stub, f"stack_exited_sidecars {PROJECT}").stdout == ""


@pytest.mark.parametrize(
    ("code", "up"),
    [("200", True), ("401", True), ("404", True), ("502", False), ("503", False), ("000", False)],
)
def test_the_front_is_down_only_on_gateway_errors_or_no_answer(
    stub: Path, code: str, up: bool
) -> None:
    r = _lib(stub, "stack_front_answers 4000", STUB_HTTP_CODE=code)
    assert (r.returncode == 0) is up


def test_an_exited_litellm_asks_for_its_respawn_on_the_first_tick(stub: Path) -> None:
    _set_state(stub, "litellm", "exited")
    r = _lib(
        stub,
        f'stack_health_init; stack_health_tick {PROJECT} 4000; echo "rc=$? $STACK_HEALTH_RESPAWN"',
    )
    assert "rc=0 litellm" in r.stdout, r.stderr


def test_a_502_front_inside_the_startup_grace_is_not_yet_a_failure(stub: Path) -> None:
    script = (
        "stack_health_init; for i in 1 2 3 4 5; do stack_health_tick "
        f'{PROJECT} 4000 && echo respawn; done; echo "streak=$_SH_FRONT_STREAK"'
    )
    r = _lib(stub, script, STUB_HTTP_CODE="502", RIVAULT_FRONT_FAIL_STREAK="3")
    assert "respawn" not in r.stdout
    assert "streak=5" in r.stdout


def test_a_front_that_answered_once_then_502s_persistently_respawns_both(stub: Path) -> None:
    script = (
        f"stack_health_init; stack_health_tick {PROJECT} 4000; "
        f"export STUB_HTTP_CODE=502; for i in 1 2 3; do "
        f'stack_health_tick {PROJECT} 4000 && echo "respawn@$i $STACK_HEALTH_RESPAWN"; done'
    )
    r = _lib(stub, script, RIVAULT_FRONT_FAIL_STREAK="3")
    assert r.stdout.strip() == "respawn@3 litellm nginx", r.stderr


def test_the_probe_is_skipped_when_litellm_is_not_part_of_the_stack(stub: Path) -> None:
    (stub / f"state.{PROJECT}_litellm_1").unlink()
    script = f"stack_health_init; stack_health_tick {PROJECT} 4000 || echo healthy"
    r = _lib(
        stub,
        script,
        STUB_HTTP_CODE="502",
        RIVAULT_FRONT_STARTUP_GRACE_SEC="0",
        RIVAULT_FRONT_FAIL_STREAK="1",
    )
    assert r.stdout.strip() == "healthy"


# ── the monitor section of serve-model.slurm against the stubs ──────────────

_PRELUDE = f"""
SLURM_JOB_ID=77
RIVAULT_COMPOSE_PROJECT={PROJECT}
RIVAULT_NETWORK_NAME=genai-net-77
RIVAULT_STOP_REQUEST_FILE=""
NGINX_LITELLM_DOCKER_PORT=4000
uuid=u77
GPUoverride=()
DbOverride=()
_profile_flags=""
"""


def _run_monitor(stub: Path, **extra: str) -> subprocess.CompletedProcess:
    """Run the wrapper's monitor (publish + loop) with its two libraries."""
    text = SERVE_MODEL_SLURM.read_text()
    section = text[text.index("RIVAULT_JOB_TIMEOUT=${RIVAULT_JOB_TIMEOUT:-0}") :]
    wrapper_dir = stub / "wrapper"
    wrapper_dir.mkdir()
    for lib in ("attempt_budget.sh", "stack_health.sh"):
        shutil.copy(MINI_RIVAULT / lib, wrapper_dir / lib)
    monitor = wrapper_dir / "monitor.sh"
    monitor.write_text(_PRELUDE + section)
    settings = {"RIVAULT_JOB_TIMEOUT": "300", "RIVAULT_ENGINE_RESPAWN_SLEEP_SEC": "0"}
    settings.update(extra)
    return subprocess.run(
        ["bash", str(monitor)],
        cwd=wrapper_dir,
        env=_env(stub, **settings),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_a_litellm_that_keeps_exiting_fails_the_job_after_the_budget(stub: Path) -> None:
    """The 8218 shape: engine up, LiteLLM dead. Old monitor: ran until the
    walltime (here the 300 s test timeout, exit 124) serving 502."""
    _set_state(stub, "litellm", "exited")
    r = _run_monitor(stub)
    assert r.returncode == 125, r.stdout[-2000:]
    assert _calls(stub).count("up --detach --no-deps litellm") == 3
    info = (stub / "info" / "rivault-job-77.info").read_text()
    assert "RIVAULT_TERMINAL_REASON=budget_exhausted" in info


def test_a_litellm_that_comes_back_keeps_the_job_running(stub: Path) -> None:
    _set_state(stub, "litellm", "exited")
    r = _run_monitor(stub, STUB_RECREATE_STATE="running")
    assert r.returncode == 124, r.stdout[-2000:]  # ran to the test timeout
    assert _calls(stub).count("up --detach --no-deps litellm") == 1
    assert "Sidecar respawn OK (litellm)" in r.stdout


def test_a_healthy_stack_is_left_alone(stub: Path) -> None:
    r = _run_monitor(stub)
    assert r.returncode == 124, r.stdout[-2000:]
    assert "podman-compose" not in _calls(stub)


def test_a_front_stuck_on_502_fails_the_job_after_the_budget(stub: Path) -> None:
    r = _run_monitor(
        stub,
        STUB_HTTP_CODE="502",
        STUB_RECREATE_STATE="running",
        RIVAULT_FRONT_STARTUP_GRACE_SEC="0",
        RIVAULT_FRONT_FAIL_STREAK="3",
    )
    assert r.returncode == 125, r.stdout[-2000:]
    calls = _calls(stub)
    assert calls.count("up --detach --no-deps litellm") == 3
    assert calls.count("up --detach --no-deps nginx") == 3


def test_a_dead_engine_still_respawns_through_the_shared_claim(stub: Path) -> None:
    _set_state(stub, "vllm", "exited")
    r = _run_monitor(stub)
    assert r.returncode == 125, r.stdout[-2000:]
    assert _calls(stub).count("up --detach --no-deps vllm") == 3
    assert "Respawn budget exhausted" in r.stdout
