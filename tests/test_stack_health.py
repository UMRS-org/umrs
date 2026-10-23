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

# State files under STUB_DIR, per container name:
#   state.<ctr>     its status (running/exited ...); absent = no such container
#   requires.<ctr>  the containers it --requires (compose depends_on)
#   exitcode.<ctr>  its exit code (default 1)     logs.<ctr>  its log
# Like real podman, rm refuses a container another container requires.
_PODMAN = r"""#!/bin/bash
echo "podman $*" >> "$STUB_DIR/calls.log"
state_of() { cat "$STUB_DIR/state.$1" 2>/dev/null; }
last="${@: -1}"
inspect() {
  [ -f "$STUB_DIR/state.$1" ] || exit 125
  case "$2" in
    *ExitCode*) cat "$STUB_DIR/exitcode.$1" 2>/dev/null || echo 1 ;;
    *.Id*) echo "id-$1" ;;
    *Dependencies*) printf '['; for d in $(cat "$STUB_DIR/requires.$1" 2>/dev/null); do printf 'id-%s ' "$d"; done; echo ']' ;;
    *) state_of "$1" ;;
  esac
  exit 0
}
remove() {
  local f
  for f in "$STUB_DIR"/requires.*; do
    [ -f "$f" ] || continue
    if grep -qx "$1" "$f"; then
      echo "Error: container $1 has dependent containers which must be removed before it: ${f##*/requires.}" >&2
      exit 125
    fi
  done
  rm -f "$STUB_DIR/state.$1" "$STUB_DIR/requires.$1" "$STUB_DIR/exitcode.$1" "$STUB_DIR/logs.$1"
  echo "$1"; exit 0
}
case "$1 $2" in
  "ps -q") ls "$STUB_DIR" | grep -c '^state\.' ; exit 0 ;;
  "ps -a") case "$*" in *name=*) ls "$STUB_DIR" | sed -n 's/^state\.//p' ;; esac; exit 0 ;;
  "container exists") [ -f "$STUB_DIR/state.$3" ]; exit $? ;;
  "container inspect") inspect "$3" "$5" ;;
  "logs --tail") cat "$STUB_DIR/logs.$last" 2>/dev/null; exit 0 ;;
  "rm -f") remove "$last" ;;
esac
exit 0
"""

# Recreate = the container comes back in the next state recreate.<svc> lists
# (one line per recreate), else STUB_RECREATE_STATE; with STUB_RECREATE_EXIT
# as its exit code. A name still in use fails, like podman create does; nginx
# comes back requiring litellm and rag requiring vllm (depends_on).
_COMPOSE = r"""#!/bin/bash
echo "podman-compose $*" >> "$STUB_DIR/calls.log"
svc="${@: -1}"
ctr="rivault-77_${svc}_1"
if [ -f "$STUB_DIR/state.$ctr" ]; then
  echo "Error: the container name \"$ctr\" is already in use" >&2
  exit 125
fi
state="${STUB_RECREATE_STATE:-exited}"
queue="$STUB_DIR/recreate.$svc"
if [ -s "$queue" ]; then
  state="$(head -n 1 "$queue")"
  sed -i 1d "$queue"
fi
echo "$state" > "$STUB_DIR/state.$ctr"
echo "${STUB_RECREATE_EXIT:-1}" > "$STUB_DIR/exitcode.$ctr"
[ "$svc" = nginx ] && echo "rivault-77_litellm_1" > "$STUB_DIR/requires.$ctr"
[ "$svc" = rag ] && echo "rivault-77_vllm_1" > "$STUB_DIR/requires.$ctr"
exit 0
"""

_CURL = r"""#!/bin/bash
echo "curl $*" >> "$STUB_DIR/calls.log"
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
    # compose created nginx with --requires on litellm (depends_on)
    (tmp_path / f"requires.{PROJECT}_nginx_1").write_text(f"{PROJECT}_litellm_1\n")
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
    """Run ``script`` with the library; the front is probed on every tick."""
    extra.setdefault("RIVAULT_FRONT_PROBE_EVERY_TICKS", "1")
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


def _write(stub: Path, kind: str, svc: str, text: str) -> None:
    (stub / f"{kind}.{PROJECT}_{svc}_1").write_text(text + "\n")


def _queue_recreates(stub: Path, svc: str, *states: str) -> None:
    (stub / f"recreate.{svc}").write_text("".join(f"{state}\n" for state in states))


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


def test_an_unplanned_exit_is_not_marked_planned(stub: Path) -> None:
    _set_state(stub, "litellm", "exited")
    r = _lib(
        stub, f'stack_health_init; stack_health_tick {PROJECT} 4000; echo "p=$STACK_HEALTH_PLANNED"'
    )
    assert "p=0" in r.stdout, r.stderr


@pytest.mark.parametrize(
    ("exit_code", "log", "planned"),
    [
        ("0", "", True),
        ("1", "INFO: Maximum request limit of 10000 exceeded. Terminating process.", True),
        ("1", "ERROR: DATABASE_URL is empty", False),
        ("137", "", False),
    ],
)
def test_a_planned_exit_is_exit_0_or_the_max_request_limit(
    stub: Path, exit_code: str, log: str, planned: bool
) -> None:
    """Review F1: uvicorn ends a one-worker LiteLLM after N requests on plan."""
    _set_state(stub, "litellm", "exited")
    _write(stub, "exitcode", "litellm", exit_code)
    _write(stub, "logs", "litellm", log)
    r = _lib(
        stub, f'stack_health_init; stack_health_tick {PROJECT} 4000; echo "p=$STACK_HEALTH_PLANNED"'
    )
    assert f"p={int(planned)}" in r.stdout, r.stderr


def test_planned_restarts_are_free_only_while_the_window_has_room(stub: Path) -> None:
    script = (
        "stack_health_init; STACK_HEALTH_PLANNED=1; for i in 1 2 3 4; do "
        "stack_restart_is_free && echo -n F || echo -n C; done; "
        "STACK_HEALTH_PLANNED=0; stack_restart_is_free && echo -n F || echo -n C"
    )
    r = _lib(stub, script, RIVAULT_SIDECAR_PLANNED_RESTARTS="2")
    assert r.stdout == "FFCCC", r.stderr


@pytest.mark.parametrize(
    ("respawn", "vision", "order"),
    [
        ("litellm", False, "nginx litellm"),
        ("litellm nginx", False, "nginx litellm"),
        ("nginx", False, "nginx"),
        ("litellm", True, "bentoml-vision nginx litellm"),
        ("litellm nginx", True, "bentoml-vision nginx litellm"),
    ],
)
def test_dependents_are_removed_before_what_they_require(
    stub: Path, respawn: str, vision: bool, order: str
) -> None:
    """Review F2: podman refuses rm of a container another one --requires."""
    if vision:  # bentoml-vision depends_on nginx
        _set_state(stub, "bentoml-vision", "running")
        _write(stub, "requires", "bentoml-vision", f"{PROJECT}_nginx_1")
    r = _lib(stub, f'stack_removal_order {PROJECT} "{respawn}"')
    assert r.stdout == order, r.stderr


def test_a_respawn_keeps_the_answered_once_state_and_uses_a_short_grace(stub: Path) -> None:
    """Review F6: a respawn restarted the 900 s start grace, so a front that
    stayed down after it was never respawned again inside that grace."""
    script = (
        f"stack_health_init; stack_health_tick {PROJECT} 4000; "
        "export STUB_HTTP_CODE=502; for i in 1 2 3 4 5 6; do "
        f'stack_health_tick {PROJECT} 4000 && echo -n "respawn@$i "; done'
    )
    r = _lib(stub, script, RIVAULT_FRONT_FAIL_STREAK="3", RIVAULT_FRONT_RESPAWN_GRACE_SEC="0")
    assert r.stdout.strip() == "respawn@3 respawn@6", r.stderr


def test_after_a_respawn_failures_inside_the_respawn_grace_do_not_count(stub: Path) -> None:
    script = (
        f"stack_health_init; stack_health_tick {PROJECT} 4000; "
        "export STUB_HTTP_CODE=502; for i in 1 2 3 4 5 6; do "
        f'stack_health_tick {PROJECT} 4000 && echo -n "respawn@$i "; done'
    )
    r = _lib(stub, script, RIVAULT_FRONT_FAIL_STREAK="3", RIVAULT_FRONT_RESPAWN_GRACE_SEC="600")
    assert r.stdout.strip() == "respawn@3", r.stderr


def test_a_respawn_before_the_front_answered_keeps_the_start_grace(stub: Path) -> None:
    """Re-review N2: a LiteLLM respawned during its cold start got the 120 s
    respawn grace instead of the rest of the 900 s start grace, so a slow
    start could be charged again and again until exit 125."""
    _set_state(stub, "litellm", "exited")
    script = (
        f"stack_health_init; stack_health_tick {PROJECT} 4000; "
        f"echo running > $STUB_DIR/state.{PROJECT}_litellm_1; "
        "for i in 1 2 3 4 5; do "
        f'stack_health_tick {PROJECT} 4000 && echo -n "respawn@$i "; done; echo done'
    )
    r = _lib(
        stub,
        script,
        STUB_HTTP_CODE="502",
        RIVAULT_FRONT_FAIL_STREAK="1",
        RIVAULT_FRONT_RESPAWN_GRACE_SEC="0",
    )
    assert r.stdout.strip() == "done", r.stderr


def test_a_respawn_grace_longer_than_the_start_grace_still_applies(stub: Path) -> None:
    script = (
        'stack_health_init; stack_health_respawned; echo "$(( _SH_GRACE_UNTIL - $(date +%s) ))"'
    )
    r = _lib(
        stub,
        script,
        RIVAULT_FRONT_STARTUP_GRACE_SEC="10",
        RIVAULT_FRONT_RESPAWN_GRACE_SEC="600",
    )
    assert 595 <= int(r.stdout.strip()) <= 600, r.stderr


def test_the_front_is_probed_every_nth_tick(stub: Path) -> None:
    script = f"stack_health_init; for i in $(seq 1 24); do stack_health_tick {PROJECT} 4000; done"
    r = _lib(stub, script, RIVAULT_FRONT_PROBE_EVERY_TICKS="12")
    assert _calls(stub).count("curl ") == 2, r.stderr


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
        RIVAULT_FRONT_PROBE_EVERY_TICKS="1",
        RIVAULT_FRONT_STARTUP_GRACE_SEC="0",
        RIVAULT_FRONT_RESPAWN_GRACE_SEC="0",
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


def test_a_respawn_removes_nginx_first_and_recreates_it_last(stub: Path) -> None:
    """Review F2: with nginx --requires litellm, ``podman rm -f litellm`` failed
    (hidden by ``|| true``), the recreate hit the name in use, and LiteLLM
    never came back while the budget was spent."""
    _set_state(stub, "litellm", "exited")
    r = _run_monitor(stub, STUB_RECREATE_STATE="running")
    assert r.returncode == 124, r.stdout[-2000:]
    calls = [line for line in _calls(stub).splitlines() if "rm -f" in line or " up " in line]
    assert calls[:4] == [
        f"podman rm -f {PROJECT}_nginx_1",
        f"podman rm -f {PROJECT}_litellm_1",
        calls[2],
        calls[3],
    ]
    assert calls[2].endswith("up --detach --no-deps litellm")
    assert calls[3].endswith("up --detach --no-deps nginx")
    assert "could not remove" not in r.stdout


def test_a_litellm_that_exits_on_plan_four_times_keeps_the_job(stub: Path) -> None:
    """Review F1: uvicorn's max-requests exit (exit 0) used to be charged to
    the never-refilling job budget; the fourth one tore a healthy endpoint
    down with exit 125."""
    _set_state(stub, "litellm", "exited")
    _write(stub, "exitcode", "litellm", "0")
    _queue_recreates(stub, "litellm", "exited", "exited", "exited", "running")
    r = _run_monitor(stub, STUB_RECREATE_STATE="running", STUB_RECREATE_EXIT="0")
    assert r.returncode == 124, r.stdout[-2000:]
    assert _calls(stub).count("up --detach --no-deps litellm") == 4
    assert r.stdout.count("not charged to the respawn budget") == 4
    assert not (stub / "info" / "rivault-job-77.info").read_text().count("budget_exhausted")


def test_a_sidecar_exiting_0_in_a_loop_still_ends_the_job(stub: Path) -> None:
    _set_state(stub, "litellm", "exited")
    _write(stub, "exitcode", "litellm", "0")
    r = _run_monitor(stub, STUB_RECREATE_STATE="exited", STUB_RECREATE_EXIT="0")
    assert r.returncode == 125, r.stdout[-2000:]
    assert r.stdout.count("not charged to the respawn budget") == 6
    assert "Respawn budget exhausted" in r.stdout


def test_the_monitor_probes_the_front_about_once_a_minute(stub: Path) -> None:
    """Review F1: a probe every 5 s tick was 17,280 requests a day."""
    r = _run_monitor(stub)  # 300 s at 5 s ticks
    assert r.returncode == 124, r.stdout[-2000:]
    assert _calls(stub).count("curl ") == 5


def _removals_and_recreates(stub: Path) -> list[str]:
    """``rm <container>`` / ``up <service>`` for each podman rm / compose up."""
    steps = []
    for line in _calls(stub).splitlines():
        if line.startswith("podman rm -f "):
            steps.append("rm " + line.rsplit(" ", 1)[1])
        elif " up --detach --no-deps " in line:
            steps.append("up " + line.rsplit(" ", 1)[1])
    return steps


def test_a_dead_engine_with_rag_on_removes_rag_first_and_recreates_it_last(stub: Path) -> None:
    """Round-1 open item: rag --requires vllm, so ``podman rm -f`` of the
    engine alone was refused (hidden by ``|| true``), the recreate hit the name
    in use, and every engine respawn failed while it spent the budget."""
    _set_state(stub, "vllm", "exited")
    _set_state(stub, "rag", "running")
    _write(stub, "requires", "rag", f"{PROJECT}_vllm_1")
    r = _run_monitor(stub, STUB_RECREATE_STATE="running")
    assert r.returncode == 124, r.stdout[-2000:]
    assert _removals_and_recreates(stub) == [
        f"rm {PROJECT}_rag_1",
        f"rm {PROJECT}_vllm_1",
        "up vllm",
        "up rag",
    ]
    assert "could not remove" not in r.stdout
    assert "Engine respawn OK" in r.stdout


def test_a_dead_engine_without_rag_recreates_only_the_engine(stub: Path) -> None:
    _set_state(stub, "vllm", "exited")
    r = _run_monitor(stub, STUB_RECREATE_STATE="running")
    assert r.returncode == 124, r.stdout[-2000:]
    assert _removals_and_recreates(stub) == [f"rm {PROJECT}_vllm_1", "up vllm"]


def _function_text(name: str) -> str:
    text = SERVE_MODEL_SLURM.read_text()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 3]


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_the_recreate_quotes_its_override_arrays(tmp_path: Path) -> None:
    """SC2068: an unquoted ${GPUoverride[@]} re-splits its elements."""
    script = tmp_path / "recreate.sh"
    script.write_text("#!/bin/bash\n" + _function_text("_compose_recreate_service"))
    r = subprocess.run(
        ["shellcheck", "-S", "warning", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "SC2068" not in r.stdout, r.stdout


def test_the_recreate_passes_the_override_files_and_profiles(stub: Path) -> None:
    body = _function_text("_compose_recreate_service")
    script = (
        f"RIVAULT_COMPOSE_PROJECT={PROJECT}; uuid=u77; "
        "GPUoverride=(--file ./x_nvidia-override.yaml); DbOverride=(); "
        '_profile_flags=" --profile harness --profile vllm"; '
        f"{body}\n_compose_recreate_service postgres"
    )
    r = subprocess.run(
        ["bash", "-c", script], env=_env(stub), capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr
    assert _calls(stub).strip() == (
        f"podman-compose --project-name {PROJECT} --in-pod false --env-file ./env.u77 "
        "--file ./serve-model.yaml --file ./x_nvidia-override.yaml "
        "--profile harness --profile vllm up --detach --no-deps postgres"
    )
