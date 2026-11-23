"""SH-H0: the inline ``HostRuntime`` (ADR-098 D-2/D-3, decision SH-D3).

Two halves:

* **Characterization** (written against the pre-extraction
  ``FrontendPodmanProvisioner`` and kept byte-identical after it): every
  podman argv and stdin, every rendered file and mode, every broker call,
  audit event and log message, for the happy paths and for a failure at each
  step. The digests below were captured on the old code (umrs ``6ffdb3c3``);
  a refactor that changes any of it fails here.
* **Core contract**: ``umrs.hosts`` owns pod and egress provisioning, a
  worker spec is the only thing an extension adds, and core never imports
  the r4 extension (ADR-117 D-3).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from umrs_r4.venues.frontend_podman import FrontendPodmanProvisioner, ProvisionConfig

HOSTS_PKG = Path(__file__).resolve().parents[1] / "src" / "umrs" / "hosts"


# ── recording fakes (deterministic: no random lease ids, no clock) ──────────


@dataclass
class _Lease:
    lease_id: str
    expires_at: Any = None


class RecordingBroker:
    def __init__(self, bind_ok: bool = True) -> None:
        self.calls: list[list[Any]] = []
        self.bind_ok = bind_ok

    def issue_lease(self, *args: Any, **kwargs: Any) -> tuple[_Lease, str]:
        self.calls.append(["issue_lease", list(args), _jsonable(kwargs)])
        return _Lease(lease_id=f"lease-{kwargs.get('session_id')}"), "tok-PLAINTEXT"

    def bind_egress_ip(self, lease_id: str, ip: str, actor: str = "") -> bool:
        self.calls.append(["bind_egress_ip", lease_id, ip, actor])
        return self.bind_ok

    def revoke_by_session(self, sid: str, actor: str = "") -> int:
        self.calls.append(["revoke_by_session", sid, actor])
        return 1


class ScriptedPodman:
    """Records argv + stdin; ``fail`` = argv prefix (or word) that fails."""

    def __init__(self, *, fail: tuple[str, ...] = (), controllers: str = "[cpu io memory pids]",
                 info_rc: int = 0, inspect_ip: str = "10.88.0.7", raise_on: str = "") -> None:
        self.calls: list[list[Any]] = []
        self.fail = fail
        self.controllers = controllers
        self.info_rc = info_rc
        self.inspect_ip = inspect_ip
        self.raise_on = raise_on

    def __call__(self, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess:
        self.calls.append([list(args), input_text])
        if self.raise_on and self.raise_on in args:
            raise OSError("podman vanished")
        if self._fails(args):
            return subprocess.CompletedProcess(args, 1, "", f"boom {' '.join(args)}")
        return subprocess.CompletedProcess(args, 0, self._stdout(args), "")

    def _fails(self, args: tuple[str, ...]) -> bool:
        if not self.fail:
            return False
        return tuple(args[: len(self.fail)]) == self.fail or (
            len(self.fail) == 1 and self.fail[0] in args
        )

    def _stdout(self, args: tuple[str, ...]) -> str:
        if args[0] == "info":
            return self.controllers + "\n" if self.info_rc == 0 else ""
        if args[:2] == ("pod", "inspect"):
            return "infra-1\n"
        if args[0] == "inspect":
            nets = {"egress-net": {"IPAddress": self.inspect_ip}} if self.inspect_ip else {}
            return json.dumps([{"NetworkSettings": {"Networks": nets}}])
        return ""


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _cfg(**over: Any) -> ProvisionConfig:
    base: dict[str, Any] = {
        "session_id": "s1",
        "owner": "alice",
        "source_ids": ("src-1", "src-2"),
        "default_source_id": "src-1",
        "subnet": "172.18.52.0/24",
        "infer_host": "umrs-infer",
        "worker_image": "r4-companion@sha256:abc",
        "nginx_image": "nginx@sha256:def",
        "squid_image": "squid@sha256:123",
        "gateway_host": "gw.example.org",
        "umrs_host": "umrs.example.org",
        "rail_upstream": "umrs-api:8000",
        "project": "proj-a",
    }
    base.update(over)
    return ProvisionConfig(**base)


def _files(root: Path) -> list[list[Any]]:
    if not root.exists():
        return []
    return [
        [str(p.relative_to(root)), oct(p.stat().st_mode & 0o777), p.read_text()]
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


@pytest.fixture
def audit_log(monkeypatch: pytest.MonkeyPatch) -> list[list[Any]]:
    events: list[list[Any]] = []

    def _emit(event: str, **kw: Any) -> None:
        events.append([event, _jsonable(kw)])

    import umrs.api.audit_events as audit_events

    monkeypatch.setattr(audit_events, "emit_audit", _emit)
    return events


def _snapshot(tmp_path: Path, caplog: Any, audit: list, cfg: ProvisionConfig,
              podman: ScriptedPodman, broker: RecordingBroker) -> str:
    root = tmp_path / "sessions"
    prov = FrontendPodmanProvisioner(broker, podman=podman, session_root=root)
    with caplog.at_level(logging.INFO):
        result = prov.provision(cfg)
    snap = {
        "result": [result.state, result.reason, result.lease_id, result.egress_ip, result.steps],
        "podman": podman.calls,
        "files": _files(root),
        "broker": broker.calls,
        "audit": audit,
        "logs": [[r.levelname, r.getMessage()] for r in caplog.records],
    }
    blob = json.dumps(snap, sort_keys=True, default=str).replace(str(tmp_path), "<TMP>")
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# Scenario → (config overrides, podman kwargs, broker kwargs, digest on 6ffdb3c3).
SCENARIOS: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]] = {
    "public-rail": ({}, {}, {}, "18409dbaec4910d7"),
    "allowlist-http-resolver-no-cpu": (
        {
            "web_policy": "allowlist",
            "web_allow_domains": ("Example.org", "*.pypi.org", "example.org"),
            "infer_uplink": "http",
            "infer_resolve_via": "10.89.0.1",
        },
        {"controllers": "[memory pids]"},
        {},
        "ef2fec4cb3fa6e42",
    ),
    "none-no-rail-probe-fails-custom-cmd": (
        {"web_policy": "none", "rail_upstream": "", "worker_command": ("venue", "plan-run"),
         "infer_host": "10.0.0.5", "infer_resolve_via": "10.89.0.1"},
        {"info_rc": 1},
        {},
        "f29f802451d77731",
    ),
    "fail-network": ({}, {"fail": ("network", "create")}, {}, "52794940b64432fe"),
    "fail-secret": ({}, {"fail": ("secret", "create")}, {}, "1c7f7e342073f646"),
    "fail-pod-create": ({}, {"fail": ("pod", "create")}, {}, "4577e3f8c5047cb9"),
    "fail-nginx": ({}, {"fail": ("nginx@sha256:def",)}, {}, "02e1a7b436c65706"),
    "fail-squid": ({}, {"fail": ("squid@sha256:123",)}, {}, "c50db0e3a83d8426"),
    "fail-pod-inspect": ({}, {"fail": ("pod", "inspect")}, {}, "b83873a1e29336e9"),
    "fail-ctr-inspect": ({}, {"fail": ("inspect", "--type")}, {}, "c7a1a8b5147e0d13"),
    "egress-ip-missing": ({}, {"inspect_ip": ""}, {}, "bda7e30f2f7e0e04"),
    "bind-refused": ({}, {}, {"bind_ok": False}, "de41f824bb818c59"),
    "fail-worker": ({}, {"fail": ("r4-companion@sha256:abc",)}, {}, "c02805694ff14730"),
    "teardown-step-raises": ({}, {"fail": ("r4-companion@sha256:abc",), "raise_on": "rm"}, {}, "7fed01a5d0ac6221"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_provisioning_is_byte_identical_to_the_pre_extraction_code(
    name: str, tmp_path: Path, caplog: Any, audit_log: list
) -> None:
    over, podman_kw, broker_kw, digest = SCENARIOS[name]
    got = _snapshot(tmp_path, caplog, audit_log, _cfg(**over),
                    ScriptedPodman(**podman_kw), RecordingBroker(**broker_kw))
    assert got == digest


# ── core contract (SH-H0) ───────────────────────────────────────────────────


def _generic_spec(cfg: Any) -> Any:
    from umrs.hosts import WorkerSpec

    return WorkerSpec(image="busybox@sha256:1", command=("sleep", "1"), env=(("A", "b"),))


def _worker_argv(podman: ScriptedPodman, sid: str) -> list[str]:
    runs = [call[0] for call in podman.calls if call[0][0] == "run"]
    return next(argv for argv in runs if f"worker-{sid}" in argv)


def test_inline_runtime_runs_any_worker_spec_without_r4_names(tmp_path: Path) -> None:
    """Core owns the pod and egress; the spec only names image, command, env."""
    from umrs.hosts import InlineHostRuntime, SandboxConfig

    podman = ScriptedPodman()
    runtime = InlineHostRuntime(
        RecordingBroker(), _generic_spec, podman=podman, session_root=tmp_path
    )
    cfg = SandboxConfig(session_id="g1", owner="o", source_ids=("s",), default_source_id="s",
                        subnet="172.18.9.0/24", infer_host="umrs-infer", rail_upstream="api:8000")
    assert runtime.provision(cfg).state == "running"
    worker = _worker_argv(podman, "g1")
    assert worker[-5:] == ["--env", "A=b", "busybox@sha256:1", "sleep", "1"]
    assert [arg for arg in worker if "R4_" in arg] == []
    assert {"--read-only", "no-new-privileges"} <= set(worker)
    nginx = (tmp_path / "g1" / "nginx.conf").read_text()
    assert "/umrs/companion" not in nginx
    assert "listen 8081" not in nginx  # no rail_path → no rail listener


def test_r4_extension_contributes_only_the_worker_spec() -> None:
    from umrs.hosts import HostRuntime, InlineHostRuntime
    from umrs_r4.venues.frontend_podman import r4_worker_spec

    spec = r4_worker_spec(_cfg())
    assert spec.image == "r4-companion@sha256:abc" and spec.command == ("venue", "worker")
    assert spec.rail_path == "/umrs/companion/sessions/s1/rail/"
    assert {k for k, _ in spec.env} >= {"R4_VENUE_SESSION_ID", "R4_WEB_ACCESS"}
    prov = FrontendPodmanProvisioner(RecordingBroker(), podman=ScriptedPodman())
    assert isinstance(prov, InlineHostRuntime) and isinstance(prov, HostRuntime)


def test_admin_views_name_the_inline_host() -> None:
    from umrs.hosts import INLINE_HOST_ID, LAUNCH_MODES
    from umrs_r4.venues import frontend_service

    assert LAUNCH_MODES == ("inline",)
    assert frontend_service.SESSION_HOST == INLINE_HOST_ID == "local"
    assert frontend_service.admin_view({"runtime": {}})["host"] == "local"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    return {m.split(".")[0] for m in names}


def test_core_hosts_imports_nothing_from_the_r4_extension() -> None:
    """ADR-117 D-3, statically and at import time."""
    files = sorted(HOSTS_PKG.rglob("*.py"))
    assert {p.name for p in files} >= {"__init__.py", "runtime.py", "inline.py"}
    for path in files:
        assert not _import_roots(path) & {"umrs_r4", "r4", "r4_mcp"}, path
    code = "import sys, umrs.hosts; print(sorted(m for m in sys.modules if m.startswith('umrs_r4')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60, check=True)
    assert out.stdout.strip() == "[]"
