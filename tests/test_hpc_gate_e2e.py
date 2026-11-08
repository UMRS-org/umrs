"""End to end: the server's gate client against a real gate (ADR-112 wave B1).

Plan §4 B1.5a binds three lanes: the gate (``r4_contracts.hpc_gate``), the
server (``umrs.api.hpc_accounts`` + ``umrs.hpc_accounts.gate_client``) and
r4 (``r4.hpc_signing`` / ``r4.hpc_remote``). Each lane tests itself against
the contract; this file runs them together:

- a temp directory is the person's HPC home, and the gate is installed into
  it exactly as ``r4 hpc add`` installs it: r4's ``build_install`` from the
  server's real ``POST transport-key`` answer, then r4's fixed install script
  run by ``sh -c`` under that HOME;
- the server reaches the gate through its own ``gate_client`` with an
  injected runner that does what sshd does for a forced-command key: find
  the ``authorized_keys`` line carrying the offered key, ignore the
  requested command and run the line's ``command=`` (``/usr/bin/python3 -I
  <home>/.umrs/gate/umrs_gate.py --site <site>``) with HOME, PATH and
  ``SSH_ORIGINAL_COMMAND`` set; a key with no line is refused (exit 255);
- a fake ``sacctmgr`` on the gate's PATH lists the person's allocations;
- the signing key is a real ``ssh-keygen`` ed25519 key in ``tmp_path``
  (``allow_non_sk``), signing through r4's ``sign`` with its test hook for a
  passphrase-less key.

Nothing here reads ``~/.ssh`` or real configuration; every key is generated
at test time under ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pwd
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
hs = pytest.importorskip("r4.hpc_signing")
hpc_remote = pytest.importorskip("r4.hpc_remote")

from umrs.api import hpc_accounts as api  # noqa: E402
from umrs.hpc_accounts import gate_client, transport_keys  # noqa: E402
from umrs.hpc_accounts.store import InMemorySiteAccountStore  # noqa: E402
from umrs.orchestrator.site_config import (  # noqa: E402
    HpcAccessConfig,
    ServiceAccountAccess,
    SiteConfig,
    UserAccountAccess,
)

GATE_PYTHON = "/usr/bin/python3"
KEYGEN = "/usr/bin/ssh-keygen"
SITE = "e2e-site"
DOMAIN = "platform.example.org"
USER = "alice"
HEADERS = {"X-User-ID": USER, "X-Verified-Project-ID": "p1"}
LOGIN = pwd.getpwuid(os.geteuid()).pw_name
SACCTMGR_ROWS = "c1|grp-a|gpu|normal\nc1|grp-b||\n"
FAKE_SACCTMGR = f"""#!/bin/sh
printf '%s\\n' "$*" > "$0.args"
printf '{SACCTMGR_ROWS}'
"""
_COMMAND_OPTION = re.compile(r'(?:^|,)command="([^"]*)"')


def _gate_python_ok() -> bool:
    if not (os.access(GATE_PYTHON, os.X_OK) and os.access(KEYGEN, os.X_OK)):
        return False
    probe = [GATE_PYTHON, "-I", "-c", "import sys; sys.exit(sys.version_info < (3, 9))"]
    return subprocess.run(probe, check=False, timeout=30).returncode == 0


pytestmark = pytest.mark.skipif(
    not _gate_python_ok(), reason="needs /usr/bin/python3 >= 3.9 and /usr/bin/ssh-keygen"
)


class Sshd:
    """sshd for forced-command keys: the line's command runs, never the request."""

    def __init__(self, home: Path, bin_dir: Path) -> None:
        self.home = home
        self.bin_dir = bin_dir
        self.calls: list[list[str]] = []

    def forced_command(self, key_b64: str) -> str | None:
        keys = self.home / ".ssh" / "authorized_keys"
        for line in keys.read_text().splitlines():
            options, sep, rest = line.partition(" ssh-ed25519 ")
            if sep and rest.split(" ")[0] == key_b64:
                found = _COMMAND_OPTION.search(options)
                return found.group(1) if found else None
        return None

    def env(self, original_command: str) -> dict[str, str]:
        return {
            "HOME": str(self.home),
            "PATH": f"{self.bin_dir}:/usr/bin:/bin",
            "LC_ALL": "C",
            "SSH_ORIGINAL_COMMAND": original_command,
        }

    async def __call__(
        self, argv: list[str], stdin: bytes, timeout: float
    ) -> tuple[int, bytes, bytes]:
        self.calls.append(argv)
        key = Path(argv[argv.index("-i") + 1])
        command = self.forced_command(Path(f"{key}.pub").read_text().split()[1])
        if command is None:
            return 255, b"", b"Permission denied (publickey).\r\n"
        proc = subprocess.run(
            shlex.split(command),
            input=stdin,
            capture_output=True,
            env=self.env(argv[-1]),
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr


@dataclass
class World:
    client: TestClient
    store: InMemorySiteAccountStore
    sshd: Sshd
    site: SiteConfig
    home: Path
    signer: Path
    offer: Any

    def key_path(self, key_id: str = "") -> Path:
        key = transport_keys.load(SITE, USER, key_id or self.offer.key_id)
        assert key is not None
        return key.path

    def call(self, verb: str, body: dict[str, Any] | None = None, key: Path | None = None):
        run = gate_client.run_verb(
            self.site, LOGIN, key or self.key_path(), verb, body=body, runner=self.sshd
        )
        return asyncio.run(run)

    def verify(self):
        body = {
            "login": LOGIN,
            "signers": [hs.public_key_info(self.signer)[2]],
            "uv_required": False,
            "allow_non_sk": True,
        }
        return self.client.post("/umrs/sites/me/hpc/verify", json=body, headers=HEADERS)

    def link(self, slurm_account: str):
        body = {"project_id": "p1", "slurm_account": slurm_account}
        return self.client.post("/umrs/sites/me/hpc/links", json=body, headers=HEADERS)

    def authorized_keys(self) -> str:
        return (self.home / ".ssh" / "authorized_keys").read_text()


def _site() -> SiteConfig:
    site = SiteConfig(name=SITE)
    site.extra = {"ssh_target": "login.example.org", "ssh_port": 22, "ssh_user": "op"}
    site.scheduler.hpc_access = HpcAccessConfig(
        operator="Centre X",
        modes_allowed=["service_account", "user_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(login="op", personal=True),
        user_account=UserAccountAccess(methods=["gated_key"], signing_domain=DOMAIN),
    )
    return site


def _signing_key(tmp_path: Path) -> Path:
    key = tmp_path / "signing" / "id_umrs_ed25519"
    key.parent.mkdir(mode=0o700)
    argv = [KEYGEN, "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)]
    subprocess.run(argv, check=True, timeout=30)
    return key


def _fake_sacctmgr(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "sacctmgr"
    script.write_text(FAKE_SACCTMGR)
    script.chmod(0o755)
    return bin_dir


def _install_as_r4(home: Path, plan: Any) -> None:
    """r4's fixed install script, as the person's login shell would run it."""
    proc = subprocess.run(
        ["sh", "-c", hpc_remote.INSTALL_SCRIPT],
        input=hpc_remote.install_payload(plan),
        capture_output=True,
        env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0 and proc.stdout.strip() == b"installed", proc.stderr


def _app(store: InMemorySiteAccountStore, sshd: Sshd, site: SiteConfig, monkeypatch) -> Any:
    monkeypatch.setattr(api, "_load_site", lambda: site)
    monkeypatch.setattr(api, "_make_store", lambda _req, _venue: store)
    monkeypatch.setattr(api, "_gate_runner", sshd)
    monkeypatch.setattr(api, "_live_memberships", lambda _req, user: {USER: ["p1"]}.get(user))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    """The person's HPC home with the gate installed the way r4 installs it."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.setenv("UMRS_TRANSPORT_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.delenv("R4_COMPANION_SESSION", raising=False)
    monkeypatch.setattr(hs, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(hs, "_ALLOW_UNENCRYPTED_KEYS", True)
    home = tmp_path / "hpc-home"
    home.mkdir(mode=0o700)
    sshd = Sshd(home, _fake_sacctmgr(tmp_path))
    site, store = _site(), InMemorySiteAccountStore(SITE)
    client = _app(store, sshd, site, monkeypatch)
    answer = client.post("/umrs/sites/me/hpc/transport-key", json={"rotate": False}, headers=HEADERS)
    assert answer.status_code == 200, answer.text
    offer = hs.parse_offer(answer.json())
    signer = _signing_key(tmp_path)
    plan = hs.build_install(offer, principal=USER, home=str(home), signer=signer, kind="ed25519")
    _install_as_r4(home, plan)
    return World(client, store, sshd, site, home, signer, offer)


def _control(world: World, op: str, key_id: str) -> dict[str, str]:
    body = hs.control_body(op, site=SITE, principal=USER, key_id=key_id)
    return hs.signed_control(body, world.signer, DOMAIN)


def _second_transport_line(world: World) -> str:
    """A second transport key and its line (as during a rotation)."""
    key = transport_keys.generate(SITE, USER)
    b64 = key.public_key.split()[1]
    line = hpc_gate.authorized_keys_line(str(world.home), SITE, key.key_id, b64)
    with (world.home / ".ssh" / "authorized_keys").open("a") as handle:
        handle.write(line + "\n")
    return key.key_id


# ── the install is the contract's layout ─────────────────────────────────


def test_r4_installs_the_contract_layout_built_by_the_gates_helpers(world):
    gate_dir = world.home / ".umrs" / "gate"
    shipped = Path(hpc_gate.__file__).read_bytes()
    assert (gate_dir / "umrs_gate.py").read_bytes() == shipped
    assert json.loads((gate_dir / "gate.json").read_text()) == {
        "site": SITE,
        "domain": DOMAIN,
        "principal": USER,
        "uv_required": False,
        "allow_non_sk": True,
        "gate_version": hpc_gate.GATE_VERSION,
    }
    key_type, signer_b64, _fp = hs.public_key_info(world.signer)
    signers = hpc_gate.allowed_signers_line(USER, DOMAIN, key_type, signer_b64)
    assert (gate_dir / "allowed_signers").read_text() == signers + "\n"
    line = hpc_gate.authorized_keys_line(
        str(world.home), SITE, world.offer.key_id, world.offer.key_b64
    )
    assert world.authorized_keys() == line + "\n"
    assert line.startswith(f'restrict,command="{GATE_PYTHON} -I {world.home}/.umrs/gate/')
    assert line.endswith(f" umrs-transport-{SITE}-{world.offer.key_id}")
    for path in (gate_dir, gate_dir / "state" / "nonces", gate_dir / "log"):
        assert path.stat().st_mode & 0o777 == 0o700


# ── probe and hello over the server's gate client ────────────────────────


def test_probe_is_refused_by_the_gate_itself(world):
    answer = world.call("probe")
    assert isinstance(answer, gate_client.GateRefusal)
    assert (answer.error, answer.exit_code) == (gate_client.GATE_PROBE_REFUSAL, 4)


def test_any_other_command_over_the_transport_key_reaches_only_the_gate(world):
    answer = world.call("id")
    assert isinstance(answer, gate_client.GateRefusal)
    assert (answer.error, answer.exit_code) == ("unknown_verb", 4)


def test_hello_reports_login_sha_signers_assocs_and_log_head(world):
    answer = world.call("hello")
    assert isinstance(answer, gate_client.GateResult), answer
    data = answer.data
    assert (data["login"], data["site"], data["gate_version"]) == (LOGIN, SITE, "1")
    assert data["gate_sha256"] == hpc_gate.gate_sha256() == hs.gate_sha256()
    assert data["log_head"] == {"n": 0, "h": "0" * 64}
    assert data["signers"] == [
        {"fingerprint": hs.public_key_info(world.signer)[2], "type": "ssh-ed25519"}
    ]
    assert [row["account"] for row in data["assocs"]] == ["grp-a", "grp-b"]
    args = (world.sshd.bin_dir / "sacctmgr.args").read_text().split()
    assert args == ["-nP", "show", "assoc", f"user={LOGIN}", "format=cluster,account,partition,qos"]


# ── the verify route and links ───────────────────────────────────────────


def test_verify_route_activates_the_account_with_the_gates_report(world):
    response = world.verify()
    assert response.status_code == 200, response.text
    account = response.json()["account"]
    assert (account["status"], account["gate_probe"]) == ("active", "refused")
    assert (account["site_login"], account["verified_login"]) == (LOGIN, LOGIN)
    assert account["gate_sha256"] == hpc_gate.gate_sha256()
    assert account["transport_key_id"] == world.offer.key_id
    assert [row["account"] for row in account["verified_assocs"]] == ["grp-a", "grp-b"]


def test_verify_reaches_the_gate_with_the_contracts_ssh_call(world):
    assert world.verify().status_code == 200
    assert [argv[-1] for argv in world.sshd.calls] == ["probe", "hello"]
    first = world.sshd.calls[0]
    assert first[first.index("-i") + 1] == str(world.key_path())
    assert first[-2] == f"{LOGIN}@login.example.org"
    assert "IdentitiesOnly=yes" in first and "ControlPath=none" in first


def test_a_gate_that_is_not_this_servers_release_is_refused(world):
    gate_file = world.home / ".umrs" / "gate" / "umrs_gate.py"
    gate_file.write_bytes(gate_file.read_bytes() + b"# changed\n")
    response = world.verify()
    assert response.headers.get("x-platform-error-code") == "hpcaccess.gate_version"
    assert world.store.get_account(USER)["status"] == "pending"


def test_links_follow_the_allocations_the_site_listed(world):
    assert world.verify().status_code == 200
    listed = world.link("grp-a")
    assert listed.status_code == 200, listed.text
    assert listed.json()["link"]["slurm_account"] == "grp-a"
    unlisted = world.link("grp-not-mine")
    assert (unlisted.status_code, unlisted.headers.get("x-platform-error-code")) == (
        403,
        "hpcaccess.allocation_not_at_site",
    )


# ── signed controls ──────────────────────────────────────────────────────


def test_a_tampered_control_is_refused_and_changes_nothing(world):
    other = _second_transport_line(world)
    before = world.authorized_keys()
    envelope = _control(world, "revoke", other)
    body = json.loads(base64.b64decode(envelope["payload"]))
    body["key_id"] = world.offer.key_id
    envelope["payload"] = base64.b64encode(hs.canonical_bytes(body)).decode("ascii")
    answer = world.call("revoke", body=envelope)
    assert isinstance(answer, gate_client.GateRefusal)
    assert (answer.error, answer.exit_code) == ("bad_signature", 4)
    assert world.authorized_keys() == before


def test_a_signed_revoke_removes_its_line_and_a_replay_is_refused(world):
    other = _second_transport_line(world)
    envelope = _control(world, "revoke", other)
    done = world.call("revoke", body=envelope)
    assert isinstance(done, gate_client.GateResult), done
    assert (done.data["ok"], done.data["key_id"], done.data["lines"]) == (True, other, 1)
    assert f"umrs-transport-{SITE}-{other}" not in world.authorized_keys()
    assert f"umrs-transport-{SITE}-{world.offer.key_id}" in world.authorized_keys()
    gone = world.call("hello", key=world.key_path(other))
    assert (gone.error, gone.exit_code) == (gate_client.UNREACHABLE, 255)
    replay = world.call("revoke", body=envelope)
    assert isinstance(replay, gate_client.GateRefusal)
    assert (replay.error, replay.exit_code) == ("replayed", 4)


def test_revoking_the_calling_key_ends_the_servers_reach(world):
    assert world.call("revoke", body=_control(world, "revoke", world.offer.key_id)).data["ok"]
    assert world.authorized_keys() == ""
    after = world.call("hello")
    assert (after.error, after.exit_code) == (gate_client.UNREACHABLE, 255)
    log = world.home / ".umrs" / "gate" / "log" / "gate.jsonl"
    assert hpc_gate.verify_log(str(log)) and hpc_gate.read_log_head(str(log))["n"] == 1
