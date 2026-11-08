"""The server's SSH call to a user's gate (plan §4 B1.5a wire contract).

The argv is the contract's, exactly: the person's transport key only
(``IdentitiesOnly``), no control master to share (``ControlPath=none``), no
``remote_env`` prefix and no agent. Answers parse into ``GateResult`` /
``GateRefusal`` by the gate's own names. The last tests talk to the real
gate file (``r4_contracts.hpc_gate``) as sshd would run it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from umrs.hpc_accounts import gate_client as gc

_KEY = "/keys/bdrt-front/alice/tk-0123456789ab"


def _site(**extra) -> SimpleNamespace:
    base = {"ssh_target": "login-0", "ssh_port": 22, "ssh_user": "operator"}
    return SimpleNamespace(name="bdrt-front", extra={**base, **extra})


_FIXED = [
    "-o", "IdentitiesOnly=yes",
    "-o", "IdentityAgent=none",
    "-o", "ForwardAgent=no",
    "-o", "ForwardX11=no",
    "-o", "ClearAllForwardings=yes",
    "-o", "PermitLocalCommand=no",
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ConnectTimeout=10",
]  # fmt: skip


def test_the_argv_is_the_contracts_exactly() -> None:
    resolved = {"hostname": "192.0.2.5", "userknownhostsfile": "/root/.ssh/known_hosts"}
    assert gc.gate_argv(_site(), "alice1", _KEY, "hello", resolved) == [
        "ssh", "-F", "none", "-i", _KEY, *_FIXED,
        "-o", "HostName=192.0.2.5",
        "-o", "UserKnownHostsFile=/root/.ssh/known_hosts",
        "-p", "22", "alice1@login-0", "hello",
    ]  # fmt: skip


def test_a_proxy_jump_and_port_come_from_the_site() -> None:
    argv = gc.gate_argv(
        _site(ssh_proxy_jump="bastion.example.org", ssh_port=2222), "a", _KEY, "probe"
    )
    hop = "ProxyCommand=ssh -W '[%h]:%p' bastion.example.org"
    assert argv[argv.index(hop) - 1] == "-o" and "-J" not in argv
    assert argv.index(hop) < argv.index("-p") < argv.index("a@login-0")
    assert argv[argv.index("-p") + 1] == "2222"


def test_the_alias_jump_and_port_apply_when_the_site_names_none() -> None:
    site = SimpleNamespace(name="s", extra={"ssh_target": "login-0"})
    resolved = {"proxyjump": "gw1,gw2", "port": "2200", "hostkeyalias": "none"}
    argv = gc.gate_argv(site, "a", _KEY, "hello", resolved)
    assert "ProxyCommand=ssh -J gw1 -W '[%h]:%p' gw2" in argv
    assert argv[argv.index("-p") + 1] == "2200" and not any("HostKeyAlias" in a for a in argv)


@pytest.mark.parametrize("jump", ["-oProxyCommand=x", "a;id", "a b", "$(id)", "[::1]"])
def test_a_jump_a_shell_could_read_is_refused(jump) -> None:
    with pytest.raises(ValueError, match="jump"):
        gc.gate_argv(_site(ssh_proxy_jump=jump), "a", _KEY, "hello")


def test_resolved_values_with_shell_or_quote_syntax_are_dropped() -> None:
    resolved = {"hostname": "h;id", "userknownhostsfile": "/k\"x", "stricthostkeychecking": "yes"}
    argv = gc.gate_argv(_site(), "a", _KEY, "hello", resolved)
    assert "StrictHostKeyChecking=yes" in argv
    assert not any(a.startswith(("HostName=", "UserKnownHostsFile=")) for a in argv)


def test_the_platform_config_identities_never_reach_the_call(tmp_path) -> None:
    """``ssh -G`` on the produced argv: only the transport key is offered,
    whatever IdentityFile the platform's config holds for the alias."""
    import shutil
    import subprocess

    if shutil.which("ssh") is None or shutil.which("ssh-keygen") is None:
        pytest.skip("needs OpenSSH")
    key = tmp_path / "tk-0123456789ab"
    keygen = ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)]
    subprocess.run(keygen, check=True, timeout=30)
    resolved = {"hostname": "192.0.2.5", "proxyjump": "bastion"}
    argv = gc.gate_argv(_site(), "alice", key, "probe", resolved)
    dump = subprocess.run(
        [argv[0], "-G", *argv[1:]], capture_output=True, text=True, timeout=30, check=True
    ).stdout.splitlines()
    assert [line for line in dump if line.startswith("identityfile ")] == [f"identityfile {key}"]
    assert {"identitiesonly yes", "identityagent none", "forwardagent no"} <= set(dump)
    assert "hostname 192.0.2.5" in dump and "user alice" in dump


def test_resolve_target_reads_ssh_g_and_keeps_only_what_the_call_takes() -> None:
    seen: dict = {}

    async def runner(argv, stdin, timeout):
        seen["argv"] = argv
        dump = (
            "user operator\nhostname 192.0.2.5\nport 2222\nidentityfile /root/.ssh/k\n"
            "proxyjump bastion\nuserknownhostsfile /root/.ssh/kh\nsendenv LC_*\n"
        )
        return 0, dump.encode(), b""

    found = asyncio.run(gc.resolve_target("nimbus-0", runner=runner))
    assert seen["argv"] == ["ssh", "-G", "--", "nimbus-0"]
    assert found == {
        "hostname": "192.0.2.5",
        "port": "2222",
        "proxyjump": "bastion",
        "userknownhostsfile": "/root/.ssh/kh",
    }
    with pytest.raises(ValueError):
        asyncio.run(gc.resolve_target("-oProxyCommand=x", runner=runner))


def test_an_unresolvable_target_is_unreachable_and_never_called() -> None:
    calls: list = []

    async def runner(argv, stdin, timeout):
        calls.append(argv)
        return 0, b"{}", b""

    async def resolver(_target):
        raise OSError("ssh not found")

    answer = asyncio.run(
        gc.run_verb(_site(), "a", _KEY, "hello", runner=runner, resolver=resolver)
    )
    assert (answer.error, answer.exit_code, calls) == (gc.UNREACHABLE, gc.SSH_FAILED, [])


def test_no_remote_env_no_platform_login_no_shell() -> None:
    site = _site(remote_env="source /opt/venue.sh &&", ssh_control_path="/run/umrs-ssh/x-%p")
    argv = gc.gate_argv(site, "alice1", _KEY, "hello")
    joined = " ".join(argv)
    assert "venue.sh" not in joined and "/run/umrs-ssh" not in joined
    assert "operator" not in joined  # never the site's platform login
    assert argv[-1] == "hello" and argv[-2] == "alice1@login-0"


@pytest.mark.parametrize(
    ("login", "verb"),
    [
        ("-oProxyCommand=evil", "hello"),
        ("alice 1", "hello"),
        ("", "hello"),
        ("alice", "hello; id"),
        ("alice", "Hello"),
        ("alice", ""),
    ],
)
def test_logins_and_verbs_that_could_become_options_or_commands_are_refused(login, verb) -> None:
    with pytest.raises(ValueError):
        gc.gate_argv(_site(), login, _KEY, verb)


def test_a_site_without_a_target_is_refused() -> None:
    with pytest.raises(ValueError, match="ssh_target"):
        gc.gate_argv(SimpleNamespace(extra={}), "alice", _KEY, "hello")


def test_the_body_goes_to_stdin_as_one_compact_json_object() -> None:
    seen: dict = {}

    async def runner(argv, stdin, timeout):
        seen.update(argv=argv, stdin=stdin, timeout=timeout)
        return 0, b'{"ok": true}', b""

    body = {"signature": "sig", "payload": "cGF5"}
    result = asyncio.run(gc.run_verb(_site(), "alice", _KEY, "revoke", body, runner=runner))
    assert result == gc.GateResult("revoke", {"ok": True})
    assert seen["stdin"] == b'{"payload":"cGF5","signature":"sig"}'
    assert seen["argv"][-1] == "revoke"
    assert asyncio.run(gc.run_verb(_site(), "a", _KEY, "hello", runner=runner)).verb == "hello"
    assert gc.encode_body(None) == b""


def test_the_body_uses_the_contracts_canonical_encoding() -> None:
    with pytest.raises(ValueError):
        gc.encode_body({"x": float("nan")})
    with pytest.raises(ValueError):
        gc.encode_body({"x": 1.5})
    assert gc.encode_body({"b": "é", "a": 1}) == '{"a":1,"b":"é"}'.encode()


@pytest.mark.parametrize(
    ("code", "out", "err", "want"),
    [
        (0, b'{"login": "a"}', b"", gc.GateResult("v", {"login": "a"})),
        (
            4,
            b"",
            b'Warning: Permanently added host\n{"error": "gate_probe_refusal", "detail": "d"}\n',
            gc.GateRefusal("v", "gate_probe_refusal", "d", 4),
        ),
        (
            3,
            b"",
            b'{"error": "absent", "detail": "x"}',
            gc.GateRefusal("v", "absent", '{"error": "absent", "detail": "x"}', 3),
        ),
        (
            1,
            b"",
            b'{"error": "gate_probe_refusal", "detail": "d"}',
            gc.GateRefusal(
                "v", "failed", '{"error": "gate_probe_refusal", "detail": "d"}', 1
            ),
        ),
        (
            255,
            b"",
            b"ssh: connect refused",
            gc.GateRefusal("v", "unreachable", "ssh: connect refused", 255),
        ),
        (1, b"", b"Traceback ...", gc.GateRefusal("v", "failed", "Traceback ...", 1)),
        (7, b"", b"", gc.GateRefusal("v", "bad_answer", "", 7)),
        (
            0,
            b"not json",
            b"",
            gc.GateRefusal("v", "bad_answer", "the gate's answer is not one JSON object", 0),
        ),
        (
            0,
            b"[1, 2]",
            b"",
            gc.GateRefusal("v", "bad_answer", "the gate's answer is not one JSON object", 0),
        ),
    ],
)
def test_answers_parse_by_the_gates_names(code, out, err, want) -> None:
    assert gc.parse_answer("v", code, out, err) == want


def test_an_oversized_answer_is_refused() -> None:
    big = json.dumps({"x": "y" * gc.MAX_ANSWER_BYTES}).encode()
    assert gc.parse_answer("hello", 0, big, b"").error == gc.BAD_ANSWER


def test_the_child_sees_path_home_and_lang_only(monkeypatch) -> None:
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    monkeypatch.setenv("UMRS_MARKER", "secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    env = gc._child_env()
    assert set(env) <= {"PATH", "HOME", "LANG"} and env["LANG"] == "C.UTF-8"


def test_the_subprocess_runner_runs_the_argv_without_a_shell(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    script = "import os,sys; print(os.environ.get('SSH_AUTH_SOCK','none')); sys.stdout.write(sys.stdin.read())"
    code, out, _err = asyncio.run(gc.subprocess_runner([sys.executable, "-c", script], b"in", 10))
    assert (code, out) == (0, b"none\nin")


def test_a_hung_call_times_out_as_unreachable() -> None:
    argv = [sys.executable, "-c", "import time; time.sleep(30)"]
    code, _out, err = asyncio.run(gc.subprocess_runner(argv, b"", 0.2))
    assert code == gc.SSH_FAILED and b"timed out" in err


# ── against the real gate file, as sshd's forced command runs it ─────────


def _gate_runner(home: Path):
    hpc_gate = pytest.importorskip("r4_contracts.hpc_gate")
    gate_file = Path(hpc_gate.__file__)

    async def runner(argv, stdin, timeout):
        env = {
            "HOME": str(home),
            "PATH": os.environ.get("PATH", ""),
            "SSH_ORIGINAL_COMMAND": argv[-1],
        }
        cmd = [sys.executable, "-I", str(gate_file), "--site", "bdrt-front"]
        return await _run(cmd, env, stdin)

    return runner


async def _run(cmd, env, stdin):
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    out, err = await proc.communicate(stdin)
    return proc.returncode, out, err


def test_the_real_gate_refuses_the_probe_by_name(tmp_path) -> None:
    runner = _gate_runner(tmp_path)
    answer = asyncio.run(gc.run_verb(_site(), "alice", _KEY, "probe", runner=runner))
    assert isinstance(answer, gc.GateRefusal)
    assert (answer.error, answer.exit_code) == (gc.GATE_PROBE_REFUSAL, gc.EXIT_REFUSED)


def test_the_real_gate_refuses_an_unknown_verb_by_name(tmp_path) -> None:
    runner = _gate_runner(tmp_path)
    answer = asyncio.run(gc.run_verb(_site(), "alice", _KEY, "shell", runner=runner))
    assert isinstance(answer, gc.GateRefusal) and answer.error == "unknown_verb"
