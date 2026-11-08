"""The server's SSH call to a user's gate (ADR-112 D-3/D-7; plan §4 B1.5a).

One call per verb, as the person's own login with the person's transport
key, and nothing else (plan §4 B1.5a, amended 2026-11-08):

    ssh -F none -i <key> -o IdentitiesOnly=yes -o IdentityAgent=none
        -o ForwardAgent=no -o ForwardX11=no -o ClearAllForwardings=yes
        -o PermitLocalCommand=no -o BatchMode=yes -o ControlMaster=no
        -o ControlPath=none -o ConnectTimeout=10 -o HostName=<host>
        [-o UserKnownHostsFile=.. -o GlobalKnownHostsFile=..
         -o HostKeyAlias=.. -o StrictHostKeyChecking=..]
        [-o ProxyCommand=<jump hop>] -p <port> <login>@<ssh_target> <verb>

- ``-F none``: the platform's ``ssh_config`` is never read for the call
  itself, so an ``IdentityFile`` there (``-i`` adds to it, it does not
  replace it) can never be offered to a user's login. ``ssh_target`` is
  resolved once with ``ssh -G`` (host name, port, known-hosts files, host
  key alias, jump): only those values reach the call;
- a jump host (``extra.ssh_proxy_jump``, else the alias's own) is a
  ``ProxyCommand`` hop, ``ssh -W [%h]:%p <jump>``: the platform's own hop to
  its bastion, which forwards TCP only; the gate login itself is
  authenticated with the transport key alone;
- no ``remote_env`` prefix, no agent, no forwarding, and a minimal child
  environment (``PATH``, ``HOME``, ``LANG``), so no other person's control
  master is ever reused and no session variable is sent;
- the request body, if any, is one JSON object on stdin;
- the gate answers one JSON object on stdout (exit 0), or refuses with
  ``{"error": "<name>", "detail": "<text>"}`` on stderr (exit 4).

Answers come back as ``GateResult`` or ``GateRefusal``. Refusal names are
the gate's own, honoured only on the gate's refusal exit (4); the client
adds names for failures the gate never saw: ``unreachable`` (ssh exit 255,
a timeout, or a target that cannot be resolved), ``bad_answer`` (output that
is not the contract's JSON) and the exit-code names ``failed`` / ``usage`` /
``absent`` / ``scheduler``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_PROBE_REFUSAL = "gate_probe_refusal"
EXIT_REFUSED = 4
SSH_FAILED = 255
MAX_ANSWER_BYTES = 1 << 20  # the gate's own body cap
DEFAULT_TIMEOUT_S = 20.0
CONNECT_TIMEOUT_S = 10
RESOLVE_TIMEOUT_S = 10.0

UNREACHABLE = "unreachable"
BAD_ANSWER = "bad_answer"
_EXIT_NAMES = {
    1: "failed",
    2: "usage",
    3: "absent",
    5: "scheduler",
    SSH_FAILED: UNREACHABLE,
}

# A POSIX-ish login (never an option: no leading dash) and a bare verb word.
_LOGIN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_VERB = re.compile(r"^[a-z][a-z_]{0,31}$")
# A host alias or jump hop: no leading dash, nothing a shell would expand.
_HOST = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@:-]{0,254}$")
# A value taken from ``ssh -G``: one word list, no quote or shell syntax.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_.@:/~%+, -]{1,1024}$")

# Everything the call fixes, whatever the platform's config says.
_FIXED_OPTIONS = (
    "IdentitiesOnly=yes",
    "IdentityAgent=none",
    "ForwardAgent=no",
    "ForwardX11=no",
    "ClearAllForwardings=yes",
    "PermitLocalCommand=no",
    "BatchMode=yes",
    "ControlMaster=no",
    "ControlPath=none",
    f"ConnectTimeout={CONNECT_TIMEOUT_S}",
)
# What ``ssh -G <ssh_target>`` may contribute, as ``-o`` options.
_RESOLVED_OPTIONS = {
    "hostname": "HostName",
    "userknownhostsfile": "UserKnownHostsFile",
    "globalknownhostsfile": "GlobalKnownHostsFile",
    "hostkeyalias": "HostKeyAlias",
    "stricthostkeychecking": "StrictHostKeyChecking",
}
_RESOLVED_KEYS = frozenset({*_RESOLVED_OPTIONS, "port", "proxyjump"})
_CHILD_ENV_KEYS = ("PATH", "HOME", "LANG")

Runner = Callable[[list[str], bytes, float], Awaitable[tuple[int, bytes, bytes]]]
Resolver = Callable[[str], Awaitable[dict[str, str]]]


@dataclass(frozen=True)
class GateResult:
    verb: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateRefusal:
    verb: str
    error: str
    detail: str = ""
    exit_code: int = EXIT_REFUSED


def _check_words(login: str, verb: str) -> None:
    """Neither may become an ssh option or a second remote word."""
    if not _LOGIN.fullmatch(login or ""):
        raise ValueError(f"not a site login: {login!r}")
    if not _VERB.fullmatch(verb or ""):
        raise ValueError(f"not a gate verb: {verb!r}")


def site_target(site: Any) -> str:
    """The site's ``extra.ssh_target`` (an alias or a host name)."""
    extra = getattr(site, "extra", None) or {}
    target = str(extra.get("ssh_target") or "").strip()
    if not target:
        raise ValueError("site has no extra.ssh_target")
    if not _HOST.fullmatch(target):
        raise ValueError(f"not an ssh target: {target!r}")
    return target


def _jump(site: Any, resolved: dict[str, str]) -> str:
    extra = getattr(site, "extra", None) or {}
    jump = str(extra.get("ssh_proxy_jump") or "").strip()
    jump = jump or resolved.get("proxyjump", "")
    return "" if jump == "none" else jump


def _port(site: Any, resolved: dict[str, str]) -> int:
    extra = getattr(site, "extra", None) or {}
    return int(extra.get("ssh_port") or resolved.get("port") or 22)


def proxy_command(jump: str) -> str:
    """The ``ProxyCommand`` hop for ``jump`` (``a`` or ``a,b``): the
    platform's own ssh, with its own config, forwarding TCP only."""
    hops = jump.split(",")
    if not all(_HOST.fullmatch(hop) for hop in hops):
        raise ValueError(f"not a jump host list: {jump!r}")
    via = ["-J", ",".join(hops[:-1])] if len(hops) > 1 else []
    return " ".join(["ssh", *via, "-W", "'[%h]:%p'", hops[-1]])


def _resolved_options(resolved: dict[str, str], jump: str) -> list[str]:
    options: list[str] = []
    for key, name in _RESOLVED_OPTIONS.items():
        value = resolved.get(key, "")
        if value and value != "none" and _SAFE_VALUE.fullmatch(value):
            options += ["-o", f"{name}={value}"]
    if jump:
        options += ["-o", f"ProxyCommand={proxy_command(jump)}"]
    return options


def gate_argv(
    site: Any,
    login: str,
    key_path: Path | str,
    verb: str,
    resolved: dict[str, str] | None = None,
) -> list[str]:
    """The contract's exact ssh argv for ``verb`` (ValueError on bad input).

    ``resolved`` is what ``resolve_target`` read for the site's target."""
    _check_words(login, verb)
    target = site_target(site)
    found = dict(resolved or {})
    argv = ["ssh", "-F", "none", "-i", str(key_path)]
    for option in _FIXED_OPTIONS:
        argv += ["-o", option]
    argv += _resolved_options(found, _jump(site, found))
    argv += ["-p", str(_port(site, found)), f"{login}@{target}", verb]
    return argv


def parse_ssh_g(out: bytes) -> dict[str, str]:
    """The ``_RESOLVED_KEYS`` of an ``ssh -G`` dump (first value wins)."""
    found: dict[str, str] = {}
    for line in out.decode("utf-8", "replace").splitlines():
        key, _, value = line.strip().partition(" ")
        if key.lower() in _RESOLVED_KEYS and key.lower() not in found:
            found[key.lower()] = value.strip()
    return found


async def resolve_target(target: str, runner: Runner | None = None) -> dict[str, str]:
    """``ssh -G -- <target>``: the platform config's view of the alias.

    Read once per call; only the values ``gate_argv`` takes are kept."""
    if not _HOST.fullmatch(target):
        raise ValueError(f"not an ssh target: {target!r}")
    argv = ["ssh", "-G", "--", target]
    code, out, err = await (runner or subprocess_runner)(argv, b"", RESOLVE_TIMEOUT_S)
    if code != 0:
        raise ValueError(f"ssh -G {target} failed: {err.decode('utf-8', 'replace')[-200:]}")
    return parse_ssh_g(out)


async def no_resolution(_target: str) -> dict[str, str]:
    """A test transport resolves nothing (no ``ssh_config`` is read)."""
    return {}


def encode_body(body: dict[str, Any] | None) -> bytes:
    """The body as the contract's canonical JSON (the gate's own encoder)."""
    if body is None:
        return b""
    from r4_contracts.hpc_gate import canonical_bytes

    return canonical_bytes(body)


def _json_object(raw: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _last_line(raw: bytes) -> bytes:
    lines = [line for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else b""


def _refusal(verb: str, code: int, err: bytes) -> GateRefusal:
    """A non-zero exit: the gate's JSON on the last stderr line, else a name
    for the exit code (ssh prints its own warnings before the gate's line).

    A refusal name counts only on the gate's refusal exit (4): a wrapper that
    prints the gate's line and exits otherwise is not the gate refusing."""
    answer = _json_object(_last_line(err)) if code == EXIT_REFUSED else None
    if answer and isinstance(answer.get("error"), str) and answer["error"]:
        return GateRefusal(verb, answer["error"], str(answer.get("detail") or ""), code)
    tail = err.decode("utf-8", "replace").strip()[-300:]
    return GateRefusal(verb, _EXIT_NAMES.get(code, BAD_ANSWER), tail, code)


def parse_answer(
    verb: str, code: int, out: bytes, err: bytes
) -> GateResult | GateRefusal:
    if code != 0:
        return _refusal(verb, code, err)
    answer = _json_object(out) if len(out) <= MAX_ANSWER_BYTES else None
    if answer is None:
        return GateRefusal(
            verb, BAD_ANSWER, "the gate's answer is not one JSON object", 0
        )
    return GateResult(verb, answer)


def _child_env() -> dict[str, str]:
    """``PATH``, ``HOME`` and ``LANG`` only: no agent socket, no server
    secret, nothing an ``ssh_config`` ``SendEnv`` could forward."""
    return {k: os.environ[k] for k in _CHILD_ENV_KEYS if k in os.environ}


async def subprocess_runner(
    argv: list[str], stdin: bytes, timeout: float
) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_child_env(),
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return SSH_FAILED, b"", b"gate call timed out"
    return proc.returncode or 0, out, err


def _resolver_for(runner: Runner | None, resolver: Resolver | None) -> Resolver:
    """``resolver``; else ``ssh -G`` for the real transport, and nothing for
    an injected one (a test runner is its own transport)."""
    if resolver is not None:
        return resolver
    return resolve_target if runner is None else no_resolution


async def run_verb(
    site: Any,
    login: str,
    key_path: Path | str,
    verb: str,
    body: dict[str, Any] | None = None,
    runner: Runner | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    resolver: Resolver | None = None,
) -> GateResult | GateRefusal:
    """Call ``verb`` on ``login``'s gate at ``site`` with the transport key."""
    _check_words(login, verb)
    target = site_target(site)
    try:
        resolved = await _resolver_for(runner, resolver)(target)
    except (OSError, ValueError) as exc:
        return GateRefusal(verb, UNREACHABLE, f"cannot resolve {target}: {exc}", SSH_FAILED)
    argv = gate_argv(site, login, key_path, verb, resolved)
    code, out, err = await (runner or subprocess_runner)(
        argv, encode_body(body), timeout
    )
    return parse_answer(verb, code, out, err)


__all__ = [
    "BAD_ANSWER",
    "GATE_PROBE_REFUSAL",
    "UNREACHABLE",
    "GateRefusal",
    "GateResult",
    "encode_body",
    "gate_argv",
    "no_resolution",
    "parse_answer",
    "parse_ssh_g",
    "proxy_command",
    "resolve_target",
    "run_verb",
    "site_target",
    "subprocess_runner",
]
