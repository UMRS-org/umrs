#!/usr/bin/env python3
"""UMRS gate: the forced command behind a per-person UMRS transport key.

ADR-112 D-3, D-4 and D-7; the wire contract is the gated-delegation plan,
section "B1.5a Wire contract". This one self-contained file has two roles:

* **An importable module** shared by UMRS and r4: strict canonical JSON
  (``canonical_bytes``, ``parse_strict``), the SSHSIG namespaces
  (``namespace``), ``intent_id``, a stdlib SSHSIG parser with the ``sk-``
  flag policy, and the ``allowed_signers`` / ``authorized_keys`` line
  builders.
* **The gate script** that r4 installs byte-identical as
  ``~/.umrs/gate/umrs_gate.py`` and that sshd runs as
  ``/usr/bin/python3 -I <home>/.umrs/gate/umrs_gate.py --site <site>``.

It runs on Python 3.9 or newer and imports only the standard library
(``fcntl`` and ``pwd`` lazily, so the module stays importable everywhere).

Runtime contract, in brief:

* ``SSH_ORIGINAL_COMMAND`` is split with ``shlex`` into ``<verb>``; a body,
  if any, is JSON on stdin (cap 1 MiB). One JSON object goes to stdout.
* Exit codes: 0 ok, 1 failed, 2 usage, 3 absent, 4 refused, 5 scheduler.
  Every non-zero exit prints ``{"detail", "error"}`` on stderr; for exit 4
  ``error`` is one of ``REFUSALS``; otherwise it is ``failed``, ``usage``,
  ``absent`` or ``scheduler``.
* Verbs: ``hello``, ``probe`` (always refuses ``gate_probe_refusal``),
  ``revoke`` and ``rotate`` (signed controls, namespace
  ``umrs-control-v1@<domain>``). Wave B2 (plan section 5, "B2.0 Wire
  contract") adds the execution verbs ``mkwd``, ``stage-in``, ``submit``
  (signed intents, namespace ``umrs-intent-v1@<domain>``), ``lookup``,
  ``status``, ``tail-log``, ``read-reply``, ``collect-out``, ``cancel``,
  ``rmwd``, ``sweep`` and ``usage``, and the shared B2 API (intent and site
  profile validation, the step graph, the job script, ``input.json``).
  Waves C and D (plan section 6, "C/D.0 Wire contract"; gate_version "3")
  add standing grants: ``submit`` with a signed grant and an unsigned run
  spec, ``grant-check``, the signed control ``grant-cancel``, per-grant
  counters under ``flock`` with reservation and settlement, the outputs
  index of ``chained`` grants, the revocation epoch and the ``hello``
  checkpoint (shared API: ``validate_grant``, ``params_problems``,
  ``grant_covers``, ``grant_cost`` and the other names listed in C3).
* A session lasts at most ``GATE_DEADLINE`` seconds (B2 verbs: ``DEADLINES``)
  and waits at most
  ``LOCK_WAIT`` seconds for the log lock (refusing ``log_unwritable``,
  ``busy``). A control's expiry is judged when its body has arrived.
* Every control attempt is logged. One with no valid signature gets a fixed
  short entry, and none once the log holds ``LOG_UNSIGNED_CAP`` bytes; a
  failed append is rolled back, so the chain never ends in a partial line.

Log chain: ``h = sha256(bytes.fromhex(prev) + canonical(entry without h))``,
written as hex; the genesis ``prev`` is 64 zeros and ``n`` starts at 1.
``verify_log`` checks a whole log file.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import copy
import errno
import hashlib
import json
import math
import os
import platform
import posixpath
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import IO, Any

GATE_VERSION = "3"
#: ``gate.json`` written by a B1 ("1") or B2 ("2") install keeps working after
#: the upgrade (plan section 6 "C/D.0", K1).
GATE_CONFIG_VERSIONS = ("1", "2", "3")

NAMESPACE_INTENT = "umrs-intent-v1@{domain}"
NAMESPACE_GRANT = "umrs-grant-v1@{domain}"
NAMESPACE_CONTROL = "umrs-control-v1@{domain}"
NAMESPACES = {
    "intent": NAMESPACE_INTENT,
    "grant": NAMESPACE_GRANT,
    "control": NAMESPACE_CONTROL,
}

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ABSENT = 3
EXIT_REFUSED = 4
EXIT_SCHEDULER = 5

REFUSALS = (
    "too_large",
    "bad_envelope",
    "not_signed",
    "bad_signature",
    "presence_required",
    "verification_required",
    "key_type_not_allowed",
    "not_canonical",
    "bad_schema",
    "wrong_site",
    "wrong_principal",
    "expired",
    "replayed",
    "no_matching_key",
    "unknown_verb",
    "no_command",
    "gate_probe_refusal",
    "log_unwritable",
    # wave B2 (plan section 5.3)
    "sealed",
    "broken",
    "busy",
    "wrong_step",
    "wrong_login",
    "wrong_intent",
    "profile_mismatch",
    "profile_unsafe",
    "resource_not_allowed",
    "account_not_held",
    "parent_not_collected",
    "src_mismatch",
    "input_mismatch",
    "image_absent",
    "image_mismatch",
    "image_label",
    "bad_member",
    "digest_mismatch",
    "too_many_workdirs",
    "low_space",
    "not_mine",
    "not_finished",
    "job_live",
    "bad_reply",
    "log_full",
    # waves C and D (plan section 6 "C/D.0", C3)
    "grant_mismatch",
    "grant_exhausted",
    "grant_expired",
    "grant_not_yet_valid",
    "grant_cancelled",
    "grant_conflict",
    "run_spec_changed",
)

B2_VERBS = (
    "mkwd",
    "stage-in",
    "submit",
    "lookup",
    "status",
    "tail-log",
    "read-reply",
    "collect-out",
    "cancel",
    "rmwd",
    "sweep",
    "usage",
)
#: Wave C (plan section 6 "C/D.0", C3): ``grant-check`` is a session verb
#: (it settles through Slurm); ``grant-cancel`` is a signed control.
C_VERBS = ("grant-check", "grant-cancel")
VERBS = ("hello", "probe", "revoke", "rotate", *B2_VERBS, *C_VERBS)
CONTROL_VERBS = ("revoke", "rotate", "grant-cancel")

MAX_INT = 2**53 - 1  # I-JSON / RFC 8785 safe range (plan B1.5a amendment 2026-11-08)
MAX_DEPTH = 64
MAX_STDIN = 1 << 20
MAX_AUTHORIZED_KEYS = 4 << 20
CONTROL_MAX_WINDOW = 600
CLOCK_SKEW = 120
NONCE_RETENTION = 2 * 86400
ASSOC_CAP = 200
SIGNER_CAP = 64
DETAIL_CAP = 512
SACCTMGR_TIMEOUT = 15
KEYGEN_TIMEOUT = 30
VERSION_TIMEOUT = 10
GATE_DEADLINE = 60
LOCK_WAIT = 10.0
LOG_UNSIGNED_CAP = 8 << 20
GENESIS = "0" * 64

SSH_KEYGEN = "/usr/bin/ssh-keygen"
SSH_CLIENT = "/usr/bin/ssh"
GATE_PYTHON = "/usr/bin/python3"
GATE_DIR = os.path.join(".umrs", "gate")
GATE_SCRIPT = "umrs_gate.py"
LOG_NAME = "gate.jsonl"

FLAG_USER_PRESENCE = 0x01
FLAG_USER_VERIFICATION = 0x04

SK_KEY_TYPES = frozenset({"sk-ssh-ed25519@openssh.com", "sk-ecdsa-sha2-nistp256@openssh.com"})
NON_SK_KEY_TYPES = frozenset({"ssh-ed25519"})
SIGNER_KEY_TYPES = SK_KEY_TYPES | NON_SK_KEY_TYPES

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_TAIL_BYTES = 1 << 16

_SITE_RE = re.compile(r"[a-z0-9-]{1,40}")
_KEY_ID_RE = re.compile(r"tk-[0-9a-f]{12}")
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_DOMAIN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
_PRINCIPAL_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._@+-]{0,127}")
_ED25519_B64_RE = re.compile(r"[A-Za-z0-9+/]{68}")
_HOME_RE = re.compile(r"/[A-Za-z0-9._+/-]{0,254}")
_FROM_ITEM_RE = re.compile(r"!?[A-Za-z0-9.:*?/_-]{1,255}")
_VALID_BEFORE_RE = re.compile(r"[0-9]{8}(?:[0-9]{4}(?:[0-9]{2})?)?Z?")
_NEW_KEY_RE = re.compile(
    r"ssh-ed25519 ([A-Za-z0-9+/]{68})(?: (umrs-transport-[a-z0-9-]{1,40}-tk-[0-9a-f]{12}))?"
)
_ARMOR_BEGIN = "-----BEGIN SSH SIGNATURE-----"
_ARMOR_END = "-----END SSH SIGNATURE-----"


class GateError(Exception):
    """A gate outcome other than success: ``error`` name, ``detail``, exit ``code``."""

    def __init__(self, error: str, detail: str = "", code: int = EXIT_REFUSED) -> None:
        super().__init__(f"{error}: {detail}")
        self.error = error
        self.detail = detail
        self.code = code

    def as_json(self) -> dict[str, str]:
        return {"error": self.error, "detail": self.detail}


def refuse(name: str, detail: str = "") -> GateError:
    """Build a refusal (exit 4); ``name`` must be one of ``REFUSALS``."""
    if name not in REFUSALS:
        raise ValueError(f"unknown refusal name: {name}")
    return GateError(name, _clip(detail), EXIT_REFUSED)


def _clip(text: str, cap: int = DETAIL_CAP) -> str:
    return text if len(text) <= cap else text[: cap - 3] + "..."


# ---------------------------------------------------------------------------
# Namespaces and canonical JSON (ADR-112 D-4; plan invariant I-8)
# ---------------------------------------------------------------------------


def namespace(kind: str, domain: str) -> str:
    """The SSHSIG namespace for ``kind`` (intent, grant, control) at ``domain``."""
    if kind not in NAMESPACES:
        raise ValueError(f"unknown namespace kind: {kind!r}")
    if not isinstance(domain, str) or not _DOMAIN_RE.fullmatch(domain):
        raise ValueError(f"invalid signing domain: {domain!r}")
    return NAMESPACES[kind].format(domain=domain)


def _is_json_extra(value: Any) -> bool:
    """``null`` or a finite float: allowed only where ``json_values`` is set."""
    return value is None or (isinstance(value, float) and math.isfinite(value))


def _check_scalar(value: Any, json_values: bool = False) -> None:
    if isinstance(value, (bool, str)):
        return
    if isinstance(value, int) and -MAX_INT <= value <= MAX_INT:
        return
    if json_values and _is_json_extra(value):
        return
    raise ValueError(f"value not allowed in canonical JSON: {type(value).__name__} {value!r}")


def _check_mapping(mapping: dict[Any, Any], depth: int, json_values: bool = False) -> None:
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"canonical JSON keys must be str, got {type(key).__name__}")
        _check_value(item, depth + 1, json_values)


def _check_value(value: Any, depth: int, json_values: bool = False) -> None:
    if depth > MAX_DEPTH:
        raise ValueError("canonical JSON nests too deeply")
    if isinstance(value, list):
        for item in value:
            _check_value(item, depth + 1, json_values)
        return
    if isinstance(value, dict):
        _check_mapping(value, depth, json_values)
        return
    _check_scalar(value, json_values)


def canonical_bytes(obj: Any, json_values: bool = False) -> bytes:
    """Contract encoding: sorted keys, compact separators, UTF-8, restricted values.

    Allowed: str, bool, int with ``|v| <= 2**53 - 1``, list, dict with str keys.
    Raises ``ValueError`` on anything else (floats, None, tuples, big ints,
    non-str keys, lone surrogates). With ``json_values`` (plan B2.0 section
    1.4: intent ``params`` and step ``config`` only) ``None`` and finite
    floats are also allowed; a float is written as Python's shortest
    round-trip ``repr``.
    """
    _check_value(obj, 0, json_values)
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def _reject_constant(name: str) -> Any:
    raise ValueError(f"JSON constant not allowed: {name}")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def _loads_no_duplicates(data: bytes) -> Any:
    """``json.loads`` refusing duplicate keys, NaN/Infinity and invalid UTF-8."""
    text = bytes(data).decode("utf-8")
    try:
        return json.loads(
            text, object_pairs_hook=_pairs_no_duplicates, parse_constant=_reject_constant
        )
    except RecursionError as exc:
        raise ValueError("JSON nests too deeply") from exc


def parse_strict(data: bytes, json_values: bool = False) -> dict[str, Any]:
    """Parse a signed body: one JSON object whose bytes equal its canonical form.

    ``json_values`` admits ``null`` and finite floats (``canonical_bytes``);
    a float that does not re-encode to the same bytes (``1E5``, ``0.50``) is
    not canonical."""
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("parse_strict takes bytes")
    obj = _loads_no_duplicates(data)
    if not isinstance(obj, dict):
        raise ValueError("top level must be a JSON object")
    if canonical_bytes(obj, json_values) != bytes(data):
        raise ValueError("body is not in canonical form")
    return obj


def intent_id(data: bytes) -> str:
    """The id of a signed body: sha256 hex of its exact bytes."""
    return hashlib.sha256(bytes(data)).hexdigest()


def gate_sha256() -> str:
    """sha256 hex of this file's bytes (what ``hello`` reports)."""
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# ---------------------------------------------------------------------------
# SSH wire format, SSHSIG parsing and the sk- flag policy (ADR-112 D-4, I-7)
# ---------------------------------------------------------------------------


class _Reader:
    """Read SSH wire encoding (RFC 4251 ``byte``, ``uint32``, ``string``)."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def take(self, count: int) -> bytes:
        end = self._pos + count
        if count < 0 or end > len(self._data):
            raise ValueError("truncated SSH wire data")
        chunk = self._data[self._pos : end]
        self._pos = end
        return chunk

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def byte(self) -> int:
        return self.take(1)[0]

    def string(self) -> bytes:
        return self.take(self.u32())

    def text(self) -> str:
        return self.string().decode("ascii")

    def end(self) -> None:
        if self._pos != len(self._data):
            raise ValueError("trailing bytes in SSH wire data")


@dataclass(frozen=True)
class SshSig:
    """The fields of an armored SSHSIG (OpenSSH PROTOCOL.sshsig)."""

    pubkey_type: str
    namespace: str
    hash_alg: str
    sig_type: str
    flags: int | None
    counter: int | None
    public_key: bytes = b""

    @property
    def signer_fingerprint(self) -> str:
        return fingerprint(self.public_key)


def _dearmor(armored: str) -> bytes:
    lines = [line.strip() for line in armored.strip().split("\n")]
    if len(lines) < 3 or lines[0] != _ARMOR_BEGIN or lines[-1] != _ARMOR_END:
        raise ValueError("not an armored SSH signature")
    try:
        return base64.b64decode("".join(lines[1:-1]), validate=True)
    except binascii.Error as exc:
        raise ValueError("SSH signature is not base64") from exc


def _parse_signature_blob(blob: bytes) -> tuple[str, int | None, int | None]:
    reader = _Reader(blob)
    sig_type = reader.text()
    reader.string()
    if not sig_type.startswith("sk-"):
        reader.end()
        return sig_type, None, None
    flags = reader.byte()
    counter = reader.u32()
    reader.end()
    return sig_type, flags, counter


def parse_sshsig(armored: str) -> SshSig:
    """Parse an armored SSHSIG; raises ``ValueError`` on any malformation.

    Layout: ``"SSHSIG"``, u32 version (1), string publickey, string namespace,
    string reserved, string hash_algorithm, string signature. For ``sk-`` keys
    the signature string holds string sig-type, string sig, byte flags,
    u32 counter. Parsing does not verify; ``ssh-keygen -Y verify`` does.
    """
    reader = _Reader(_dearmor(armored))
    if reader.take(6) != b"SSHSIG" or reader.u32() != 1:
        raise ValueError("not an SSHSIG v1 blob")
    public_key = reader.string()
    ns = reader.string().decode("utf-8")
    reader.string()
    hash_alg = reader.text()
    sig_type, flags, counter = _parse_signature_blob(reader.string())
    reader.end()
    return SshSig(
        pubkey_type=_Reader(public_key).text(),
        namespace=ns,
        hash_alg=hash_alg,
        sig_type=sig_type,
        flags=flags,
        counter=counter,
        public_key=public_key,
    )


def _check_sk_flags(flags: int | None, uv_required: bool) -> None:
    if flags is None or not flags & FLAG_USER_PRESENCE:
        raise refuse("presence_required", "the security key did not assert user presence")
    if uv_required and not flags & FLAG_USER_VERIFICATION:
        raise refuse("verification_required", "the security key did not assert user verification")


def check_signature_policy(sig: SshSig, *, uv_required: bool, allow_non_sk: bool) -> None:
    """Enforce the key policy on a verified signature, raising a refusal.

    ``sk-`` keys need UP (0x01) always and UV (0x04) when ``uv_required``;
    a plain ``ssh-ed25519`` key is accepted only with ``allow_non_sk``.
    """
    if sig.pubkey_type in SK_KEY_TYPES:
        _check_sk_flags(sig.flags, uv_required)
        return
    if sig.pubkey_type in NON_SK_KEY_TYPES and allow_non_sk:
        return
    raise refuse("key_type_not_allowed", f"signing key type {sig.pubkey_type} is not allowed")


# ---------------------------------------------------------------------------
# Keys and the lines r4 installs
# ---------------------------------------------------------------------------


def fingerprint(blob: bytes) -> str:
    """OpenSSH SHA256 fingerprint of a public key blob."""
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode("ascii")
    return "SHA256:" + digest.rstrip("=")


def _b64_strict(text: str) -> bytes:
    try:
        blob = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("key is not base64") from exc
    if base64.b64encode(blob).decode("ascii") != text:
        raise ValueError("key is not canonical base64")
    return blob


def ed25519_blob(key_b64: str) -> bytes:
    """Decode a strict ``ssh-ed25519`` base64 key to its 51-byte blob."""
    if not isinstance(key_b64, str) or not _ED25519_B64_RE.fullmatch(key_b64):
        raise ValueError("not an ssh-ed25519 base64 key")
    blob = _b64_strict(key_b64)
    reader = _Reader(blob)
    if reader.text() != "ssh-ed25519" or len(reader.string()) != 32:
        raise ValueError("not an ssh-ed25519 key blob")
    reader.end()
    return blob


def _signer_blob(key_type: str, key_b64: str) -> bytes:
    if key_type not in SIGNER_KEY_TYPES:
        raise ValueError(f"signing key type not allowed: {key_type!r}")
    blob = _b64_strict(key_b64)
    if _Reader(blob).text() != key_type:
        raise ValueError("key type does not match the key blob")
    return blob


def _require(pattern: re.Pattern[str], value: Any, label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def is_site_name(value: Any) -> bool:
    """True when ``value`` can name a site in the forced command and comment."""
    return isinstance(value, str) and _SITE_RE.fullmatch(value) is not None


def transport_comment(site: str, key_id: str) -> str:
    """The comment that names a transport line: ``umrs-transport-<site>-<key_id>``."""
    _require(_SITE_RE, site, "site")
    _require(_KEY_ID_RE, key_id, "key_id")
    return f"umrs-transport-{site}-{key_id}"


def _check_home(home: str) -> str:
    _require(_HOME_RE, home, "home")
    if any(part in (".", "..") for part in home.split("/")):
        raise ValueError(f"invalid home: {home!r}")
    return home.rstrip("/") or "/"


def gate_command(home: str, site: str) -> str:
    """The forced command for ``site`` with the gate installed under ``home``."""
    home = _check_home(home)
    _require(_SITE_RE, site, "site")
    script = f"{home.rstrip('/')}/{GATE_DIR}/{GATE_SCRIPT}"
    return f"{GATE_PYTHON} -I {script} --site {site}"


def check_from_address(from_address: str) -> str:
    """A non-empty sshd ``from=`` pattern list the gate writes; ``ValueError`` else."""
    if not isinstance(from_address, str):
        raise ValueError("from address must be a string")
    return _check_from(from_address)


def _check_from(from_address: str) -> str:
    items = from_address.split(",")
    if len(items) > 32:
        raise ValueError("too many from= patterns")
    for item in items:
        _require(_FROM_ITEM_RE, item, "from address")
    return from_address


def authorized_keys_line(
    home: str, site: str, key_id: str, public_key_b64: str, from_address: str = ""
) -> str:
    """The transport line: ``restrict[,from=..],command=".." ssh-ed25519 <b64> <comment>``."""
    ed25519_blob(public_key_b64)
    comment = transport_comment(site, key_id)
    options = "restrict"
    if from_address:
        options += f',from="{_check_from(from_address)}"'
    options += f',command="{gate_command(home, site)}"'
    return f"{options} ssh-ed25519 {public_key_b64} {comment}"


def allowed_signers_line(
    principal: str, domain: str, key_type: str, key_b64: str, valid_before: str = ""
) -> str:
    """``<principal> namespaces="<intent>,<grant>,<control>"[,valid-before=..] <type> <key>``."""
    _require(_PRINCIPAL_RE, principal, "principal")
    _signer_blob(key_type, key_b64)
    names = ",".join(namespace(kind, domain) for kind in ("intent", "grant", "control"))
    options = f'namespaces="{names}"'
    if valid_before:
        options += f',valid-before="{_require(_VALID_BEFORE_RE, valid_before, "valid_before")}"'
    return f"{principal} {options} {key_type} {key_b64}"


def _token_spans(line: str) -> list[tuple[int, int]]:
    """Whitespace-separated token spans; double quotes keep spaces inside a token."""
    spans: list[tuple[int, int]] = []
    start = -1
    quoted = False
    for index, char in enumerate(line):
        if char == '"':
            quoted = not quoted
        if char in " \t" and not quoted:
            if start >= 0:
                spans.append((start, index))
            start = -1
        elif start < 0:
            start = index
    if start >= 0:
        spans.append((start, len(line)))
    return spans


# ---------------------------------------------------------------------------
# Hash-chained log (ADR-112 D-5 last bullets; plan I-9)
# ---------------------------------------------------------------------------

_LOG_KEYS = frozenset({"n", "t", "verb", "result", "detail", "prev", "h"})


def log_entry_hash(prev: str, entry: dict[str, Any]) -> str:
    """``sha256(bytes.fromhex(prev) + canonical(entry without "h"))`` as hex."""
    body = {key: value for key, value in entry.items() if key != "h"}
    return hashlib.sha256(bytes.fromhex(prev) + canonical_bytes(body)).hexdigest()


def _check_log_entry(raw: bytes, prev: str, index: int) -> str:
    entry = parse_strict(raw)
    if set(entry) != _LOG_KEYS or type(entry["n"]) is not int or entry["n"] != index + 1:
        raise ValueError(f"log entry {index + 1}: bad shape or sequence")
    if entry["prev"] != prev or entry["h"] != log_entry_hash(prev, entry):
        raise ValueError(f"log entry {index + 1}: chain broken")
    return str(entry["h"])


def _log_lines(path: str) -> Iterator[bytes]:
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC)
    with os.fdopen(fd, "rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise ValueError("log ends with a partial line")
            yield line[:-1]


def verify_log(path: str) -> bool:
    """True when every entry of the log at ``path`` chains to the one before.

    An absent or empty log is a valid (empty) chain.
    """
    prev = GENESIS
    try:
        for index, raw in enumerate(_log_lines(path)):
            prev = _check_log_entry(raw, prev, index)
    except FileNotFoundError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def _pread_all(fd: int, size: int, start: int) -> bytes:
    """Bytes ``start..size`` of ``fd``; a short ``pread`` is read again."""
    chunks: list[bytes] = []
    offset = start
    while offset < size:
        chunk = os.pread(fd, size - offset, offset)
        if not chunk:
            raise ValueError("the log shrank while it was read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _tail_head(fd: int) -> tuple[int, str]:
    size = os.fstat(fd).st_size
    if size == 0:
        return 0, GENESIS
    start = max(0, size - _TAIL_BYTES)
    chunk = _pread_all(fd, size, start)
    if not chunk.endswith(b"\n"):
        raise ValueError("log ends with a partial line")
    entry = parse_strict(chunk[:-1].rsplit(b"\n", 1)[-1])
    count, head = entry.get("n"), entry.get("h")
    if type(count) is not int or not isinstance(head, str) or not _HEX64_RE.fullmatch(head):
        raise ValueError("log tail entry is malformed")
    return count, head


def read_log_head(path: str) -> dict[str, Any]:
    """``{"n", "h"}`` of the last log entry; ``{0, GENESIS}`` when there is none."""
    try:
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC)
    except FileNotFoundError:
        return {"n": 0, "h": GENESIS}
    try:
        count, head = _tail_head(fd)
    finally:
        os.close(fd)
    return {"n": count, "h": head}


def _require_owned_regular(fd: int) -> None:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise OSError("not a regular file owned by this user")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _lock(fcntl: Any, fd: int, wait: float) -> None:
    """``flock`` the log, waiting at most ``wait`` seconds (never forever)."""
    end = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic() >= end:
                raise OSError("busy: another control holds the log") from None
            time.sleep(0.05)


class GateLog:
    """The append-only, fsync'd, hash-chained gate log, held under ``flock``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fd = -1
        self._held = False
        self.n = 0
        self.h = GENESIS

    def open(self, lock: bool = True) -> None:
        """Open for append; with ``lock`` the ``flock`` is held until ``close``
        (B1 controls). Without it, each ``record`` takes the lock for its own
        append only (B2 verbs, plan section 5 "Locks")."""
        import fcntl

        flags = os.O_RDWR | os.O_APPEND | os.O_CREAT | _O_NOFOLLOW | _O_CLOEXEC
        fd = os.open(self.path, flags, 0o600)
        try:
            _require_owned_regular(fd)
            if lock:
                _lock(fcntl, fd, LOCK_WAIT)
                self.n, self.h = _tail_head(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._held = lock

    def append_locked(self, verb: str, result: str, detail: str, now: int) -> None:
        """Take the lock (``LOCK_WAIT``), re-read the head, append, release."""
        import fcntl

        _lock(fcntl, self._fd, LOCK_WAIT)
        try:
            self.n, self.h = _tail_head(self._fd)
            self.append(verb, result, detail, now)
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def record(self, verb: str, result: str, detail: str, now: int) -> None:
        """Append under the held lock, or under a lock taken for this entry."""
        if self._held:
            self.append(verb, result, detail, now)
        else:
            self.append_locked(verb, result, detail, now)

    def append(self, verb: str, result: str, detail: str, now: int) -> None:
        entry: dict[str, Any] = {
            "n": self.n + 1,
            "t": now,
            "verb": verb,
            "result": result,
            "detail": _clip(detail),
            "prev": self.h,
        }
        entry["h"] = log_entry_hash(self.h, entry)
        offset = os.fstat(self._fd).st_size
        try:
            _write_all(self._fd, canonical_bytes(entry) + b"\n")
            os.fsync(self._fd)
        except BaseException:
            self._roll_back(offset)
            raise
        self.n, self.h = entry["n"], entry["h"]

    def _roll_back(self, offset: int) -> None:
        """A failed append leaves no partial line: the chain stays intact."""
        with contextlib.suppress(OSError):
            os.ftruncate(self._fd, offset)
            os.fsync(self._fd)

    def size(self) -> int:
        return os.fstat(self._fd).st_size

    def head(self) -> dict[str, Any]:
        return {"n": self.n, "h": self.h}

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


# ---------------------------------------------------------------------------
# Files: config, layout, nonces, authorized_keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """``~/.umrs/gate/gate.json`` as written by ``r4 hpc add``."""

    site: str
    domain: str
    principal: str
    uv_required: bool
    allow_non_sk: bool
    gate_version: str


def _matches(pattern: re.Pattern[str]) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, str) and pattern.fullmatch(value) is not None


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


_CONFIG_FIELDS: dict[str, Callable[[Any], bool]] = {
    "site": _matches(_SITE_RE),
    "domain": _matches(_DOMAIN_RE),
    "principal": _matches(_PRINCIPAL_RE),
    "uv_required": _is_bool,
    "allow_non_sk": _is_bool,
    "gate_version": lambda value: value in GATE_CONFIG_VERSIONS,
}


def _read_upto(fd: int, cap: int) -> bytes:
    """Up to ``cap + 1`` bytes of ``fd``: a short ``read`` is read again."""
    chunks: list[bytes] = []
    total = 0
    while total <= cap:
        chunk = os.read(fd, cap + 1 - total)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _read_nofollow(path: str, cap: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC)
    try:
        _require_owned_regular(fd)
        data = _read_upto(fd, cap)
        size = os.fstat(fd).st_size
    finally:
        os.close(fd)
    if len(data) > cap:
        raise ValueError(f"{os.path.basename(path)} is too large")
    if len(data) != size:
        raise ValueError(f"{os.path.basename(path)} changed while it was read")
    return data


def _validate_config(raw: Any) -> GateConfig:
    if not isinstance(raw, dict) or set(raw) != set(_CONFIG_FIELDS):
        raise ValueError(f"gate.json must hold exactly {sorted(_CONFIG_FIELDS)}")
    for key, check in _CONFIG_FIELDS.items():
        if not check(raw[key]):
            raise ValueError(f"gate.json: invalid {key}")
    return GateConfig(**raw)


def load_config(root: str) -> GateConfig:
    """Read and validate ``gate.json``; ``GateError`` absent (3) or failed (1)."""
    path = os.path.join(root, "gate.json")
    try:
        return _validate_config(_loads_no_duplicates(_read_nofollow(path, 1 << 16)))
    except FileNotFoundError:
        raise GateError("absent", "gate.json is missing", EXIT_ABSENT) from None
    except (OSError, ValueError) as exc:
        raise GateError("failed", _clip(f"gate.json: {exc}"), EXIT_FAILED) from None


def _home() -> str:
    home = os.environ.get("HOME", "")
    if os.path.isabs(home):
        return home
    import pwd

    return pwd.getpwuid(os.geteuid()).pw_dir


def gate_root(home: str) -> str:
    return os.path.join(home, GATE_DIR)


def _require_owned_dir(path: str) -> None:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise OSError(f"{path} is not a directory owned by this user")


def _ensure_dir(path: str) -> None:
    with contextlib.suppress(FileExistsError):
        os.mkdir(path, 0o700)
    _require_owned_dir(path)


def _ensure_layout(root: str) -> None:
    _require_owned_dir(root)
    for sub in ("state", os.path.join("state", "nonces"), "log"):
        _ensure_dir(os.path.join(root, sub))


def _is_stale(entry: os.DirEntry[str], cutoff: float) -> bool:
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def prune_nonces(nonce_dir: str, now: float) -> None:
    """Remove nonce files older than ``NONCE_RETENTION`` (two days)."""
    cutoff = now - NONCE_RETENTION
    with os.scandir(nonce_dir) as entries:
        for entry in entries:
            if _is_stale(entry, cutoff):
                _unlink_quiet(entry.path)


def consume_nonce(nonce_dir: str, nonce: str, now: int) -> None:
    """Record ``nonce`` with ``O_EXCL``; a second use refuses ``replayed``."""
    _require(_NONCE_RE, nonce, "nonce")
    prune_nonces(nonce_dir, now)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
    try:
        fd = os.open(os.path.join(nonce_dir, nonce), flags, 0o600)
    except FileExistsError:
        raise refuse("replayed", f"nonce {nonce} was already used") from None
    try:
        _write_all(fd, str(now).encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_authorized_keys(path: str) -> str:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        raise refuse("no_matching_key", "no authorized_keys file") from None
    if stat.S_ISLNK(info.st_mode):
        raise GateError("failed", "authorized_keys is a symlink; refusing to edit", EXIT_FAILED)
    try:
        return _read_nofollow(path, MAX_AUTHORIZED_KEYS).decode("utf-8")
    except (OSError, ValueError) as exc:
        raise GateError("failed", _clip(f"authorized_keys: {exc}"), EXIT_FAILED) from None


def _line_comment(line: str) -> str:
    text = line.strip()
    if not text or text.startswith("#"):
        return ""
    spans = _token_spans(text)
    start, end = spans[-1]
    return text[start:end]


def _replace_key(line: str, b64: str, comment: str | None) -> str:
    body = line.rstrip("\r\n")
    ending = line[len(body) :]
    spans = _token_spans(body)
    first = body[spans[0][0] : spans[0][1]]
    index = 0 if first.startswith(("ssh-", "ecdsa-", "sk-")) else 1
    if index + 2 >= len(spans):
        raise GateError("failed", "matching authorized_keys line has no key blob", EXIT_FAILED)
    old_comment = body[spans[-1][0] : spans[-1][1]]
    middle = body[spans[index + 1][1] : spans[-1][0]]
    prefix = body[: spans[index][0]]
    return f"{prefix}ssh-ed25519 {b64}{middle}{comment or old_comment}{ending}"


def split_lines(text: str) -> list[str]:
    """Lines as sshd reads them: split at ``\\n`` only, endings kept.

    ``str.splitlines`` also breaks at ``\\x0b``, ``\\x1c``, U+2028 and more,
    which sshd does not; a line edited by that reading is not the line sshd
    applies."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    return lines + ([parts[-1]] if parts[-1] else [])


def _holds_key(line: str, key_b64: str) -> bool:
    text = line.strip()
    if not text or text.startswith("#"):
        return False
    return any(text[start:end] == key_b64 for start, end in _token_spans(text))


def _require_unused(lines: list[str], comment: str, key_b64: str) -> None:
    """A rotation never installs a key another line already authorizes (the
    gate's line would then capture that key's logins)."""
    for line in lines:
        if _line_comment(line) != comment and _holds_key(line, key_b64):
            raise refuse("bad_schema", "new_public_key is already authorized")


def _rewrite_lines(
    lines: list[str], comment: str, new_key: tuple[str, str | None] | None
) -> tuple[list[str], int]:
    out: list[str] = []
    changed = 0
    for line in lines:
        if _line_comment(line) != comment:
            out.append(line)
            continue
        changed += 1
        if new_key is not None:
            out.append(_replace_key(line, new_key[0], new_key[1]))
    return out, changed


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | _O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(directory: str, path: str, data: bytes, prefix: str = ".authorized_keys") -> None:
    temp = os.path.join(directory, f"{prefix}.umrs-{os.getpid()}-{secrets.token_hex(4)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
    fd = os.open(temp, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        _write_all(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        _unlink_quiet(temp)
        raise
    _fsync_dir(directory)


def edit_authorized_keys(home: str, comment: str, new_key: tuple[str, str | None] | None) -> int:
    """Remove (``new_key`` None) or re-key every line whose comment is ``comment``.

    Only ``~/.ssh/authorized_keys`` of the running user, never through a
    symlink, rewritten atomically with mode 0600. Returns the lines touched;
    refuses ``no_matching_key`` when there are none.
    """
    ssh_dir = os.path.join(home, ".ssh")
    try:
        _require_owned_dir(ssh_dir)
    except FileNotFoundError:
        raise refuse("no_matching_key", "no ~/.ssh directory") from None
    except OSError as exc:
        raise GateError("failed", _clip(str(exc)), EXIT_FAILED) from None
    path = os.path.join(ssh_dir, "authorized_keys")
    lines = split_lines(_read_authorized_keys(path))
    if new_key is not None:
        _require_unused(lines, comment, new_key[0])
    out, changed = _rewrite_lines(lines, comment, new_key)
    if not changed:
        raise refuse("no_matching_key", f"no authorized_keys line named {comment}")
    try:
        _atomic_write(ssh_dir, path, "".join(out).encode("utf-8"))
    except OSError as exc:
        raise GateError("failed", _clip(f"authorized_keys: {exc}"), EXIT_FAILED) from None
    return changed


# ---------------------------------------------------------------------------
# Subprocesses (absolute paths, argv lists, no shell, timeouts)
# ---------------------------------------------------------------------------


def _run(
    argv: list[str], stdin: bytes, timeout: float, env: dict[str, str] | None
) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _tool_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "HOME": os.environ.get("HOME", "/")}


def _scheduler_env() -> dict[str, str]:
    """sacctmgr sees PATH, HOME and the C locale, never the rest of the
    session environment (what the SSH client sent through AcceptEnv)."""
    path = os.environ.get("PATH") or "/usr/bin:/bin"
    return {"PATH": path, "LC_ALL": "C", "HOME": os.environ.get("HOME", "/")}


def ssh_keygen_verify(
    signers_path: str, principal: str, ns: str, armored: str, payload: bytes, scratch: str
) -> None:
    """``ssh-keygen -Y verify`` of ``payload``; refuses ``bad_signature`` on failure."""
    fd, sig_path = tempfile.mkstemp(prefix=".sig-", dir=scratch)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(armored.encode("utf-8"))
        argv = [SSH_KEYGEN, "-Y", "verify", "-f", signers_path, "-I", principal, "-n", ns]
        proc = _run([*argv, "-s", sig_path], payload, KEYGEN_TIMEOUT, _tool_env())
    finally:
        _unlink_quiet(sig_path)
    if proc is None or proc.returncode != 0:
        reason = proc.stderr.decode("utf-8", "replace").strip() if proc else "ssh-keygen failed"
        raise refuse("bad_signature", reason)


def _openssh_version() -> str:
    exe = SSH_CLIENT if os.path.exists(SSH_CLIENT) else shutil.which("ssh")
    if not exe:
        return ""
    proc = _run([exe, "-V"], b"", VERSION_TIMEOUT, _tool_env())
    if proc is None:
        return ""
    return _clip((proc.stderr or proc.stdout).decode("utf-8", "replace").strip(), 200)


def parse_assocs(text: str) -> list[dict[str, str]]:
    """Rows of ``sacctmgr -nP ... format=cluster,account,partition,qos`` (capped)."""
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 4:
            continue
        cluster, account, partition, qos = fields
        rows.append({"cluster": cluster, "account": account, "partition": partition, "qos": qos})
        if len(rows) >= ASSOC_CAP:
            break
    return rows


def _assocs(login: str) -> list[dict[str, str]]:
    exe = shutil.which("sacctmgr")
    if not exe:
        return []
    argv = [exe, "-nP", "show", "assoc", f"user={login}", "format=cluster,account,partition,qos"]
    proc = _run(argv, b"", SACCTMGR_TIMEOUT, _scheduler_env())
    if proc is None or proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip() if proc else "did not finish"
        raise GateError("scheduler", _clip(f"sacctmgr: {detail}"), EXIT_SCHEDULER)
    return parse_assocs(proc.stdout.decode("utf-8", "replace"))


def _signer_entry(line: str) -> dict[str, str] | None:
    text = line.strip()
    spans = _token_spans(text) if text and not text.startswith("#") else []
    tokens = [text[start:end] for start, end in spans]
    for index, token in enumerate(tokens[:-1]):
        if token in SIGNER_KEY_TYPES:
            try:
                blob = _signer_blob(token, tokens[index + 1])
            except ValueError:
                return None
            return {"fingerprint": fingerprint(blob), "type": token}
    return None


def read_signers(path: str) -> list[dict[str, str]]:
    """Fingerprints and types of the keys in ``allowed_signers`` (capped)."""
    try:
        text = _read_nofollow(path, 1 << 20).decode("utf-8", "replace")
    except (OSError, ValueError):
        return []
    entries = (_signer_entry(line) for line in split_lines(text))
    return [entry for entry in entries if entry is not None][:SIGNER_CAP]


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------


def _whoami() -> tuple[str, int]:
    import pwd

    uid = os.geteuid()
    try:
        return pwd.getpwuid(uid).pw_name, uid
    except KeyError:
        return str(uid), uid


def cmd_hello(cfg: GateConfig, root: str) -> dict[str, Any]:
    login, uid = _whoami()
    try:
        log_head = read_log_head(os.path.join(root, "log", LOG_NAME))
    except (OSError, ValueError) as exc:
        raise GateError("failed", _clip(f"log: {exc}"), EXIT_FAILED) from None
    return {
        "gate_version": GATE_VERSION,
        "gate_sha256": gate_sha256(),
        "site": cfg.site,
        "login": login,
        "uid": uid,
        "python": platform.python_version(),
        "openssh": _openssh_version(),
        "signers": read_signers(os.path.join(root, "allowed_signers")),
        "uv_required": cfg.uv_required,
        "allow_non_sk": cfg.allow_non_sk,
        "assocs": _assocs(login),
        "log_head": log_head,
        "site_profile_sha256": _site_profile_digest(root),
        "verbs": list(VERBS),
        "checkpoint": checkpoint_text(cfg.site, login, log_head),
    }


def checkpoint_text(site: str, login: str, head: dict[str, Any]) -> str:
    """The C2SP tlog-checkpoint body of the log head, unsigned (plan section
    6 "C/D.0", D5): origin ``umrs-gate/<site>/<login>``, the size ``n`` and
    the base64 of the 32 bytes of ``h``."""
    root_hash = base64.b64encode(bytes.fromhex(str(head["h"]))).decode("ascii")
    return f"umrs-gate/{site}/{login}\n{int(head['n'])}\n{root_hash}\n"


def _site_profile_digest(root: str) -> str:
    """sha256 of ``site.json`` as it is on disk (no validation); "" when absent
    or unreadable (plan B2.0 section 2, "Verify")."""
    try:
        data = _read_nofollow(os.path.join(root, "site.json"), PROFILE_CAP)
    except (OSError, ValueError):
        return ""
    return hashlib.sha256(data).hexdigest()


def _read_capped(stream: IO[bytes]) -> bytes:
    data = stream.read(MAX_STDIN + 1)
    if len(data) > MAX_STDIN:
        raise refuse("too_large", f"request exceeds {MAX_STDIN} bytes")
    return data


def _envelope_fields(data: bytes) -> tuple[Any, Any]:
    try:
        envelope = _loads_no_duplicates(data)
    except ValueError:
        raise refuse("bad_envelope", "the request is not a JSON object") from None
    if not isinstance(envelope, dict) or not set(envelope) <= {"payload", "signature"}:
        raise refuse("bad_envelope", 'expected {"payload", "signature"}')
    return envelope.get("payload"), envelope.get("signature")


def read_envelope(data: bytes) -> tuple[bytes, str]:
    """Envelope ``{"payload": <base64>, "signature": <armored SSHSIG>}``."""
    return _envelope_values(*_envelope_fields(data))


def _envelope_values(payload_b64: Any, signature: Any) -> tuple[bytes, str]:
    if not isinstance(payload_b64, str):
        raise refuse("bad_envelope", "payload must be a base64 string")
    try:
        payload = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError):
        raise refuse("bad_envelope", "payload is not base64") from None
    if not isinstance(signature, str) or not signature.strip():
        raise refuse("not_signed", "the request carries no signature")
    if not signature.isascii():
        raise refuse("bad_envelope", "the signature is not ASCII armor")
    return payload, signature


def _keygen_verified(cfg: GateConfig, root: str, stdin: IO[bytes]) -> tuple[bytes, str]:
    """Steps 1-3: size cap, envelope, ``ssh-keygen -Y verify``. Until this
    returns, the request carries no valid signature (the unsigned stage)."""
    payload, armored = read_envelope(_read_capped(stdin))
    ns = namespace("control", cfg.domain)
    signers = os.path.join(root, "allowed_signers")
    ssh_keygen_verify(signers, cfg.principal, ns, armored, payload, os.path.join(root, "state"))
    return payload, armored


def _signature_policy(cfg: GateConfig, armored: str, kind: str = "control") -> SshSig:
    """Steps 4-5 on a signature ``ssh-keygen`` verified: SSHSIG and flags."""
    ns = namespace(kind, cfg.domain)
    try:
        sig = parse_sshsig(armored)
    except ValueError as exc:
        raise refuse("bad_signature", f"unparsable SSHSIG: {exc}") from None
    if sig.namespace != ns:
        raise refuse("bad_signature", "signature namespace mismatch")
    check_signature_policy(sig, uv_required=cfg.uv_required, allow_non_sk=cfg.allow_non_sk)
    return sig


def _parse_body(payload: bytes) -> dict[str, Any]:
    try:
        return parse_strict(payload)
    except ValueError as exc:
        raise refuse("not_canonical", str(exc)) from None


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256


def _is_time(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_INT


def _is_new_key(value: Any) -> bool:
    match = _NEW_KEY_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return False
    try:
        ed25519_blob(match.group(1))
    except ValueError:
        return False
    return True


_CONTROL_FIELDS: dict[str, Callable[[Any], bool]] = {
    "v": lambda value: type(value) is int and value == 1,
    "type": lambda value: value == "control",
    "op": lambda value: value in CONTROL_VERBS,
    "site": _is_text,
    "principal": _is_text,
    "key_id": _matches(_KEY_ID_RE),
    "nonce": _matches(_NONCE_RE),
    "issued": _is_time,
    "expires": _is_time,
    "new_public_key": _is_new_key,
    "grant_id": _matches(_NONCE_RE),
}


def _control_fields(verb: str) -> set[str]:
    """``new_public_key`` only (and always) for rotate; ``grant-cancel`` names
    a ``grant_id`` and no ``key_id`` (plan section 6 "C/D.0", C3)."""
    expected = set(_CONTROL_FIELDS) - {"new_public_key", "grant_id"}
    if verb == "rotate":
        expected.add("new_public_key")
    if verb == "grant-cancel":
        expected = (expected - {"key_id"}) | {"grant_id"}
    return expected


def validate_control(body: dict[str, Any], verb: str) -> None:
    """Control body v1 schema (``_control_fields``); ``op`` is the verb."""
    expected = _control_fields(verb)
    if set(body) != expected:
        raise refuse("bad_schema", f"control fields must be {sorted(expected)}")
    for key in sorted(expected):
        if not _CONTROL_FIELDS[key](body[key]):
            raise refuse("bad_schema", f"invalid {key}")
    if body["op"] != verb:
        raise refuse("bad_schema", f"op {body['op']} does not match the verb {verb}")


def _check_new_key_site(text: str | None, site: str) -> None:
    match = _NEW_KEY_RE.fullmatch(text) if text else None
    comment = match.group(2) if match else None
    if comment and comment != f"umrs-transport-{site}-tk-{comment[-12:]}":
        raise refuse("wrong_site", "new_public_key names another site")


def _check_window(issued: int, expires: int, now: int) -> None:
    if expires < issued or expires - issued > CONTROL_MAX_WINDOW:
        raise refuse("expired", f"validity window must be 0..{CONTROL_MAX_WINDOW} s")
    if now + CLOCK_SKEW < issued:
        raise refuse("expired", "the control is not valid yet")
    if now - CLOCK_SKEW > expires:
        raise refuse("expired", "the control has expired")


def check_binding(body: dict[str, Any], cfg: GateConfig, now: int) -> None:
    """Site, principal and expiry of a control body (step 8)."""
    if body["site"] != cfg.site:
        raise refuse("wrong_site", f"control is for site {body['site']}")
    if body["principal"] != cfg.principal:
        raise refuse("wrong_principal", "control is for another principal")
    _check_new_key_site(body.get("new_public_key"), cfg.site)
    _check_window(body["issued"], body["expires"], now)


def _new_key_parts(text: str) -> tuple[str, str | None]:
    match = _NEW_KEY_RE.fullmatch(text)
    if match is None:
        raise refuse("bad_schema", "invalid new_public_key")
    return match.group(1), match.group(2)


def _signers_hold(root: str, key_b64: str) -> bool:
    try:
        text = _read_nofollow(os.path.join(root, "allowed_signers"), 1 << 20).decode("utf-8")
    except (OSError, ValueError) as exc:
        raise GateError("failed", _clip(f"allowed_signers: {exc}"), EXIT_FAILED) from None
    return any(_holds_key(line, key_b64) for line in split_lines(text))


def _rotation_key(verb: str, body: dict[str, Any], root: str) -> tuple[str, str | None] | None:
    """The new key of a rotate (never one of the person's signing keys)."""
    if verb != "rotate":
        return None
    new_key = _new_key_parts(body["new_public_key"])
    if _signers_hold(root, new_key[0]):
        raise refuse("bad_schema", "new_public_key is a signing key")
    return new_key


def _verify_and_act(
    verb: str, cfg: GateConfig, home: str, payload: bytes, armored: str
) -> tuple[dict[str, Any], str]:
    """Steps 4-10 on a request whose signature ``ssh-keygen`` verified; the
    answer fields and the log detail.

    ``now`` is read here, after the body arrived: a control held back on an
    open session is judged at the time it is acted on."""
    root = gate_root(home)
    sig = _signature_policy(cfg, armored)
    body = _parse_body(payload)
    validate_control(body, verb)
    now = int(time.time())
    check_binding(body, cfg, now)
    consume_nonce(os.path.join(root, "state", "nonces"), body["nonce"], now)
    signer = sig.signer_fingerprint
    if verb == "grant-cancel":
        return _cancel_grant(root, body["grant_id"], now, signer)
    if verb == "revoke":  # K7: grants issued before a revoke end with it
        write_epoch(root, body["issued"], advance=True)
    comment = transport_comment(cfg.site, body["key_id"])
    lines = edit_authorized_keys(home, comment, _rotation_key(verb, body, root))
    done = {"key_id": body["key_id"], "lines": lines, "signer": signer}
    return done, f"key_id={body['key_id']} lines={lines} signer={signer}"


def _open_log(root: str) -> GateLog:
    log = GateLog(os.path.join(root, "log", LOG_NAME))
    try:
        _ensure_layout(root)
        log.open()
    except (OSError, ValueError) as exc:
        raise refuse("log_unwritable", str(exc)) from None
    return log


def _as_gate_error(exc: Exception) -> GateError:
    if isinstance(exc, GateError):
        return exc
    return GateError("failed", type(exc).__name__, EXIT_FAILED)


def _log_refusal(log: GateLog, verb: str, err: GateError, detail: str) -> None:
    result = "refused" if err.code == EXIT_REFUSED else "failed"
    with contextlib.suppress(OSError, ValueError):
        log.record(verb, result, detail, int(time.time()))


def _log_unsigned(log: GateLog, verb: str, err: GateError) -> None:
    """A request with no valid signature costs the person's quota little: a
    fixed short entry (no tool output), and none once the log is large."""
    try:
        full = log.size() >= LOG_UNSIGNED_CAP
    except OSError:
        return
    if not full:
        _log_refusal(log, verb, err, f"{err.error}: before a valid signature")


def _run_logged(
    verb: str, cfg: GateConfig, home: str, stdin: IO[bytes], log: GateLog
) -> dict[str, Any]:
    try:
        payload, armored = _keygen_verified(cfg, gate_root(home), stdin)
    except Exception as exc:  # every attempt is logged, whatever failed
        err = _as_gate_error(exc)
        _log_unsigned(log, verb, err)
        raise err from None
    try:
        done, detail = _verify_and_act(verb, cfg, home, payload, armored)
    except Exception as exc:
        err = _as_gate_error(exc)
        _log_refusal(log, verb, err, f"{err.error}: {err.detail}")
        raise err from None
    try:
        log.record(verb, "ok", detail, int(time.time()))
    except (OSError, ValueError) as exc:
        raise refuse("log_unwritable", str(exc)) from None
    return {"ok": True, "op": verb, **done, "log_head": log.head()}


def cmd_control(verb: str, cfg: GateConfig, home: str, stdin: IO[bytes]) -> dict[str, Any]:
    """``revoke`` / ``rotate`` / ``grant-cancel``: verify the signed envelope
    in contract order, act, log.

    The log is opened (and locked) first, so a log that cannot be written
    refuses ``log_unwritable`` before anything is read or changed.
    """
    log = _open_log(gate_root(home))
    try:
        return _run_logged(verb, cfg, home, stdin, log)
    finally:
        log.close()


# ===========================================================================
# Wave B2: gated execution with intents (plan section 5, "B2.0 Wire contract")
# ===========================================================================
#
# Shared API (one copy for the gate, the UMRS server and r4; plan B2.0 lane G):
# ``validate_intent``, ``validate_site_profile``, ``site_profile_bytes``,
# ``src_file_entry``, ``render_job_script``, ``input_json``, ``physical_key``,
# ``out_member``, ``step_graph``, ``expected_step_config``, ``canonical_bytes``
# / ``parse_strict`` (``json_values``) and the constants below.

INTENT_MAX_BYTES = 8 << 20
INTENT_MAX_WINDOW = 3600
RUN_MIN_SECONDS = 60
RUN_MAX_SECONDS = 604800
INPUT_JSON_MAX_BYTES = 48 << 20
#: ``remote_slurm_step_executor.REPLY_CAP_BYTES`` and ``workflow_export.MAX_EXPORT_BYTES``.
REPLY_CAP_BYTES = 4 << 20
MAX_EXPORT_BYTES = 200 * 1024 * 1024
MAX_KEY_BYTES = 1024
MAX_SEGMENT_BYTES = 255
MAX_PATH_BYTES = 4096
MAX_STEPS = 64
MAX_INPUT_OBJECTS = 200_000
MAX_INPUT_PREFIXES = 10_000
MAX_SRC_LIST = 200_000
MAX_PARTITIONS = 64
MAX_HEADERS = 32
PROFILE_CAP = 1 << 20
#: K1: a maximal grant envelope (a 2 MiB grant, an 8 MiB run spec and 1 MiB
#: of ``workflow_steps``, base64) is about 14.7 MiB.
SUBMIT_MAX_STDIN = 16 << 20
STAGE_CALLS_MAX = 8
LOG_FULL_BYTES = 64 << 20
UNSIGNED_DETAIL_CAP = 200
SUBMIT_RESERVE = 120
SWEEP_MIN_AGE = 3600
RETENTION_SECONDS = 7 * 86400
USAGE_MAX_JOBS = 100
SLURM_TIMEOUT = 60
SBATCH_TIMEOUT = 90
REMOVE_TIMEOUT = 600
STATE_CAP = 64 << 20
CHUNK = 1 << 20
LOG_HEAD_BYTES = 63 * 1024 * 1024
LOG_TAIL_BYTES = 1024 * 1024
#: Session deadline per verb (plan B2.0 section 5); every other verb: ``GATE_DEADLINE``.
DEADLINES = {
    "submit": 600,
    "stage-in": 7200,
    "collect-out": 7200,
    "rmwd": 1800,
    "sweep": 1800,
    # ssh-keygen, sacctmgr and one sacct call (each capped): more than 60 s.
    "grant-check": 180,
}

INPUT_MODES = ("listed", "server_listed")
ENDPOINT_KINDS = ("vllm", "bentoml")
#: ``workflow_runner._PARAM_RESERVED_KEYS`` and the staged lane's drop
#: (``remote_slurm_step_executor._PLATFORM_CONFIG_KEYS``), plan B2.0 section 1.3.
PARAM_RESERVED_KEYS = frozenset({"api_base", "api_key", "headers", "_routing", "image_param"})
PLATFORM_CONFIG_KEYS = ("api_key", "headers", "_routing")
TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
STEP_CONTRACT_LABEL = "org.umrs.step-contract"
STEP_CONTRACT_VERSION = "1"
STEP_PAYLOAD_LABEL = "org.umrs.step-payload"
WORKDIR_DIRS = ("in", "out", "reply", "src", "log")
#: ``state/intents/<i>/intent.json`` holds the signed intent, so no step may
#: be named ``intent`` (its nonce ``<step>.json`` would be that file).
RESERVED_STEP_NAME = "intent"

STEP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
WD_RE = re.compile(r"[0-9a-f]{16}-[A-Za-z0-9][A-Za-z0-9_-]{0,39}-[0-9a-f]{8}")
_LOGIN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
_PARTITION_RE = _LOGIN_RE
_RUN_ID_RE = re.compile(r"wf-[0-9a-f]{12}")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_KEY_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_TEXT_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_IMAGE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,511}")
_ENTRYPOINT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*"
)
_MEM_RE = re.compile(r"(?:[1-9][0-9]{0,8}[KMGT]?)?")
_HOST_RE = re.compile(r"[A-Za-z0-9.-]{1,253}")
_GRES_RE = re.compile(r"[A-Za-z0-9_:.,-]{0,128}")
#: Profile paths land in TOML strings and shell words: a plain character set.
_PATH_CHARS_RE = re.compile(r"/[A-Za-z0-9._+@%,=~/-]{0,4095}")
_JOB_ID_RE = re.compile(r"[0-9]{1,20}")
_STATE_RE = re.compile(r"[A-Z_]{1,32}")
_BLOB_MEMBER_RE = re.compile(r"in/blobs/([0-9a-f]{64})")
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")
_HEADER_VALUE_RE = re.compile(r"[\x20-\x7e\t]*")
_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _O_NOFOLLOW | _O_CLOEXEC
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW | _O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | _O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0) | _O_CLOEXEC


class ContractError(GateError, ValueError):
    """What the shared B2 validators raise: a gate outcome (``error``,
    ``detail``, ``code``) that a caller outside the gate may also catch as a
    ``ValueError``."""


class _Invalid(ValueError):
    """A schema check failed (internal; mapped to a named outcome)."""


# ---------------------------------------------------------------------------
# A small declarative schema (every check is one closure)
# ---------------------------------------------------------------------------


def _at(where: str, key: Any) -> str:
    return f"{where}.{key}"


def _pred(test: Callable[[Any], bool], what: str) -> Callable[[Any, str], None]:
    def check(value: Any, where: str) -> None:
        if not test(value):
            raise _Invalid(f"{where} {what}")

    return check


def _int_in(low: int, high: int) -> Callable[[Any, str], None]:
    return _pred(lambda v: type(v) is int and low <= v <= high, f"must be an int in {low}..{high}")


def _const(expected: Any) -> Callable[[Any, str], None]:
    return _pred(lambda v: type(v) is type(expected) and v == expected, f"must be {expected!r}")


def _one_of(*options: str) -> Callable[[Any, str], None]:
    return _pred(lambda v: isinstance(v, str) and v in options, f"must be one of {list(options)}")


def _matching(pattern: re.Pattern[str]) -> Callable[[Any, str], None]:
    return _pred(lambda v: isinstance(v, str) and pattern.fullmatch(v) is not None, "is malformed")


def _plain_text(value: Any, low: int, high: int) -> bool:
    if not isinstance(value, str) or not low <= len(value) <= high:
        return False
    if _TEXT_CONTROL_RE.search(value):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _text(low: int, high: int) -> Callable[[Any, str], None]:
    return _pred(lambda v: _plain_text(v, low, high), f"must be {low}..{high} plain characters")


def _check_keys(value: Any, where: str, required: frozenset[str], names: frozenset[str]) -> None:
    if not isinstance(value, dict):
        raise _Invalid(f"{where} must be an object")
    keys = set(value)
    if not required <= keys <= names:
        shown = sorted(str(key) for key in keys)[:40]
        raise _Invalid(
            f"{where} keys must be {sorted(required)} (+{sorted(names - required)}); got {shown}"
        )


def _obj(
    fields: dict[str, Callable[[Any, str], None]], optional: tuple[str, ...] = ()
) -> Callable[[Any, str], None]:
    names = frozenset(fields)
    required = names - frozenset(optional)

    def check(value: Any, where: str) -> None:
        _check_keys(value, where, required, names)
        for key in sorted(value):
            fields[key](value[key], _at(where, key))

    return check


def _strictly_increasing(keys: list[Any], where: str) -> None:
    for index in range(1, len(keys)):
        if not keys[index - 1] < keys[index]:
            raise _Invalid(f"{where} must be strictly increasing (no duplicates)")


def _list_of(
    item: Callable[[Any, str], None],
    low: int,
    high: int,
    order: Callable[[Any], Any] | None = None,
) -> Callable[[Any, str], None]:
    def check(value: Any, where: str) -> None:
        if not isinstance(value, list) or not low <= len(value) <= high:
            raise _Invalid(f"{where} must be a list of {low}..{high} entries")
        for index, entry in enumerate(value):
            item(entry, f"{where}[{index}]")
        if order is not None:
            _strictly_increasing([order(entry) for entry in value], where)

    return check


def _map_of(
    key: Callable[[Any, str], None], item: Callable[[Any, str], None], high: int
) -> Callable[[Any, str], None]:
    def check(value: Any, where: str) -> None:
        if not isinstance(value, dict) or len(value) > high:
            raise _Invalid(f"{where} must be an object of at most {high} entries")
        for name, entry in value.items():
            key(name, f"{where} key")
            item(entry, _at(where, name))

    return check


def _json_object(value: Any, where: str) -> None:
    """An object of JSON values (B2.0 section 1.4: floats and null allowed)."""
    if not isinstance(value, dict):
        raise _Invalid(f"{where} must be an object")
    try:
        _check_value(value, 0, True)
    except ValueError as exc:
        raise _Invalid(f"{where}: {exc}") from None


def _identity(value: Any) -> Any:
    return value


_BOOL = _pred(lambda v: type(v) is bool, "must be a bool")
_UINT = _int_in(0, MAX_INT)
_HEX64 = _matching(_HEX64_RE)
_STEP_NAME = _matching(STEP_RE)


# ---------------------------------------------------------------------------
# Keys, buckets and output members (staged_s3 / step_staging rules)
# ---------------------------------------------------------------------------


def _segment_ok(segment: str) -> bool:
    return segment not in ("", ".", "..") and len(segment.encode("utf-8")) <= MAX_SEGMENT_BYTES


def is_normal_key(value: Any) -> bool:
    """True when ``value`` is a fixed point of ``staged_s3.normalize_key``.

    No leading ``/``, no empty, ``.`` or ``..`` segment, no control
    character, 1..1024 UTF-8 bytes, segments of at most 255 bytes. Nothing is
    normalized on the way: ``a//b`` is not a key (B2.0 section 1)."""
    if not isinstance(value, str) or not value or _KEY_CONTROL_RE.search(value):
        return False
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    return size <= MAX_KEY_BYTES and all(_segment_ok(seg) for seg in value.split("/"))


def is_bucket_name(value: Any) -> bool:
    """``staged_s3.check_bucket_name`` as a predicate."""
    if not isinstance(value, str) or _BUCKET_RE.fullmatch(value) is None:
        return False
    return ".." not in value and value != "manifest.json"


def _is_prefix_ref(value: Any) -> bool:
    """``<bucket>/<prefix>``: the prefix is empty, a key, or a key plus ``/``."""
    if not isinstance(value, str):
        return False
    bucket, sep, prefix = value.partition("/")
    if not sep or not is_bucket_name(bucket):
        return False
    if not prefix:
        return True
    return is_normal_key(prefix[:-1] if prefix.endswith("/") else prefix)


_KEY = _pred(is_normal_key, "is not a normalize_key fixed point")
_BUCKET = _pred(is_bucket_name, "is not a bucket name")
_PREFIX = _pred(_is_prefix_ref, "is not <bucket>/<prefix>")


def physical_key(run_id: str, step: str, key: str) -> str:
    """``runs/<run_id>/<step>/<key>`` (``step_staging.physical_prefix`` + key)."""
    _require(_RUN_ID_RE, run_id, "run_id")
    _require(STEP_RE, step, "step")
    if not is_normal_key(key):
        raise ValueError(f"not an object key: {key!r}")
    return f"runs/{run_id}/{step}/{key}"


def out_member(raw: Any, output_bucket: str) -> str | None:
    """The ``out`` member a reply's ``output_keys`` entry names, or None.

    As ``step_staging._physical_output_key``: ``s3://<b>/<k>`` names bucket
    ``b``, a bare key names ``output_bucket``; the bucket must be
    ``output_bucket`` and the key a ``normalize_key`` fixed point. The member
    is ``<output_bucket>/<k>``."""
    if not isinstance(raw, str):
        return None
    bucket, key = output_bucket, raw
    if raw.startswith("s3://"):
        bucket, _, key = raw[len("s3://") :].partition("/")
    if bucket != output_bucket or not is_bucket_name(bucket) or not is_normal_key(key):
        return None
    return f"{bucket}/{key}"


def _rewritten_key(raw: str, run_id: str, step: str, output_bucket: str) -> str:
    """A parent's output key as its physical key; the ``s3://`` form is kept
    (``step_staging.rewrite_output_keys``)."""
    member = out_member(raw, output_bucket)
    if member is None:
        raise ContractError(
            "parent_not_collected", _clip(f"step {step} output key {raw!r} names no output")
        )
    physical = physical_key(run_id, step, member[len(output_bucket) + 1 :])
    return f"s3://{output_bucket}/{physical}" if raw.startswith("s3://") else physical


def reply_output_keys(reply: Any, step: str) -> list[str]:
    """``result.output_keys`` of a recorded ``ok`` reply (``null`` is empty)."""
    result = reply.get("result") if isinstance(reply, dict) else None
    if not isinstance(result, dict):
        raise ContractError("parent_not_collected", f"step {step} has no result")
    keys = result.get("output_keys")
    if keys is None:
        return []
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise ContractError("parent_not_collected", f"step {step} output_keys are not strings")
    return list(keys)


# ---------------------------------------------------------------------------
# Intent body v1 (B2.0 section 1)
# ---------------------------------------------------------------------------

_IMAGE = _obj({"ref": _matching(_IMAGE_REF_RE), "image_id": _HEX64, "archive_sha256": _HEX64})
_OBJECT = _obj(
    {
        "mode": _one_of(*INPUT_MODES),
        "bucket": _BUCKET,
        "key": _KEY,
        "sha256": _HEX64,
        "size": _UINT,
    }
)
_INPUTS = _obj(
    {
        "objects": _list_of(_OBJECT, 0, MAX_INPUT_OBJECTS, lambda o: (o["bucket"], o["key"])),
        "prefixes": _list_of(_PREFIX, 0, MAX_INPUT_PREFIXES, _identity),
        "chained": _list_of(_STEP_NAME, 0, MAX_STEPS - 1),
    }
)
_ENDPOINT = _obj(
    {"host": _matching(_HOST_RE), "port": _int_in(1, 65535), "kind": _one_of(*ENDPOINT_KINDS)}
)
_STEP = _obj(
    {
        "name": _STEP_NAME,
        "after": _list_of(_STEP_NAME, 0, MAX_STEPS - 1, _identity),
        "config": _json_object,
        "image": _IMAGE,
        "partition": _matching(_PARTITION_RE),
        "cpus": _int_in(1, 1 << 20),
        "mem": _matching(_MEM_RE),
        "gpus": _int_in(0, 1 << 16),
        "time": _int_in(RUN_MIN_SECONDS, RUN_MAX_SECONDS),
        "inputs": _INPUTS,
        "endpoint": _ENDPOINT,
    },
    optional=("endpoint",),
)
_SRC_FILE = _obj({"path": _KEY, "sha256": _HEX64, "size": _UINT})
_ENTRYPOINT = _pred(
    lambda v: isinstance(v, str) and len(v) <= 300 and _ENTRYPOINT_RE.fullmatch(v) is not None,
    "is not a module:Class entrypoint",
)
_SRC_MANIFEST = _obj(
    {
        "files": _list_of(_SRC_FILE, 0, MAX_SRC_LIST, lambda f: f["path"]),
        "entrypoints": _map_of(_STEP_NAME, _ENTRYPOINT, MAX_STEPS),
    }
)
_INTENT = _obj(
    {
        "v": _const(1),
        "type": _const("intent"),
        "site": _matching(_SITE_RE),
        "principal": _matching(_PRINCIPAL_RE),
        "login": _matching(_LOGIN_RE),
        "account": _matching(_LOGIN_RE),
        "project": _text(1, 128),
        "workflow": _text(1, 128),
        "version": _text(1, 64),
        "source_sha256": _HEX64,
        "run_id": _matching(_RUN_ID_RE),
        "input_bucket": _BUCKET,
        "output_bucket": _BUCKET,
        "params": _json_object,
        "site_profile_sha256": _HEX64,
        "trusted_image": _IMAGE,
        "src_manifest": _SRC_MANIFEST,
        "steps": _list_of(_STEP, 1, MAX_STEPS),
        "nonce": _matching(_NONCE_RE),
        "issued": _int_in(1, MAX_INT),
        "expires": _int_in(1, MAX_INT),
        "max_run_seconds": _int_in(RUN_MIN_SECONDS, RUN_MAX_SECONDS),
    }
)


def _src_rules(files: list[dict[str, Any]]) -> None:
    """The manifest sums to at most ``MAX_EXPORT_BYTES`` and is a tree: no
    path is a directory prefix of another (``a`` and ``a/b``)."""
    if sum(entry["size"] for entry in files) > MAX_EXPORT_BYTES:
        raise _Invalid(f"intent.src_manifest.files sum to more than {MAX_EXPORT_BYTES} bytes")
    paths = {entry["path"] for entry in files}
    for path in paths:
        parts = path.split("/")
        for depth in range(1, len(parts)):
            if "/".join(parts[:depth]) in paths:
                raise _Invalid(f"intent.src_manifest.files: {path!r} is below a file")


def _step_graph_rules(step: dict[str, Any], after: dict[str, set[str]]) -> None:
    name, mine = step["name"], after[step["name"]]
    if name in mine or not mine <= set(after):
        raise _Invalid(f"step {name}: after names itself or an unknown step")
    if any(not after[parent] <= mine for parent in mine):
        raise _Invalid(f"step {name}: after is not transitively closed")
    chained = step["inputs"]["chained"]
    if len(set(chained)) != len(chained) or not set(chained) <= mine:
        raise _Invalid(f"step {name}: chained must be distinct names from after")


def _intent_rules(body: dict[str, Any]) -> None:
    window = body["expires"] - body["issued"]
    if not 0 < window <= INTENT_MAX_WINDOW:
        raise _Invalid(f"intent: expires - issued must be in 1..{INTENT_MAX_WINDOW}")
    names = [step["name"] for step in body["steps"]]
    if len(set(names)) != len(names) or RESERVED_STEP_NAME in names:
        raise _Invalid(f"intent.steps: names must be unique and not {RESERVED_STEP_NAME!r}")
    if set(body["src_manifest"]["entrypoints"]) != set(names):
        raise _Invalid("intent.src_manifest.entrypoints must name each step once")
    _src_rules(body["src_manifest"]["files"])
    after = {step["name"]: set(step["after"]) for step in body["steps"]}
    for step in body["steps"]:
        _step_graph_rules(step, after)


def validate_intent(body: Any) -> dict[str, Any]:
    """Check an intent body v1 (B2.0 section 1); returns it.

    Every key is required, unknown keys refuse, ints are ``type(v) is int``,
    floats and ``null`` only inside ``params`` and ``steps[].config``. Raises
    ``ContractError`` ``bad_schema`` (also a ``ValueError``)."""
    try:
        _INTENT(body, "intent")
        _intent_rules(body)
    except _Invalid as exc:
        raise ContractError("bad_schema", _clip(str(exc)), EXIT_REFUSED) from None
    return body


def _step_named(intent: dict[str, Any], step: Any) -> dict[str, Any]:
    """The step object of ``intent`` named ``step`` (a name or the object)."""
    name = step.get("name") if isinstance(step, dict) else step
    for entry in intent["steps"]:
        if entry["name"] == name:
            return entry
    raise ContractError("wrong_step", _clip(f"the intent has no step {name!r}"))


# ---------------------------------------------------------------------------
# Graph and config (B2.0 section 1.3; DAGScheduler + _assemble_step_config)
# ---------------------------------------------------------------------------


def _graph_name(raw: Any, index: int) -> str:
    if not isinstance(raw, dict):
        raise ValueError(f"workflow step {index} is not an object")
    name = raw.get("name") or f"step_{index}"
    if not isinstance(name, str):
        raise ValueError(f"workflow step {index} name is not a string")
    return name


def _declared_parents(raw: dict[str, Any]) -> list[str] | None:
    deps = raw.get("depends_on")
    if deps is None:
        return None
    if not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps):
        raise ValueError("depends_on must be a list of step names")
    return list(dict.fromkeys(deps))


def _graph_parents(names: list[str], declared: list[list[str] | None]) -> dict[str, set[str]]:
    """DAGScheduler's rule: no ``depends_on`` anywhere chains each step to the
    one before; otherwise parents are the declared ones (None and [] are roots)."""
    if all(deps is None for deps in declared):
        return {name: ({names[index - 1]} if index else set()) for index, name in enumerate(names)}
    known = set(names)
    parents: dict[str, set[str]] = {}
    for index, name in enumerate(names):
        deps = declared[index]
        unknown = set(deps or []) - known
        if unknown:
            raise ValueError(f"step {name!r} depends on unknown step(s) {sorted(unknown)}")
        parents[name] = set(deps or [])
    return parents


def _closure(name: str, parents: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    todo = list(parents[name])
    while todo:
        parent = todo.pop()
        if parent not in seen:
            seen.add(parent)
            todo.extend(parents[parent])
    return seen


def step_graph(workflow_steps: Any) -> list[tuple[str, list[str], list[str]]]:
    """Per ``workflow.json`` step, in file order: ``(name, after, chained)``.

    ``name`` is the step's ``name``, else ``step_<index>``; ``after`` is the
    sorted transitive parent closure (``workflow_runner._upstream_closure``);
    ``chained`` the declared ``depends_on`` in order without duplicates
    (``[]`` when absent). An unknown name, a duplicate name or a cycle raises
    ``ValueError``."""
    if not isinstance(workflow_steps, list):
        raise ValueError("workflow steps must be a list")
    names = [_graph_name(raw, index) for index, raw in enumerate(workflow_steps)]
    if len(set(names)) != len(names):
        raise ValueError("workflow step names are not unique")
    declared = [_declared_parents(raw) for raw in workflow_steps]
    closures = _acyclic_closures(names, _graph_parents(names, declared))
    return [
        (name, sorted(closures[name]), list(declared[index] or []))
        for index, name in enumerate(names)
    ]


def _acyclic_closures(names: list[str], parents: dict[str, set[str]]) -> dict[str, set[str]]:
    """Each step's transitive parents; a step among its own is a cycle."""
    closures = {name: _closure(name, parents) for name in names}
    cyclic = sorted(name for name in names if name in closures[name])
    if cyclic:
        raise ValueError(f"cycle detected involving steps: {cyclic}")
    return closures


def expected_step_config(
    workflow_step: dict[str, Any], name: str, params: dict[str, Any]
) -> dict[str, Any]:
    """``_assemble_step_config`` followed by the staged lane's drop of
    ``api_key``, ``headers`` and ``_routing`` (B2.0 section 1.3)."""
    config = copy.deepcopy(dict(workflow_step.get("config") or {}))
    params = dict(params or {})
    config.setdefault("step_name", name)
    config.setdefault("_workflow_params", copy.deepcopy(params))
    for key, value in params.items():
        if key not in PARAM_RESERVED_KEYS:
            config.setdefault(key, copy.deepcopy(value))
    for key in PLATFORM_CONFIG_KEYS:
        config.pop(key, None)
    return config


def src_file_entry(path: str, data: bytes) -> dict[str, Any]:
    """One ``src_manifest.files`` entry ``{path, sha256, size}`` of ``data``."""
    if not is_normal_key(path):
        raise ValueError(f"source path {path!r} is not a normalize_key fixed point")
    blob = bytes(data)
    return {"path": path, "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)}


# ---------------------------------------------------------------------------
# Site profile ~/.umrs/gate/site.json (B2.0 section 2)
# ---------------------------------------------------------------------------

CAP_KEYS = (
    "max_object_bytes",
    "max_in_bytes",
    "max_in_objects",
    "max_src_bytes",
    "max_src_files",
    "max_out_files",
    "max_out_bytes",
    "max_out_file_bytes",
    "min_free_bytes",
    "reply_bytes",
    "log_tail_bytes",
    "max_workdirs",
)
ENDPOINT_CAP_KEYS = ("max_request_bytes", "max_response_bytes", "max_total_bytes")


def _is_abs_path(value: Any) -> bool:
    """Absolute, ``normpath``-equal, plain characters (no ``:``, quote, space or
    control character), at most 4096 bytes."""
    if not isinstance(value, str) or _PATH_CHARS_RE.fullmatch(value) is None:
        return False
    return posixpath.normpath(value) == value and not value.startswith("//")


def _is_slurm_path(value: Any) -> bool:
    return isinstance(value, str) and all(_is_abs_path(part) for part in value.split(":"))


_PATH = _pred(_is_abs_path, "must be a plain absolute normalized path")
_OPT_PATH = _pred(lambda v: v == "" or _is_abs_path(v), "must be empty or an absolute path")
_PARTITION = _obj(
    {
        "arch": _one_of("x86_64", "aarch64"),
        "max_cpus": _UINT,
        "max_gpus": _UINT,
        "max_time": _UINT,
        "mem": _BOOL,
        "gpu_request": _one_of("", "none", "gres"),
        "gres": _matching(_GRES_RE),
        "gpu_flag": _one_of("gres", "gpus"),
        "exclusive": _BOOL,
    }
)
_PROFILE = _obj(
    {
        "v": _const(1),
        "site": _matching(_SITE_RE),
        "domain": _matching(_DOMAIN_RE),
        "slurm": _obj(
            {
                "path": _pred(_is_slurm_path, "must be a colon list of absolute directories"),
                "owner_uids": _list_of(_UINT, 0, 64),
                "queue_seconds": _UINT,
                "grace_seconds": _UINT,
            }
        ),
        "partitions": _map_of(_matching(_PARTITION_RE), _PARTITION, MAX_PARTITIONS),
        "gpu": _obj({"enabled": _BOOL, "form": _one_of("cdi", "rocm", "nvidia-devices")}),
        "podman": _obj(
            {
                "pids_limit": _UINT,
                "cgroups": _one_of("enabled", "disabled"),
                "nproc_limit": _UINT,
            }
        ),
        "venue": _obj(
            {
                "root": _OPT_PATH,
                "storage_root": _OPT_PATH,
                "network_config_dir": _OPT_PATH,
                "network_backend": _one_of("", "netavark", "cni"),
            }
        ),
        "image_archive_dir": _PATH,
        "caps": _obj(dict.fromkeys(CAP_KEYS, _UINT)),
        "endpoint": _obj(dict.fromkeys(ENDPOINT_CAP_KEYS, _UINT)),
        "watch_seconds": _int_in(1, MAX_INT),
    }
)


def _profile_rules(obj: dict[str, Any]) -> None:
    caps = obj["caps"]
    if caps["reply_bytes"] > REPLY_CAP_BYTES:
        raise _Invalid(f"profile.caps.reply_bytes must be at most {REPLY_CAP_BYTES}")
    if caps["max_src_bytes"] > MAX_EXPORT_BYTES:
        raise _Invalid(f"profile.caps.max_src_bytes must be at most {MAX_EXPORT_BYTES}")


def validate_site_profile(obj: Any) -> dict[str, Any]:
    """Check a site profile (B2.0 section 2); returns it. Raises
    ``ContractError`` ``failed`` (exit 1 in the gate; also a ``ValueError``)."""
    try:
        _PROFILE(obj, "profile")
        _profile_rules(obj)
    except _Invalid as exc:
        raise ContractError("failed", _clip(f"site profile: {exc}"), EXIT_FAILED) from None
    return obj


def site_profile_bytes(obj: Any) -> bytes:
    """The canonical bytes ``site.json`` holds (their sha256 is the profile sha)."""
    return canonical_bytes(validate_site_profile(obj))


# ---------------------------------------------------------------------------
# input.json and in/manifest.json (B2.0 sections 7 and 8)
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> bytes:
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def _parent_result(intent: dict[str, Any], parent: str, reply: Any) -> dict[str, Any]:
    """A parent's recorded ``result`` with ``output_keys`` made physical."""
    if not isinstance(reply, dict) or reply.get("status") != "ok":
        raise ContractError("parent_not_collected", f"step {parent} has no ok reply")
    keys = reply_output_keys(reply, parent)
    run_id, bucket = intent["run_id"], intent["output_bucket"]
    result = dict(reply["result"])
    result["output_keys"] = [_rewritten_key(raw, run_id, parent, bucket) for raw in keys]
    return result


def input_json(intent: dict[str, Any], step: Any, parents: dict[str, Any]) -> bytes:
    """The bytes of ``input.json`` for ``step`` (a name or the step object).

    ``parents`` maps every name in the step's ``after`` to its recorded reply
    (the parsed ``<p>.reply.json``). Raises ``ContractError`` with
    ``parent_not_collected`` or ``too_large`` (above ``INPUT_JSON_MAX_BYTES``)."""
    sobj = _step_named(intent, step)
    results = {p: _parent_result(intent, p, parents.get(p)) for p in sobj["after"]}
    input_keys = [key for p in sobj["inputs"]["chained"] for key in results[p]["output_keys"]]
    bucket = intent["output_bucket"]
    doc = {
        "v": 1,
        "entrypoint": intent["src_manifest"]["entrypoints"][sobj["name"]],
        "config": sobj["config"],
        "parent_results": results,
        "ctx": {
            "run_tag": intent["run_id"],
            "step": sobj["name"],
            "attempt": 1,
            "input_bucket": intent["input_bucket"],
            "output_bucket": bucket,
            "parent_output_bucket": bucket if sobj["after"] else "",
            "input_keys": input_keys,
        },
    }
    data = _json_dumps(doc)
    if len(data) > INPUT_JSON_MAX_BYTES:
        raise ContractError("too_large", f"input.json is {len(data)} bytes")
    return data


def _manifest_entry(sha: str, size: int) -> dict[str, Any]:
    return {"path": f"blobs/{sha}", "size": size, "etag": "", "content_type": ""}


def _chained_objects(intent: dict[str, Any], parent: str, reply: Any, out: Any) -> dict[str, Any]:
    """A chained parent's outputs keyed by their physical keys."""
    bucket, run_id = intent["output_bucket"], intent["run_id"]
    objects: dict[str, Any] = {}
    for raw in reply_output_keys(reply, parent):
        member = out_member(raw, bucket)
        if member is None or not isinstance(out, dict) or member not in out:
            raise ContractError(
                "parent_not_collected", _clip(f"{parent} output {raw!r} was not collected")
            )
        sha, size = out[member]
        key = physical_key(run_id, parent, member[len(bucket) + 1 :])
        objects[f"{bucket}/{key}"] = _manifest_entry(sha, int(size))
    return objects


def manifest_json(
    intent: dict[str, Any], step: Any, parents: dict[str, Any], outs: dict[str, Any]
) -> bytes:
    """``in/manifest.json``: the signed objects plus each chained parent's
    outputs under their physical keys (``outs``: the recorded ``out`` maps)."""
    sobj = _step_named(intent, step)
    bucket = intent["output_bucket"]
    objects = {
        f"{o['bucket']}/{o['key']}": _manifest_entry(o["sha256"], o["size"])
        for o in sobj["inputs"]["objects"]
    }
    for parent in sobj["inputs"]["chained"]:
        objects.update(_chained_objects(intent, parent, parents.get(parent), outs.get(parent)))
    doc = {"objects": objects, "prefixes": sobj["inputs"]["prefixes"], "output_bucket": bucket}
    return _json_dumps(doc)


# ---------------------------------------------------------------------------
# Grants (waves C and D): the shared API (plan section 6 "C/D.0", C1-C3)
# ---------------------------------------------------------------------------
#
# ``GRANT_MAX_BYTES``, ``GRANT_MAX_WINDOW``, ``validate_grant``,
# ``grant_param_schema`` (+ ``merge_grant_param_schemas``), ``params_problems``,
# ``workflow_steps_skeleton``, ``steps_sha256``, ``src_manifest_sha256``,
# ``grant_covers``, ``grant_run_key``, ``grant_cost`` (+ ``grant_wall``),
# ``grant_control_body``, ``checkpoint_text``, ``GRANT_CHECK_MAX_STDIN`` and
# ``SUBMIT_MAX_STDIN``: one copy for the gate, the UMRS server and r4.

GRANT_MAX_BYTES = 2 << 20
GRANT_MAX_WINDOW = 30 * 86400
GRANT_CHECK_MAX_STDIN = 4 << 20
WORKFLOW_STEPS_MAX_BYTES = 1 << 20
#: K8: a cancel record outlives every grant issued before it.
GRANT_CANCEL_RETENTION = 400 * 86400
GRANT_INPUT_MODES = ("listed", "chained", "any")
GRANT_STATES = ("active", "not_yet_valid", "expired", "cancelled")
PARAM_TYPES = ("string", "integer", "number", "boolean")
MAX_GRANT_PARAMS = 128
MAX_LISTED_INPUTS = 10_000
MAX_INPUT_FILES = 10_000
MAX_INPUT_FILE_BYTES = 1100
PARAM_MAX_LENGTH = 65_536
PARAM_ENUM_MAX = 64
#: K9c: what r4 puts on a ``string`` parameter the workflow leaves unbounded.
PARAM_DEFAULT_MAX_LENGTH = 1024
#: Injected by the platform, checked by ``params_problems``, never declared.
PARAM_INJECTED = ("project_id", "input_files", "mcp_tools")
#: K9b: never a property of a grant's ``param_schema``.
GRANT_PARAM_FORBIDDEN = PARAM_RESERVED_KEYS | {*PARAM_INJECTED, "step_name", "_workflow_params"}
_PARAM_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
#: C1 "r4 building the schema": a parameter that could carry code or a place.
#: The name is matched after :func:`_snake_name` (camelCase splits into
#: words), a word may be plural, and a few compound words count too
#: (K9b, review 2026-11-09: ``scriptPath``, ``commands`` and ``hostname``
#: passed before).
_CODE_PARAM_RE = re.compile(
    r"(?i)(^|_)(code|script|source|cmd|command|exec|shell|eval|path|file|filename|dir"
    r"|directory|url|uri|endpoint|host|prompt|template|completion|response|model_?output"
    r"|entry_?point|module|import|plugin|file_?path|host_?name|cmd_?line|path_?name|dir_?name"
    r"|call_?back|snippet|program|binary|executable|api_?base|base_?url|image_?param)(e?s)?(_|$)"
)
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_CODE_PARAM_NAMES = frozenset({"api_base", "base_url", "image_param"})
_DECLARED_PARAM_KEYS = frozenset(
    {"type", "enum", "minimum", "maximum", "maxLength", "description", "title", "default"}
)
#: Per type: the keywords a granted property must carry and may carry.
_PARAM_KEYS = {
    "string": (frozenset({"type", "maxLength"}), frozenset({"type", "enum", "maxLength"})),
    "integer": (frozenset({"type"}), frozenset({"type", "enum", "minimum", "maximum"})),
    "number": (frozenset({"type"}), frozenset({"type", "enum", "minimum", "maximum"})),
    "boolean": (frozenset({"type"}), frozenset({"type", "enum"})),
}
_SKELETON_KEYS = ("name", "depends_on", "config")
_GRANT_RUN_PREFIX = b"umrs-grant-run-v1\n"


def _utf8(value: Any) -> bytes:
    """Sort key of a "strictly increasing" grant list (UTF-8 bytes)."""
    return str(value).encode("utf-8")


def _param_name_ok(name: Any) -> bool:
    if not isinstance(name, str) or _PARAM_NAME_RE.fullmatch(name) is None:
        return False
    return name not in GRANT_PARAM_FORBIDDEN


_PARAM_NAME = _pred(_param_name_ok, "is not a grantable parameter name")


def _param_type_ok(kind: str, value: Any) -> bool:
    """A value a grant may carry for ``kind`` (an ``enum`` entry: no floats)."""
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return type(value) is bool
    return type(value) is int


def _param_bounds(prop: dict[str, Any], where: str) -> None:
    if "maxLength" in prop:
        _int_in(1, PARAM_MAX_LENGTH)(prop["maxLength"], f"{where}.maxLength")
    for key in ("minimum", "maximum"):
        if key in prop:
            _int_in(-MAX_INT, MAX_INT)(prop[key], f"{where}.{key}")
    if prop.get("minimum", -MAX_INT) > prop.get("maximum", MAX_INT):
        raise _Invalid(f"{where}: minimum must be at most maximum")


def _param_enum(prop: dict[str, Any], kind: str, where: str) -> None:
    if "enum" not in prop:
        return
    values = prop["enum"]
    if not isinstance(values, list) or not 1 <= len(values) <= PARAM_ENUM_MAX:
        raise _Invalid(f"{where}.enum must be a list of 1..{PARAM_ENUM_MAX} values")
    limit = prop.get("maxLength", PARAM_MAX_LENGTH)
    for value in values:
        if not _param_type_ok(kind, value) or (isinstance(value, str) and len(value) > limit):
            raise _Invalid(f"{where}.enum holds a value that is not a {kind} within maxLength")


def _param_prop(prop: Any, where: str) -> None:
    """One ``param_schema`` property: ``type`` plus the keywords of its type."""
    kind = prop.get("type") if isinstance(prop, dict) else None
    if not isinstance(kind, str) or kind not in _PARAM_KEYS:
        raise _Invalid(f"{where}.type must be one of {list(PARAM_TYPES)}")
    required, allowed = _PARAM_KEYS[kind]
    _check_keys(prop, where, required, allowed)
    _param_bounds(prop, where)
    _param_enum(prop, kind, where)


def _param_schema(value: Any, where: str) -> None:
    """The JSON-Schema subset of C1: a closed object of typed properties."""
    names = frozenset({"type", "additionalProperties", "required", "properties"})
    _check_keys(value, where, names, names)
    if value["type"] != "object" or value["additionalProperties"] is not False:
        raise _Invalid(f"{where} must be a closed object schema")
    _map_of(_PARAM_NAME, _param_prop, MAX_GRANT_PARAMS)(value["properties"], f"{where}.properties")
    _list_of(_PARAM_NAME, 0, MAX_GRANT_PARAMS, _utf8)(value["required"], f"{where}.required")
    if not set(value["required"]) <= set(value["properties"]):
        raise _Invalid(f"{where}.required names an undeclared property")


_GRANT_WORKFLOW = _obj(
    {
        "name": _text(1, 128),
        "version": _text(1, 64),
        "src_manifest_sha256": _HEX64,
        "steps_sha256": _HEX64,
        "params": _list_of(_PARAM_NAME, 0, MAX_GRANT_PARAMS, _utf8),
    }
)
_LISTED = _obj({"sha256": _HEX64, "size": _UINT})
_GRANT = _obj(
    {
        "v": _const(1),
        "type": _const("grant"),
        "grant_id": _matching(_NONCE_RE),
        "site": _matching(_SITE_RE),
        "login": _matching(_LOGIN_RE),
        "principal": _matching(_PRINCIPAL_RE),
        "accounts": _list_of(_matching(_LOGIN_RE), 1, 16, _utf8),
        "projects": _list_of(_text(1, 128), 1, 16, _utf8),
        "workflows": _list_of(
            _GRANT_WORKFLOW, 1, 16, lambda w: (_utf8(w["name"]), _utf8(w["version"]))
        ),
        "images": _list_of(_IMAGE, 1, 64, lambda i: i["image_id"]),
        "trusted_images": _list_of(_IMAGE, 1, 64, lambda i: i["image_id"]),
        "param_schema": _param_schema,
        "inputs": _obj(
            {
                "mode": _one_of(*GRANT_INPUT_MODES),
                "listed": _list_of(_LISTED, 0, MAX_LISTED_INPUTS, lambda e: e["sha256"]),
            }
        ),
        "caps": _obj(
            {
                "max_jobs": _int_in(1, 10_000),
                "max_gpu_hours": _int_in(0, 1_000_000),
                "max_core_hours": _int_in(0, 10_000_000),
                "max_concurrent": _int_in(1, 64),
            }
        ),
        "not_before": _int_in(1, MAX_INT),
        "expires": _int_in(1, MAX_INT),
        "issued": _int_in(1, MAX_INT),
        "delegable": _const(False),
        "nonce": _matching(_NONCE_RE),
    },
    ("trusted_images",),
)


def _trusted_image_rules(body: dict[str, Any]) -> None:
    if any(image not in body["images"] for image in body.get("trusted_images", [])):
        raise _Invalid("grant.trusted_images: every helper image must also be listed in images")


def _grant_rules(body: dict[str, Any]) -> None:
    if not body["issued"] <= body["not_before"] < body["expires"]:
        raise _Invalid("grant: issued <= not_before < expires must hold")
    if body["expires"] - body["issued"] > GRANT_MAX_WINDOW:
        raise _Invalid(f"grant: expires - issued must be at most {GRANT_MAX_WINDOW}")
    mode, listed = body["inputs"]["mode"], body["inputs"]["listed"]
    if (mode == "listed" and not listed) or (mode == "any" and listed):
        raise _Invalid(f"grant.inputs: mode {mode} with {len(listed)} listed inputs")
    declared = set(body["param_schema"]["properties"])
    for entry in body["workflows"]:
        if not set(entry["params"]) <= declared:
            raise _Invalid(f"grant.workflows: {entry['name']} lists an undeclared parameter")
    _trusted_image_rules(body)


def validate_grant(body: Any) -> dict[str, Any]:
    """Check a grant body v1 (C1); returns it. Unknown keys refuse, and only
    ``trusted_images`` is optional (its absence keeps legacy combined-role
    coverage). Ints are ``type(v) is int``; no floats or ``null`` are allowed."""
    try:
        _check_value(body, 0, False)
        _GRANT(body, "grant")
        _grant_rules(body)
    except _Invalid as exc:
        raise ContractError("bad_schema", _clip(str(exc))) from None
    except ValueError as exc:  # a float, null or big int somewhere
        raise ContractError("bad_schema", _clip(f"grant: {exc}")) from None
    checked: dict[str, Any] = body
    return checked


# -- r4 building a grant's param_schema (C1, K9) ----------------------------


def _declared_schema(workflow_json: Any) -> dict[str, Any]:
    """The workflow's parameter schema, the list form normalized as
    ``api/workflows._workflow_parameter_schema`` does."""
    raw = workflow_json if isinstance(workflow_json, dict) else {}
    for key in ("parameter_schema", "parameters_schema", "parameters"):
        candidate = raw.get(key)
        if isinstance(candidate, dict) and candidate.get("type"):
            return candidate
        if isinstance(candidate, list):
            return _schema_from_items(candidate)
    return {"type": "object", "properties": {}}


def _schema_from_items(items: list[Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name:
            continue
        properties[name] = {k: v for k, v in item.items() if k not in ("name", "required")}
        required += [name] if item.get("required") else []
    return {"type": "object", "properties": properties, "required": required}


def _snake_name(name: str) -> str:
    """``scriptPath`` -> ``script_Path``, ``HTTPUrl`` -> ``HTTP_Url``."""
    return _CAMEL_RE.sub("_", name)


def _code_like(name: str) -> bool:
    return _CODE_PARAM_RE.search(_snake_name(name).lower()) is not None


def _grant_name(name: Any) -> str:
    if not isinstance(name, str) or _PARAM_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"parameter {str(name)[:64]!r}: the name cannot be granted")
    normalized = _snake_name(name).lower()
    if name in GRANT_PARAM_FORBIDDEN or normalized in _CODE_PARAM_NAMES or _code_like(name):
        raise ValueError(f"parameter {name}: the name may carry code or a place (not grantable)")
    return name


def _bound(name: str, key: str, value: Any) -> int:
    """K9d: a float ``minimum`` narrows up to its ceiling, a ``maximum`` down."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"parameter {name}: {key} must be a number")
    return int(math.ceil(value) if key == "minimum" else math.floor(value))


def _shape_enum(name: str, prop: dict[str, Any], kind: str) -> None:
    values = prop.get("enum", [])
    if not isinstance(values, list) or not all(_param_type_ok(kind, v) for v in values):
        raise ValueError(f"parameter {name}: every enum value must be a {kind} (no float enum)")


def _shape_string(name: str, prop: dict[str, Any]) -> None:
    prop.setdefault("maxLength", PARAM_DEFAULT_MAX_LENGTH)
    _shape_enum(name, prop, "string")


def _shape_number(name: str, prop: dict[str, Any]) -> None:
    for key in ("minimum", "maximum"):
        if key in prop:
            prop[key] = _bound(name, key, prop[key])
    if prop.get("minimum", -MAX_INT) > prop.get("maximum", MAX_INT):
        raise ValueError(f"parameter {name}: no integer lies within minimum..maximum")
    _shape_enum(name, prop, "integer")


_SHAPERS: dict[str, Callable[[str, dict[str, Any]], None]] = {
    "string": _shape_string,
    "integer": _shape_number,
    "number": _shape_number,
}


def _check_declared(name: str, raw: Any) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"parameter {name}: the declaration is not an object")
    extra = sorted(str(key) for key in set(raw) - _DECLARED_PARAM_KEYS)
    if extra:
        raise ValueError(f"parameter {name}: keyword(s) {extra[:5]} cannot be granted")
    kind = raw.get("type")
    if not isinstance(kind, str) or kind not in PARAM_TYPES:
        raise ValueError(f"parameter {name}: type {str(kind)[:40]} cannot be granted")


def _grant_prop(name: str, raw: Any) -> dict[str, Any]:
    """One declared property as a grant carries it; ``ValueError`` names why
    not. ``description``, ``title`` and ``default`` are dropped."""
    _check_declared(name, raw)
    keys = ("type", "enum", "minimum", "maximum", "maxLength")
    prop = {key: copy.deepcopy(raw[key]) for key in keys if key in raw}
    shaper = _SHAPERS.get(prop["type"])
    if shaper is not None:
        shaper(name, prop)
    try:
        _param_prop(prop, name)
    except _Invalid as exc:
        raise ValueError(f"parameter {exc}") from None
    return prop


def _schema_top(schema: Any) -> tuple[dict[str, Any], list[Any]]:
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ValueError("the parameter schema is not an object schema")
    properties, required = schema.get("properties") or {}, schema.get("required") or []
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("the parameter schema's properties or required are malformed")
    return properties, required


def grant_param_schema(workflow_json: Any) -> dict[str, Any]:
    """The C1 ``param_schema`` for one workflow, or ``ValueError`` naming the
    parameter r4 refuses to sign for (a code-typed name or keyword, a type
    outside the four, a float enum, an empty narrowed range). The injected
    ``project_id``, ``input_files`` and ``mcp_tools`` are left out; a
    ``string`` without ``maxLength`` gets ``PARAM_DEFAULT_MAX_LENGTH``."""
    properties, required = _schema_top(_declared_schema(workflow_json))
    granted = {
        _grant_name(name): _grant_prop(name, raw)
        for name, raw in properties.items()
        if name not in PARAM_INJECTED
    }
    names = sorted(granted, key=_utf8)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted({n for n in required if isinstance(n, str) and n in granted}, key=_utf8),
        "properties": {name: granted[name] for name in names},
    }


def merge_grant_param_schemas(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    """C1 "More than one workflow": the union of properties (a name declared
    differently refuses, "grant them separately") and the intersection of
    ``required``."""
    properties: dict[str, Any] = {}
    for schema in schemas:
        for name, prop in schema["properties"].items():
            if properties.setdefault(name, prop) != prop:
                raise ValueError(f"parameter {name} is declared differently: grant them separately")
    required = (
        set.intersection(*(set(schema["required"]) for schema in schemas)) if schemas else set()
    )
    names = sorted(properties, key=_utf8)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(required, key=_utf8),
        "properties": {name: properties[name] for name in names},
    }


# -- run params against a grant (C1 "Run params validate") ------------------


def _value_type_ok(kind: str, value: Any) -> bool:
    if kind == "number":
        return type(value) is int or (type(value) is float and math.isfinite(value))
    return _param_type_ok(kind, value)


def _value_problem(prop: dict[str, Any], value: Any) -> str:
    if not _value_type_ok(prop["type"], value):
        return f"must be a {prop['type']}"
    if isinstance(value, str) and len(value) > prop["maxLength"]:
        return f"is longer than {prop['maxLength']} characters"
    if "enum" in prop and value not in prop["enum"]:
        return "is not one of the granted values"
    if value < prop.get("minimum", value) or value > prop.get("maximum", value):
        return "is outside the granted range"
    return ""


def _input_file_ok(item: Any) -> bool:
    """A normal key or ``s3://<bucket>/<normal key>``, at most 1100 bytes."""
    if not isinstance(item, str) or len(item) > MAX_INPUT_FILE_BYTES:
        return False
    if not item.startswith("s3://"):
        return is_normal_key(item)
    bucket, sep, key = item[len("s3://") :].partition("/")
    return bool(sep) and is_bucket_name(bucket) and is_normal_key(key)


def _injected_problem(params: dict[str, Any], project: str) -> str:
    if params.get("project_id") != project:
        return "project_id: must equal the run's project"
    files = params.get("input_files", [])
    if not isinstance(files, list) or len(files) > MAX_INPUT_FILES:
        return f"input_files: must be a list of at most {MAX_INPUT_FILES} names"
    bad = [item for item in files if not _input_file_ok(item)]
    if bad:
        return f"input_files: {str(bad[0])[:80]!r} is not an object key"
    return "mcp_tools: cannot be granted" if "mcp_tools" in params else ""


def _declared_problem(schema: dict[str, Any], params: dict[str, Any], names: set[str]) -> str:
    keys = set(params) - {"project_id", "input_files"}
    stray = sorted(keys - names)
    if stray:
        return f"{stray[0]}: not a parameter this workflow declares under the grant"
    missing = sorted((set(schema["required"]) & names) - keys)
    if missing:
        return f"{missing[0]}: required"
    for name in sorted(keys):
        why = _value_problem(schema["properties"][name], params[name])
        if why:
            return f"{name}: {why}"
    return ""


def params_problems(
    schema: dict[str, Any],
    params: Any,
    project: str,
    declared: list[str] | None = None,
) -> str:
    """ "" when the run's ``params`` validate against a grant's ``param_schema``
    for a workflow whose granted names are ``declared`` (K9a; None: every
    property), else the first problem as ``<name>: <why>``."""
    if not isinstance(params, dict):
        return "params: must be an object"
    problem = _injected_problem(params, project)
    names = set(schema["properties"]) if declared is None else set(declared)
    return problem or _declared_problem(schema, params, names & set(schema["properties"]))


# -- digests, run key, cost ---------------------------------------------------


def _skeleton_step(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("a workflow step is not an object")
    return {key: raw[key] for key in _SKELETON_KEYS if key in raw}


def workflow_steps_skeleton(workflow_json: Any) -> dict[str, Any]:
    """``{"steps": [{name, depends_on, config} of each step, in file order]}``
    (X1); the skeleton of a skeleton is itself."""
    steps = workflow_json.get("steps") if isinstance(workflow_json, dict) else None
    if not isinstance(steps, list):
        raise ValueError("the workflow has no steps list")
    return {"steps": [_skeleton_step(raw) for raw in steps]}


def steps_sha256(workflow_json: Any) -> str:
    """sha256 of the canonical skeleton (``json_values``: configs may hold
    floats and ``null``), from a ``workflow.json`` or a skeleton."""
    skeleton = workflow_steps_skeleton(workflow_json)
    return hashlib.sha256(canonical_bytes(skeleton, True)).hexdigest()


def src_manifest_sha256(src_manifest: Any) -> str:
    """sha256 of an intent's ``src_manifest`` object, canonical, no floats."""
    return hashlib.sha256(canonical_bytes(src_manifest)).hexdigest()


def grant_run_key(grant_id: str, run_id: str) -> str:
    """X2: the per-run key of a grant run (``<i>`` of mkwd, lookup, state)."""
    _require(_NONCE_RE, grant_id, "grant_id")
    _require(_RUN_ID_RE, run_id, "run_id")
    data = _GRANT_RUN_PREFIX + grant_id.encode("ascii") + b"\n" + run_id.encode("ascii")
    return hashlib.sha256(data).hexdigest()


def grant_wall(step: dict[str, Any]) -> int:
    """K4: the walltime Slurm enforces, ``--time`` in whole minutes."""
    return -(-int(step["time"]) // 60) * 60


def grant_cost(step: dict[str, Any], partition: dict[str, Any]) -> tuple[int, int]:
    """K4: ``(gpu_seconds, core_seconds)`` a step may use: the walltime
    rounded up to whole minutes, and the whole node on an exclusive partition."""
    wall = grant_wall(step)
    exclusive = bool(partition["exclusive"])
    cpus = int(partition["max_cpus"]) if exclusive else int(step["cpus"])
    gpus = int(step["gpus"])
    if exclusive and gpus > 0:
        gpus = int(partition["max_gpus"])
    return gpus * wall, cpus * wall


# -- coverage: grant ∩ run spec (C2 step 11) --------------------------------


def _granted_workflow(grant: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    for entry in grant["workflows"]:
        if (entry["name"], entry["version"]) == (spec["workflow"], spec["version"]):
            found: dict[str, Any] = entry
            return found
    return None


def _cover_account(grant: dict[str, Any], spec: dict[str, Any], _skeleton: Any) -> str:
    if spec["account"] not in grant["accounts"]:
        return f"account: {spec['account']} is not granted"
    if spec["project"] not in grant["projects"]:
        return f"project: {spec['project'][:128]} is not granted"
    return ""


def _cover_workflow(grant: dict[str, Any], spec: dict[str, Any], skeleton: Any) -> str:
    entry = _granted_workflow(grant, spec)
    label = f"{spec['workflow'][:128]}@{spec['version'][:64]}"
    if entry is None:
        return f"workflow: {label} is not granted"
    if entry["src_manifest_sha256"] != src_manifest_sha256(spec["src_manifest"]):
        return f"workflow: the src_manifest of {label} is not the granted one"
    try:
        same = entry["steps_sha256"] == steps_sha256(skeleton)
    except ValueError as exc:
        return f"workflow_steps: {exc}"
    return "" if same else f"steps_sha256: the step skeleton of {label} is not the granted one"


def _same_json(left: Any, right: Any) -> bool:
    """Equal as canonical JSON (``1`` and ``1.0`` differ)."""
    return canonical_bytes(left, True) == canonical_bytes(right, True)


def _steps_problem(spec: dict[str, Any], skeleton: dict[str, Any]) -> str:
    raws = skeleton["steps"]
    graph = step_graph(raws)
    if [name for name, _after, _chained in graph] != [step["name"] for step in spec["steps"]]:
        return "steps: the run's steps are not the granted step graph"
    for index, (name, after, chained) in enumerate(graph):
        raw, step = raws[index], spec["steps"][index]
        if step["after"] != after or step["inputs"]["chained"] != chained:
            return f"steps: {name} is ordered or chained otherwise than granted"
        if not _same_json(step["config"], expected_step_config(raw, name, spec["params"])):
            return f"config: step {name} has a config the grant does not pin"
    return ""


def _cover_steps(_grant: dict[str, Any], spec: dict[str, Any], skeleton: Any) -> str:
    try:
        return _steps_problem(spec, skeleton)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return _clip(f"steps: {exc}")


def _cover_images(grant: dict[str, Any], spec: dict[str, Any], _skeleton: Any) -> str:
    """Check step images and the independently granted helper image role.

    Combined-role legacy grants stay readable/cancellable. They execute only
    when one image makes both roles unambiguous; multi-image grants must be
    reissued with explicit roles, never subject to role-substitution fallback.
    """
    if "trusted_images" not in grant and len(grant["images"]) != 1:
        return "images: legacy grant lacks trusted_images; revoke and reissue with explicit roles"
    helpers = grant.get("trusted_images", grant["images"])
    if spec["trusted_image"] not in helpers:
        return f"images: trusted_image {spec['trusted_image']['image_id'][:12]} is not granted"
    for step in spec["steps"]:
        if step["image"] not in grant["images"]:
            return f"images: {step['image']['image_id'][:12]} is not granted"
    return ""


def _cover_endpoint(_grant: dict[str, Any], spec: dict[str, Any], _skeleton: Any) -> str:
    for step in spec["steps"]:
        if "endpoint" in step:
            return f"endpoint: step {step['name']} has an endpoint (not grantable)"
    return ""


def _cover_params(grant: dict[str, Any], spec: dict[str, Any], _skeleton: Any) -> str:
    entry = _granted_workflow(grant, spec) or {"params": []}
    problem = params_problems(
        grant["param_schema"], spec["params"], spec["project"], entry["params"]
    )
    return f"params: {problem}" if problem else ""


def _cover_inputs(grant: dict[str, Any], spec: dict[str, Any], _skeleton: Any) -> str:
    """``listed``: every object is listed. ``chained``: a listed object (the
    seed, K10) passes; the others need gate state (C2 step 13)."""
    if grant["inputs"]["mode"] != "listed":
        return ""
    listed = {(entry["sha256"], entry["size"]) for entry in grant["inputs"]["listed"]}
    for step in spec["steps"]:
        for obj in step["inputs"]["objects"]:
            if (obj["sha256"], obj["size"]) not in listed:
                return f"inputs: {obj['bucket']}/{obj['key'][:200]} is not listed"
    return ""


_COVERAGE = (
    _cover_account,
    _cover_workflow,
    _cover_steps,
    _cover_images,
    _cover_endpoint,
    _cover_params,
    _cover_inputs,
)


def grant_covers(grant: dict[str, Any], run_spec: dict[str, Any], skeleton: Any) -> str:
    """ "" when the validated ``grant`` covers the validated ``run_spec`` and
    its step ``skeleton``, else the first problem, ``<field>: <why>`` (C2
    step 11). The chained-output and cap checks need gate state: not here."""
    for check in _COVERAGE:
        problem = check(grant, run_spec, skeleton)
        if problem:
            return problem
    return ""


def grant_control_body(
    site: str, principal: str, grant_id: str, now: float | None = None, ttl: int = 300
) -> dict[str, Any]:
    """A ``grant-cancel`` control body (C3): ``{v, type, op, site, principal,
    grant_id, nonce, issued, expires}``, no ``key_id``; ``ValueError`` when
    it would not validate."""
    if not 0 < int(ttl) <= CONTROL_MAX_WINDOW:
        raise ValueError(f"a control lives at most {CONTROL_MAX_WINDOW} s")
    issued = int(time.time() if now is None else now)
    body = {
        "v": 1,
        "type": "control",
        "op": "grant-cancel",
        "site": site,
        "principal": principal,
        "grant_id": grant_id,
        "nonce": secrets.token_hex(16),
        "issued": issued,
        "expires": issued + int(ttl),
    }
    try:
        validate_control(body, "grant-cancel")
    except GateError as exc:
        raise ValueError(exc.detail) from None
    return body


# ---------------------------------------------------------------------------
# The job script (B2.0 section 6): composed from signed fields and the profile
# ---------------------------------------------------------------------------

#: The step container's posture: ``slurm_step_executor._POD_POSTURE`` and
#: ``_POD_BASE_ENV`` (a shared-vector test keeps them identical).
POD_POSTURE = (
    "--rm",
    "--init",
    "--pull=never",
    "--network=none",
    "--http-proxy=false",
    "--unsetenv-all",
    "--image-volume=ignore",
    "--workdir=/tmp",
    "--read-only",
    "--tmpfs /tmp:rw,size=4g",
    "--cap-drop=all",
    "--security-opt no-new-privileges",
    "--userns=keep-id",
    '--user "$(id -u):$(id -g)"',
)
POD_BASE_ENV = (
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME=/tmp",
    "LANG=C.UTF-8",
    "PYTHONUNBUFFERED=1",
    "PYTHONDONTWRITEBYTECODE=1",
)
#: (workdir entry, container path, mode): the staged mounts with ``input.json``
#: in place of ``input.pkl``.
GATED_MOUNTS = (
    ("in", "/work/in", "ro"),
    ("out", "/work/out", "rw"),
    ("reply", "/work/reply", "rw"),
    ("src", "/src", "ro"),
    ("input.json", "/work/input.json", "ro"),
)
GATED_ENV = (
    "UMRS_STEP_INPUT_JSON=/work/input.json",
    "UMRS_STEP_OUTPUT_JSON=/work/reply/output.json",
    "UMRS_STEP_STAGED=1",
    "UMRS_STEP_SOURCE_DIR=/src",
)
ENDPOINT_SOCKET_ENV = "UMRS_STEP_ENDPOINT_SOCKET"
ENDPOINT_SOCKET_PATH = "/run/umrs/endpoint.sock"
ENDPOINT_CREDENTIAL_FILE = "endpoint.json"
#: The only ``#SBATCH`` options the template writes (B2.0 section 6 table).
SBATCH_OPTIONS = (
    "--job-name",
    "--comment",
    "--no-requeue",
    "--export",
    "--time",
    "--partition",
    "--account",
    "--cpus-per-task",
    "--mem",
    "--gres",
    "--gpus",
    "--exclusive",
    "--deadline",
    "--output",
    "--error",
)
CLEAN_PATH = "/usr/local/bin:/usr/bin:/bin"
_EX_CONFIG = 78
_ENDPOINT_START_TICKS = 120
_GPU_DEVICES = {
    "cdi": ("--device", "nvidia.com/gpu=all"),
    "rocm": ("--device", "/dev/kfd", "--device", "/dev/dri"),
    "nvidia-devices": ('"${_step_gpu[@]}"',),
}
_NVIDIA_HOST_LIBS = ("libcuda.so.1", "libnvidia-ml.so.1", "libnvidia-ptxjitcompiler.so.1")
_NVIDIA_LIB_DIRS = ("/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
_NVIDIA_DEVICE_GLOBS = (
    "/dev/nvidia[0-9]*",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
    "/dev/nvidia-caps/nvidia-cap[0-9]*",
)
#: The env names the clean re-exec may carry over, each with its value pattern.
_PASS_ENV = (
    ("SLURM_JOB_GPUS", "^[0-9,]{1,512}$"),
    ("CUDA_VISIBLE_DEVICES", "^[A-Za-z0-9,:-]{1,1024}$"),
    ("XDG_RUNTIME_DIR", "^/run/user/[0-9]{1,10}$"),
)
SECCOMP_FILTER_PY = """import json, sys
profile = json.load(open(sys.argv[1]))
kept = []
for rule in profile.get("syscalls") or []:
    if rule.get("name") == "fallocate":
        continue
    if "names" in rule:
        rule["names"] = [n for n in rule["names"] if n != "fallocate"]
        if not rule["names"]:
            continue
    kept.append(rule)
kept.append({"names": ["fallocate"], "action": "SCMP_ACT_ERRNO", "errnoRet": 95})
profile["syscalls"] = kept
json.dump(profile, sys.stdout)
"""
_WATCH_STATS_AWK = (
    "awk '{n += 1; b = $2 * 512; t += ($1 > b ? $1 : b)} END {printf \"%.0f %.0f\\n\", n, t}'"
)


@dataclass(frozen=True)
class _Job:
    """What one gated job script is rendered from (all of it validated)."""

    iid: str
    intent: dict[str, Any]
    step: dict[str, Any]
    profile: dict[str, Any]
    part: dict[str, Any]
    wd: str
    login: str
    home: str
    #: a grant run: seconds until the grant expires, the most ``--deadline``
    #: may give (review 2026-11-09, K19a)
    deadline_max: int | None = None
    #: Stronger alternative: the absolute epoch, immune to pre-sbatch delay.
    deadline_epoch: int | None = None

    @property
    def name(self) -> str:
        return str(self.step["name"])

    @property
    def needs_gpu(self) -> bool:
        return int(self.step["gpus"]) > 0

    @property
    def cgroups_disabled(self) -> bool:
        return self.profile["podman"]["cgroups"] == "disabled"

    @property
    def endpoint(self) -> dict[str, Any] | None:
        return self.step.get("endpoint")

    @property
    def images(self) -> list[dict[str, Any]]:
        """The step image and the trusted image, once when they are equal."""
        step, trusted = self.step["image"], self.intent["trusted_image"]
        return [step] if trusted == step else [step, trusted]


def _q(word: Any) -> str:
    return shlex.quote(str(word))


def _fatal(command: str, message: str) -> str:
    return f'{command} || _step_fatal {_EX_CONFIG} "{message}"'


def _gpu_request(job: _Job) -> str | None:
    """``_staged_gpu_request`` on the step's partition profile."""
    part, gpus = job.part, int(job.step["gpus"])
    style = part["gpu_request"]
    if part["gpu_flag"] == "gpus" and not style:
        return f"--gpus={gpus}"
    if style == "none":
        return None
    if style == "gres":
        return f"--gres={part['gres'] or f'gpu:{gpus}'}"
    return f"--gres=gpu:{gpus}"


def _gpu_directives(job: _Job) -> list[str]:
    request = _gpu_request(job) if job.needs_gpu else None
    lines = [f"#SBATCH {request}"] if request else []
    if job.part["exclusive"]:
        lines.append("#SBATCH --exclusive")
    return lines


def _directives(job: _Job) -> list[str]:
    step, slurm = job.step, job.profile["slurm"]
    i16 = job.iid[:16]
    lines = [
        "#!/bin/bash -p",
        f"#SBATCH --job-name=umrs-gated-{i16}-{job.name}",
        f"#SBATCH --comment=umrs-gate:{i16}:{job.name}",
        "#SBATCH --no-requeue",
        "#SBATCH --export=NONE",
        f"#SBATCH --time={-(-int(step['time']) // 60)}",
        f"#SBATCH --partition={step['partition']}",
        f"#SBATCH --account={job.intent['account']}",
        f"#SBATCH --cpus-per-task={int(step['cpus'])}",
    ]
    if step["mem"]:
        lines.append(f"#SBATCH --mem={step['mem']}")
    lines += _gpu_directives(job)
    deadline = int(slurm["queue_seconds"]) + int(step["time"]) + int(slurm["grace_seconds"])
    if job.deadline_max is not None:
        deadline = min(deadline, job.deadline_max)
    stamp = f"now+{deadline}"
    if job.deadline_epoch is not None:
        stamp = _slurm_time(job.deadline_epoch)
    return [
        *lines,
        f"#SBATCH --deadline={stamp}",
        f"#SBATCH --output={job.wd}/log/slurm.out",
        f"#SBATCH --error={job.wd}/log/slurm.err",
    ]


def _clean_env_lines(job: _Job) -> list[str]:
    """Re-exec under ``env -i`` with a fixed environment (``bash -p``: no
    ``BASH_ENV``, no imported functions); only a few Slurm-given names with
    a fixed value shape are carried over."""
    fixed = [
        "UMRS_GATE_CLEAN=1",
        f"PATH={CLEAN_PATH}",
        f"HOME={job.home}",
        "LANG=C.UTF-8",
        f"USER={job.login}",
        f"LOGNAME={job.login}",
    ]
    lines = [
        'if [[ "${UMRS_GATE_CLEAN:-}" != "1" ]]; then',
        f"    _gate_env=({' '.join(_q(item) for item in fixed)})",
        '    if [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]{1,20}$ ]]; then',
        '        _gate_env+=("SLURM_JOB_ID=${SLURM_JOB_ID}")',
        "    fi",
    ]
    for name, pattern in _PASS_ENV:
        lines += [
            f'    if [[ "${{{name}:-}}" =~ {pattern} ]]; then',
            f'        _gate_env+=("{name}=${{{name}}}")',
            "    fi",
        ]
    return [
        *lines,
        '    exec /usr/bin/env -i "${_gate_env[@]}" /bin/bash -p "$0"',
        "fi",
        'if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]{1,20}$ ]]; then',
        "    printf '[umrs-gate] FATAL: no Slurm job id\\n' >&2",
        f"    exit {_EX_CONFIG}",
        "fi",
    ]


def _prologue(job: _Job) -> list[str]:
    stop = []
    if job.endpoint is not None:
        stop = [
            '    podman stop -t 5 "${_step_ctr}-endpoint" >/dev/null 2>&1 || true',
            '    podman rm -f "${_step_ctr}-endpoint" >/dev/null 2>&1 || true',
        ]
    return [
        "",
        "set -euo pipefail",
        "umask 077",
        '_step_ctr="umrs-step-${SLURM_JOB_ID}"',
        '_step_watch_pid=""',
        "_gate_copies=()",
        "_step_fatal() {",
        "    printf '[umrs-step] FATAL: %s\\n' \"$2\" >&2",
        '    exit "$1"',
        "}",
        "_step_cleanup() {",
        "    trap '' TERM INT",
        '    if [[ -n "${_step_watch_pid}" ]]; then',
        '        kill "${_step_watch_pid}" 2>/dev/null || true',
        "    fi",
        '    for _c in ${_gate_copies[@]+"${_gate_copies[@]}"}; do',
        '        rm -f -- "${_c}" 2>/dev/null || true',
        "    done",
        '    podman kill "${_step_ctr}" >/dev/null 2>&1 || true',
        '    podman rm -f "${_step_ctr}" >/dev/null 2>&1 || true',
        *stop,
        "}",
        "trap _step_cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        "",
    ]


def _venue_paths(job: _Job) -> list[str]:
    venue = job.profile["venue"]
    root = _q(venue["root"]) if venue["root"] else f'"/var/tmp/podman_{job.login}"'
    storage = (
        _q(venue["storage_root"]) if venue["storage_root"] else '"${RIVAULT_VENUE_ROOT}/storage"'
    )
    nets = venue["network_config_dir"]
    networks = _q(nets) if nets else '"${RIVAULT_VENUE_ROOT}/networks"'
    return [
        f"_gate_login={_q(job.login)}",
        _fatal(
            '[[ "$(id -un)" == "${_gate_login}" ]]',
            "this job runs as $(id -un), not ${_gate_login}",
        ),
        '_gate_uid="$(id -u)"',
        f"RIVAULT_VENUE_ROOT={root}",
        f"RIVAULT_STORAGE_ROOT={storage}",
        f"RIVAULT_NETWORK_CONFIG_DIR={networks}",
    ]


def _safe_path_function() -> list[str]:
    """``_gate_safe_path <dir>``: the directory and every ancestor up to ``/``
    is a directory (no link in any component), owned by root or the login,
    and not group- or other-writable unless root-owned and sticky (``/tmp``).
    Nobody else can then rename or replace anything on that path (B2.0
    section 6, fixer amendment 2026-11-08)."""
    return [
        "_gate_safe_path() {",
        '    local p="$1" s m',
        "    while :; do",
        '        [[ -d "${p}" && ! -L "${p}" ]] || return 1',
        '        s="$(stat -c \'%u %a\' -- "${p}")" || return 1',
        "        m=$((8#${s##* }))",
        '        [[ "${s%% *}" == 0 || "${s%% *}" == "${_gate_uid}" ]] || return 1',
        "        if (( m & 8#022 )); then",
        '            [[ "${s%% *}" == 0 ]] && (( m & 8#1000 )) || return 1',
        "        fi",
        '        [[ "${p}" != / ]] || return 0',
        '        p="$(dirname -- "${p}")"',
        "    done",
        "}",
    ]


_PRIVATE = (
    "is not a private directory of $(id -un) "
    "(0700, not a link, under directories only root or $(id -un) can change)"
)


def _venue_owner_lines(job: _Job) -> list[str]:
    """Each venue directory must be a directory (not a link) of the login with
    mode 0700 under ancestors nobody else can change, created so when absent;
    otherwise exit 78 (B2.0 section 6). The runtime directory is the node's
    own ``/run/user/<uid>`` when that passes, else ``<venue root>/run``: never
    a guessable name in a directory another account could create first. The
    workdir's own path is checked the same way before anything is mounted."""
    return [
        *_safe_path_function(),
        "_gate_private_dir() {",
        '    if [[ ! -e "$1" && ! -L "$1" ]]; then',
        '        mkdir -p -m 0700 -- "$1" || return 1',
        "    fi",
        '    [[ -d "$1" && ! -L "$1" ]] || return 1',
        '    [[ "$(stat -c \'%u %a\' -- "$1")" == "${_gate_uid} 700" ]] || return 1',
        '    _gate_safe_path "$(dirname -- "$1")"',
        "}",
        _fatal(
            '_gate_private_dir "${RIVAULT_VENUE_ROOT}"',
            "venue directory ${RIVAULT_VENUE_ROOT} " + _PRIVATE,
        ),
        'if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR}" ]] \\',
        '    || ! _gate_private_dir "${XDG_RUNTIME_DIR}"; then',
        '    XDG_RUNTIME_DIR="${RIVAULT_VENUE_ROOT}/run"',
        "fi",
        'RIVAULT_RUNROOT="${XDG_RUNTIME_DIR}/containers"',
        'RIVAULT_JOB_CONF_DIR="${XDG_RUNTIME_DIR}/rivault-umrs-gated-${SLURM_JOB_ID}/containers"',
        'for _d in "${XDG_RUNTIME_DIR}" "${RIVAULT_STORAGE_ROOT}" "${RIVAULT_RUNROOT}" \\',
        '    "${RIVAULT_NETWORK_CONFIG_DIR}" "${RIVAULT_JOB_CONF_DIR}"; do',
        "    " + _fatal('_gate_private_dir "${_d}"', "venue directory ${_d} " + _PRIVATE),
        "done",
        _fatal(
            f"_gate_safe_path {_q(job.wd)}",
            "the workdir sits under a directory another account can change",
        ),
    ]


def _container_conf_lines(job: _Job) -> list[str]:
    """``rivault_venue_write_container_configs cgroupfs <backend>``, inlined."""
    backend = job.profile["venue"]["network_backend"]
    backend_lines = [f"    echo 'network_backend = \"{backend}\"'"] if backend else []
    persist = []
    if backend:
        persist = [
            f"printf '%s' {_q(backend)} > \"${{RIVAULT_STORAGE_ROOT}}/defaultNetworkBackend\" 2>/dev/null || true"
        ]
    return [
        '_gate_mount_program=""',
        "if [[ -x /usr/bin/fuse-overlayfs ]]; then",
        "    _gate_mount_program=/usr/bin/fuse-overlayfs",
        "fi",
        "{",
        "    echo '[storage]'",
        "    echo 'driver = \"overlay\"'",
        '    echo "runroot = \\"${RIVAULT_RUNROOT}\\""',
        '    echo "graphroot = \\"${RIVAULT_STORAGE_ROOT}\\""',
        '    echo "rootless_storage_path = \\"${RIVAULT_STORAGE_ROOT}\\""',
        '    if [[ -n "${_gate_mount_program}" ]]; then',
        "        echo '[storage.options]'",
        '        echo "mount_program = \\"${_gate_mount_program}\\""',
        "        echo '[storage.options.overlay]'",
        "        echo 'mountopt = \"xattr_permissions=2,threaded=1\"'",
        "    fi",
        '} > "${RIVAULT_JOB_CONF_DIR}/storage.conf"',
        "{",
        "    echo '[engine]'",
        "    echo 'cgroup_manager = \"cgroupfs\"'",
        "    echo '[containers]'",
        "    echo 'label = false'",
        "    echo '[network]'",
        '    echo "network_config_dir = \\"${RIVAULT_NETWORK_CONFIG_DIR}\\""',
        *backend_lines,
        '} > "${RIVAULT_JOB_CONF_DIR}/containers.conf"',
        'CONTAINERS_STORAGE_CONF="${RIVAULT_JOB_CONF_DIR}/storage.conf"',
        'CONTAINERS_CONF="${RIVAULT_JOB_CONF_DIR}/containers.conf"',
        "export XDG_RUNTIME_DIR CONTAINERS_STORAGE_CONF CONTAINERS_CONF",
        *persist,
    ]


def _adopt_runroot_lines() -> list[str]:
    """``rivault_venue_adopt_runroot``, inlined (nothing is sourced)."""
    return [
        "_gate_runroot=\"$(podman info --format '{{.Store.RunRoot}}' 2>/dev/null || true)\"",
        'if [[ -n "${_gate_runroot}" && "${_gate_runroot}" != "${RIVAULT_RUNROOT}" ]]; then',
        '    if [[ -z "$(podman ps -q 2>/dev/null | head -1)" ]]; then',
        '        rm -rf -- "${RIVAULT_STORAGE_ROOT}/libpod" 2>/dev/null || true',
        '        rm -f -- "${RIVAULT_STORAGE_ROOT}/db.sql" 2>/dev/null || true',
        '        if [[ -d "${RIVAULT_STORAGE_ROOT}/overlay-containers" ]]; then',
        "            printf '[]' > \"${RIVAULT_STORAGE_ROOT}/overlay-containers/containers.json\" 2>/dev/null || true",
        '            for _e in "${RIVAULT_STORAGE_ROOT}"/overlay-containers/*/; do',
        '                if [[ -d "${_e}" ]]; then',
        '                    rm -rf -- "${_e}" 2>/dev/null || true',
        "                fi",
        "            done",
        "        fi",
        "    fi",
        "fi",
        "",
    ]


def _arch_lines(job: _Job) -> list[str]:
    arch = job.part["arch"]
    return [
        _fatal(f'[[ "$(uname -m)" == {_q(arch)} ]]', f"this node is $(uname -m), not {arch}"),
        "",
    ]


def _image_function_lines(job: _Job) -> list[str]:
    """``_gate_ensure_image <image_id> <archive_sha256>``: use the image when the
    login's own storage holds it; otherwise copy the archive to a private
    file, hash THE COPY, load THE COPY, check the loaded id; then the labels."""
    inspect = "podman image inspect --format"
    label = "'{{ index .Config.Labels \"%s\" }}'"
    return [
        f"_gate_archive_dir={_q(job.profile['image_archive_dir'])}",
        "_gate_ensure_image() {",
        '    local want_id="$1" want_sha="$2" have copy contract payload',
        f'    have="$({inspect} \'{{{{.Id}}}}\' "${{want_id}}" 2>/dev/null || true)"',
        '    if [[ "${have}" != "${want_id}" ]]; then',
        '        copy="$(mktemp -p "${RIVAULT_VENUE_ROOT}" umrs-image.XXXXXXXXXX)" || _step_fatal 78 "no private image copy"',
        '        _gate_copies+=("${copy}")',
        '        chmod 0600 -- "${copy}"',
        '        cp -- "${_gate_archive_dir}/${want_sha}.tar" "${copy}" || _step_fatal 78 "image archive ${want_sha}.tar could not be copied"',
        '        [[ "$(sha256sum -- "${copy}" | cut -d" " -f1)" == "${want_sha}" ]] || _step_fatal 78 "image archive ${want_sha}.tar does not hash to its name"',
        '        podman load -i "${copy}" >&2 || _step_fatal 78 "podman load of ${want_sha}.tar failed"',
        '        rm -f -- "${copy}"',
        f'        have="$({inspect} \'{{{{.Id}}}}\' "${{want_id}}" 2>/dev/null || true)"',
        '        [[ "${have}" == "${want_id}" ]] || _step_fatal 78 "the loaded image is not ${want_id}"',
        "    fi",
        f'    contract="$({inspect} {label % STEP_CONTRACT_LABEL} "${{want_id}}" 2>/dev/null || true)"',
        f'    [[ "${{contract}}" == "{STEP_CONTRACT_VERSION}" ]] || _step_fatal 78 "image ${{want_id}} lacks {STEP_CONTRACT_LABEL}={STEP_CONTRACT_VERSION}"',
        f'    payload="$({inspect} {label % STEP_PAYLOAD_LABEL} "${{want_id}}" 2>/dev/null || true)"',
        f'    [[ ",${{payload}}," == *",json,"* ]] || _step_fatal 78 "image ${{want_id}} lacks json in {STEP_PAYLOAD_LABEL}"',
        "}",
    ]


def _image_lines(job: _Job) -> list[str]:
    calls = [
        f"_gate_ensure_image {_q(image['image_id'])} {_q(image['archive_sha256'])}"
        for image in job.images
    ]
    return [*_image_function_lines(job), *calls, ""]


def _podman_run(job: _Job) -> list[str]:
    if job.cgroups_disabled:
        return ["podman", "--cgroup-manager=cgroupfs", "run"]
    return ["podman", "run"]


def _pod_posture(job: _Job, name: str, seccomp: bool = True) -> list[str]:
    podman = job.profile["podman"]
    parts = [*POD_POSTURE[:2], f"--name {name}", *POD_POSTURE[2:]]
    parts += [f"-e {assignment}" for assignment in POD_BASE_ENV]
    if seccomp:
        parts.append('--security-opt seccomp="${_step_seccomp}"')
    if job.cgroups_disabled:
        parts.append("--cgroups=disabled")
    elif podman["pids_limit"] > 0:
        parts.append(f"--pids-limit={int(podman['pids_limit'])}")
    if podman["nproc_limit"] > 0:
        parts.append(f"--ulimit nproc={int(podman['nproc_limit'])}")
    return [*parts, f"--ulimit fsize={int(job.profile['caps']['max_out_file_bytes'])}"]


def _trusted(job: _Job) -> str:
    return _q(job.intent["trusted_image"]["image_id"])


def _seccomp_lines(job: _Job) -> list[str]:
    info = "podman info --format '{{.Host.Security.SECCOMPProfilePath}}'"
    transform = " ".join(
        [
            *_podman_run(job),
            *_pod_posture(job, '"${_step_ctr}-seccomp"', seccomp=False),
            '-v "${_step_seccomp_base}:/umrs-seccomp-base.json:ro"',
            "--entrypoint python",
            _trusted(job),
            "-I -c",
            _q(SECCOMP_FILTER_PY),
            "/umrs-seccomp-base.json",
        ]
    )
    return [
        f'_step_seccomp_base="$({info} 2>/dev/null || true)"',
        _fatal('[[ -f "${_step_seccomp_base}" ]]', "no default seccomp profile on $(hostname)"),
        f"_step_seccomp={_q(job.wd + '/seccomp.json')}",
        _fatal(
            f'{transform} > "${{_step_seccomp}}"',
            "the step seccomp profile could not be written on $(hostname)",
        ),
        "",
    ]


def _pids_lines(job: _Job) -> list[str]:
    limit = int(job.profile["podman"]["pids_limit"])
    if limit <= 0 or job.cgroups_disabled:
        return []
    probe = " ".join(
        [
            *_podman_run(job),
            *_pod_posture(job, '"${_step_ctr}-pids"'),
            "--entrypoint cat",
            _trusted(job),
            "/sys/fs/cgroup/pids.max",
        ]
    )
    return [
        _fatal(f'_step_pids="$({probe} 2>&1)"', "the step posture was refused on $(hostname)"),
        '_step_pids_max="$(printf \'%s\\n\' "${_step_pids}" | tail -n 1)"',
        _fatal(
            f'[[ "${{_step_pids_max}}" == "{limit}" ]]',
            f"--pids-limit={limit} is not applied on $(hostname)",
        ),
        "",
    ]


def _stale_lines() -> list[str]:
    listing = (
        "podman ps -a --filter name=^umrs-step- --filter status=exited "
        "--filter status=created --format '{{.Names}}' 2>/dev/null || true"
    )
    return [
        'if _step_live="$(squeue --me -h -o %i 2>/dev/null)"; then',
        "    while read -r _step_old; do",
        '        _step_id="${_step_old#umrs-step-}"',
        '        _step_id="${_step_id%%-*}"',
        '        if [[ -n "${_step_id}" && "${_step_id}" != "${SLURM_JOB_ID}" ]] \\',
        '            && ! grep -qx -- "${_step_id}" <<<"${_step_live}"; then',
        '            podman rm -f "${_step_old}" >/dev/null 2>&1 || true',
        "        fi",
        f"    done < <({listing})",
        "fi",
        "",
    ]


def _watch_stop(condition: str, why: str) -> list[str]:
    return [
        f"        if {condition}; then",
        f"            printf '[umrs-step] FATAL: %s\\n' \"{why}; the step was stopped\" >&2",
        '            podman kill "${_step_ctr}" >/dev/null 2>&1',
        "            return",
        "        fi",
    ]


def _watch_floor(job: _Job) -> list[str]:
    floor = int(job.profile["caps"]["min_free_bytes"])
    if floor <= 0:
        return []
    return [
        f'        avail="$(df -B1 --output=avail -- {_q(job.wd)} 2>/dev/null | tail -n 1 | tr -dc 0-9)"',
        *_watch_stop(
            f'[[ -n "${{avail}}" ]] && (( avail < {floor} ))',
            f"the workdir filesystem has ${{avail}} bytes free (floor {floor})",
        ),
    ]


def _watch_lines(job: _Job) -> list[str]:
    caps = job.profile["caps"]
    out, reply = _q(job.wd + "/out"), _q(job.wd + "/reply")
    entries = 2 * int(caps["max_out_files"]) + 8
    total = int(caps["max_out_bytes"]) + int(caps["reply_bytes"])
    hidden = r"-type d \( ! -readable -o ! -executable \) -print -quit"
    return [
        "_step_watch() {",
        "    set +eo pipefail",
        "    local stats used entries hidden avail",
        f"    while sleep {int(job.profile['watch_seconds'])} </dev/null >/dev/null 2>&1; do",
        *_watch_floor(job),
        f"        stats=\"$(find {out} {reply} -printf '%s %b\\n' 2>/dev/null | head -n {entries + 1} | {_WATCH_STATS_AWK})\"",
        '        entries="${stats%% *}"',
        '        used="${stats##* }"',
        *_watch_stop(
            f"(( used > {total} || entries > {entries} ))",
            "out/ and reply/ passed their caps (${used} bytes, ${entries} entries)",
        ),
        f'        hidden="$(find {out} {reply} {hidden} 2>/dev/null)"',
        *_watch_stop(
            '[[ -n "${hidden}" ]]', "a directory under out/ or reply/ is hidden from the watchdog"
        ),
        "    done",
        "}",
        "_step_watch &",
        "_step_watch_pid=$!",
        "if command -v stdbuf >/dev/null 2>&1; then",
        "    _step_head=(stdbuf -o0 head)",
        "else",
        "    _step_head=(head)",
        "fi",
        "_step_log() {",
        f'    "${{_step_head[@]}}" -c {LOG_HEAD_BYTES}',
        f"    tail -c {LOG_TAIL_BYTES}",
        "}",
        "",
    ]


def _gpu_lines(job: _Job) -> list[str]:
    if not (job.needs_gpu and job.profile["gpu"]["form"] == "nvidia-devices"):
        return []
    return [
        "_step_gpu=()",
        f"for _d in {' '.join(_NVIDIA_DEVICE_GLOBS)}; do",
        '    if [[ -e "${_d}" ]]; then',
        '        _step_gpu+=(--device "${_d}")',
        "    fi",
        "done",
        _fatal("[[ ${#_step_gpu[@]} -gt 0 ]]", "no NVIDIA device nodes on $(hostname)"),
        f"for _l in {' '.join(_NVIDIA_HOST_LIBS)}; do",
        f"    for _p in {' '.join(_NVIDIA_LIB_DIRS)}; do",
        '        if [[ -e "${_p}/${_l}" ]]; then',
        '            _step_gpu+=(-v "${_p}/${_l}:${_p}/${_l}:ro")',
        "            break",
        "        fi",
        "    done",
        "done",
        "",
    ]


def _forwarder_argv(job: _Job, endpoint: dict[str, Any]) -> list[str]:
    """The forwarder runs trusted code (the trusted image), with the host
    network, the step's ``reply/`` (its socket) and ``endpoint.json``; the
    credential file is never mounted into the step's own pod."""
    caps = job.profile["endpoint"]
    posture = [
        "--network=host" if part == "--network=none" else part
        for part in _pod_posture(job, '"${_step_ctr}-endpoint"')
    ]
    mounts = (
        f"{job.wd}/reply:{ENDPOINT_SOCKET_PATH.rsplit('/', 1)[0]}:rw",
        f"{job.wd}/{ENDPOINT_CREDENTIAL_FILE}:/umrs-endpoint.json:ro",
    )
    args = (
        ("--socket", ENDPOINT_SOCKET_PATH),
        ("--credential", "/umrs-endpoint.json"),
        ("--upstream", f"{endpoint['host']}:{int(endpoint['port'])}"),
        ("--kind", endpoint["kind"]),
        ("--max-request-bytes", str(int(caps["max_request_bytes"]))),
        ("--max-response-bytes", str(int(caps["max_response_bytes"]))),
        ("--max-total-bytes", str(int(caps["max_total_bytes"]))),
    )
    volumes = [word for mount in mounts for word in ("-v", _q(mount))]
    command = ["--entrypoint python", _trusted(job), "-I -m umrs.server.step_endpoint_forwarder"]
    return [
        *_podman_run(job),
        *posture,
        *volumes,
        *command,
        *(_q(w) for pair in args for w in pair),
    ]


def _endpoint_lines(job: _Job) -> list[str]:
    endpoint = job.endpoint
    if endpoint is None:
        return []
    return [
        f"_step_sock={_q(job.wd + '/reply/endpoint.sock')}",
        " ".join(_forwarder_argv(job, endpoint)) + " >&2 &",
        "_step_fwd_pid=$!",
        f"for _ in $(seq 1 {_ENDPOINT_START_TICKS}); do",
        '    [[ -S "${_step_sock}" ]] && break',
        '    kill -0 "${_step_fwd_pid}" 2>/dev/null || break',
        "    sleep 0.5",
        "done",
        _fatal('[[ -S "${_step_sock}" ]]', "the endpoint forwarder did not start on $(hostname)"),
        "",
    ]


def _pod_argv(job: _Job) -> list[str]:
    mounts = list(GATED_MOUNTS)
    env = [
        *GATED_ENV,
        f"UMRS_STEP_ENTRYPOINT={job.intent['src_manifest']['entrypoints'][job.name]}",
    ]
    if job.endpoint is not None:
        mounts.append(("reply/endpoint.sock", ENDPOINT_SOCKET_PATH, "rw"))
        env.append(f"{ENDPOINT_SOCKET_ENV}={ENDPOINT_SOCKET_PATH}")
    parts = [*_podman_run(job), *_pod_posture(job, '"${_step_ctr}"')]
    for entry, target, mode in mounts:
        parts += ["-v", _q(f"{job.wd}/{entry}:{target}:{mode}")]
    for assignment in env:
        parts += ["-e", _q(assignment)]
    if job.needs_gpu:
        parts += list(_GPU_DEVICES[job.profile["gpu"]["form"]])
    image = _q(job.step["image"]["image_id"])
    return [*parts, "--entrypoint", "python", image, "-m", "umrs.server.step_worker"]


def _job_spec(
    intent: dict[str, Any],
    step: Any,
    profile: dict[str, Any],
    wd_path: str,
    login: str,
    home: str,
    run_key: str | None = None,
    deadline_max: int | None = None,
    deadline_epoch: int | None = None,
) -> _Job:
    validate_intent(intent)
    validate_site_profile(profile)
    sobj = _step_named(intent, step)
    part = profile["partitions"].get(sobj["partition"])
    if part is None:
        raise ContractError(
            "resource_not_allowed", f"partition {sobj['partition']} is not in the profile"
        )
    if not _is_abs_path(wd_path) or not _is_abs_path(home):
        raise ValueError("the workdir and home must be plain absolute paths")
    _require(_LOGIN_RE, login, "login")
    iid = intent_id(canonical_bytes(intent, True)) if run_key is None else run_key
    _require(_HEX64_RE, iid, "run key")
    return _Job(
        iid, intent, sobj, profile, part, wd_path, login, home, deadline_max, deadline_epoch
    )


def render_job_script(
    intent: dict[str, Any],
    step: Any,
    profile: dict[str, Any],
    wd_path: str,
    login: str,
    home: str = "/tmp",
    run_key: str | None = None,
    deadline_max: int | None = None,
    deadline_epoch: int | None = None,
) -> str:
    """The gated job script for ``step`` (a name or the step object).

    Composed only from the signed intent and the site profile: every value is
    schema-checked and shell-quoted, nothing from a caller becomes a flag or a
    path. ``#!/bin/bash -p``, a clean ``env -i`` re-exec, the inlined venue
    preparation with ownership checks, the image ensure by literal image id
    from a private, hashed archive copy, the staged lane's posture, watchdog
    and forwarder. ``home`` is ``HOME`` in the job's clean environment.
    ``run_key`` (a grant run, plan section 6 "C/D.0", C2) names the job and
    its comment in place of the intent id; ``deadline_max`` (a grant run:
    the seconds left until the grant expires) caps ``--deadline``, so Slurm
    never starts a job that cannot end before the grant does. ``deadline_epoch``
    pins an absolute endpoint instead, so a slow pre-sbatch path cannot move it."""
    job = _job_spec(
        intent, step, profile, wd_path, login, home, run_key, deadline_max, deadline_epoch
    )
    lines = _directives(job) + _clean_env_lines(job) + _prologue(job)
    lines += _venue_paths(job) + _venue_owner_lines(job) + _container_conf_lines(job)
    lines += _adopt_runroot_lines() + _arch_lines(job) + _image_lines(job)
    lines += _seccomp_lines(job) + _pids_lines(job) + _stale_lines() + _watch_lines(job)
    lines += _gpu_lines(job) + _endpoint_lines(job)
    lines += [" ".join(_pod_argv(job)) + " 2>&1 | _step_log >&2", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate runtime for the B2 verbs: session, state files, locks, Slurm
# ---------------------------------------------------------------------------

_ABSENT = object()


@dataclass
class _Session:
    """One gate session of a B2 verb."""

    cfg: GateConfig
    home: str
    root: str
    stdin: IO[bytes]
    stdout: IO[bytes]
    end: float
    login: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    profile_sha: str = ""
    work: str = ""

    @property
    def caps(self) -> dict[str, int]:
        return dict(self.profile["caps"])

    def time_left(self) -> float:
        return self.end - time.monotonic()


@dataclass
class _Streamed:
    """A verb that wrote its own stdout; ``summary`` goes to stderr."""

    summary: dict[str, Any]


def _usage(text: str) -> GateError:
    return GateError("usage", _clip(text), EXIT_USAGE)


def _absent(text: str) -> GateError:
    return GateError("absent", _clip(text), EXIT_ABSENT)


def _failed(text: str) -> GateError:
    return GateError("failed", _clip(text), EXIT_FAILED)


def _arg(pattern: re.Pattern[str], value: str, label: str) -> str:
    if not pattern.fullmatch(value):
        raise _usage(f"invalid {label}: {value[:80]!r}")
    return value


def _arity(args: list[str], count: int, verb: str) -> list[str]:
    if len(args) != count:
        raise _usage(f"{verb} takes {count} argument(s)")
    return args


def load_site_profile(root: str, cfg: GateConfig) -> tuple[dict[str, Any], str]:
    """``site.json`` checked and its sha256; absent: exit 3, invalid: exit 1."""
    path = os.path.join(root, "site.json")
    try:
        data = _read_nofollow(path, PROFILE_CAP)
        obj = parse_strict(data)
    except FileNotFoundError:
        raise _absent("site.json is missing") from None
    except (OSError, ValueError) as exc:
        raise _failed(f"site.json: {exc}") from None
    validate_site_profile(obj)
    if obj["site"] != cfg.site or obj["domain"] != cfg.domain:
        raise _failed("site.json is for another site or signing domain")
    return obj, hashlib.sha256(data).hexdigest()


def _entry_problem(path: str, owners: set[int], sticky_ok: bool, is_dir: bool = True) -> str:
    """Why ``path`` (lstat, never followed) is not a directory (or regular
    file) owned by ``owners`` that nobody else can write; '' when it is. A
    root-owned sticky directory (``/tmp``) passes when ``sticky_ok``."""
    try:
        info = os.lstat(path)
    except OSError:
        return f"{path} does not exist"
    kind = "directory" if is_dir else "regular file"
    if not (stat.S_ISDIR if is_dir else stat.S_ISREG)(info.st_mode):
        return f"{path} is not a {kind} (or is a link)"
    if info.st_uid not in owners:
        return f"{path} is owned by uid {info.st_uid}"
    if _others_can_write(info, sticky_ok):
        return f"{path} is group- or other-writable"
    return ""


def _others_can_write(info: os.stat_result, sticky_ok: bool) -> bool:
    """Group- or other-writable, unless (``sticky_ok``) root-owned and
    sticky: nobody can then rename or remove another account's entry."""
    if not info.st_mode & 0o022:
        return False
    return not (sticky_ok and info.st_uid == 0 and info.st_mode & stat.S_ISVTX)


def ancestors_problem(path: str, owners: set[int]) -> str:
    """Why some ancestor of the absolute ``path`` could be changed by another
    account ('' when none can): each one up to ``/`` must be a directory (no
    link), owned by root or ``owners``, and not group- or other-writable
    unless root-owned and sticky. Without this, an account that can write a
    parent renames the checked entry away and puts its own in its place
    (fixer amendment 2026-11-08 to B2.0 section 2)."""
    current = os.path.dirname(path)
    while True:
        problem = _entry_problem(current, {0, *owners}, sticky_ok=True)
        if problem or current == "/":
            return problem
        current = os.path.dirname(current)


def _check_tool_dir(directory: str, owners: set[int]) -> None:
    problem = _entry_problem(directory, owners, sticky_ok=False) or ancestors_problem(
        directory, owners
    )
    if problem:
        raise refuse("profile_unsafe", f"slurm.path {problem}")


def check_slurm_path(profile: dict[str, Any]) -> None:
    """Every ``slurm.path`` directory exists, is no link, is owned by root (or
    a listed ``owner_uids`` uid), is not group- or other-writable, and sits
    under ancestors nobody else can change: a fake ``sbatch`` in an
    operator-writable directory would run as the user."""
    owners = {0, *profile["slurm"]["owner_uids"]}
    for directory in profile["slurm"]["path"].split(":"):
        _check_tool_dir(directory, owners)


def _require_private_dir(path: str) -> None:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise _failed(f"{path} is not a directory owned by this login")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise _failed(f"{path} must have mode 0700")
    problem = ancestors_problem(path, {os.geteuid()})
    if problem:
        raise _failed(f"the work directory sits where another account can change it: {problem}")


def _work_dir(root: str) -> str:
    """``work/``: a 0700 directory, or ONE symlink to such a directory of the
    login (scratch); resolved once per session, nothing below it followed.
    The resolved directory's every ancestor must be one nobody else can
    change (a group scratch directory without the sticky bit is refused)."""
    path = os.path.join(root, "work")
    if not os.path.lexists(path):
        os.mkdir(path, 0o700)
        os.chmod(path, 0o700)
    real = os.path.realpath(path)
    _require_private_dir(real)
    if not _is_abs_path(real):
        raise _failed("the work directory path has characters a job script cannot carry")
    return real


def _ensure_b2_layout(root: str) -> None:
    _ensure_layout(root)
    for sub in ("wd", "intents", "jobs", "images", "grants"):
        _ensure_dir(os.path.join(root, "state", sub))


def _open_session(
    cfg: GateConfig, home: str, stdin: IO[bytes], stdout: IO[bytes], end: float
) -> _Session:
    root = gate_root(home)
    session = _Session(cfg, home, root, stdin, stdout, end, login=_whoami()[0])
    session.profile, session.profile_sha = load_site_profile(root, cfg)
    check_slurm_path(session.profile)
    try:
        _ensure_b2_layout(root)
        session.work = _work_dir(root)
    except OSError as exc:
        raise _failed(str(exc)) from None
    return session


# -- state files ------------------------------------------------------------


def _state(root: str, *parts: str) -> str:
    return os.path.join(root, "state", *parts)


def _wd_record_path(s: _Session, wd: str) -> str:
    return _state(s.root, "wd", wd + ".json")


def _intent_dir(s: _Session, iid: str) -> str:
    return _state(s.root, "intents", iid)


def _nonce_path(s: _Session, iid: str, step: str) -> str:
    return os.path.join(_intent_dir(s, iid), step + ".json")


def _read_json(path: str, cap: int = STATE_CAP) -> Any:
    return _loads_no_duplicates(_read_nofollow(path, cap))


def _write_state(path: str, obj: Any) -> None:
    """Atomically (temp, fsync, rename), mode 0600."""
    _atomic_write(os.path.dirname(path), path, canonical_bytes(obj, True), prefix=".state")


def _create_exclusive(path: str, data: bytes) -> bool:
    """Create ``path`` holding ``data`` only if absent (``link`` of a fsynced
    temp file: the file appears whole or not at all). False when it exists."""
    directory = os.path.dirname(path)
    temp = os.path.join(directory, f".new.umrs-{os.getpid()}-{secrets.token_hex(4)}")
    fd = os.open(temp, _CREATE_FLAGS, 0o600)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temp, path)
    except FileExistsError:
        return False
    finally:
        _unlink_quiet(temp)
    _fsync_dir(directory)
    return True


def _new_record(iid: str, step: str) -> dict[str, Any]:
    return {
        "intent_id": iid,
        "step": step,
        "created": int(time.time()),
        "sealed": False,
        "broken": False,
        "removed": False,
        "stage_calls": 0,
        "src": {},
        "blobs": {},
        "job_id": "",
    }


def _load_record(s: _Session, wd: str) -> dict[str, Any]:
    """The workdir's record; a name the gate did not create (or removed) is absent."""
    try:
        rec = _read_json(_wd_record_path(s, wd))
    except FileNotFoundError:
        raise _absent(f"no gate workdir {wd}") from None
    except (OSError, ValueError) as exc:
        raise _failed(f"workdir record {wd}: {exc}") from None
    if not isinstance(rec, dict) or rec.get("removed"):
        raise _absent(f"no gate workdir {wd}")
    return rec


def _open_record(s: _Session, wd: str, step: str = "") -> dict[str, Any]:
    """The record of a workdir that still takes files: neither sealed nor
    broken (and, for ``submit``, made for ``step``: checked first)."""
    rec = _load_record(s, wd)
    if step and rec["step"] != step:
        raise refuse("wrong_step", f"workdir {wd} was made for step {rec['step']}")
    if rec["sealed"]:
        raise refuse("sealed", f"workdir {wd} is sealed")
    if rec["broken"]:
        raise refuse("broken", f"workdir {wd} is broken; only rmwd is accepted")
    return rec


def _save_record(s: _Session, wd: str, rec: dict[str, Any]) -> None:
    _write_state(_wd_record_path(s, wd), rec)


def _intent_body(s: _Session, iid: str) -> dict[str, Any]:
    """The signed intent this gate recorded at the first submit, or a grant
    run's run spec (plan section 6 "C/D.0", K2)."""
    try:
        stored = _read_json(os.path.join(_intent_dir(s, iid), "intent.json"))
        raw = stored["run_spec"] if "run_spec" in stored else stored["payload"]
        return parse_strict(base64.b64decode(raw, validate=True), True)
    except FileNotFoundError:
        raise refuse("not_finished", "nothing of this intent was submitted") from None
    except (OSError, ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise _failed(f"intent record: {exc}") from None


# -- locks --------------------------------------------------------------------


@contextlib.contextmanager
def _flocked(path: str, wait: float) -> Iterator[None]:
    """``flock`` ``path`` (created 0600), waiting at most ``wait`` s (``busy``)."""
    import fcntl

    fd = os.open(path, os.O_RDWR | os.O_CREAT | _O_NOFOLLOW | _O_CLOEXEC, 0o600)
    try:
        try:
            _lock(fcntl, fd, max(0.0, wait))
        except OSError:
            raise refuse("busy", f"{os.path.basename(path)} is held by another session") from None
        yield
    finally:
        os.close(fd)


def _wd_lock(s: _Session, wd: str) -> Any:
    return _flocked(_state(s.root, "wd", wd + ".lock"), LOCK_WAIT)


def _intent_lock(s: _Session, iid: str) -> Any:
    directory = _intent_dir(s, iid)
    _ensure_dir(directory)
    return _flocked(os.path.join(directory, ".lock"), s.time_left())


# -- logging ------------------------------------------------------------------


def _log_path(root: str) -> str:
    return os.path.join(root, "log", LOG_NAME)


def _append_unsigned(s: _Session, verb: str, detail: str) -> dict[str, Any]:
    """One fixed-form entry (at most 200 bytes) for a state-changing unsigned verb."""
    log = GateLog(_log_path(s.root))
    try:
        log.open(lock=False)
        log.record(verb, "ok", _clip(detail, UNSIGNED_DETAIL_CAP), int(time.time()))
        return log.head()
    except (OSError, ValueError) as exc:
        raise refuse("log_unwritable", str(exc)) from None
    finally:
        log.close()


def _log_size(root: str) -> int:
    try:
        return os.lstat(_log_path(root)).st_size
    except FileNotFoundError:
        return 0


# -- Slurm --------------------------------------------------------------------


@dataclass(frozen=True)
class _Proc:
    rc: int
    out: str
    err: str


def _trusted_tool(path: str, owners: set[int]) -> str:
    """The resolved file ``path`` names, when it and every directory on its
    resolved path are owned by root or ``owners`` and nobody else can write
    them; a link into an operator-writable tree is ``profile_unsafe``."""
    real = os.path.realpath(path)
    problem = _entry_problem(real, owners, sticky_ok=False, is_dir=False)
    problem = problem or _entry_problem(os.path.dirname(real), owners, sticky_ok=False)
    problem = problem or ancestors_problem(os.path.dirname(real), owners)
    if problem:
        raise refuse("profile_unsafe", f"slurm tool {path}: {problem}")
    return real


def _slurm_tool(s: _Session, tool: str) -> str:
    """The first ``slurm.path`` entry holding an executable ``tool``,
    resolved and checked (``_trusted_tool``); the gate executes the resolved
    path, so nothing can be swapped between the check and the exec."""
    owners = {0, *s.profile["slurm"]["owner_uids"]}
    for directory in s.profile["slurm"]["path"].split(":"):
        path = os.path.join(directory, tool)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return _trusted_tool(path, owners)
    raise GateError("scheduler", f"{tool} is not in slurm.path", EXIT_SCHEDULER)


def _slurm_env(s: _Session) -> dict[str, str]:
    """``env -i PATH=<slurm.path> HOME USER LOGNAME LC_ALL=C``: nothing else of
    the session reaches a Slurm command."""
    login = s.login
    return {
        "PATH": s.profile["slurm"]["path"],
        "HOME": s.home,
        "USER": login,
        "LOGNAME": login,
        "LC_ALL": "C",
    }


def _slurm(
    s: _Session, tool: str, args: list[str], timeout: float = SLURM_TIMEOUT, cwd: str = "/"
) -> _Proc:
    argv = [_slurm_tool(s, tool), *args]
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
            env=_slurm_env(s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _Proc(124, "", f"{tool} timed out")
    except (OSError, subprocess.SubprocessError) as exc:
        return _Proc(127, "", f"{tool}: {exc}")
    return _Proc(
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def _scheduler_error(tool: str, proc: _Proc) -> GateError:
    detail = f"{tool} failed (rc={proc.rc}): {proc.err.strip()[-300:]}"
    return GateError("scheduler", _clip(detail), EXIT_SCHEDULER)


def _first_state(text: str) -> str:
    lines = text.strip().splitlines()
    words = lines[0].split() if lines else []
    token = words[0].upper() if words else ""
    return token if _STATE_RE.fullmatch(token) else ""


def _is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES or state.startswith("CANCELLED")


def job_state(s: _Session, job_id: str) -> str:
    """squeue ``%T``, else ``sacct -n -X -P -o State``; a sacct failure exits 5."""
    queued = _slurm(s, "squeue", ["-h", "-j", job_id, "-o", "%T"])
    state = _first_state(queued.out) if queued.rc == 0 else ""
    if state:
        return state
    final = _slurm(s, "sacct", ["-n", "-X", "-P", "-j", job_id, "-o", "State"])
    if final.rc != 0:
        raise _scheduler_error("sacct", final)
    return _first_state(final.out) or "UNKNOWN"


def _job_live(s: _Session, job_id: str) -> bool:
    """squeue lists it (any state), or sacct reports a non-terminal state; a
    Slurm failure counts as live (fail closed)."""
    queued = _slurm(s, "squeue", ["-h", "-j", job_id, "-o", "%T"])
    if queued.rc == 0 and _first_state(queued.out):
        return True
    final = _slurm(s, "sacct", ["-n", "-X", "-P", "-j", job_id, "-o", "State"])
    if final.rc != 0:
        return True
    state = _first_state(final.out)
    return bool(state) and not _is_terminal(state)


def _own_job(s: _Session, job_id: str) -> str:
    """The workdir a job of this gate ran in; ``not_mine`` for any other id."""
    _arg(_JOB_ID_RE, job_id, "job id")
    try:
        return _read_nofollow(_state(s.root, "jobs", job_id), 4096).decode("ascii").strip()
    except (OSError, ValueError):
        raise refuse("not_mine", f"job {job_id} was not submitted by this gate") from None


def _require_completed(s: _Session, rec: dict[str, Any]) -> None:
    job_id = rec["job_id"]
    if not job_id:
        raise refuse("not_finished", "the workdir has no job")
    state = job_state(s, job_id)
    if state != "COMPLETED":
        raise refuse("not_finished", f"job {job_id} is {state}")


def _slurm_time(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))


def _rows_matching(text: str, sep: str, path: str) -> set[str]:
    found = set()
    for line in text.splitlines():
        parts = line.strip().split(sep, 1)
        if len(parts) == 2 and _JOB_ID_RE.fullmatch(parts[0]) and parts[1].rstrip("/") == path:
            found.add(parts[0])
    return found


def _match_workdir(s: _Session, path: str, submitted: int) -> tuple[str | None, bool]:
    """The one job whose WorkDir is ``path`` (squeue, then sacct from
    ``submitted - 60``): ``(job_id, True)``; ``(None, True)`` when Slurm knows
    none; ``(None, False)`` when Slurm did not answer or several match."""
    queued = _slurm(s, "squeue", ["--me", "-h", "-o", "%i %Z"])
    if queued.rc != 0:
        return None, False
    since = _slurm_time(max(0, submitted - 60))
    done = _slurm(s, "sacct", ["-n", "-X", "-P", "-S", since, "-o", "JobID,WorkDir"])
    if done.rc != 0:
        return None, False
    ids = _rows_matching(queued.out, " ", path) | _rows_matching(done.out, "|", path)
    if len(ids) == 1:
        return ids.pop(), True
    return None, not ids


def _cancel_quiet(s: _Session, job_id: str) -> None:
    with contextlib.suppress(GateError):
        _slurm(s, "scancel", [job_id])


# -- workdir files --------------------------------------------------------------


def _wd_path(s: _Session, wd: str) -> str:
    return os.path.join(s.work, wd)


def _open_dir_at(parent_fd: int | None, name: str) -> int:
    if parent_fd is None:
        return os.open(name, _DIR_FLAGS)
    return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)


def _open_path(base_fd: int, parts: list[str]) -> int:
    """A descriptor of ``parts`` below ``base_fd``; no link is followed."""
    fd = os.dup(base_fd)
    for part in parts:
        try:
            nxt = _open_dir_at(fd, part)
        finally:
            os.close(fd)
        fd = nxt
    return fd


def _open_wd_fd(s: _Session, wd: str) -> int:
    work_fd = _open_dir_at(None, s.work)
    try:
        return _open_dir_at(work_fd, wd)
    finally:
        os.close(work_fd)


def _free_bytes(fd: int) -> int:
    info = os.fstatvfs(fd)
    return info.f_bavail * info.f_frsize


def _open_plain(dir_fd: int, name: str) -> tuple[int, os.stat_result]:
    """A regular, singly-linked file, checked on its descriptor."""
    mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} is not a regular file")
    fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise ValueError(f"{name} is not a regular, singly-linked file")
    return fd, info


def _read_capped_fd(fd: int, cap: int) -> bytes:
    data = _read_upto(fd, cap)
    if len(data) > cap:
        raise refuse("too_large", f"file exceeds {cap} bytes")
    return data


def _write_new(base_fd: int, parts: list[str], data: bytes) -> None:
    """Create ``parts`` below ``base_fd`` (O_EXCL, O_NOFOLLOW, 0600), fsynced."""
    dir_fd = _open_path(base_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], _CREATE_FLAGS, 0o600, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_tree(path: str) -> None:
    """``rmtree`` (fd-based, links never followed); a tree the step made
    unreadable falls back to ``chmod -R u+rwX`` and ``rm -rf``."""
    try:
        shutil.rmtree(path)
    except Exception:  # RecursionError too: python3.9 recurses per level
        for argv in (
            ["chmod", "-R", "u+rwX", "--", path],
            ["rm", "-rf", "--one-file-system", "--", path],
        ):
            exe = shutil.which(argv[0], path="/usr/bin:/bin")
            if exe:
                _run([exe, *argv[1:]], b"", REMOVE_TIMEOUT, _tool_env())
    if os.path.lexists(path):
        raise _failed(f"{os.path.basename(path)} could not be removed")


# ---------------------------------------------------------------------------
# B2 verbs: mkwd
# ---------------------------------------------------------------------------


def _workdir_count(s: _Session) -> int:
    with os.scandir(s.work) as entries:
        return sum(1 for entry in entries if WD_RE.fullmatch(entry.name))


def _make_workdir(s: _Session, iid: str, step: str) -> str:
    work_fd = _open_dir_at(None, s.work)
    try:
        for _ in range(16):
            name = f"{iid[:16]}-{step}-{secrets.token_hex(4)}"
            with contextlib.suppress(FileExistsError):
                os.mkdir(name, 0o700, dir_fd=work_fd)
                _make_subdirs(work_fd, name)
                return name
    finally:
        os.close(work_fd)
    raise _failed("no free workdir name")


def _make_subdirs(work_fd: int, name: str) -> None:
    wd_fd = _open_dir_at(work_fd, name)
    try:
        for sub in WORKDIR_DIRS:
            os.mkdir(sub, 0o700, dir_fd=wd_fd)
        in_fd = _open_dir_at(wd_fd, "in")
        try:
            os.mkdir("blobs", 0o700, dir_fd=in_fd)
        finally:
            os.close(in_fd)
    finally:
        os.close(wd_fd)


def cmd_mkwd(s: _Session, args: list[str]) -> dict[str, Any]:
    """``mkwd <i> <step>``: a fresh workdir ``<i16>-<step>-<8 hex>``."""
    iid, step = _arity(args, 2, "mkwd")
    _arg(_HEX64_RE, iid, "intent id")
    _arg(STEP_RE, step, "step")
    if _log_size(s.root) >= LOG_FULL_BYTES:
        raise refuse("log_full", f"the gate log holds {LOG_FULL_BYTES} bytes")
    if os.path.lexists(_nonce_path(s, iid, step)):
        raise refuse("replayed", f"step {step} of this intent was already submitted")
    if _workdir_count(s) >= s.caps["max_workdirs"]:
        raise refuse("too_many_workdirs", f"at most {s.caps['max_workdirs']} workdirs")
    fd = _open_dir_at(None, s.work)
    try:
        free = _free_bytes(fd)
    finally:
        os.close(fd)
    if free < s.caps["min_free_bytes"]:
        raise refuse("low_space", f"{free} bytes free")
    wd = _make_workdir(s, iid, step)
    _save_record(s, wd, _new_record(iid, step))
    _append_unsigned(s, "mkwd", f"wd={wd}")
    return {"wd": wd}


# ---------------------------------------------------------------------------
# stage-in (B2.0 section 5.1)
# ---------------------------------------------------------------------------


class _CappedReader:
    """A stdin that refuses ``too_large`` past ``cap`` bytes."""

    def __init__(self, stream: IO[bytes], cap: int) -> None:
        self.stream, self.cap, self.total = stream, cap, 0
        self.head = b""

    def peek(self) -> bytes:
        self.head = self.stream.read(tarfile.BLOCKSIZE)
        self.total = len(self.head)
        return self.head

    def read(self, size: int = -1) -> bytes:
        want = self.cap + 1 if size is None or size < 0 else size
        head, self.head = self.head[:want], self.head[want:]
        rest = self.stream.read(want - len(head)) if want > len(head) else b""
        self.total += len(rest)
        if self.total > self.cap:
            raise refuse("too_large", f"stage-in exceeds {self.cap} bytes")
        return head + rest


def _stage_cap(caps: dict[str, int]) -> int:
    files = caps["max_src_files"] + caps["max_in_objects"]
    return caps["max_src_bytes"] + caps["max_in_bytes"] + 1024 * files + (1 << 20)


def _stage_target(member: tarfile.TarInfo) -> tuple[str, str]:
    """``("src", path)`` or ``("blob", sha)``; anything else is ``bad_member``."""
    name = member.name
    regular = member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and not member.issparse()
    if not regular:
        raise refuse("bad_member", f"{name[:120]!r} is not a regular file")
    blob = _BLOB_MEMBER_RE.fullmatch(name)
    if blob:
        return "blob", blob.group(1)
    if name.startswith("src/") and is_normal_key(name[4:]):
        return "src", name[4:]
    raise refuse("bad_member", f"{name[:120]!r} is not src/<path> or in/blobs/<sha256>")


def _make_dirs(base_fd: int, parts: list[str]) -> int:
    """Create (0700) and open the directories ``parts``; never through a link."""
    fd = os.dup(base_fd)
    for part in parts:
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd=fd)
            nxt = _open_dir_at(fd, part)
        except OSError:
            os.close(fd)
            raise refuse("bad_member", f"{part!r} is not a directory") from None
        os.close(fd)
        fd = nxt
    return fd


def _copy_hashed(source: IO[bytes], fd: int, size: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    written = 0
    while written < size:
        chunk = source.read(min(CHUNK, size - written))
        if not chunk:
            break
        digest.update(chunk)
        _write_all(fd, chunk)
        written += len(chunk)
    return digest.hexdigest(), written


class _Stager:
    """One ``stage-in`` call: every file created here is undone on failure."""

    def __init__(self, s: _Session, wd: str, wd_fd: int, rec: dict[str, Any]) -> None:
        self.s, self.wd, self.wd_fd, self.rec = s, wd, wd_fd, rec
        self.caps = s.caps
        self.created: list[list[str]] = []
        self.src: dict[str, list[Any]] = {}
        self.blobs: dict[str, int] = {}
        self.src_bytes = sum(int(entry[1]) for entry in rec["src"].values())
        self.blob_bytes = sum(int(size) for size in rec["blobs"].values())
        self.bytes = 0

    def add(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> None:
        kind, name = _stage_target(member)
        source = tar.extractfile(member)
        if source is None:
            raise refuse("bad_member", f"{member.name[:120]!r} has no data")
        if kind == "src":
            self._add_src(name, member.size, source)
        else:
            self._add_blob(name, member.size, source)

    def _add_src(self, path: str, size: int, source: IO[bytes]) -> None:
        if path in self.rec["src"] or path in self.src:
            raise refuse("bad_member", f"src/{path[:120]} is already staged")
        self.src_bytes += size
        files = len(self.rec["src"]) + len(self.src) + 1
        if files > self.caps["max_src_files"] or self.src_bytes > self.caps["max_src_bytes"]:
            raise refuse("too_large", "src/ passes max_src_files or max_src_bytes")
        sha = self._write(["src", *path.split("/")], size, source)
        self.src[path] = [sha, size]

    def _add_blob(self, sha: str, size: int, source: IO[bytes]) -> None:
        if sha in self.rec["blobs"] or sha in self.blobs:
            raise refuse("bad_member", f"blob {sha} is already staged")
        self.blob_bytes += size
        count = len(self.rec["blobs"]) + len(self.blobs) + 1
        over = count > self.caps["max_in_objects"] or self.blob_bytes > self.caps["max_in_bytes"]
        if over or size > self.caps["max_object_bytes"]:
            raise refuse("too_large", "in/ passes max_in_objects, max_in_bytes or max_object_bytes")
        if self._write(["in", "blobs", sha], size, source) != sha:
            raise refuse("digest_mismatch", f"blob {sha} does not hash to its name")
        self.blobs[sha] = size

    def _write(self, parts: list[str], size: int, source: IO[bytes]) -> str:
        dir_fd = _make_dirs(self.wd_fd, parts[:-1])
        try:
            fd = os.open(parts[-1], _CREATE_FLAGS, 0o600, dir_fd=dir_fd)
        except OSError:
            raise refuse("bad_member", f"{'/'.join(parts)[:120]} already exists") from None
        finally:
            os.close(dir_fd)
        self.created.append(parts)
        try:
            sha, written = _copy_hashed(source, fd, size)
            os.fsync(fd)
            free = _free_bytes(fd)
        finally:
            os.close(fd)
        if written != size:
            raise refuse("bad_member", f"{'/'.join(parts)[:120]} is truncated")
        if free < self.caps["min_free_bytes"]:
            raise refuse("low_space", f"{free} bytes free")
        self.bytes += size
        return sha

    def poison(self) -> None:
        """Unlink what this call created and mark the workdir ``broken``."""
        for parts in reversed(self.created):
            with contextlib.suppress(OSError):
                dir_fd = _open_path(self.wd_fd, parts[:-1])
                try:
                    os.unlink(parts[-1], dir_fd=dir_fd)
                finally:
                    os.close(dir_fd)
        self.rec["broken"] = True
        _save_record(self.s, self.wd, self.rec)

    def commit(self) -> dict[str, Any]:
        self.rec["src"].update(self.src)
        self.rec["blobs"].update(self.blobs)
        _save_record(self.s, self.wd, self.rec)
        return {
            "files": len(self.src) + len(self.blobs),
            "bytes": self.bytes,
            "free_bytes": _free_bytes(self.wd_fd),
        }


def _stage_failure(exc: BaseException) -> GateError:
    if isinstance(exc, GateError):
        return exc
    if isinstance(exc, (tarfile.TarError, EOFError, UnicodeError)):
        return refuse("bad_member", f"the stage-in stream is not an acceptable tar: {exc}")
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
        return refuse("low_space", str(exc))
    return _failed(type(exc).__name__)


def _stage_stream(stager: _Stager, stream: _CappedReader) -> None:
    with tarfile.open(fileobj=stream, mode="r|", encoding="utf-8", errors="strict") as tar:
        for member in tar:
            stager.add(tar, member)


def _stage(s: _Session, wd: str, rec: dict[str, Any]) -> dict[str, Any]:
    stream = _CappedReader(s.stdin, _stage_cap(s.caps))
    wd_fd = _open_wd_fd(s, wd)
    stager = _Stager(s, wd, wd_fd, rec)
    try:
        if stream.peek():
            _stage_stream(stager, stream)
        return stager.commit()
    except BaseException as exc:  # any failure poisons the workdir (B2.0 section 5.1)
        stager.poison()
        raise _stage_failure(exc) from None
    finally:
        os.close(wd_fd)


def cmd_stage_in(s: _Session, args: list[str]) -> dict[str, Any]:
    """``stage-in <wd>``: a tar of ``src/<path>`` and ``in/blobs/<sha256>`` files."""
    (wd,) = _arity(args, 1, "stage-in")
    _arg(WD_RE, wd, "workdir")
    with _wd_lock(s, wd):
        rec = _open_record(s, wd)
        if rec["stage_calls"] >= STAGE_CALLS_MAX:
            raise refuse("too_large", f"at most {STAGE_CALLS_MAX} stage-in calls per workdir")
        rec["stage_calls"] += 1
        _save_record(s, wd, rec)
        result = _stage(s, wd, rec)
    _append_unsigned(s, "stage-in", f"wd={wd} files={result['files']} bytes={result['bytes']}")
    return result


# ---------------------------------------------------------------------------
# submit (B2.0 section 5.3)
# ---------------------------------------------------------------------------


@dataclass
class _Submit:
    s: _Session
    wd: str
    step: str
    rec: dict[str, Any]
    payload: bytes
    armored: str
    credential: Any
    iid: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    sobj: dict[str, Any] = field(default_factory=dict)
    signer: str = ""
    replies: dict[str, Any] = field(default_factory=dict)
    outs: dict[str, Any] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    submitted: int = 0
    job_id: str = ""
    # a grant submit (plan section 6 "C/D.0", C2): ``payload`` and
    # ``armored`` are the signed grant, ``body`` the unsigned run spec and
    # ``iid`` the run key
    kind: str = "intent"
    spec: bytes = b""
    steps_raw: bytes = b""
    grant: dict[str, Any] = field(default_factory=dict)
    gid: str = ""
    grant_sha: str = ""
    spec_id: str = ""
    skeleton: dict[str, Any] = field(default_factory=dict)
    reserved: list[int] = field(default_factory=list)
    entry_key: str = ""
    holds: bool = False


def _read_submit_json(s: _Session) -> dict[str, Any]:
    data = s.stdin.read(SUBMIT_MAX_STDIN + 1)
    if len(data) > SUBMIT_MAX_STDIN:
        raise refuse("too_large", f"request exceeds {SUBMIT_MAX_STDIN} bytes")
    try:
        envelope = _loads_no_duplicates(data)
    except ValueError:
        raise refuse("bad_envelope", "the request is not a JSON object") from None
    if not isinstance(envelope, dict):
        raise refuse("bad_envelope", "the request is not a JSON object")
    return envelope


def _submit_envelope(s: _Session, wd: str, step: str, rec: dict[str, Any]) -> _Submit:
    """Steps 4-6: stdin cap, envelope keys and payload size, ``ssh-keygen``.
    An envelope with a ``grant`` key is a grant submit (C2)."""
    envelope = _read_submit_json(s)
    if "grant" in envelope:
        return _grant_envelope(s, wd, step, rec, envelope)
    if not set(envelope) <= {"payload", "signature", "endpoint_credential"}:
        raise refuse("bad_envelope", 'expected {"payload", "signature"[, "endpoint_credential"]}')
    payload, armored = _envelope_values(envelope.get("payload"), envelope.get("signature"))
    if len(payload) > INTENT_MAX_BYTES:
        raise refuse("too_large", f"the intent exceeds {INTENT_MAX_BYTES} bytes")
    ns = namespace("intent", s.cfg.domain)
    signers = os.path.join(s.root, "allowed_signers")
    ssh_keygen_verify(signers, s.cfg.principal, ns, armored, payload, _state(s.root))
    credential = envelope.get("endpoint_credential", _ABSENT)
    return _Submit(s, wd, step, rec, payload, armored, credential)


def _parse_intent(sub: _Submit) -> None:
    """Steps 7-9: SSHSIG policy, strict canonical JSON, ``validate_intent``."""
    sub.signer = _signature_policy(sub.s.cfg, sub.armored, "intent").signer_fingerprint
    try:
        body = parse_strict(sub.payload, json_values=True)
    except ValueError as exc:
        raise refuse("not_canonical", str(exc)) from None
    sub.body = validate_intent(body)
    sub.iid = intent_id(sub.payload)


def _bind_intent(sub: _Submit) -> None:
    """Step 10: site, principal, login, full intent id, step, profile sha."""
    s, body = sub.s, sub.body
    if body["site"] != s.cfg.site:
        raise refuse("wrong_site", f"intent is for site {body['site']}")
    if body["principal"] != s.cfg.principal:
        raise refuse("wrong_principal", "intent is for another principal")
    if body["login"] != s.login:
        raise refuse("wrong_login", f"intent is for login {body['login']}")
    if sub.iid != sub.rec["intent_id"]:
        raise refuse("wrong_intent", "the workdir was made for another intent")
    if sub.step not in {step["name"] for step in body["steps"]}:
        raise refuse("wrong_step", f"the intent has no step {sub.step}")
    if body["site_profile_sha256"] != s.profile_sha:
        raise refuse("profile_mismatch", "the intent was proposed against another site.json")
    sub.sobj = _step_named(body, sub.step)


def _read_started(s: _Session, iid: str) -> int | None:
    try:
        return int(_read_nofollow(os.path.join(_intent_dir(s, iid), "started"), 64).decode("ascii"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise _failed(f"started record: {exc}") from None


def _check_times(sub: _Submit) -> None:
    """Step 11 (under the intent lock): never before ``issued`` (120 s skew);
    the first submit by ``expires``, later ones within ``max_run_seconds``."""
    now, body = int(time.time()), sub.body
    if body["issued"] - CLOCK_SKEW > now:
        raise refuse("expired", "the intent is not valid yet")
    started = _read_started(sub.s, sub.iid)
    if started is None and now > body["expires"] + CLOCK_SKEW:
        raise refuse("expired", "the intent has expired")
    if started is not None and now > started + body["max_run_seconds"]:
        raise refuse("expired", "the run's max_run_seconds have passed")


def _profile_assocs(s: _Session) -> set[str]:
    args = ["-nP", "show", "assoc", f"user={s.login}", "format=cluster,account,partition,qos"]
    proc = _slurm(s, "sacctmgr", args, timeout=SACCTMGR_TIMEOUT)
    if proc.rc != 0:
        raise _scheduler_error("sacctmgr", proc)
    return {row["account"] for row in parse_assocs(proc.out)}


def _resource_problem(step: dict[str, Any], part: dict[str, Any], gpu_enabled: bool) -> str:
    if step["cpus"] > part["max_cpus"] or step["gpus"] > part["max_gpus"]:
        return "cpus or gpus above the partition maximum"
    if step["time"] > part["max_time"]:
        return "time above the partition maximum"
    if step["mem"] and not part["mem"]:
        return "the partition takes no --mem"
    if step["gpus"] > 0 and not gpu_enabled:
        return "the site enables no GPU steps"
    return ""


def _check_resources(sub: _Submit) -> None:
    """Step 12: the profile's partition limits, then the login's associations."""
    profile, step = sub.s.profile, sub.sobj
    part = profile["partitions"].get(step["partition"])
    problem = "the partition is not in the profile" if part is None else ""
    problem = problem or _resource_problem(step, part, profile["gpu"]["enabled"])
    if problem:
        raise refuse("resource_not_allowed", f"{step['partition']}: {problem}")
    if sub.body["account"] not in _profile_assocs(sub.s):
        raise refuse(
            "account_not_held", f"login {sub.s.login} holds no account {sub.body['account']}"
        )


def _recorded_reply(s: _Session, iid: str, step: str) -> Any:
    try:
        return _read_json(os.path.join(_intent_dir(s, iid), step + ".reply.json"), REPLY_CAP_BYTES)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise _failed(f"reply record of {step}: {exc}") from None


def _nonce_record(s: _Session, iid: str, step: str) -> dict[str, Any] | None:
    try:
        rec = _read_json(_nonce_path(s, iid, step))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise _failed(f"nonce record of {step}: {exc}") from None
    return rec if isinstance(rec, dict) else None


def _collected_out(sub: _Submit, parent: str) -> dict[str, Any]:
    record = _nonce_record(sub.s, sub.iid, parent) or {}
    out = record.get("out")
    if not isinstance(out, dict):
        raise refuse("parent_not_collected", f"the outputs of {parent} were not collected")
    bucket = sub.body["output_bucket"]
    for raw in reply_output_keys(sub.replies[parent], parent):
        if out_member(raw, bucket) not in out:
            raise refuse("parent_not_collected", f"{parent} output {raw[:120]!r} was not collected")
    return out


def _check_parents(sub: _Submit) -> None:
    """Step 13: every ``after`` step replied ``ok``; every ``chained`` step's
    outputs were collected, each ``output_keys`` entry among them."""
    for parent in sub.sobj["after"]:
        reply = _recorded_reply(sub.s, sub.iid, parent)
        if not isinstance(reply, dict) or reply.get("status") != "ok":
            raise refuse("parent_not_collected", f"step {parent} has no ok reply")
        sub.replies[parent] = reply
    for parent in sub.sobj["inputs"]["chained"]:
        sub.outs[parent] = _collected_out(sub, parent)


def _parent_dirs(paths: Iterator[str] | list[str]) -> set[str]:
    dirs: set[str] = set()
    for path in paths:
        parts = path.split("/")
        dirs.update("/".join(parts[:depth]) for depth in range(1, len(parts)))
    return dirs


def _walk_dir(
    fd: int, prefix: str, want: tuple[dict[str, int], set[str], str], stack: list[tuple[str, int]]
) -> int:
    files, dirs, refusal = want
    found = 0
    for name in os.listdir(fd):
        rel = prefix + name
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        size = files.get(rel)
        if stat.S_ISDIR(info.st_mode) and rel in dirs:
            stack.append((rel + "/", _open_dir_at(fd, name)))
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and size == info.st_size:
            found += 1
        else:
            raise refuse(refusal, f"unexpected entry {rel[:200]!r}")
    return found


def _walk_exact(base_fd: int, files: dict[str, int], dirs: set[str], refusal: str) -> None:
    """The tree below ``base_fd`` holds exactly ``files`` (regular, singly
    linked, of their size) and the directories ``dirs``; ``lstat`` only."""
    stack: list[tuple[str, int]] = [("", os.dup(base_fd))]
    found = 0
    try:
        while stack:
            prefix, fd = stack.pop()
            try:
                found += _walk_dir(fd, prefix, (files, dirs, refusal), stack)
            finally:
                os.close(fd)
    finally:
        for _prefix, fd in stack:
            os.close(fd)
    if found != len(files):
        raise refuse(refusal, "a staged file is missing")


def _check_src(sub: _Submit) -> None:
    """Step 14: the ``src`` record is the manifest, the entrypoint exists, and
    the disk holds exactly those files."""
    manifest = sub.body["src_manifest"]
    want = {entry["path"]: [entry["sha256"], entry["size"]] for entry in manifest["files"]}
    if sub.rec["src"] != want or sub.step not in manifest["entrypoints"]:
        raise refuse("src_mismatch", "the staged src/ is not the signed src_manifest")
    sizes = {path: int(entry[1]) for path, entry in want.items()}
    wd_fd = _open_wd_fd(sub.s, sub.wd)
    try:
        src_fd = _open_dir_at(wd_fd, "src")
        try:
            _walk_exact(src_fd, sizes, _parent_dirs(list(sizes)), "src_mismatch")
        finally:
            os.close(src_fd)
    finally:
        os.close(wd_fd)


def _need(required: dict[str, int], sha: str, size: int) -> None:
    if required.setdefault(sha, size) != size:
        raise refuse("input_mismatch", f"digest {sha} is required with two sizes")


def _required_blobs(sub: _Submit) -> tuple[dict[str, int], set[str]]:
    required: dict[str, int] = {}
    for obj in sub.sobj["inputs"]["objects"]:
        _need(required, obj["sha256"], obj["size"])
    bucket, run_id = sub.body["output_bucket"], sub.body["run_id"]
    chained_keys: set[str] = set()
    for parent in sub.sobj["inputs"]["chained"]:
        for raw in reply_output_keys(sub.replies[parent], parent):
            member = out_member(raw, bucket) or ""
            sha, size = sub.outs[parent][member]
            _need(required, sha, int(size))
            chained_keys.add(f"{bucket}/{physical_key(run_id, parent, member[len(bucket) + 1 :])}")
    return required, chained_keys


def _check_inputs(sub: _Submit) -> None:
    """Step 15: the blobs are exactly the signed digests and sizes, on disk too."""
    required, chained_keys = _required_blobs(sub)
    if {sha: int(size) for sha, size in sub.rec["blobs"].items()} != required:
        raise refuse("input_mismatch", "the staged blobs are not the signed inputs")
    listed = {f"{obj['bucket']}/{obj['key']}" for obj in sub.sobj["inputs"]["objects"]}
    if chained_keys & listed:
        raise refuse("input_mismatch", "a listed object has a chained output's physical key")
    files = {f"blobs/{sha}": size for sha, size in required.items()}
    wd_fd = _open_wd_fd(sub.s, sub.wd)
    try:
        in_fd = _open_dir_at(wd_fd, "in")
        try:
            _walk_exact(in_fd, files, {"blobs"}, "input_mismatch")
        finally:
            os.close(in_fd)
    finally:
        os.close(wd_fd)


def _hash_stream(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _tar_member_bytes(tar: tarfile.TarFile, name: Any, cap: int) -> bytes:
    member = tar.getmember(name) if isinstance(name, str) else None
    if member is None or not member.isreg() or member.size > cap:
        raise ValueError(f"archive member {str(name)[:80]!r} is missing or too large")
    source = tar.extractfile(member)
    return source.read() if source is not None else b""


def archive_config(handle: IO[bytes]) -> bytes:
    """The image config of a docker archive (``manifest.json``, exactly one entry)."""
    with tarfile.open(fileobj=handle, mode="r:") as tar:
        entries = _loads_no_duplicates(_tar_member_bytes(tar, "manifest.json", PROFILE_CAP))
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise ValueError("manifest.json must hold exactly one image")
        return _tar_member_bytes(tar, entries[0].get("Config"), 16 << 20)


def image_labels_ok(config: bytes) -> bool:
    """``org.umrs.step-contract`` = "1" and ``json`` in ``org.umrs.step-payload``."""
    try:
        obj = _loads_no_duplicates(config)
    except ValueError:
        return False
    inner = obj.get("config") if isinstance(obj, dict) else None
    labels = inner.get("Labels") if isinstance(inner, dict) else None
    if not isinstance(labels, dict) or labels.get(STEP_CONTRACT_LABEL) != STEP_CONTRACT_VERSION:
        return False
    payload = labels.get(STEP_PAYLOAD_LABEL)
    return isinstance(payload, str) and "json" in payload.split(",")


def _stat_key(info: os.stat_result) -> list[str]:
    """``(dev, ino, size, mtime_ns, ctime_ns)`` as decimal strings (nanosecond
    times pass the canonical int range)."""
    fields = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return [str(value) for value in fields]


def _cache_path(s: _Session, image: dict[str, Any]) -> str:
    return _state(s.root, "images", image["archive_sha256"] + ".ok")


def _cache_hit(s: _Session, image: dict[str, Any], info: os.stat_result) -> bool:
    try:
        cached = _read_json(_cache_path(s, image), 4096)
    except (OSError, ValueError):
        return False
    return cached == {"stat": _stat_key(info), "image_id": image["image_id"]}


def _check_archive(handle: IO[bytes], image: dict[str, Any]) -> None:
    if _hash_stream(handle) != image["archive_sha256"]:
        raise refuse("image_mismatch", f"{image['archive_sha256']}.tar does not hash to its name")
    handle.seek(0)
    try:
        config = archive_config(handle)
    except (tarfile.TarError, ValueError, KeyError, OSError) as exc:
        raise refuse("image_mismatch", f"not a docker archive: {exc}") from None
    if hashlib.sha256(config).hexdigest() != image["image_id"]:
        raise refuse(
            "image_mismatch", f"the archive's config is not image {image['image_id'][:12]}"
        )
    if not image_labels_ok(config):
        raise refuse(
            "image_label", f"image {image['image_id'][:12]} lacks the step contract labels"
        )


def _check_image(s: _Session, image: dict[str, Any]) -> None:
    """Step 16 for one image: the archive is ``<archive_sha256>.tar`` in the
    profile's archive directory (a link there is followed: only content
    counts), its config hashes to ``image_id``, and its labels hold."""
    path = os.path.join(s.profile["image_archive_dir"], image["archive_sha256"] + ".tar")
    try:
        handle = open(path, "rb")  # noqa: SIM115 - closed below
    except OSError:
        raise refuse("image_absent", f"no image archive {image['archive_sha256']}.tar") from None
    with handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise refuse("image_absent", f"{image['archive_sha256']}.tar is not a file")
        if _cache_hit(s, image, info):
            return
        _check_archive(handle, image)
    with contextlib.suppress(OSError, ValueError):
        _write_state(
            _cache_path(s, image), {"stat": _stat_key(info), "image_id": image["image_id"]}
        )


def _check_images(sub: _Submit) -> None:
    images = [sub.sobj["image"], sub.body["trusted_image"]]
    for index, image in enumerate(images):
        if image not in images[:index]:
            _check_image(sub.s, image)


def _check_credential(sub: _Submit) -> None:
    """Step 17: a credential exactly when the step has an endpoint, shaped
    ``{"headers": {name: value}}`` as the forwarder's ``check_headers`` wants."""
    present = sub.credential is not _ABSENT
    if present != ("endpoint" in sub.sobj):
        raise refuse("bad_envelope", "endpoint_credential must come exactly with a step endpoint")
    if present:
        check_credential(sub.credential)


def _header_ok(name: Any, value: Any) -> bool:
    if not isinstance(name, str) or not isinstance(value, str):
        return False
    return bool(_HEADER_NAME_RE.fullmatch(name) and _HEADER_VALUE_RE.fullmatch(value))


def check_credential(cred: Any) -> None:
    """``{"headers": {name: value}}`` as ``step_endpoint_forwarder.check_headers``
    accepts them (at most ``MAX_HEADERS``); else refuse ``bad_envelope``."""
    headers = cred.get("headers") if isinstance(cred, dict) and set(cred) == {"headers"} else None
    if not isinstance(headers, dict) or len(headers) > MAX_HEADERS:
        raise refuse("bad_envelope", 'endpoint_credential must be {"headers": {...}}')
    for name, value in headers.items():
        if not _header_ok(name, value):
            raise refuse(
                "bad_envelope", f"credential header {str(name)[:40]!r} cannot be forwarded"
            )


def _compose(sub: _Submit) -> None:
    """Step 18: ``input.json`` (capped), ``in/manifest.json`` and the script."""
    s = sub.s
    sub.files["input.json"] = input_json(sub.body, sub.sobj, sub.replies)
    sub.files["manifest.json"] = manifest_json(sub.body, sub.sobj, sub.replies, sub.outs)
    expiry = sub.grant["expires"] if sub.gid else None
    script = render_job_script(
        sub.body,
        sub.sobj,
        s.profile,
        _wd_path(s, sub.wd),
        s.login,
        s.home,
        run_key=sub.iid,
        deadline_epoch=expiry,
    )
    sub.files["job.sbatch"] = script.encode("utf-8")


def _submit_checks(sub: _Submit) -> None:
    """Steps 11-19, under the intent lock."""
    _check_times(sub)
    _check_resources(sub)
    _check_parents(sub)
    _check_src(sub)
    _check_inputs(sub)
    _check_images(sub)
    _check_credential(sub)
    _compose(sub)
    if sub.s.time_left() < SUBMIT_RESERVE:
        raise _failed("deadline")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _stored_request(sub: _Submit) -> list[tuple[str, bytes]]:
    """What step 20 records beside the nonce: ``started``, ``intent.json``
    (the signed intent, or for a grant run the grant, the run spec and the
    skeleton) and, for a grant run, ``grant`` (its id)."""
    files = [("started", str(sub.submitted).encode("ascii"))]
    if not sub.gid:
        stored = {"payload": _b64(sub.payload), "signature": sub.armored}
        return [*files, ("intent.json", canonical_bytes(stored))]
    stored_grant = {
        "grant": {"payload": _b64(sub.payload), "signature": sub.armored},
        "run_spec": _b64(sub.spec),
        "workflow_steps": _b64(sub.steps_raw),
    }
    return [
        *files,
        ("intent.json", canonical_bytes(stored_grant)),
        ("grant", sub.gid.encode("ascii")),
    ]


def _claim_nonce(sub: _Submit) -> None:
    """Step 20: the (intent, step) nonce, O_EXCL; ``started`` and
    ``intent.json`` when absent. A grant run's refusal here deletes this
    call's reservation."""
    directory = _intent_dir(sub.s, sub.iid)
    sub.submitted = int(time.time())
    record = {"wd": sub.wd, "job_id": "", "submitted": sub.submitted}
    if not _create_exclusive(_nonce_path(sub.s, sub.iid, sub.step), canonical_bytes(record)):
        _drop_reservation(sub)
        raise refuse("replayed", f"step {sub.step} of this intent was already submitted")
    for name, data in _stored_request(sub):
        path = os.path.join(directory, name)
        if _create_exclusive(path, data):
            sub.created.append(path)


def _release_nonce(sub: _Submit) -> None:
    """Nothing was queued: the step may be submitted again from a fresh
    workdir; this one stays sealed. A grant run's reservation goes too."""
    for path in [_nonce_path(sub.s, sub.iid, sub.step), *sub.created]:
        _unlink_quiet(path)
    _drop_reservation(sub)
    sub.rec["sealed"] = True
    with contextlib.suppress(OSError, ValueError):
        _save_record(sub.s, sub.wd, sub.rec)


def _write_job_files(sub: _Submit) -> None:
    """Step 21: write the gate's files and seal the workdir."""
    wd_fd = _open_wd_fd(sub.s, sub.wd)
    try:
        _write_new(wd_fd, ["in", "manifest.json"], sub.files["manifest.json"])
        _write_new(wd_fd, ["input.json"], sub.files["input.json"])
        if sub.credential is not _ABSENT:
            _write_new(wd_fd, [ENDPOINT_CREDENTIAL_FILE], canonical_bytes(sub.credential))
        _write_new(wd_fd, ["job.sbatch"], sub.files["job.sbatch"])
    finally:
        os.close(wd_fd)
    sub.rec["sealed"] = True
    _save_record(sub.s, sub.wd, sub.rec)


def _parsable_job_id(proc: _Proc) -> str:
    lines = proc.out.strip().splitlines()
    job_id = lines[0].split(";")[0].strip() if proc.rc == 0 and lines else ""
    return job_id if _JOB_ID_RE.fullmatch(job_id) else ""


def _run_sbatch(sub: _Submit) -> str:
    """Step 22: ``sbatch --parsable --export=NONE --chdir=<wd>`` in a clean
    environment. A failed or timed-out sbatch may still have queued the job:
    the workdir match finds it. Only a certain "no job" frees the nonce."""
    path = _wd_path(sub.s, sub.wd)
    args = ["--parsable", "--export=NONE", f"--chdir={path}", f"{path}/job.sbatch"]
    proc = _slurm(sub.s, "sbatch", args, timeout=SBATCH_TIMEOUT, cwd=path)
    job_id = _parsable_job_id(proc)
    if job_id:
        return job_id
    matched, certain = _match_workdir(sub.s, path, sub.submitted)
    if matched:
        return matched
    if certain and proc.rc != 124:
        _release_nonce(sub)
    raise _scheduler_error("sbatch", proc)


def _record_job(s: _Session, iid: str, step: str, wd: str, job_id: str) -> None:
    """Step 23: ``state/jobs/<id>`` and the job id in the nonce; for a grant
    run, the job's grant entry becomes ``live`` (C2 step 23)."""
    path = _state(s.root, "jobs", job_id)
    _atomic_write(os.path.dirname(path), path, wd.encode("ascii"), prefix=".job")
    nonce = _nonce_record(s, iid, step) or {}
    nonce.update({"wd": wd, "job_id": job_id})
    _write_state(_nonce_path(s, iid, step), nonce)
    gid = _grant_of(s, iid)
    if gid:
        _grant_job_live(s, gid, f"{_intent_body(s, iid)['run_id']}/{step}", job_id)


def _submit_act(sub: _Submit) -> None:
    """Steps 20-23."""
    _claim_nonce(sub)
    try:
        _write_job_files(sub)
    except BaseException:
        _release_nonce(sub)
        raise
    sub.job_id = _run_sbatch(sub)
    _record_job(sub.s, sub.iid, sub.step, sub.wd, sub.job_id)
    sub.rec["job_id"] = sub.job_id
    _save_record(sub.s, sub.wd, sub.rec)


def _submit_signed(sub: _Submit) -> None:
    """Steps 7-23 on a request whose signature ``ssh-keygen`` verified."""
    if sub.kind == "grant":
        _submit_grant(sub)
        return
    _parse_intent(sub)
    _bind_intent(sub)
    with _intent_lock(sub.s, sub.iid):
        _submit_checks(sub)
        _submit_act(sub)


def _submit_logged(
    s: _Session, wd: str, step: str, rec: dict[str, Any], log: GateLog
) -> dict[str, Any]:
    try:
        sub = _submit_envelope(s, wd, step, rec)
    except Exception as exc:  # every attempt is logged, whatever failed
        err = _as_gate_error(exc)
        _log_unsigned(log, "submit", err)
        raise err from None
    try:
        _submit_signed(sub)
    except Exception as exc:
        err = _as_gate_error(exc)
        _log_refusal(log, "submit", err, f"{err.error}: {err.detail}")
        raise err from None
    try:
        log.record("submit", "ok", _submit_detail(sub), int(time.time()))
    except (OSError, ValueError) as exc:  # fail closed: the job does not run unlogged
        _cancel_quiet(s, sub.job_id)
        _settle_zero(sub)
        raise refuse("log_unwritable", str(exc)) from None
    return _submit_answer(sub, log.head())


def _submit_detail(sub: _Submit) -> str:
    """The fixed-form ``submit ok`` detail; a grant run's carries the
    reservation, for the witness (K17)."""
    if not sub.gid:
        return f"intent={sub.iid} step={sub.step} wd={sub.wd} job={sub.job_id} signer={sub.signer}"
    gpu_s, core_s = sub.reserved
    return (
        f"grant={sub.gid} run={sub.body['run_id']} spec={sub.spec_id} step={sub.step} "
        f"wd={sub.wd} job={sub.job_id} gpu_s={gpu_s} core_s={core_s} signer={sub.signer}"
    )


def _submit_answer(sub: _Submit, head: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"job_id": sub.job_id, "wd": sub.wd, "step": sub.step, "log_head": head}
    if not sub.gid:
        return {**out, "intent_id": sub.iid}
    reserved = {"gpu_seconds": sub.reserved[0], "core_seconds": sub.reserved[1]}
    return {**out, "grant_id": sub.gid, "run_key": sub.iid, "reserved": reserved}


def cmd_submit(s: _Session, args: list[str]) -> dict[str, Any]:
    """``submit <wd> <step>``: verify the signed intent in contract order
    (B2.0 section 5.3), seal the workdir and ``sbatch`` the gate's script."""
    wd, step = _arity(args, 2, "submit")
    _arg(WD_RE, wd, "workdir")
    _arg(STEP_RE, step, "step")
    with _wd_lock(s, wd):
        rec = _open_record(s, wd, step)
        log = GateLog(_log_path(s.root))
        try:
            log.open(lock=False)
        except (OSError, ValueError) as exc:
            raise refuse("log_unwritable", str(exc)) from None
        try:
            return _submit_logged(s, wd, step, rec, log)
        finally:
            log.close()


# ---------------------------------------------------------------------------
# lookup, status, cancel, usage
# ---------------------------------------------------------------------------


def _resolve_unknown_job(s: _Session, iid: str, step: str, nonce: dict[str, Any]) -> str:
    """A nonce left with ``job_id: ""`` (an sbatch that timed out): the one
    job Slurm ran in that workdir, recorded; "" when none or several."""
    matched, _certain = _match_workdir(s, _wd_path(s, nonce["wd"]), int(nonce["submitted"]))
    if not matched:
        return ""
    _record_job(s, iid, step, nonce["wd"], matched)
    with contextlib.suppress(GateError, OSError, ValueError), _wd_lock(s, nonce["wd"]):
        rec = _load_record(s, nonce["wd"])
        rec["job_id"] = matched
        _save_record(s, nonce["wd"], rec)
    _append_unsigned(s, "lookup", f"intent={iid[:16]} step={step} job={matched}")
    return matched


def cmd_lookup(s: _Session, args: list[str]) -> dict[str, Any]:
    """``lookup <i> <step>``: the nonce's workdir and job, or exit 3."""
    iid, step = _arity(args, 2, "lookup")
    _arg(_HEX64_RE, iid, "intent id")
    _arg(STEP_RE, step, "step")
    if _nonce_record(s, iid, step) is None:
        raise _absent(f"step {step} of this intent has no nonce")
    with _intent_lock(s, iid):
        nonce = _nonce_record(s, iid, step) or {}
        job_id = nonce.get("job_id") or _resolve_unknown_job(s, iid, step, nonce)
    state = job_state(s, job_id) if job_id else "UNKNOWN"
    return {"wd": nonce["wd"], "job_id": job_id, "submitted": nonce["submitted"], "state": state}


def cmd_status(s: _Session, args: list[str]) -> dict[str, Any]:
    (job_id,) = _arity(args, 1, "status")
    _own_job(s, job_id)
    return {"job_id": job_id, "state": job_state(s, job_id)}


def cmd_cancel(s: _Session, args: list[str]) -> dict[str, Any]:
    (job_id,) = _arity(args, 1, "cancel")
    _own_job(s, job_id)
    proc = _slurm(s, "scancel", [job_id])
    if proc.rc != 0:
        raise _scheduler_error("scancel", proc)
    _append_unsigned(s, "cancel", f"job={job_id}")
    return {"job_id": job_id, "cancelled": True}


_USAGE_FIELDS = (
    "job_id",
    "state",
    "elapsed",
    "alloc_tres",
    "alloc_cpus",
    "account",
    "partition",
    "start",
    "end",
)


_USAGE_FORMAT = "JobID,State,Elapsed,AllocTRES,AllocCPUS,Account,Partition,Start,End"


def _usage_proc(s: _Session, job_ids: list[str]) -> _Proc:
    return _slurm(s, "sacct", ["-n", "-X", "-P", "-j", ",".join(job_ids), "-o", _USAGE_FORMAT])


def _usage_rows(text: str, job_ids: list[str]) -> list[dict[str, str]]:
    wanted = set(job_ids)
    rows = [line.strip().split("|") for line in text.splitlines()]
    return [
        {key: row[index] for index, key in enumerate(_USAGE_FIELDS)}
        for row in rows
        if len(row) == len(_USAGE_FIELDS) and row[0] in wanted
    ]


def cmd_usage(s: _Session, args: list[str]) -> dict[str, Any]:
    """``usage <jobid>...``: raw ``sacct`` accounting of this gate's own jobs.
    Each row also names its grant (``""`` for an intent job) and whether the
    grant's counters hold it ``settled``; a terminal row settles its entry
    (plan section 6 "C/D.0", C3)."""
    if not 1 <= len(args) <= USAGE_MAX_JOBS:
        raise _usage(f"usage takes 1..{USAGE_MAX_JOBS} job ids")
    for job_id in args:
        _own_job(s, job_id)
    proc = _usage_proc(s, args)
    if proc.rc != 0:
        raise _scheduler_error("sacct", proc)
    jobs = _usage_rows(proc.out, args)
    _settle_rows(s, jobs)
    return {"jobs": [_usage_grant_fields(s, job) for job in jobs]}


# ---------------------------------------------------------------------------
# tail-log, read-reply, collect-out (B2.0 section 5.2)
# ---------------------------------------------------------------------------


def _read_tail(s: _Session, wd: str, count: int) -> bytes:
    wd_fd = _open_wd_fd(s, wd)
    try:
        log_fd = _open_dir_at(wd_fd, "log")
    finally:
        os.close(wd_fd)
    try:
        fd, info = _open_plain(log_fd, "slurm.err")
    except FileNotFoundError:
        return b""
    finally:
        os.close(log_fd)
    try:
        os.lseek(fd, max(0, info.st_size - count), os.SEEK_SET)
        return _read_upto(fd, count)[:count]
    finally:
        os.close(fd)


def cmd_tail_log(s: _Session, args: list[str]) -> dict[str, Any]:
    if not 1 <= len(args) <= 2:
        raise _usage("tail-log takes <wd> [<n>]")
    wd = _arg(WD_RE, args[0], "workdir")
    cap = s.caps["log_tail_bytes"]
    count = int(_arg(_JOB_ID_RE, args[1], "byte count")) if len(args) == 2 else cap
    if count > cap:
        raise _usage(f"tail-log reads at most {cap} bytes")
    _load_record(s, wd)
    data = _read_tail(s, wd, count) if count else b""
    return {"tail": data.decode("utf-8", "replace")}


def _step_dir(s: _Session, wd: str, name: str, refusal: str) -> int:
    """``out/`` or ``reply/``: the step owns them, so a link or a withheld
    permission there is the step's doing, a refusal."""
    wd_fd = _open_wd_fd(s, wd)
    try:
        return _open_dir_at(wd_fd, name)
    except OSError as exc:
        raise refuse(refusal, f"{name}/ cannot be opened: {exc.strerror}") from None
    finally:
        os.close(wd_fd)


def _read_reply_bytes(s: _Session, wd: str) -> bytes:
    reply_fd = _step_dir(s, wd, "reply", "bad_reply")
    try:
        fd, _info = _open_plain(reply_fd, "output.json")
    except (OSError, ValueError) as exc:
        raise refuse("bad_reply", f"reply/output.json: {exc}") from None
    finally:
        os.close(reply_fd)
    try:
        return _read_capped_fd(fd, s.caps["reply_bytes"])
    finally:
        os.close(fd)


def _check_reply(data: bytes) -> None:
    try:
        obj = _loads_no_duplicates(data)
    except (ValueError, RecursionError) as exc:
        raise refuse("bad_reply", f"the reply is not strict JSON: {exc}") from None
    if not isinstance(obj, dict):
        raise refuse("bad_reply", "the reply is not a JSON object")


def _record_reply(s: _Session, rec: dict[str, Any], data: bytes) -> None:
    """The first record wins; a later read that differs is ``digest_mismatch``."""
    iid, step = rec["intent_id"], rec["step"]
    path = os.path.join(_intent_dir(s, iid), step + ".reply.json")
    digest = hashlib.sha256(data).hexdigest()
    if not _create_exclusive(path, data):
        if hashlib.sha256(_read_nofollow(path, REPLY_CAP_BYTES)).hexdigest() != digest:
            raise refuse("digest_mismatch", "reply/output.json changed since it was recorded")
        return
    nonce = _nonce_record(s, iid, step) or {}
    nonce["reply_sha256"] = digest
    _write_state(_nonce_path(s, iid, step), nonce)


def cmd_read_reply(s: _Session, args: list[str]) -> bytes:
    """``read-reply <wd>``: the completed job's reply, recorded and returned."""
    (wd,) = _arity(args, 1, "read-reply")
    _arg(WD_RE, wd, "workdir")
    with _wd_lock(s, wd):
        rec = _load_record(s, wd)
        _require_completed(s, rec)
        data = _read_reply_bytes(s, wd)
        _check_reply(data)
        with _intent_lock(s, rec["intent_id"]):
            _record_reply(s, rec, data)
        _settle_job_quiet(s, rec["job_id"])
    _append_unsigned(s, "read-reply", f"wd={wd} sha256={hashlib.sha256(data).hexdigest()[:16]}")
    return data


class _OutScan:
    """``out/`` as ``umrs_step_helper.scan_out`` sees it, plus the bucket rule."""

    def __init__(self, caps: dict[str, int], bucket: str) -> None:
        self.caps, self.bucket = caps, bucket
        self.files: list[tuple[str, int]] = []
        self.dirs = 0
        self.bytes = 0

    def entry(self, fd: int, prefix: str, name: str, stack: list[tuple[str, int]]) -> None:
        rel = _out_rel(prefix, name)
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if info.st_uid != os.geteuid():
            raise refuse("bad_member", f"out/{rel[:120]} is not this login's")
        if stat.S_ISDIR(info.st_mode):
            self._add_dir(rel, info)
            stack.append((rel + "/", _open_dir_at(fd, name)))
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            self._add_file(rel, info)
        else:
            raise refuse("bad_member", f"out/{rel[:120]} is not a regular, singly-linked file")

    def _add_dir(self, rel: str, info: os.stat_result) -> None:
        self.dirs += 1
        if self.dirs > self.caps["max_out_files"]:
            raise refuse(
                "too_large", f"out/ holds more than {self.caps['max_out_files']} directories"
            )
        if info.st_mode & 0o500 != 0o500:
            raise refuse("bad_member", f"out/{rel[:120]} is hidden (not u+rx)")

    def _add_file(self, rel: str, info: os.stat_result) -> None:
        bucket, _, key = rel.partition("/")
        if bucket != self.bucket or not is_normal_key(key):
            raise refuse("bad_member", f"out/{rel[:120]} is not <output_bucket>/<key>")
        if not info.st_mode & 0o400:
            raise refuse("bad_member", f"out/{rel[:120]} is not readable")
        self.files.append((rel, info.st_size))
        self.bytes += info.st_size
        caps = self.caps
        if len(self.files) > caps["max_out_files"] or info.st_size > caps["max_out_file_bytes"]:
            raise refuse("too_large", "out/ passes max_out_files or max_out_file_bytes")
        if self.bytes > caps["max_out_bytes"]:
            raise refuse("too_large", f"out/ holds more than {caps['max_out_bytes']} bytes")


def _out_rel(prefix: str, name: str) -> str:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        raise refuse("bad_member", f"out/ entry {name!r} is not a UTF-8 name") from None
    return prefix + name


def _scan_out(out_fd: int, scan: _OutScan) -> list[tuple[str, int]]:
    stack: list[tuple[str, int]] = [("", os.dup(out_fd))]
    try:
        while stack:
            prefix, fd = stack.pop()
            try:
                for name in sorted(os.listdir(fd)):
                    scan.entry(fd, prefix, name, stack)
            finally:
                os.close(fd)
    finally:
        for _prefix, fd in stack:
            os.close(fd)
    return sorted(scan.files)


class _Hashing:
    """A reader that hashes what passes through it."""

    def __init__(self, handle: IO[bytes]) -> None:
        self.handle, self.digest = handle, hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.handle.read(size)
        self.digest.update(chunk)
        return chunk


def _add_out_file(tar: tarfile.TarFile, out_fd: int, rel: str, size: int) -> str:
    parts = rel.split("/")
    dir_fd = _open_path(out_fd, parts[:-1])
    try:
        fd, info = _open_plain(dir_fd, parts[-1])
    except (OSError, ValueError) as exc:
        raise refuse("bad_member", f"out/{rel[:120]}: {exc}") from None
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as handle:
        if info.st_size != size:
            raise refuse("bad_member", f"out/{rel[:120]} changed while it was collected")
        member = tarfile.TarInfo(rel)
        member.size, member.mode, member.mtime = size, 0o600, 0
        reader = _Hashing(handle)
        tar.addfile(member, reader)  # type: ignore[arg-type]
    return reader.digest.hexdigest()


def _stream_out(out_fd: int, files: list[tuple[str, int]], sink: IO[bytes]) -> dict[str, list[Any]]:
    digests: dict[str, list[Any]] = {}
    with tarfile.open(fileobj=sink, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for rel, size in files:
            digests[rel] = [_add_out_file(tar, out_fd, rel, size), size]
    return digests


def _record_out(s: _Session, rec: dict[str, Any], digests: dict[str, list[Any]]) -> None:
    iid, step = rec["intent_id"], rec["step"]
    with _intent_lock(s, iid):
        nonce = _nonce_record(s, iid, step)
        if nonce is None:
            raise _failed("the workdir's nonce is missing")
        if "out" in nonce:
            if nonce["out"] != digests:
                raise refuse("digest_mismatch", "out/ changed since it was recorded")
            return
        nonce["out"] = digests
        _write_state(_nonce_path(s, iid, step), nonce)


def cmd_collect_out(s: _Session, args: list[str]) -> _Streamed:
    """``collect-out <wd>``: a tar of ``out/`` on stdout, every file hashed
    while it streams and the digests recorded (the first record wins)."""
    (wd,) = _arity(args, 1, "collect-out")
    _arg(WD_RE, wd, "workdir")
    with _wd_lock(s, wd):
        rec = _load_record(s, wd)
        _require_completed(s, rec)
        bucket = _intent_body(s, rec["intent_id"])["output_bucket"]
        out_fd = _step_dir(s, wd, "out", "bad_member")
        try:
            files = _scan_out(out_fd, _OutScan(s.caps, bucket))
            digests = _stream_out(out_fd, files, s.stdout)
        finally:
            os.close(out_fd)
        _record_out(s, rec, digests)
        _grant_after_collect(s, rec, digests)
    total = sum(size for _rel, size in files)
    _append_unsigned(s, "collect-out", f"wd={wd} files={len(files)} bytes={total}")
    return _Streamed({"files": len(files), "bytes": total})


# ===========================================================================
# Waves C and D: grants in the gate (plan section 6 "C/D.0": C2, C3, K1-K20)
# ===========================================================================
#
# State in ``state/grants/``: ``<id>.json`` (the counters, replaced by
# rename), ``<id>.lock`` (their ``flock``, X3), ``<id>.cancelled``,
# ``<id>/outputs/<sha256>`` (the ``chained`` index) and ``epoch`` (K7). Lock
# order: workdir, run, grant. The grant lock is never held across sbatch,
# image hashing or sacct (K5).

_SETTLE_FIRST = ("concurrent", "gpu_hours", "core_hours")
_GRANT_ENVELOPE_KEYS = frozenset({"grant", "run_spec", "workflow_steps"})
#: ``shared/slurm_accounting.py``'s rules, as stdlib code (C2 "Settlement").
_ELAPSED_RE = re.compile(r"(?:([0-9]+)-)?(?:([0-9]+):)?([0-9]{1,2}):([0-9]{1,2})")
#: One ``AllocTRES`` entry naming GPUs: untyped (``gres/gpu=2``, ``gpu=2``,
#: ``gres:gpu:2``) or typed (``gres/gpu:h100=2``); group 1 is the type.
_GPU_TRES_RE = re.compile(
    r"\s*(?:gres[:/])?gpu(?::([A-Za-z0-9_.-]*[A-Za-z_.-][A-Za-z0-9_.-]*))?[=:]([0-9]+)\s*"
)
#: A reservation with no nonce this old belongs to no live submit session.
_ORPHAN_AGE = DEADLINES["submit"] + 60


def _grants_dir(root: str) -> str:
    return os.path.join(root, "state", "grants")


def _grant_file(root: str, gid: str, suffix: str) -> str:
    return os.path.join(_grants_dir(root), gid + suffix)


def _outputs_dir(root: str, gid: str) -> str:
    return os.path.join(_grants_dir(root), gid, "outputs")


def _grant_lock(root: str, gid: str) -> Any:
    """The grant's ``flock`` (X3), waited for at most ``LOCK_WAIT`` (``busy``)."""
    return _flocked(_grant_file(root, gid, ".lock"), LOCK_WAIT)


# -- epoch and cancel (K7, C3) ----------------------------------------------


def _read_epoch(root: str, absent: int | None) -> int:
    """Read an epoch, with zero only for a writer's first initialization."""
    try:
        data = _read_json(os.path.join(_grants_dir(root), "epoch"), 4096)
    except FileNotFoundError:
        if absent is None:
            raise _failed("grant epoch is missing") from None
        return absent
    except (OSError, ValueError) as exc:
        raise _failed(f"grant epoch: {exc}") from None
    epoch = data.get("epoch") if isinstance(data, dict) else None
    if type(epoch) is not int or epoch < 0:
        raise _failed("grant epoch is malformed")
    return epoch


def read_epoch(root: str) -> int:
    """A bootstrapped epoch; missing or malformed state fails closed (exit 1)."""
    return _read_epoch(root, None)


def write_epoch(root: str, epoch: int, *, advance: bool = False) -> None:
    """Initialize an epoch or raise it without lowering it. With ``advance``,
    also step past the current value, making a revoke a fresh boundary."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("grant epoch must be a non-negative integer")
    directory = _grants_dir(root)
    _ensure_dir(os.path.dirname(directory))
    _ensure_dir(directory)
    path = os.path.join(directory, "epoch")
    with _flocked(os.path.join(directory, "epoch.lock"), LOCK_WAIT):
        current = _read_epoch(root, 0)
        target = max(epoch, current + 1) if advance else epoch
        if target > current or not os.path.lexists(path):
            data = canonical_bytes({"epoch": target})
            _atomic_write(directory, path, data, prefix=".epoch")


def _cancel_grant(root: str, gid: str, now: int, signer: str) -> tuple[dict[str, Any], str]:
    """``grant-cancel`` (C3): ``<id>.cancelled`` under the grant lock; a
    running job is not touched."""
    _ensure_dir(_grants_dir(root))
    with _grant_lock(root, gid):
        record = canonical_bytes({"cancelled": now})
        created = _create_exclusive(_grant_file(root, gid, ".cancelled"), record)
    done = {"grant_id": gid, "already": not created, "signer": signer}
    return done, f"grant={gid} signer={signer}"


def _grant_ended(root: str, grant: dict[str, Any]) -> str:
    if os.path.lexists(_grant_file(root, grant["grant_id"], ".cancelled")):
        return "cancelled"
    if grant["issued"] < read_epoch(root):
        return "account re-enrolled"
    return ""


def _grant_window(grant: dict[str, Any], now: int) -> str:
    if now + CLOCK_SKEW < grant["not_before"] or now + CLOCK_SKEW < grant["issued"]:
        return "grant_not_yet_valid"
    if now - CLOCK_SKEW > grant["expires"]:
        return "grant_expired"
    return ""


def _grant_state_name(root: str, grant: dict[str, Any], now: int) -> str:
    """``grant-check``'s ``state``: cancelled, not_yet_valid, expired, active."""
    if _grant_ended(root, grant):
        return "cancelled"
    window = _grant_window(grant, now)
    return {"grant_not_yet_valid": "not_yet_valid", "grant_expired": "expired"}.get(
        window, "active"
    )


# -- counters -----------------------------------------------------------------


def _load_counters(root: str, gid: str) -> dict[str, Any] | None:
    try:
        data = _read_json(_grant_file(root, gid, ".json"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise _failed(f"grant counters: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise _failed("grant counters are malformed")
    return data


def _save_counters(root: str, counters: dict[str, Any]) -> None:
    """Canonical JSON, 0600, temp file, fsync and rename (X3)."""
    _write_state(_grant_file(root, counters["grant_id"], ".json"), counters)


def _fresh_counters(grant: dict[str, Any], sha: str) -> dict[str, Any]:
    return {
        "v": 1,
        "grant_id": grant["grant_id"],
        "grant_sha256": sha,
        "expires": grant["expires"],
        "entries": {},
    }


def _new_entry(now: int, wall: int, reserved: list[int]) -> dict[str, Any]:
    return {
        "job_id": "",
        "state": "reserved",
        "submitted": now,
        "wall": wall,
        "reserved": list(reserved),
        "charged": [0, 0],
        "actual": [],
        "bound": False,
    }


def _entries(counters: dict[str, Any] | None) -> dict[str, Any]:
    return counters["entries"] if counters is not None else {}


def _check_conflict(root: str, gid: str, sha: str) -> None:
    counters = _load_counters(root, gid)
    if counters is not None and counters.get("grant_sha256") != sha:
        raise refuse("grant_conflict", "another grant with this id was used at this gate")


def _check_grant_state(s: _Session, grant: dict[str, Any], sha: str) -> None:
    """C2 step 13 (under the grant lock): conflict, cancel or epoch, window;
    ``now`` is read under the lock."""
    _check_conflict(s.root, grant["grant_id"], sha)
    ended = _grant_ended(s.root, grant)
    if ended:
        raise refuse("grant_cancelled", ended)
    window = _grant_window(grant, int(time.time()))
    if window:
        raise refuse(window, f"the grant is valid from {grant['not_before']} to {grant['expires']}")


def _produced(root: str, gid: str, obj: dict[str, Any]) -> bool:
    try:
        data = _read_nofollow(os.path.join(_outputs_dir(root, gid), obj["sha256"]), 64)
    except (OSError, ValueError):
        return False
    return data.strip() == str(obj["size"]).encode("ascii")


def _check_chained(sub: _Submit) -> None:
    """``chained`` mode: an object outside the seed (K10) must be an output
    this grant's own ``collect-out`` indexed, with its size."""
    if sub.grant["inputs"]["mode"] != "chained":
        return
    seed = {(entry["sha256"], entry["size"]) for entry in sub.grant["inputs"]["listed"]}
    for obj in sub.sobj["inputs"]["objects"]:
        if (obj["sha256"], obj["size"]) not in seed and not _produced(sub.s.root, sub.gid, obj):
            raise refuse(
                "grant_mismatch",
                f"inputs: {obj['bucket']}/{obj['key'][:200]} was not produced under this grant",
            )


def _check_ends_in_window(sub: _Submit) -> None:
    """K19a (review 2026-11-09): a step whose walltime would run past the
    grant's expiry refuses ``grant_expired``; the job's ``--deadline`` is
    capped at the expiry too (:func:`_compose`)."""
    if int(time.time()) + grant_wall(sub.sobj) > sub.grant["expires"]:
        raise refuse("grant_expired", "the step's walltime runs past the grant's expiry")


def _grant_gate(sub: _Submit) -> None:
    """C2 step 13; every check repeats at step 19b."""
    with _grant_lock(sub.s.root, sub.gid):
        _check_grant_state(sub.s, sub.grant, sub.grant_sha)
        _check_ends_in_window(sub)
        _check_chained(sub)


# -- the grant envelope and body (C2 steps 5-12) ------------------------------


def _b64_field(value: Any, name: str) -> bytes:
    if not isinstance(value, str):
        raise refuse("bad_envelope", f"{name} must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise refuse("bad_envelope", f"{name} is not base64") from None


def _grant_sizes(payload: bytes, spec: bytes, steps_raw: bytes) -> None:
    for data, cap, name in (
        (payload, GRANT_MAX_BYTES, "the grant"),
        (spec, INTENT_MAX_BYTES, "the run spec"),
        (steps_raw, WORKFLOW_STEPS_MAX_BYTES, "workflow_steps"),
    ):
        if len(data) > cap:
            raise refuse("too_large", f"{name} exceeds {cap} bytes")


def _verify_grant_signature(s: _Session, payload: bytes, armored: str) -> None:
    ns = namespace("grant", s.cfg.domain)
    signers = os.path.join(s.root, "allowed_signers")
    ssh_keygen_verify(signers, s.cfg.principal, ns, armored, payload, _state(s.root))


def _grant_envelope(
    s: _Session, wd: str, step: str, rec: dict[str, Any], envelope: dict[str, Any]
) -> _Submit:
    """C2 step 5 (keys and sizes) and step 6 (``ssh-keygen`` on the grant)."""
    if set(envelope) != _GRANT_ENVELOPE_KEYS:
        raise refuse("bad_envelope", 'expected {"grant", "run_spec", "workflow_steps"}')
    inner = envelope["grant"]
    if not isinstance(inner, dict) or not set(inner) <= {"payload", "signature"}:
        raise refuse("bad_envelope", 'grant must be {"payload", "signature"}')
    payload, armored = _envelope_values(inner.get("payload"), inner.get("signature"))
    spec = _b64_field(envelope["run_spec"], "run_spec")
    steps_raw = _b64_field(envelope["workflow_steps"], "workflow_steps")
    _grant_sizes(payload, spec, steps_raw)
    _verify_grant_signature(s, payload, armored)
    sub = _Submit(s, wd, step, rec, payload, armored, _ABSENT, kind="grant")
    sub.spec, sub.steps_raw = spec, steps_raw
    return sub


def _bind_grant(s: _Session, grant: dict[str, Any]) -> None:
    if grant["site"] != s.cfg.site:
        raise refuse("wrong_site", f"grant is for site {grant['site']}")
    if grant["principal"] != s.cfg.principal:
        raise refuse("wrong_principal", "grant is for another principal")
    if grant["login"] != s.login:
        raise refuse("wrong_login", f"grant is for login {grant['login']}")


def _grant_body(s: _Session, payload: bytes, armored: str) -> tuple[dict[str, Any], str]:
    """C2 steps 7-9 on a grant ``ssh-keygen`` verified: the SSHSIG policy,
    strict canonical JSON (no floats), ``validate_grant``, the binding."""
    signer = _signature_policy(s.cfg, armored, "grant").signer_fingerprint
    try:
        grant = parse_strict(payload)
    except ValueError as exc:
        raise refuse("not_canonical", str(exc)) from None
    validate_grant(grant)
    _bind_grant(s, grant)
    return grant, signer


def _parse_grant(sub: _Submit) -> None:
    sub.grant, sub.signer = _grant_body(sub.s, sub.payload, sub.armored)
    sub.gid = sub.grant["grant_id"]
    sub.grant_sha = hashlib.sha256(sub.payload).hexdigest()


def _parse_run_spec(sub: _Submit) -> None:
    """C2 step 10: the run spec is a strict canonical B2 intent body (not
    signed); the B2 binding; the workdir's key is this run's key."""
    try:
        body = parse_strict(sub.spec, json_values=True)
    except ValueError as exc:
        raise refuse("not_canonical", f"run_spec: {exc}") from None
    sub.body = validate_intent(body)
    sub.iid = grant_run_key(sub.gid, body["run_id"])
    sub.spec_id = hashlib.sha256(sub.spec).hexdigest()
    sub.entry_key = f"{body['run_id']}/{sub.step}"
    _bind_intent(sub)


def _parse_skeleton(data: bytes) -> dict[str, Any]:
    try:
        obj = parse_strict(data, json_values=True)
    except ValueError as exc:
        raise refuse("not_canonical", f"workflow_steps: {exc}") from None
    try:
        exact = set(obj) == {"steps"} and workflow_steps_skeleton(obj) == obj
    except ValueError:
        exact = False
    if not exact:
        raise refuse("grant_mismatch", "workflow_steps: not a step skeleton")
    return obj


def _check_coverage(sub: _Submit) -> None:
    """C2 step 11: ``grant_covers`` (the first problem refuses)."""
    sub.skeleton = _parse_skeleton(sub.steps_raw)
    problem = grant_covers(sub.grant, sub.body, sub.skeleton)
    if problem:
        raise refuse("grant_mismatch", problem)


def _same_request(stored: Any, sub: _Submit) -> bool:
    try:
        raws = (stored["grant"]["payload"], stored["run_spec"], stored["workflow_steps"])
        decoded = [base64.b64decode(raw, validate=True) for raw in raws]
    except (KeyError, TypeError, ValueError):
        return False
    return decoded == [sub.payload, sub.spec, sub.steps_raw]


def _check_same_run(sub: _Submit) -> None:
    """C2 step 12 (under the run lock): the first submit of a run fixed its
    grant, run spec and skeleton bytes (X2)."""
    try:
        stored = _read_json(os.path.join(_intent_dir(sub.s, sub.iid), "intent.json"))
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        raise _failed(f"run record: {exc}") from None
    if not _same_request(stored, sub):
        raise refuse("run_spec_changed", "this run was first submitted with other bytes")


# -- reservation (C2 step 19b) --------------------------------------------------


def _used_seconds(entries: list[dict[str, Any]]) -> tuple[int, int]:
    """Charged of settled entries plus reserved of the others."""
    gpu = core = 0
    for entry in entries:
        amount = entry["charged"] if entry["state"] == "settled" else entry["reserved"]
        gpu += int(amount[0])
        core += int(amount[1])
    return gpu, core


def _cap_problem(entries: dict[str, Any], caps: dict[str, int], req: list[int]) -> str:
    """The first cap a new entry of ``req`` seconds would pass (integers)."""
    values = list(entries.values())
    used_gpu, used_core = _used_seconds(values)
    open_count = sum(1 for entry in values if entry["state"] != "settled")
    core_cap = caps["max_core_hours"] * 3600
    checks = (
        ("jobs", len(values) + 1 <= caps["max_jobs"]),
        ("concurrent", open_count + 1 <= caps["max_concurrent"]),
        ("gpu_hours", used_gpu + req[0] <= caps["max_gpu_hours"] * 3600),
        ("core_hours", core_cap == 0 or used_core + req[1] <= core_cap),
    )
    return next((name for name, fine in checks if not fine), "")


def _drop_stale_reservation(sub: _Submit, entries: dict[str, Any]) -> None:
    """K3: a ``reserved`` entry of this step with no nonce file (a gate
    killed between steps 19b and 20) is replaced, never counted twice."""
    entry = entries.get(sub.entry_key)
    if entry is None or entry["state"] != "reserved":
        return
    if not os.path.lexists(_nonce_path(sub.s, sub.iid, sub.step)):
        del entries[sub.entry_key]


def _try_reserve(sub: _Submit) -> str:
    """One pass of step 19b under the grant lock: "" (reserved) or the cap."""
    with _grant_lock(sub.s.root, sub.gid):
        _check_grant_state(sub.s, sub.grant, sub.grant_sha)
        _check_ends_in_window(sub)
        _check_chained(sub)
        counters = _load_counters(sub.s.root, sub.gid) or _fresh_counters(sub.grant, sub.grant_sha)
        entries = counters["entries"]
        _drop_stale_reservation(sub, entries)
        if sub.entry_key in entries:  # K3: live, settled, or in flight
            raise refuse("replayed", f"step {sub.step} of this run already ran under the grant")
        problem = _cap_problem(entries, sub.grant["caps"], sub.reserved)
        if not problem:
            entries[sub.entry_key] = _new_entry(
                int(time.time()), grant_wall(sub.sobj), sub.reserved
            )
            _save_counters(sub.s.root, counters)
            sub.holds = True
        return problem


def _reserve(sub: _Submit) -> None:
    """C2 step 19b: reserve ``grant_cost`` (K4) within the caps. Only when
    ``concurrent`` or an hour cap would refuse, settle first (``sacct``
    without the lock, K5) and try once more; ``jobs`` refuses at once."""
    part = sub.s.profile["partitions"][sub.sobj["partition"]]
    sub.reserved = list(grant_cost(sub.sobj, part))
    problem = _try_reserve(sub)
    if problem in _SETTLE_FIRST:
        _settle_grant(sub.s, sub.gid)
        problem = _try_reserve(sub)
    if problem:
        raise refuse("grant_exhausted", problem)


def _drop_reservation(sub: _Submit) -> None:
    """Delete this call's reservation (a refusal at step 20, or an sbatch
    that certainly queued nothing)."""
    if not sub.holds:
        return

    def unqueued(entry: dict[str, Any]) -> bool:
        return entry["state"] == "reserved" and not entry["job_id"]

    with _grant_lock(sub.s.root, sub.gid):
        counters = _load_counters(sub.s.root, sub.gid)
        entry = _entries(counters).get(sub.entry_key)
        if counters is not None and entry is not None and unqueued(entry):
            del counters["entries"][sub.entry_key]
            _save_counters(sub.s.root, counters)
    sub.holds = False


def _settle_zero(sub: _Submit) -> None:
    """C2 step 24: the job of an unlogged submit was cancelled; its entry
    settles at zero, with receipt and counter updates using the same retry rule."""
    if not sub.holds:
        return
    with _grant_lock(sub.s.root, sub.gid):
        counters = _load_counters(sub.s.root, sub.gid)
        entry = _entries(counters).get(sub.entry_key)
        if counters is None or entry is None or entry["state"] == "settled":
            return
        before = {key: value["state"] for key, value in counters["entries"].items()}
        entry.update(state="settled", charged=[0, 0], actual=[], bound=False)
        _append_new_settlements(sub.s, sub.gid, before, counters)
        _save_counters(sub.s.root, counters)


def _grant_of(s: _Session, iid: str) -> str:
    """The grant id of a grant run (``state/intents/<run key>/grant``), else ""."""
    try:
        text = _read_nofollow(os.path.join(_intent_dir(s, iid), "grant"), 64).decode("ascii")
    except FileNotFoundError:
        return ""
    except (OSError, ValueError) as exc:
        raise _failed(f"grant record: {exc}") from None
    if not _NONCE_RE.fullmatch(text.strip()):
        raise _failed("grant record is malformed")
    return text.strip()


def _grant_job_live(s: _Session, gid: str, key: str, job_id: str) -> None:
    """A grant entry learns its job (``live``); ``state/jobs/<id>.grant``
    maps the job to its grant and entry for ``usage``."""
    path = _state(s.root, "jobs", job_id + ".grant")
    _atomic_write(os.path.dirname(path), path, f"{gid} {key}".encode("ascii"), prefix=".job")
    with _grant_lock(s.root, gid):
        counters = _load_counters(s.root, gid)
        entry = _entries(counters).get(key)
        if counters is not None and entry is not None and entry["state"] != "settled":
            entry.update(job_id=job_id, state="live")
            _save_counters(s.root, counters)


def _job_grant(s: _Session, job_id: str) -> tuple[str, str]:
    """``(grant_id, entry key)`` of a job, ``("", "")`` for an intent job."""
    try:
        text = _read_nofollow(_state(s.root, "jobs", job_id + ".grant"), 256).decode("ascii")
    except (OSError, ValueError):
        return "", ""
    gid, _, key = text.partition(" ")
    return (gid, key) if _NONCE_RE.fullmatch(gid) else ("", "")


def _submit_grant(sub: _Submit) -> None:
    """C2 steps 7-24 for a grant whose signature ``ssh-keygen`` verified."""
    _parse_grant(sub)
    _parse_run_spec(sub)
    _check_coverage(sub)
    with _intent_lock(sub.s, sub.iid):
        _check_same_run(sub)
        _grant_gate(sub)
        _submit_checks(sub)
        _reserve(sub)
        _submit_act(sub)


# -- settlement (C2 "Settlement", K5) -------------------------------------------


@dataclass
class _Observed:
    """What Slurm said, asked without the grant lock (K5)."""

    now: int
    asked: set[str]
    rows: dict[str, dict[str, str]] | None
    queued: set[str] | None = None
    matches: dict[str, tuple[str | None, bool]] = field(default_factory=dict)


def _sacct_rows(s: _Session, job_ids: list[str]) -> dict[str, dict[str, str]] | None:
    rows: dict[str, dict[str, str]] = {}
    for start in range(0, len(job_ids), USAGE_MAX_JOBS):
        batch = job_ids[start : start + USAGE_MAX_JOBS]
        proc = _usage_proc(s, batch)
        if proc.rc != 0:
            return None
        rows.update((row["job_id"], row) for row in _usage_rows(proc.out, batch))
    return rows


def _queued_ids(s: _Session) -> set[str] | None:
    proc = _slurm(s, "squeue", ["--me", "-h", "-o", "%i"])
    if proc.rc != 0:
        return None
    return {line.strip() for line in proc.out.splitlines() if line.strip()}


def _past_bound(s: _Session, entry: dict[str, Any], now: int) -> bool:
    """K5: past ``submitted + queue_seconds + wall + grace_seconds + 120``,
    the ``--deadline`` and ``--time`` the gate wrote have ended the job."""
    slurm = s.profile["slurm"]
    bound = int(entry["submitted"]) + int(entry["wall"]) + 120
    return now > bound + int(slurm["queue_seconds"]) + int(slurm["grace_seconds"])


def _match_entry(s: _Session, gid: str, key: str) -> tuple[str | None, bool] | None:
    """The job of a ``reserved`` entry with ``job_id: ""``: its nonce's job,
    else the one Slurm ran in its workdir; None when it has no nonce."""
    run_id, _, step = key.partition("/")
    nonce = _nonce_record(s, grant_run_key(gid, run_id), step)
    if nonce is None:
        return None
    if nonce.get("job_id"):
        return str(nonce["job_id"]), True
    return _match_workdir(s, _wd_path(s, nonce["wd"]), int(nonce["submitted"]))


def _late_matches(
    s: _Session, gid: str, entries: dict[str, Any], now: int
) -> dict[str, tuple[str | None, bool]]:
    matches: dict[str, tuple[str | None, bool]] = {}
    for key, entry in entries.items():
        if entry["state"] == "reserved" and not entry["job_id"] and _past_bound(s, entry, now):
            found = _match_entry(s, gid, key)
            if found is not None:
                matches[key] = found
    return matches


def _live_ids(entries: dict[str, Any], only: set[str] | None) -> list[str]:
    live = {entry["job_id"] for entry in entries.values() if entry["state"] == "live"}
    return sorted(job for job in live if job and (only is None or job in only))


def _any_late(s: _Session, entries: dict[str, Any], now: int) -> bool:
    pending = [entry for entry in entries.values() if entry["state"] != "settled"]
    return any(_past_bound(s, entry, now) for entry in pending)


def _observe(s: _Session, gid: str, counters: dict[str, Any], only: set[str] | None) -> _Observed:
    """One ``sacct`` call for the live jobs (or ``only`` these), and, when an
    open entry is past its bound, ``squeue`` and the workdir matches."""
    now, entries = int(time.time()), counters["entries"]
    live = _live_ids(entries, only)
    obs = _Observed(now, set(live), _sacct_rows(s, live) if live else {})
    if only is None and _any_late(s, entries, now):
        obs.queued = _queued_ids(s)
        obs.matches = _late_matches(s, gid, entries, now)
    return obs


def _parse_elapsed(text: str) -> int | None:
    match = _ELAPSED_RE.fullmatch(str(text or "").strip().upper())
    if match is None:
        return None
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _alloc_gpus(text: str) -> int:
    """The GPUs of an ``AllocTRES`` value: the untyped entry when there is
    one, else the sum of the typed ones (a typed-only configuration)."""
    typed = 0
    for part in str(text or "").split(","):
        match = _GPU_TRES_RE.fullmatch(part)
        if match is None:
            continue
        if not match.group(1):
            return int(match.group(2))
        typed += int(match.group(2))
    return typed


def _alloc_cpus(text: str) -> int:
    raw = str(text or "").strip()
    return int(raw) if _JOB_ID_RE.fullmatch(raw) else 0


def _settle_measured(entry: dict[str, Any], row: dict[str, str]) -> None:
    """``actual = [elapsed x gpus, elapsed x cpus]``, where the GPUs and CPUs
    are never fewer than the step requested (K4 rate: ``reserved / wall``;
    an accounting that does not track GPUs reports none). ``charged`` is
    ``actual``, also above the reservation (K21: what Slurm allocated beyond
    the request counts); an unparsable ``Elapsed`` charges the reservation."""
    elapsed = _parse_elapsed(row["elapsed"])
    reserved = [int(part) for part in entry["reserved"]]
    if elapsed is None:
        entry.update(state="settled", actual=[], charged=reserved)
        return
    wall = int(entry.get("wall") or 0)
    gpus = max(_alloc_gpus(row.get("alloc_tres", "")), reserved[0] // wall if wall > 0 else 0)
    cpus = max(_alloc_cpus(row["alloc_cpus"]), reserved[1] // wall if wall > 0 else 0)
    actual = [elapsed * gpus, elapsed * cpus]
    entry.update(state="settled", actual=actual, charged=list(actual))


def _settle_bound(entry: dict[str, Any]) -> None:
    """K5: a job Slurm cannot report, past its bound, settles at its reservation."""
    entry.update(
        state="settled", actual=[], charged=[int(p) for p in entry["reserved"]], bound=True
    )


def _apply_live(s: _Session, entry: dict[str, Any], obs: _Observed) -> bool:
    job = entry["job_id"]
    if job not in obs.asked:
        return False
    row = (obs.rows or {}).get(job)
    if row is not None and _is_terminal(_first_state(row["state"])):
        _settle_measured(entry, row)
        return True
    gone = obs.queued is not None and job not in obs.queued
    if row is None and gone and _past_bound(s, entry, obs.now):
        _settle_bound(entry)
        return True
    return False


def _drop_orphan(s: _Session, gid: str, key: str, entry: dict[str, Any], now: int) -> bool:
    """A reservation whose submit never claimed a nonce, older than any
    submit session: no job can exist for it, so it is removed (not charged)."""
    if now <= entry["submitted"] + _ORPHAN_AGE:
        return False
    run_id, _, step = key.partition("/")
    return not os.path.lexists(_nonce_path_for(s, gid, run_id, step))


def _nonce_path_for(s: _Session, gid: str, run_id: str, step: str) -> str:
    return _nonce_path(s, grant_run_key(gid, run_id), step)


def _apply_reserved(
    s: _Session, gid: str, key: str, entries: dict[str, Any], obs: _Observed
) -> bool:
    """Returns True when the entry changed (re-keyed, settled or dropped)."""
    entry = entries[key]
    match = obs.matches.get(key)
    if match is None:
        if _drop_orphan(s, gid, key, entry, obs.now):
            del entries[key]
            return True
        return False
    job, certain = match
    if job:
        entry.update(job_id=job, state="live")
        return True
    if certain and obs.queued is not None:
        _settle_bound(entry)
        return True
    return False


def _apply_observed(s: _Session, gid: str, counters: dict[str, Any], obs: _Observed) -> bool:
    """Terminal results and bounds, applied once (a terminal state is final:
    applied late or twice, it changes nothing)."""
    changed = False
    entries = counters["entries"]
    for key in sorted(entries):
        entry = entries[key]
        if entry["state"] == "live":
            changed = _apply_live(s, entry, obs) or changed
        elif entry["state"] == "reserved" and not entry["job_id"]:
            changed = _apply_reserved(s, gid, key, entries, obs) or changed
    return changed


def _settlement_detail(gid: str, key: str, entry: dict[str, Any]) -> str:
    """The fixed witness receipt for one terminal charge (at most 200 bytes)."""
    run_id, _sep, step = key.partition("/")
    gpu, core = (int(value) for value in entry["charged"])
    return (
        f"grant={gid} run={run_id} step={step} job={entry.get('job_id', '')} "
        f"gpu_s={gpu} core_s={core} bound={int(bool(entry.get('bound')))}"
    )


def _logged_settlement_details(root: str) -> set[str]:
    """Details of chain-verified settle receipts already present on disk."""
    details: set[str] = set()
    previous = GENESIS
    try:
        for index, raw in enumerate(_log_lines(_log_path(root))):
            previous = _check_log_entry(raw, previous, index)
            entry = parse_strict(raw)
            if entry.get("verb") == "settle" and entry.get("result") == "ok":
                details.add(str(entry.get("detail", "")))
    except FileNotFoundError:
        pass
    return details


def _append_new_settlements(
    s: _Session, gid: str, before: dict[str, str], counters: dict[str, Any]
) -> None:
    """Append new evidence idempotently, so a retry after a save failure is safe."""
    logged = _logged_settlement_details(s.root)
    for key, entry in sorted(counters["entries"].items()):
        if entry["state"] != "settled" or before.get(key) == "settled":
            continue
        detail = _settlement_detail(gid, key, entry)
        if detail not in logged:
            _append_unsigned(s, "settle", detail)
            logged.add(detail)


def _commit_observed(s: _Session, gid: str, obs: _Observed) -> None:
    with _grant_lock(s.root, gid):
        counters = _load_counters(s.root, gid)
        if counters is not None:
            before = {key: entry["state"] for key, entry in counters["entries"].items()}
            if _apply_observed(s, gid, counters, obs):
                _append_new_settlements(s, gid, before, counters)
                _save_counters(s.root, counters)


def _settle_grant(s: _Session, gid: str, only: set[str] | None = None) -> None:
    """K5: read the open entries under the grant lock, release it for
    ``sacct`` (and ``squeue``), take it again to apply the results."""
    with _grant_lock(s.root, gid):
        counters = _load_counters(s.root, gid)
    if counters is not None:
        _commit_observed(s, gid, _observe(s, gid, counters, only))


def _settle_job_quiet(s: _Session, job_id: str) -> None:
    """``collect-out`` and ``read-reply`` of a grant run settle its job."""
    gid = _job_grant(s, job_id)[0] if job_id else ""
    if gid:
        with contextlib.suppress(GateError, OSError, ValueError):
            _settle_grant(s, gid, only={job_id})


def _settle_rows(s: _Session, jobs: list[dict[str, str]]) -> None:
    """``usage``: its own ``sacct`` rows settle their grant entries."""
    by_grant: dict[str, dict[str, dict[str, str]]] = {}
    for job in jobs:
        gid = _job_grant(s, job["job_id"])[0]
        if gid:
            by_grant.setdefault(gid, {})[job["job_id"]] = job
    for gid, rows in sorted(by_grant.items()):
        with contextlib.suppress(GateError, OSError, ValueError):
            _commit_observed(s, gid, _Observed(int(time.time()), set(rows), rows))


def _usage_grant_fields(s: _Session, job: dict[str, str]) -> dict[str, Any]:
    gid, key = _job_grant(s, job["job_id"])
    settled = False
    if gid:
        with contextlib.suppress(GateError, OSError, ValueError):
            settled = _entries(_load_counters(s.root, gid)).get(key, {}).get("state") == "settled"
    return {**job, "grant_id": gid, "settled": settled}


# -- outputs index and limits ---------------------------------------------------


def _stored_grant(s: _Session, iid: str) -> dict[str, Any]:
    """The grant recorded with a grant run's first submit."""
    stored = _read_json(os.path.join(_intent_dir(s, iid), "intent.json"))
    return parse_strict(base64.b64decode(stored["grant"]["payload"], validate=True))


def _grant_run_limit(s: _Session, iid: str) -> float:
    """K3: a grant run's state stays until its grant's ``expires + 7 d``."""
    try:
        return float(_stored_grant(s, iid)["expires"] + RETENTION_SECONDS)
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _index_outputs(s: _Session, iid: str, gid: str, digests: dict[str, list[Any]]) -> None:
    """C2 "Outputs index": ``outputs/<sha256>`` (its size) for every out
    member, O_EXCL; at most ``max_jobs x max_out_files`` (``outputs_capped``)."""
    cap = _stored_grant(s, iid)["caps"]["max_jobs"] * s.caps["max_out_files"]
    directory = _outputs_dir(s.root, gid)
    _ensure_dir(os.path.dirname(directory))
    _ensure_dir(directory)
    count = sum(1 for name in os.listdir(directory) if _HEX64_RE.fullmatch(name))
    skipped = 0
    for sha, size in sorted({(str(v[0]), int(v[1])) for v in digests.values()}):
        path = os.path.join(directory, sha)
        if os.path.lexists(path):
            continue
        if count >= cap:
            skipped += 1
            continue
        count += 1 if _create_exclusive(path, str(size).encode("ascii")) else 0
    if skipped:
        _append_unsigned(s, "collect-out", f"outputs_capped grant={gid} skipped={skipped}")


def _grant_after_collect(s: _Session, rec: dict[str, Any], digests: dict[str, list[Any]]) -> None:
    gid = _grant_of(s, rec["intent_id"])
    if not gid:
        return
    try:
        _index_outputs(s, rec["intent_id"], gid, digests)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _failed(f"outputs index: {exc}") from None
    _settle_job_quiet(s, rec["job_id"])


# -- grant-check (C3) -------------------------------------------------------------


def _read_grant_check(s: _Session) -> tuple[bytes, str]:
    """C2 steps 5-6 for ``grant-check``: its own stdin cap (K1), the
    envelope, the grant's size, ``ssh-keygen``."""
    data = s.stdin.read(GRANT_CHECK_MAX_STDIN + 1)
    if len(data) > GRANT_CHECK_MAX_STDIN:
        raise refuse("too_large", f"request exceeds {GRANT_CHECK_MAX_STDIN} bytes")
    payload, armored = read_envelope(data)
    if len(payload) > GRANT_MAX_BYTES:
        raise refuse("too_large", f"the grant exceeds {GRANT_MAX_BYTES} bytes")
    _verify_grant_signature(s, payload, armored)
    return payload, armored


def _grant_bounds(grant: dict[str, Any]) -> dict[str, Any]:
    return {
        "site": grant["site"],
        "login": grant["login"],
        "principal": grant["principal"],
        "accounts": grant["accounts"],
        "projects": grant["projects"],
        "workflows": [{"name": w["name"], "version": w["version"]} for w in grant["workflows"]],
        "images": [image["image_id"] for image in grant["images"]],
        "inputs_mode": grant["inputs"]["mode"],
        "listed_count": len(grant["inputs"]["listed"]),
        "caps": grant["caps"],
        "not_before": grant["not_before"],
        "expires": grant["expires"],
    }


def _grant_usage(counters: dict[str, Any] | None) -> dict[str, int]:
    """Jobs, open entries, charged seconds (settled) and reserved (open)."""
    entries = list(_entries(counters).values())
    done = [entry for entry in entries if entry["state"] == "settled"]
    pending = [entry for entry in entries if entry["state"] != "settled"]
    return {
        "jobs": len(entries),
        "open": len(pending),
        "gpu_seconds": sum(int(entry["charged"][0]) for entry in done),
        "core_seconds": sum(int(entry["charged"][1]) for entry in done),
        "reserved_gpu_seconds": sum(int(entry["reserved"][0]) for entry in pending),
        "reserved_core_seconds": sum(int(entry["reserved"][1]) for entry in pending),
    }


def cmd_grant_check(s: _Session, args: list[str]) -> dict[str, Any]:
    """``grant-check``: verify a signed grant (C2 steps 5-9), settle its open
    entries and report its state, bounds and usage. Time and cancel are
    reported in ``state``, not refused; no counters file is created."""
    _arity(args, 0, "grant-check")
    if _log_size(s.root) >= LOG_FULL_BYTES:
        raise refuse("log_full", f"the gate log holds {LOG_FULL_BYTES} bytes")
    payload, armored = _read_grant_check(s)
    grant, _signer = _grant_body(s, payload, armored)
    gid, sha = grant["grant_id"], hashlib.sha256(payload).hexdigest()
    held = sorted(set(grant["accounts"]) & _profile_assocs(s), key=_utf8)
    if not held:
        raise refuse("account_not_held", f"login {s.login} holds none of the granted accounts")
    _check_conflict(s.root, gid, sha)
    with contextlib.suppress(GateError, OSError, ValueError):
        _settle_grant(s, gid)
    state = _grant_state_name(s.root, grant, int(time.time()))
    head = _append_unsigned(s, "grant-check", f"grant={gid} state={state}")
    return {
        "grant_id": gid,
        "grant_sha256": sha,
        "state": state,
        "bounds": _grant_bounds(grant),
        "usage": _grant_usage(_load_counters(s.root, gid)),
        "held_accounts": held,
        "log_head": head,
    }


# -- sweep (C3) --------------------------------------------------------------------


def _cancel_time(path: str) -> float:
    try:
        return float(_read_json(path, 4096)["cancelled"])
    except (ValueError, KeyError, TypeError):
        return os.lstat(path).st_mtime


def _prune_cancel(s: _Session, gid: str, now: float) -> int:
    """K8: a cancel record is kept for ``GRANT_CANCEL_RETENTION`` (400 days)."""
    path = _grant_file(s.root, gid, ".cancelled")
    with contextlib.suppress(OSError):
        if now - _cancel_time(path) > GRANT_CANCEL_RETENTION:
            os.unlink(path)
            return 1
    return 0


def _counters_done(root: str, gid: str, now: float) -> bool:
    counters = _load_counters(root, gid)
    if counters is None:
        return False
    pending = any(entry["state"] != "settled" for entry in counters["entries"].values())
    return not pending and now > counters["expires"] + RETENTION_SECONDS


def _prune_counters(s: _Session, gid: str, now: float) -> int:
    """Settle, then remove the counters, the outputs index and the lock once
    ``expires + 7 d`` has passed and no entry is open."""
    with contextlib.suppress(GateError, OSError, ValueError):
        _settle_grant(s, gid)
    try:
        with _grant_lock(s.root, gid):
            done = _counters_done(s.root, gid, now)
            if done:
                _unlink_quiet(_grant_file(s.root, gid, ".json"))
                shutil.rmtree(os.path.join(_grants_dir(s.root), gid), ignore_errors=True)
    except (GateError, OSError, ValueError):
        return 0
    if done:
        _unlink_quiet(_grant_file(s.root, gid, ".lock"))
    return int(done)


def _sweep_grants(s: _Session, now: float) -> int:
    """Settle open grant entries, prune ended grants and old cancel records."""
    pruned = 0
    for name in sorted(os.listdir(_grants_dir(s.root))):
        stem, ext = os.path.splitext(name)
        if not _NONCE_RE.fullmatch(stem):
            continue
        if ext == ".cancelled":
            pruned += _prune_cancel(s, stem, now)
        elif ext == ".json":
            pruned += _prune_counters(s, stem, now)
    return pruned


# ---------------------------------------------------------------------------
# rmwd and sweep (B2.0 section 5.4)
# ---------------------------------------------------------------------------


def _remove_workdir(s: _Session, wd: str, rec: dict[str, Any] | None) -> None:
    path = _wd_path(s, wd)
    if os.path.lexists(path):
        _remove_tree(path)
    if rec is not None:
        rec["removed"] = True
        _save_record(s, wd, rec)


def _wd_job_live(s: _Session, wd: str, rec: dict[str, Any]) -> bool:
    """Whether ``wd`` may still have a live job. A sealed record with no job
    id (an sbatch that timed out) counts as live unless Slurm certainly ran
    no job in it; a single match is recorded (Slurm uncertainty = live)."""
    if not rec.get("job_id") and rec.get("sealed"):
        matched, certain = _match_workdir(s, _wd_path(s, wd), int(rec.get("created") or 0))
        if not matched:
            return not certain
        rec["job_id"] = matched
        _save_record(s, wd, rec)
    return bool(rec.get("job_id")) and _job_live(s, rec["job_id"])


def cmd_rmwd(s: _Session, args: list[str]) -> dict[str, Any]:
    """``rmwd <wd>``: remove a gate workdir whose job is not live."""
    (wd,) = _arity(args, 1, "rmwd")
    _arg(WD_RE, wd, "workdir")
    with _wd_lock(s, wd):
        rec = _load_record(s, wd)
        if _wd_job_live(s, wd, rec):
            raise refuse("job_live", f"job {rec['job_id'] or '(unknown)'} may still be live")
        _remove_workdir(s, wd, rec)
    _append_unsigned(s, "rmwd", f"wd={wd}")
    return {"removed": True}


def _own_job_ids(s: _Session) -> list[str]:
    with os.scandir(_state(s.root, "jobs")) as entries:
        return sorted(entry.name for entry in entries if _JOB_ID_RE.fullmatch(entry.name))


def _sacct_live(s: _Session, job_ids: list[str]) -> set[str] | None:
    live: set[str] = set()
    for start in range(0, len(job_ids), USAGE_MAX_JOBS):
        batch = job_ids[start : start + USAGE_MAX_JOBS]
        proc = _slurm(s, "sacct", ["-n", "-X", "-P", "-j", ",".join(batch), "-o", "JobID,State"])
        if proc.rc != 0:
            return None
        for row in (line.strip().split("|") for line in proc.out.splitlines()):
            if (
                len(row) == 2
                and row[0] in batch
                and not _is_terminal(_first_state(row[1]) or "COMPLETED")
            ):
                live.add(row[0])
    return live


def _live_jobs(s: _Session) -> set[str] | None:
    """This gate's jobs that are live; None when Slurm did not answer."""
    queued = _slurm(s, "squeue", ["--me", "-h", "-o", "%i"])
    if queued.rc != 0:
        return None
    mine = _own_job_ids(s)
    listed = {line.strip() for line in queued.out.splitlines()} & set(mine)
    rest = _sacct_live(s, [job for job in mine if job not in listed])
    return None if rest is None else listed | rest


def _last_active(path: str, rec: dict[str, Any] | None) -> float:
    newest = float(rec["created"]) if rec else 0.0
    for sub in ("", "log", os.path.join("log", "slurm.err")):
        with contextlib.suppress(OSError):
            newest = max(newest, os.lstat(os.path.join(path, sub)).st_mtime)
    return newest


def _record_or_none(s: _Session, wd: str) -> dict[str, Any] | None:
    try:
        rec = _read_json(_wd_record_path(s, wd))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _sweep_keeps_job(
    s: _Session, wd: str, rec: dict[str, Any] | None, live: set[str] | None
) -> bool:
    """Its job is live, Slurm did not answer, or (sealed, no job id) Slurm
    cannot say that no job ran in it."""
    job_id = rec.get("job_id", "") if rec else ""
    if job_id:
        return live is None or job_id in live
    if rec and rec.get("sealed"):
        return live is None or _wd_job_live(s, wd, rec)
    return False


def _sweep_one(s: _Session, wd: str, cutoff: float, live: set[str] | None) -> bool:
    """Remove ``wd`` when idle since before ``cutoff`` and its job (if any) is
    not live; a held workdir lock keeps it."""
    rec = _record_or_none(s, wd)
    if _sweep_keeps_job(s, wd, rec, live):
        return False
    if _last_active(_wd_path(s, wd), rec) >= cutoff:
        return False
    try:
        with _flocked(_state(s.root, "wd", wd + ".lock"), 0.0):
            _remove_workdir(s, wd, rec)
    except GateError:
        return False
    return True


def _sweep_workdirs(s: _Session, age: int, live: set[str] | None) -> tuple[list[str], int]:
    cutoff = time.time() - age
    with os.scandir(s.work) as entries:
        names = sorted(entry.name for entry in entries if WD_RE.fullmatch(entry.name))
    removed = [wd for wd in names if _sweep_one(s, wd, cutoff, live)]
    return removed, len(names) - len(removed)


def _intent_limit(s: _Session, iid: str) -> float:
    """When an intent's state may go (``_b2_intent_limit``); a grant run's
    stays at least until its grant's ``expires + 7 d`` (K3)."""
    return max(_b2_intent_limit(s, iid), _grant_run_limit(s, iid))


def _b2_intent_limit(s: _Session, iid: str) -> float:
    """``started + max_run_seconds + 7 d``, else ``expires + 7 d``, else
    (nothing signed was recorded) its age."""
    directory = _intent_dir(s, iid)
    started = _read_started(s, iid)
    try:
        body = _intent_body(s, iid)
    except GateError:
        return os.lstat(directory).st_mtime + INTENT_MAX_WINDOW + CLOCK_SKEW + RETENTION_SECONDS
    if started is None:
        return body["expires"] + RETENTION_SECONDS
    return started + body["max_run_seconds"] + RETENTION_SECONDS


def _intent_jobs(s: _Session, iid: str) -> list[str]:
    jobs = []
    for name in os.listdir(_intent_dir(s, iid)):
        if name.endswith(".json") and STEP_RE.fullmatch(name[:-5]):
            job_id = (_nonce_record(s, iid, name[:-5]) or {}).get("job_id")
            jobs += [job_id] if job_id else []
    return jobs


def _prune_intent(s: _Session, iid: str, live: set[str], now: float) -> bool:
    if now <= _intent_limit(s, iid):
        return False
    jobs = _intent_jobs(s, iid)
    if live & set(jobs):
        return False
    for job_id in jobs:
        _unlink_quiet(_state(s.root, "jobs", job_id))
        _unlink_quiet(_state(s.root, "jobs", job_id + ".grant"))
    shutil.rmtree(_intent_dir(s, iid), ignore_errors=True)
    return True


def _prune_state(s: _Session, live: set[str] | None) -> int:
    pruned = 0
    now = time.time()
    if live is not None:
        for iid in sorted(os.listdir(_state(s.root, "intents"))):
            if _HEX64_RE.fullmatch(iid) and _prune_intent(s, iid, live, now):
                pruned += 1
    for name in sorted(os.listdir(_state(s.root, "wd"))):
        rec = _record_or_none(s, name[:-5]) if name.endswith(".json") else None
        if rec and rec.get("removed"):
            _unlink_quiet(_wd_record_path(s, name[:-5]))
            pruned += 1
    return pruned + _prune_images(s, now)


def _prune_images(s: _Session, now: float) -> int:
    pruned = 0
    with os.scandir(_state(s.root, "images")) as entries:
        for entry in entries:
            if entry.name.endswith(".ok") and _is_stale(entry, now - RETENTION_SECONDS):
                _unlink_quiet(entry.path)
                pruned += 1
    return pruned


def cmd_sweep(s: _Session, args: list[str]) -> dict[str, Any]:
    """``sweep [--older-than <s>]``: remove idle workdirs with no live job and
    prune the state of intents past their window (Slurm silent: keep all);
    settle open grant entries and prune ended grants (C3)."""
    if args and (len(args) != 2 or args[0] != "--older-than"):
        raise _usage("sweep takes [--older-than <seconds>]")
    age = max(int(_arg(_JOB_ID_RE, args[1], "age")) if args else 0, SWEEP_MIN_AGE)
    live = _live_jobs(s)
    removed, kept = _sweep_workdirs(s, age, live)
    pruned = _prune_state(s, live) + _sweep_grants(s, time.time())
    _append_unsigned(s, "sweep", f"removed={len(removed)} kept={kept} pruned={pruned}")
    return {"removed": removed, "kept": kept, "pruned": pruned}


_B2_HANDLERS: dict[str, Callable[[_Session, list[str]], Any]] = {
    "mkwd": cmd_mkwd,
    "stage-in": cmd_stage_in,
    "submit": cmd_submit,
    "lookup": cmd_lookup,
    "status": cmd_status,
    "tail-log": cmd_tail_log,
    "read-reply": cmd_read_reply,
    "collect-out": cmd_collect_out,
    "cancel": cmd_cancel,
    "rmwd": cmd_rmwd,
    "sweep": cmd_sweep,
    "usage": cmd_usage,
    "grant-check": cmd_grant_check,
}


def _parse_argv(args: list[str]) -> str:
    if len(args) != 2 or args[0] != "--site" or not _SITE_RE.fullmatch(args[1]):
        raise GateError("usage", "usage: umrs_gate.py --site <site>", EXIT_USAGE)
    return args[1]


def _parse_command(command: str | None) -> tuple[str, list[str]]:
    """``<verb> [args]`` of ``SSH_ORIGINAL_COMMAND`` (``shlex``, never a shell)."""
    if command is None or not command.strip():
        raise refuse("no_command", "no command was given")
    try:
        parts = shlex.split(command)
    except ValueError:
        raise refuse("unknown_verb", "unparsable command") from None
    if not parts or parts[0] not in VERBS:
        shown = parts[0] if parts else ""
        raise refuse("unknown_verb", f"unknown verb {shown[:40]!r}")
    if parts[0] not in B2_VERBS and len(parts) > 1:
        raise GateError("usage", f"{parts[0]} takes no arguments", EXIT_USAGE)
    return parts[0], parts[1:]


def _deadline_seconds(command: str | None) -> int:
    """The session deadline of the command's verb (``DEADLINES``)."""
    try:
        parts = shlex.split(command or "")
    except ValueError:
        parts = []
    return DEADLINES.get(parts[0] if parts else "", GATE_DEADLINE)


def dispatch(
    args: list[str],
    command: str | None,
    stdin: IO[bytes],
    stdout: IO[bytes] | None = None,
    deadline_end: float | None = None,
) -> Any:
    """Run one verb: a dict (JSON on stdout), bytes (verbatim) or ``_Streamed``."""
    site = _parse_argv(args)
    verb, rest = _parse_command(command)
    if verb == "probe":
        raise refuse("gate_probe_refusal", "the UMRS gate answered the probe")
    home = _home()
    cfg = load_config(gate_root(home))
    if cfg.site != site:
        raise refuse("wrong_site", f"gate.json is for site {cfg.site}, not {site}")
    if verb == "hello":
        return cmd_hello(cfg, gate_root(home))
    if verb in CONTROL_VERBS:
        return cmd_control(verb, cfg, home, stdin)
    end = time.monotonic() + _deadline_seconds(command) if deadline_end is None else deadline_end
    session = _open_session(cfg, home, stdin, stdout or sys.stdout.buffer, end)
    return _B2_HANDLERS[verb](session, rest)


def _emit_error(err: GateError) -> None:
    sys.stderr.write(canonical_bytes(err.as_json()).decode("utf-8") + "\n")
    sys.stderr.flush()


def _expire(_signum: int, _frame: Any) -> None:
    raise GateError("failed", "deadline", EXIT_FAILED)


class _Deadline:
    """``SIGALRM`` after ``seconds``: no session holds the gate (its stdin,
    the log lock) longer than that. A no-op off the main thread."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self.end = time.monotonic() + seconds
        self._previous: Any = None
        self._armed = False

    def __enter__(self) -> _Deadline:
        import signal

        try:
            self._previous = signal.signal(signal.SIGALRM, _expire)
        except (AttributeError, ValueError):
            return self
        self._armed = True
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if not self._armed:
            return
        import signal

        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._previous)


def _emit(out: Any) -> None:
    if isinstance(out, _Streamed):
        sys.stdout.buffer.flush()
        sys.stderr.write(canonical_bytes(out.summary).decode("utf-8") + "\n")
        sys.stderr.flush()
        return
    data = out if isinstance(out, bytes) else canonical_bytes(out, True) + b"\n"
    sys.stdout.buffer.write(data)
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Run the gate; returns the exit code (see the module docstring)."""
    args = sys.argv[1:] if argv is None else list(argv)
    command = os.environ.get("SSH_ORIGINAL_COMMAND")
    try:
        with _Deadline(_deadline_seconds(command)) as deadline:
            out = dispatch(args, command, sys.stdin.buffer, sys.stdout.buffer, deadline.end)
    except GateError as err:
        _emit_error(err)
        return err.code
    except Exception as exc:  # fail closed: never a traceback, never success
        _emit_error(GateError("failed", type(exc).__name__, EXIT_FAILED))
        return EXIT_FAILED
    _emit(out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
