"""Per-person, per-site transport keys (ADR-112 D-3; plan §4 B1.2, B1.5a).

The server generates one ed25519 key per person per site with
``ssh-keygen -t ed25519 -N '' -C umrs-transport-<site>-<key_id> -f <path>``
and keeps it at ``UMRS_TRANSPORT_KEY_DIR/<venue>/<username>/<key_id>``
(``.pub`` beside it). Every directory on that path is 0700 and the private
key is 0600. The database holds only the key id and the public fingerprint,
so a dump or backup carries no key (investigation §4.6). ``security/seal.py``
is not used: only ``umrs-infer`` can unseal (R1-1), and the claim rests on
the user's signature, not on this key's secrecy (ADR-112 D-3).

The key reaches only the user's gate: their ``authorized_keys`` line pins it
to ``command="... umrs_gate.py --site <site>"``.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEY_DIR_ENV = "UMRS_TRANSPORT_KEY_DIR"
DEFAULT_KEY_DIR = "/var/lib/umrs/transport-keys"
COMMENT_PREFIX = "umrs-transport-"

_KEY_ID = re.compile(r"^tk-[0-9a-f]{12}$")
# One path component: no separator, never "." or "..", no leading dash.
_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}$")
_PUBLIC_LINE = re.compile(r"^ssh-ed25519 ([A-Za-z0-9+/]{68}) (\S+)$")

Runner = Callable[..., Any]


class TransportKeyError(RuntimeError):
    """A transport key could not be made, read or removed safely."""


@dataclass(frozen=True)
class TransportKey:
    key_id: str
    path: Path
    public_key: str  # "ssh-ed25519 <b64> umrs-transport-<site>-<key_id>"
    fingerprint: str  # "SHA256:<b64, unpadded>", as ssh-keygen -l prints it


def key_root() -> Path:
    return Path(os.environ.get(KEY_DIR_ENV, "").strip() or DEFAULT_KEY_DIR)


def new_key_id() -> str:
    return f"tk-{secrets.token_hex(6)}"


def comment_for(site: str, key_id: str) -> str:
    return f"{COMMENT_PREFIX}{site}-{key_id}"


def _component(value: str, what: str) -> str:
    if not _COMPONENT.fullmatch(value or ""):
        raise TransportKeyError(f"unsafe {what} for a key path: {value!r}")
    return value


def user_dir(venue: str, username: str, *, root: Path | None = None) -> Path:
    base = root or key_root()
    return base / _component(venue, "venue") / _component(username, "username")


def key_path(
    venue: str, username: str, key_id: str, *, root: Path | None = None
) -> Path:
    if not _KEY_ID.fullmatch(key_id or ""):
        raise TransportKeyError(f"not a transport key id: {key_id!r}")
    return user_dir(venue, username, root=root) / key_id


def keygen_argv(path: Path, comment: str) -> list[str]:
    """The exact ``ssh-keygen`` call of the contract (an argv list, no shell)."""
    return ["ssh-keygen", "-t", "ed25519", "-N", "", "-C", comment, "-f", str(path)]


def fingerprint(public_key: str) -> str:
    """``SHA256:...`` of an ``ssh-ed25519`` line, as ``ssh-keygen -l`` prints it."""
    match = _PUBLIC_LINE.fullmatch(public_key.strip())
    if not match:
        raise TransportKeyError("not an ssh-ed25519 public key line")
    digest = hashlib.sha256(base64.b64decode(match.group(1))).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def _private_dirs(venue: str, username: str, root: Path | None) -> Path:
    """Create the key directories 0700 and tighten any that are wider."""
    base = root or key_root()
    target = user_dir(venue, username, root=base)
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    for directory in (base, target.parent, target):
        directory.mkdir(mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)
    return target


def _read_key(path: Path, key_id: str) -> TransportKey:
    public = path.with_name(path.name + ".pub").read_text().strip()
    _, _, comment = public.rpartition(" ")
    if not comment.endswith(f"-{key_id}") or not comment.startswith(COMMENT_PREFIX):
        raise TransportKeyError(
            f"transport key {key_id}: unexpected comment {comment!r}"
        )
    return TransportKey(
        key_id=key_id, path=path, public_key=public, fingerprint=fingerprint(public)
    )


def _keygen(runner: Runner, argv: list[str]) -> None:
    """``ssh-keygen``; any failure of the tool is a ``TransportKeyError``."""
    try:
        runner(argv, check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=30)
    except (subprocess.SubprocessError, OSError) as exc:
        raise TransportKeyError(f"ssh-keygen failed: {type(exc).__name__}") from exc


def generate(
    venue: str,
    username: str,
    *,
    site: str | None = None,
    root: Path | None = None,
    runner: Runner = subprocess.run,
) -> TransportKey:
    """A new transport key for ``username`` at ``venue``; the caller drops
    the keys it replaces (``destroy_others``)."""
    try:
        directory = _private_dirs(venue, username, root)
    except OSError as exc:
        raise TransportKeyError(f"key directory unusable: {exc}") from exc
    key_id = new_key_id()
    path = directory / key_id
    _keygen(runner, keygen_argv(path, comment_for(site or venue, key_id)))
    if not path.is_file():
        raise TransportKeyError(f"ssh-keygen wrote no key at {path}")
    os.chmod(path, 0o600)
    return _read_key(path, key_id)


def load(
    venue: str, username: str, key_id: str, *, root: Path | None = None
) -> TransportKey | None:
    """The stored key, or None when its file is gone; a key that cannot be
    read is a ``TransportKeyError``."""
    path = key_path(venue, username, key_id, root=root)
    if not path.is_file():
        return None
    try:
        return _read_key(path, key_id)
    except OSError as exc:
        raise TransportKeyError(f"transport key {key_id} unreadable: {exc}") from exc


def _key_files(venue: str, username: str, root: Path | None) -> list[Path]:
    directory = user_dir(venue, username, root=root)
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if _KEY_ID.fullmatch(p.name)]


def _unlink(path: Path) -> None:
    path.unlink(missing_ok=True)
    path.with_name(path.name + ".pub").unlink(missing_ok=True)


def destroy(
    venue: str, username: str, key_id: str = "", *, root: Path | None = None
) -> int:
    """Remove one key (``key_id``) or every key of the person at the venue;
    the number of private keys removed."""
    paths = (
        [key_path(venue, username, key_id, root=root)]
        if key_id
        else _key_files(venue, username, root)
    )
    present = [p for p in paths if p.is_file()]
    for path in present:
        _unlink(path)
    return len(present)


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return -1


def destroy_others(
    venue: str, username: str, keep: str | Iterable[str], *, root: Path | None = None
) -> int:
    """Drop the keys that ``keep`` replaced: every key not kept and no newer
    than the newest kept key. A key made meanwhile (a concurrent rotation,
    not yet recorded) is newer, so it stays."""
    kept = {keep} if isinstance(keep, str) else set(keep)
    files = _key_files(venue, username, root)
    cutoff = max((_mtime_ns(p) for p in files if p.name in kept), default=-1)
    others = [p for p in files if p.name not in kept and _mtime_ns(p) <= cutoff]
    for path in others:
        _unlink(path)
    return len(others)


__all__ = [
    "DEFAULT_KEY_DIR",
    "KEY_DIR_ENV",
    "TransportKey",
    "TransportKeyError",
    "comment_for",
    "destroy",
    "destroy_others",
    "fingerprint",
    "generate",
    "key_path",
    "key_root",
    "keygen_argv",
    "load",
    "new_key_id",
]
