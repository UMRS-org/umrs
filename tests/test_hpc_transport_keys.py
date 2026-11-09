"""Transport keys (ADR-112 D-3; plan §4 B1.2 and the B1.5a wire contract).

Real ``ssh-keygen`` writes every key into ``tmp_path``; nothing reads
``~/.ssh`` and no key file is committed.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from umrs.hpc_accounts import transport_keys as tk

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_generate_writes_the_key_where_the_contract_says_with_tight_modes(tmp_path) -> None:
    root = tmp_path / "keys"
    key = tk.generate("bdrt-front", "alice", root=root)
    assert key.path == root / "bdrt-front" / "alice" / key.key_id
    dirs = (root, root / "bdrt-front", root / "bdrt-front" / "alice")
    assert [_mode(d) for d in dirs] + [_mode(key.path)] == [0o700, 0o700, 0o700, 0o600]
    assert key.path.read_text().startswith("-----BEGIN OPENSSH PRIVATE KEY-----")


def test_the_public_half_is_an_ed25519_line_named_for_site_and_key(tmp_path) -> None:
    key = tk.generate("bdrt-front", "alice", root=tmp_path)
    kind, b64, comment = key.public_key.split(" ")
    want_comment = f"umrs-transport-bdrt-front-{key.key_id}"
    assert (kind, len(b64), comment) == ("ssh-ed25519", 68, want_comment)
    assert (key.key_id[:3], len(key.key_id)) == ("tk-", 15)
    pub = key.path.with_name(key.key_id + ".pub").read_text().strip()
    assert pub == key.public_key


def test_the_fingerprint_is_what_ssh_keygen_prints(tmp_path) -> None:
    key = tk.generate("v1", "alice", root=tmp_path)
    listed = subprocess.run(
        ["ssh-keygen", "-l", "-E", "sha256", "-f", str(key.path) + ".pub"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert key.fingerprint == listed[1]
    assert "PRIVATE" not in key.fingerprint + key.public_key


def test_keygen_is_an_argv_list_with_an_empty_passphrase() -> None:
    argv = tk.keygen_argv(Path("/k/tk-0123456789ab"), "umrs-transport-s-tk-0123456789ab")
    assert argv == [
        "ssh-keygen",
        "-t",
        "ed25519",
        "-N",
        "",
        "-C",
        "umrs-transport-s-tk-0123456789ab",
        "-f",
        "/k/tk-0123456789ab",
    ]


def test_generate_runs_exactly_that_argv(tmp_path) -> None:
    seen: list[list[str]] = []

    def runner(argv, **kwargs):
        seen.append(argv)
        assert kwargs["check"] is True and "shell" not in kwargs
        return subprocess.run(argv, **kwargs)

    key = tk.generate("v1", "alice", root=tmp_path, runner=runner)
    assert seen == [tk.keygen_argv(key.path, f"umrs-transport-v1-{key.key_id}")]


def test_wider_directories_are_tightened(tmp_path) -> None:
    user_dir = tmp_path / "v1" / "alice"
    user_dir.mkdir(parents=True)
    for directory in (tmp_path, tmp_path / "v1", user_dir):
        directory.chmod(0o755)
    tk.generate("v1", "alice", root=tmp_path)
    assert {_mode(d) for d in (tmp_path, tmp_path / "v1", user_dir)} == {0o700}


def test_the_root_comes_from_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(tk.KEY_DIR_ENV, str(tmp_path / "env-root"))
    assert tk.key_root() == tmp_path / "env-root"
    monkeypatch.delenv(tk.KEY_DIR_ENV)
    assert tk.key_root() == Path("/var/lib/umrs/transport-keys")


def test_rotation_keeps_the_old_key_until_told(tmp_path) -> None:
    old = tk.generate("v1", "alice", root=tmp_path)
    new = tk.generate("v1", "alice", root=tmp_path)
    assert old.key_id != new.key_id
    assert tk.load("v1", "alice", old.key_id, root=tmp_path) == old
    assert tk.destroy_others("v1", "alice", keep=new.key_id, root=tmp_path) == 1
    assert tk.load("v1", "alice", old.key_id, root=tmp_path) is None
    assert not old.path.with_name(old.key_id + ".pub").exists()
    assert tk.load("v1", "alice", new.key_id, root=tmp_path) == new


def test_destroy_one_or_every_key_of_the_person_only(tmp_path) -> None:
    first = tk.generate("v1", "alice", root=tmp_path)
    tk.generate("v1", "alice", root=tmp_path)
    bob = tk.generate("v1", "bob", root=tmp_path)
    other_site = tk.generate("v2", "alice", root=tmp_path)
    assert tk.destroy("v1", "alice", first.key_id, root=tmp_path) == 1
    assert tk.destroy("v1", "alice", first.key_id, root=tmp_path) == 0
    assert tk.destroy("v1", "alice", root=tmp_path) == 1
    assert list((tmp_path / "v1" / "alice").iterdir()) == []
    assert bob.path.exists() and other_site.path.exists()
    assert tk.destroy("v1", "nobody", root=tmp_path) == 0


@pytest.mark.parametrize(
    ("venue", "user", "key_id"),
    [
        ("..", "alice", "tk-0123456789ab"),
        ("v1", "../etc", "tk-0123456789ab"),
        ("v1", "a/b", "tk-0123456789ab"),
        ("v1", "-oProxy", "tk-0123456789ab"),
        ("v1", "alice", "../tk-0123456789ab"),
        ("v1", "alice", "tk-XYZ"),
        ("", "alice", "tk-0123456789ab"),
    ],
)
def test_unsafe_path_parts_are_refused(tmp_path, venue, user, key_id) -> None:
    with pytest.raises(tk.TransportKeyError):
        tk.key_path(venue, user, key_id, root=tmp_path)


def test_a_key_with_a_foreign_comment_is_refused(tmp_path) -> None:
    key = tk.generate("v1", "alice", root=tmp_path)
    pub = key.path.with_name(key.key_id + ".pub")
    pub.write_text(pub.read_text().replace("umrs-transport-", "someone-else-"))
    with pytest.raises(tk.TransportKeyError, match="comment"):
        tk.load("v1", "alice", key.key_id, root=tmp_path)


def test_a_keygen_that_writes_nothing_is_an_error(tmp_path) -> None:
    with pytest.raises(tk.TransportKeyError, match="no key"):
        tk.generate("v1", "alice", root=tmp_path, runner=lambda *_a, **_k: None)


def test_fingerprint_refuses_anything_but_an_ed25519_line() -> None:
    with pytest.raises(tk.TransportKeyError):
        tk.fingerprint("ssh-rsa AAAA x")


# ── B1 review fixes (2026-11-08) ─────────────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, ["ssh-keygen"]),
        subprocess.TimeoutExpired(["ssh-keygen"], 30),
        FileNotFoundError("ssh-keygen"),
    ],
)
def test_a_failing_keygen_is_a_transport_key_error(tmp_path, error) -> None:
    def runner(*_a, **_k):
        raise error

    with pytest.raises(tk.TransportKeyError, match="ssh-keygen failed"):
        tk.generate("v1", "alice", root=tmp_path, runner=runner)


def test_an_unusable_key_directory_is_a_transport_key_error(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.raises(tk.TransportKeyError, match="directory"):
        tk.generate("v1", "alice", root=blocker)


def test_a_key_without_its_public_half_is_a_transport_key_error(tmp_path) -> None:
    key = tk.generate("v1", "alice", root=tmp_path)
    key.path.with_name(key.key_id + ".pub").unlink()
    with pytest.raises(tk.TransportKeyError, match="unreadable"):
        tk.load("v1", "alice", key.key_id, root=tmp_path)


def test_destroy_others_keeps_every_kept_key_and_any_newer_one(tmp_path) -> None:
    import os

    old, previous, new, newer = (tk.generate("v1", "alice", root=tmp_path) for _ in range(4))
    for age, key in zip((40, 30, 20, 10), (old, previous, new, newer), strict=True):
        when = 1_700_000_000 - age
        os.utime(key.path, (when, when))
    kept = {previous.key_id, new.key_id}
    assert tk.destroy_others("v1", "alice", keep=kept, root=tmp_path) == 1
    left = {p.name for p in (tmp_path / "v1" / "alice").iterdir() if not p.suffix}
    assert left == {previous.key_id, new.key_id, newer.key_id}
    assert tk.destroy_others("v1", "alice", keep="tk-000000000000", root=tmp_path) == 0
