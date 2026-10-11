"""Login-node step helper (``deploy/_common/umrs_step_helper.py``, build spec §4).

Every verb runs the real helper as the server runs it (``python3 -I helper
<verb> ...``) against a tmp workdir root; the Slurm verbs get fake
``sbatch`` / ``squeue`` / ``sacct`` / ``scancel`` on PATH. The hostile cases
are what a step can leave in its workdir (links, FIFOs, hard links, huge
files, unreadable directories) and malformed server input.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parents[1] / "deploy" / "_common" / "umrs_step_helper.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("umrs_step_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def run(*args, stdin: bytes | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", str(HELPER), *map(str, args)],
        input=stdin,
        capture_output=True,
        timeout=60,
        check=False,
        env=env,
    )


def _gnu_rm() -> bool:
    """Whether this host's rm takes ``--one-file-system``, as the login nodes'
    GNU rm does. The helper's forced removal needs it and runs only there;
    BSD rm (macOS) refuses it, so those tests failed on a workstation and left
    their deep or locked trees behind, which crashed the next pytest
    session's temp cleanup (2026-10-11)."""
    probe = subprocess.run(
        ["rm", "-rf", "--one-file-system", "--", "/nonexistent-umrs-rm-probe"],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0


needs_gnu_rm = pytest.mark.skipif(
    not _gnu_rm(), reason="forced removal needs GNU rm (--one-file-system), as on login nodes"
)


def reason(proc: subprocess.CompletedProcess) -> str:
    return json.loads(proc.stderr.decode().strip().splitlines()[-1])["detail"]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "steps"


@pytest.fixture
def wd(root: Path) -> Path:
    proc = run("mkwd", root, "wf-abc-s1-1-")
    assert proc.returncode == 0, proc.stderr
    return Path(proc.stdout.decode().strip())


def _tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def _file(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _special(name: str, kind: bytes, linkname: str = "") -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = linkname
    return info, None


def stage(root: Path, wd: Path, payload: bytes, max_files=100, max_bytes=1 << 20):
    return run(
        "stage-in", root, wd, "--max-files", max_files, "--max-bytes", max_bytes, stdin=payload
    )


# ── mkwd ────────────────────────────────────────────────────────────────────


def test_mkwd_creates_a_private_workdir_with_its_five_dirs(root, wd):
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert (wd.parent, wd.name[: len("wf-abc-s1-1-")]) == (root, "wf-abc-s1-1-")
    assert len(wd.name) == len("wf-abc-s1-1-") + 8
    modes = {p.name: stat.S_IMODE(p.stat().st_mode) for p in [wd, *wd.iterdir()]}
    assert modes == dict.fromkeys((wd.name, "in", "log", "out", "reply", "src"), 448)


def test_mkwd_takes_the_longest_tagged_prefix_the_server_makes(root):
    # <run 40>-<step 40>-<attempt>-<inputs tag 12>- (review round 3)
    prefix = f"{'r' * 40}-{'s' * 40}-999-{'a' * 12}-"
    proc = run("mkwd", root, prefix)
    assert proc.returncode == 0, proc.stderr
    name = Path(proc.stdout.decode().strip()).name
    assert name.startswith(prefix) and helper.WORKDIR_NAME_RE.fullmatch(name)


@pytest.mark.parametrize("prefix", ["../x-", "a/b-", "-lead", "", "x" * 140])
def test_mkwd_refuses_a_prefix_outside_the_name_rule(root, prefix):
    proc = run("mkwd", root, prefix)
    assert proc.returncode in (helper.EXIT_REFUSED, helper.EXIT_USAGE)
    assert not root.exists() or not any(root.iterdir())


@pytest.mark.parametrize("bad_root", ["relative/steps", "/", "/tmp/../etc"])
def test_mkwd_refuses_a_root_that_is_not_absolute_and_normalised(bad_root):
    assert run("mkwd", bad_root, "p-").returncode == helper.EXIT_REFUSED


def test_mkwd_refuses_a_symlinked_or_shared_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    assert run("mkwd", link, "p-").returncode == helper.EXIT_REFUSED
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    assert run("mkwd", shared, "p-").returncode == helper.EXIT_REFUSED
    assert list(real.iterdir()) == [] and list(shared.iterdir()) == []


# ── stage-in ────────────────────────────────────────────────────────────────


def test_stage_in_extracts_in_and_src_and_reports_the_summary(root, wd):
    payload = _tar(
        [
            _file("in/manifest.json", b"{}"),
            _file("in/b/k/deep/x.bin", b"12345"),
            _file("src/pkg/step.py", b"X = 1\n"),
        ]
    )
    proc = stage(root, wd, payload)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["files"] == 3 and summary["bytes"] == 2 + 5 + 6
    assert summary["free_bytes"] > 0
    assert (wd / "in/b/k/deep/x.bin").read_bytes() == b"12345"
    assert (wd / "src/pkg/step.py").read_text() == "X = 1\n"
    assert stat.S_IMODE((wd / "in/b/k/deep/x.bin").stat().st_mode) == 0o600
    assert stat.S_IMODE((wd / "in/b/k/deep").stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "member",
    [
        _special("in/link", tarfile.SYMTYPE, "/etc/passwd"),
        _special("in/hard", tarfile.LNKTYPE, "in/manifest.json"),
        _special("in/dev", tarfile.CHRTYPE),
        _special("in/blk", tarfile.BLKTYPE),
        _special("in/fifo", tarfile.FIFOTYPE),
        _special("in/cont", tarfile.CONTTYPE),
        _file("in/../../escape", b"x"),
        _file("/abs/path", b"x"),
        _file("out/x", b"x"),
        _file("reply/output.json", b"{}"),
        _file("job.sbatch", b"#!/bin/bash"),
        _file("in/ctrl\nname", b"x"),
        _file("in", b"x"),
    ],
    ids=lambda m: f"{m[0].name}-{m[0].type!r}",
)
def test_stage_in_refuses_links_devices_and_names_outside_in_and_src(root, wd, member):
    proc = stage(root, wd, _tar([member]))
    assert proc.returncode == helper.EXIT_REFUSED, proc.stderr
    assert not (wd / "escape").exists() and not (root.parent / "escape").exists()
    assert list((wd / "out").iterdir()) == [] and list((wd / "reply").iterdir()) == []


def test_stage_in_refuses_a_sparse_member(root, wd):
    info, data = _file("in/sparse", b"abc")
    info.type = tarfile.GNUTYPE_SPARSE
    assert stage(root, wd, _tar([(info, data)])).returncode == helper.EXIT_REFUSED


def test_stage_in_enforces_its_file_and_byte_caps(root, wd):
    files = _tar([_file(f"in/f{i}", b"x") for i in range(3)])
    assert "more than 2 files" in reason(stage(root, wd, files, max_files=2))
    big = _tar([_file("in/big", b"x" * 100)])
    assert "more than 50 bytes" in reason(stage(root, wd, big, max_bytes=50))


def test_stage_in_refuses_a_duplicate_member(root, wd):
    payload = _tar([_file("in/a", b"1"), _file("in/a", b"2")])
    proc = stage(root, wd, payload)
    assert proc.returncode == helper.EXIT_REFUSED
    assert (wd / "in/a").read_bytes() == b"1"


def test_stage_in_never_follows_a_link_planted_under_in(root, wd, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (wd / "in" / "b").symlink_to(outside)
    proc = stage(root, wd, _tar([_file("in/b/k", b"x")]))
    assert proc.returncode == helper.EXIT_REFUSED
    assert list(outside.iterdir()) == []


def test_stage_in_refuses_a_workdir_outside_the_root(root, wd, tmp_path):
    other = tmp_path / "elsewhere" / "wf-abc-s1-1-abcdefgh"
    other.mkdir(parents=True)
    assert stage(root, other, _tar([_file("in/a", b"1")])).returncode == helper.EXIT_REFUSED
    assert stage(root, root, b"").returncode == helper.EXIT_REFUSED


def test_stage_in_refuses_garbage_input(root, wd):
    assert stage(root, wd, b"not a tar at all" * 100).returncode == helper.EXIT_REFUSED


# ── put ─────────────────────────────────────────────────────────────────────


def test_put_writes_a_new_private_file(root, wd):
    assert run("put", root, wd, "input.pkl", stdin=b"payload").returncode == 0
    assert (wd / "input.pkl").read_bytes() == b"payload"
    assert stat.S_IMODE((wd / "input.pkl").stat().st_mode) == 0o600
    again = run("put", root, wd, "input.pkl", stdin=b"other")
    assert again.returncode == helper.EXIT_REFUSED
    assert (wd / "input.pkl").read_bytes() == b"payload"


@pytest.mark.parametrize("name", ["job.sh", "../x", "out/x", "reply/output.json", "log"])
def test_put_refuses_any_other_name(root, wd, name):
    assert run("put", root, wd, name, stdin=b"x").returncode == helper.EXIT_REFUSED


def test_put_never_writes_through_a_planted_link(root, wd, tmp_path):
    target = tmp_path / "victim"
    target.write_text("keep")
    (wd / "job.sbatch").symlink_to(target)
    assert run("put", root, wd, "job.sbatch", stdin=b"x").returncode == helper.EXIT_REFUSED
    assert target.read_text() == "keep"


def test_put_caps_the_file_and_leaves_nothing_behind(root, wd, monkeypatch):
    monkeypatch.setitem(helper.PUT_CAPS, "job.sbatch", 4)
    with pytest.raises(helper.Refused):
        helper.put(str(root), str(wd), "job.sbatch", io.BytesIO(b"12345"))
    assert not (wd / "job.sbatch").exists()


# ── read-reply / tail-log ───────────────────────────────────────────────────


def test_read_reply_returns_the_reply_and_3_when_absent(root, wd):
    assert run("read-reply", root, wd, 1024).returncode == helper.EXIT_ABSENT
    (wd / "reply" / "output.json").write_bytes(b'{"status": "ok"}')
    proc = run("read-reply", root, wd, 1024)
    assert proc.returncode == 0 and proc.stdout == b'{"status": "ok"}'


def test_read_reply_refuses_a_symlink_to_a_secret(root, wd, tmp_path):
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    (wd / "reply" / "output.json").symlink_to(secret)
    proc = run("read-reply", root, wd, 1024)
    assert proc.returncode == helper.EXIT_REFUSED and b"PRIVATE" not in proc.stdout


def test_read_reply_refuses_a_hard_link_a_fifo_a_directory_and_a_linked_reply_dir(
    root, wd, tmp_path
):
    reply = wd / "reply" / "output.json"
    other = wd / "out" / "x"
    other.write_text("{}")
    os.link(other, reply)
    assert run("read-reply", root, wd, 1024).returncode == helper.EXIT_REFUSED
    reply.unlink()
    os.mkfifo(reply)
    assert run("read-reply", root, wd, 1024).returncode == helper.EXIT_REFUSED  # no hang
    reply.unlink()
    reply.mkdir()
    assert run("read-reply", root, wd, 1024).returncode == helper.EXIT_REFUSED
    reply.rmdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "output.json").write_text('{"status": "ok"}')
    (wd / "reply").rmdir()
    (wd / "reply").symlink_to(elsewhere)
    assert run("read-reply", root, wd, 1024).returncode == helper.EXIT_REFUSED


@pytest.fixture
def not_root():
    if os.geteuid() == 0:
        pytest.skip("root reads past any mode")


@pytest.mark.parametrize(("target", "mode"), [("reply", 0o000), ("reply", 0o300),
                                              ("reply/output.json", 0o000)])  # fmt: skip
def test_read_reply_refuses_a_reply_the_step_made_unreadable(root, wd, not_root, target, mode):
    # The step owns reply/ (keep-id): a withheld permission is its doing, a
    # refusal (the workdir goes), never a failure (kept and retried).
    (wd / "reply" / "output.json").write_bytes(b'{"status": "ok"}')
    (wd / target).chmod(mode)
    try:
        proc = run("read-reply", root, wd, 1024)
    finally:
        (wd / target).chmod(0o700)
    assert proc.returncode == helper.EXIT_REFUSED and proc.stdout == b""
    assert "not readable: the step hid it" in reason(proc)


def _bind_socket(directory: Path, name: str, monkeypatch) -> socket.socket:
    # Relative: a tmp path easily passes the 107-byte AF_UNIX path limit.
    monkeypatch.chdir(directory)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(name)
    return sock


def test_read_reply_refuses_a_socket_the_step_bound_at_the_reply(root, wd, monkeypatch):
    # --network=none leaves AF_UNIX usable; open() on a socket fails ENXIO,
    # which was a helper failure: the server kept the workdir for every
    # retry to adopt and fail on again (review round 4).
    with _bind_socket(wd / "reply", "output.json", monkeypatch):
        proc = run("read-reply", root, wd, 1024)
    assert proc.returncode == helper.EXIT_REFUSED and proc.stdout == b""


def test_open_plain_refuses_a_socket_that_replaced_a_scanned_file(wd, monkeypatch):
    # collect-out opens each scanned file again: a socket swapped in between.
    with _bind_socket(wd / "out", "r.csv", monkeypatch):
        dir_fd = os.open(wd / "out", os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(helper.Refused):
                helper.open_plain(dir_fd, "r.csv")
        finally:
            os.close(dir_fd)


def test_read_reply_refuses_a_reply_over_the_cap(root, wd):
    (wd / "reply" / "output.json").write_bytes(b"x" * 101)
    proc = run("read-reply", root, wd, 100)
    assert proc.returncode == helper.EXIT_REFUSED and proc.stdout == b""
    assert run("read-reply", root, wd, 0).returncode == helper.EXIT_REFUSED
    too_big = helper.MAX_REPLY_CAP + 1
    assert run("read-reply", root, wd, too_big).returncode == helper.EXIT_REFUSED


def test_tail_log_reads_only_the_tail_and_never_a_link(root, wd, tmp_path):
    assert run("tail-log", root, wd, 10).returncode == helper.EXIT_ABSENT
    (wd / "log" / "slurm.err").write_bytes(b"a" * 1000 + b"THE END")
    proc = run("tail-log", root, wd, 7)
    assert proc.returncode == 0 and proc.stdout == b"THE END"
    (wd / "log" / "slurm.err").unlink()
    secret = tmp_path / "secret"
    secret.write_text("TOKEN")
    (wd / "log" / "slurm.err").symlink_to(secret)
    assert run("tail-log", root, wd, 7).returncode == helper.EXIT_REFUSED


# ── collect-out ─────────────────────────────────────────────────────────────


def collect(root, wd, max_files=100, max_bytes=1 << 20, max_file_bytes=1 << 20):
    return run(
        "collect-out", root, wd,
        "--max-files", max_files, "--max-bytes", max_bytes, "--max-file-bytes", max_file_bytes,
    )  # fmt: skip


def _members(stream: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(stream), mode="r|") as tar:
        return {m.name: tar.extractfile(m).read() for m in tar if m.isfile()}


def test_collect_out_streams_regular_files_with_relative_names(root, wd):
    (wd / "out" / "bkt" / "a" / "b").mkdir(parents=True)
    (wd / "out" / "bkt" / "a" / "b" / "c.txt").write_bytes(b"deep")
    (wd / "out" / "bkt" / "top.json").write_bytes(b"{}")
    (wd / "out" / "empty").mkdir()
    proc = collect(root, wd)
    assert proc.returncode == 0, proc.stderr
    assert _members(proc.stdout) == {"bkt/a/b/c.txt": b"deep", "bkt/top.json": b"{}"}
    assert json.loads(proc.stderr) == {"files": 2, "bytes": 6}


def test_collect_out_of_an_empty_out_is_an_empty_archive(root, wd):
    proc = collect(root, wd)
    assert proc.returncode == 0 and _members(proc.stdout) == {}
    assert json.loads(proc.stderr) == {"files": 0, "bytes": 0}


def test_collect_out_refuses_a_symlink_before_sending_anything(root, wd):
    (wd / "out" / "bkt").mkdir()
    (wd / "out" / "bkt" / "ok").write_text("fine")
    (wd / "out" / "bkt" / "key").symlink_to(Path.home() / ".ssh" / "id_rsa")
    proc = collect(root, wd)
    assert proc.returncode == helper.EXIT_REFUSED
    assert proc.stdout == b""
    assert "not a regular" in reason(proc)


def test_collect_out_refuses_hard_links_fifos_and_sockets(root, wd):
    out = wd / "out"
    (out / "a").write_text("x")
    os.link(out / "a", out / "b")
    assert collect(root, wd).returncode == helper.EXIT_REFUSED
    (out / "b").unlink()
    os.mkfifo(out / "pipe")
    assert collect(root, wd).returncode == helper.EXIT_REFUSED


def test_collect_out_enforces_its_caps_up_front(root, wd):
    out = wd / "out"
    for i in range(3):
        (out / f"f{i}").write_bytes(b"x" * 10)
    assert "more than 2 files" in reason(collect(root, wd, max_files=2))
    assert "more than 25 bytes" in reason(collect(root, wd, max_bytes=25))
    assert "cap 5" in reason(collect(root, wd, max_file_bytes=5))
    for proc in (collect(root, wd, max_files=2), collect(root, wd, max_bytes=25)):
        assert proc.stdout == b""


@pytest.mark.parametrize(
    ("target", "mode"),
    [("bkt/z.txt", 0o000), ("bkt/z.txt", 0o200), ("bkt", 0o300), ("bkt", 0o600), (".", 0o000)],
)
def test_collect_out_refuses_what_the_step_made_unreadable_before_sending(
    root, wd, not_root, target, mode
):
    out = wd / "out"
    (out / "bkt").mkdir()
    (out / "bkt" / "a.txt").write_text("sorted first")
    (out / "bkt" / "z.txt").write_text("hidden")
    (out / target).chmod(mode)
    try:
        proc = collect(root, wd)
    finally:
        (out / target).chmod(0o700)
    assert proc.returncode == helper.EXIT_REFUSED
    assert proc.stdout == b""  # nothing uploaded before the refusal
    assert "not readable: the step hid it" in reason(proc)


def test_collect_out_refuses_a_linked_out_dir(root, wd, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "loot").write_text("secret")
    (wd / "out").rmdir()
    (wd / "out").symlink_to(elsewhere)
    proc = collect(root, wd)
    assert proc.returncode == helper.EXIT_REFUSED and b"secret" not in proc.stdout


def test_collect_out_round_trips_through_the_server_collector(root, wd):
    from umrs.server.step_staging import collect_out

    class _S3:
        def __init__(self):
            self.objects = {}

        def upload_fileobj(self, fileobj, bucket, key):
            self.objects[(bucket, key)] = fileobj.read()

    (wd / "out" / "u.p.a-output" / "res").mkdir(parents=True)
    (wd / "out" / "u.p.a-output" / "res" / "r.csv").write_bytes(b"1,2")
    proc = collect(root, wd)
    s3 = _S3()
    mapping = collect_out(io.BytesIO(proc.stdout), s3, "u.p.a-output", "wf-1", "s1")
    assert mapping == {"res/r.csv": "runs/wf-1/s1/res/r.csv"}
    assert s3.objects == {("u.p.a-output", "runs/wf-1/s1/res/r.csv"): b"1,2"}


# ── rmwd ────────────────────────────────────────────────────────────────────


def test_rmwd_removes_the_workdir_and_is_idempotent(root, wd):
    (wd / "out" / "f").write_text("x")
    assert run("rmwd", root, wd).returncode == 0
    assert not wd.exists() and root.exists()
    assert run("rmwd", root, wd).returncode == 0


@needs_gnu_rm
def test_rmwd_removes_what_a_step_made_unreadable(root, wd):
    locked = wd / "out" / "locked" / "inner"
    locked.mkdir(parents=True)
    (locked / "f").write_text("x")
    locked.chmod(0o000)
    (wd / "out" / "locked").chmod(0o000)
    (wd / "out").chmod(0o500)
    assert run("rmwd", root, wd).returncode == 0
    assert not wd.exists()


def test_rmwd_never_follows_a_link_out_of_the_workdir(root, wd, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("x")
    (wd / "out" / "link").symlink_to(victim)
    assert run("rmwd", root, wd).returncode == 0
    assert (victim / "keep").read_text() == "x"


@needs_gnu_rm
def test_rmwd_removes_a_tree_deeper_than_python_recurses(root, wd):
    # python3 on the login node is 3.9: shutil.rmtree recurses once per level
    # and a step's 1100-deep tree raised RecursionError past every handler.
    fd = os.open(wd / "out", os.O_RDONLY | os.O_DIRECTORY)
    for _ in range(1100):
        os.mkdir("d", dir_fd=fd)
        nxt = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        os.close(fd)
        fd = nxt
    os.close(fd)
    proc = run("rmwd", root, wd)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert not wd.exists()


@needs_gnu_rm
def test_rmwd_removes_thousands_of_locked_dirs_under_a_low_fd_limit(root, wd):
    import resource

    for i in range(2000):
        (wd / "out" / f"d{i}").mkdir(mode=0o000)

    def low_fd_limit():
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    proc = subprocess.run(
        [sys.executable, "-I", str(HELPER), "rmwd", str(root), str(wd)],
        capture_output=True, timeout=120, check=False, preexec_fn=low_fd_limit,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr[-500:]
    assert not wd.exists()


@needs_gnu_rm
def test_rmwd_forced_removal_never_follows_a_link(root, wd, tmp_path):
    victim = tmp_path / "victim"
    (victim / "sub").mkdir(parents=True)
    (victim / "sub" / "keep").write_text("x")
    victim.chmod(0o500)
    (wd / "out" / "link").symlink_to(victim)
    (wd / "out" / "locked").mkdir(mode=0o000)  # forces the fallback
    try:
        assert run("rmwd", root, wd).returncode == 0
        assert not wd.exists()
        assert stat.S_IMODE(victim.stat().st_mode) == 0o500
        assert (victim / "sub" / "keep").read_text() == "x"
    finally:
        victim.chmod(0o700)


@pytest.mark.parametrize(
    "target",
    ["{root}", "{root}/..", "/etc", "{root}/wf-abc-s1-1-abcdefgh/../../x", "{root}/short"],
)
def test_rmwd_refuses_anything_but_a_workdir_under_the_root(root, wd, target):
    path = target.format(root=root)
    assert run("rmwd", root, path).returncode == helper.EXIT_REFUSED
    assert wd.exists()


def test_rmwd_refuses_a_symlinked_workdir(root, wd, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("x")
    fake = root / "wf-abc-s1-1-zzzzzzzz"
    fake.symlink_to(victim)
    assert run("rmwd", root, fake).returncode == helper.EXIT_REFUSED
    assert (victim / "keep").exists()


# ── Slurm verbs ─────────────────────────────────────────────────────────────


FAKE = r"""#!/usr/bin/env python3
import json, os, sys
state = os.environ["FAKE_SLURM"]
name = os.path.basename(sys.argv[0])
with open(os.path.join(state, "calls.log"), "a") as fh:
    fh.write(json.dumps([name, os.getcwd(), *sys.argv[1:]]) + "\n")
spec = json.load(open(os.path.join(state, name + ".json")))
sys.stdout.write(spec.get("out", ""))
sys.stderr.write(spec.get("err", ""))
sys.exit(spec.get("rc", 0))
"""


@pytest.fixture
def slurm(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        (bin_dir / name).write_text(FAKE)
        (bin_dir / name).chmod(0o755)
        (tmp_path / f"{name}.json").write_text("{}")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_SLURM": str(tmp_path)}

    def answer(name, out="", rc=0, err=""):
        (tmp_path / f"{name}.json").write_text(json.dumps({"out": out, "rc": rc, "err": err}))

    def calls():
        log = tmp_path / "calls.log"
        return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

    return env, answer, calls


def test_submit_runs_sbatch_parsable_from_the_workdir(root, wd, slurm):
    env, answer, calls = slurm
    assert run("put", root, wd, "job.sbatch", stdin=b"#!/bin/bash\n").returncode == 0
    answer("sbatch", out="4242;cluster\n")
    proc = run("submit", root, wd, env=env)
    assert proc.returncode == 0 and proc.stdout.decode().strip() == "4242"
    name, cwd, *args = calls()[-1]
    assert (name, cwd) == ("sbatch", str(wd))
    assert args == ["--parsable", f"--chdir={wd}", f"{wd}/job.sbatch"]


def test_submit_names_the_workdir_as_given_under_a_symlinked_root(tmp_path, slurm):
    # RCCS: /lvs0/rccs-dmp -> /lvs0/dne1/rccs-dmp. Without --chdir Slurm
    # records getcwd(), the physical path, and reconciliation never matches.
    env, answer, calls = slurm
    (tmp_path / "dne1").mkdir()
    (tmp_path / "rccs-dmp").symlink_to(tmp_path / "dne1")
    root = tmp_path / "rccs-dmp" / "steps"
    wd = Path(run("mkwd", root, "wf-abc-s1-1-").stdout.decode().strip())
    assert run("put", root, wd, "job.sbatch", stdin=b"#!/bin/bash\n").returncode == 0
    answer("sbatch", out="4242\n")
    assert run("submit", root, wd, env=env).returncode == 0
    _name, cwd, *args = calls()[-1]
    assert cwd == os.path.realpath(wd) != str(wd)  # what getcwd() would record
    assert f"--chdir={wd}" in args


def test_submit_refuses_a_missing_or_linked_script_and_garbage_output(root, wd, slurm, tmp_path):
    env, answer, calls = slurm
    assert run("submit", root, wd, env=env).returncode == helper.EXIT_ABSENT
    (wd / "job.sbatch").symlink_to(tmp_path / "evil.sh")
    assert run("submit", root, wd, env=env).returncode == helper.EXIT_REFUSED
    assert calls() == []
    (wd / "job.sbatch").unlink()
    (wd / "job.sbatch").write_text("#!/bin/bash\n")
    answer("sbatch", out="Submitted? maybe\n")
    assert run("submit", root, wd, env=env).returncode == helper.EXIT_FAILED
    answer("sbatch", rc=1, err="sbatch: error: Socket timed out on send/recv")
    proc = run("submit", root, wd, env=env)
    assert proc.returncode == helper.EXIT_FAILED and "timed out" in reason(proc)


def test_status_prefers_squeue_then_sacct(slurm):
    env, answer, calls = slurm
    answer("squeue", out="RUNNING\n")
    assert run("status", "77", env=env).stdout.decode().strip() == "RUNNING"
    answer("squeue", rc=1, err="slurm_load_jobs error: Invalid job id specified")
    answer("sacct", out="CANCELLED by 1624\n")
    assert run("status", "77", env=env).stdout.decode().strip() == "CANCELLED"
    answer("sacct", out="")
    assert run("status", "77", env=env).stdout.decode().strip() == "UNKNOWN"
    answer("sacct", rc=1, err="sacct: error: slurmdbd down")
    assert run("status", "77", env=env).returncode == helper.EXIT_FAILED
    sacct_call = next(c for c in calls() if c[0] == "sacct")
    assert sacct_call[2:] == ["-n", "-X", "-P", "-j", "77", "-o", "State"]


@pytest.mark.parametrize("bad", ["77;rm -rf ~", "-1", "abc", "1 2"])
def test_status_and_cancel_refuse_a_job_id_that_is_not_a_number(slurm, bad):
    env, _answer, calls = slurm
    assert run("status", bad, env=env).returncode == helper.EXIT_REFUSED
    assert run("cancel", bad, env=env).returncode == helper.EXIT_REFUSED
    assert calls() == []


def test_cancel_passes_scancel_outcome(slurm):
    env, answer, calls = slurm
    assert run("cancel", "12", env=env).returncode == 0
    assert calls()[-1][2:] == ["12"]
    answer("scancel", rc=1, err="scancel: error: Kill job error")
    assert run("cancel", "12", env=env).returncode == helper.EXIT_FAILED


def test_find_jobs_merges_live_and_finished_jobs_by_exact_name(slurm):
    env, answer, calls = slurm
    name = "umrs-step-wf-abc-s1"
    answer(
        "squeue",
        out=f"12|RUNNING|{name}|/r/wd2|umrs-step:wf-abc:2\n13|PENDING|{name}-x|/r/other|\n",
    )
    answer(
        "sacct",
        out=f"11|COMPLETED|{name}|/r/wd1|\n12|RUNNING|{name}|/r/wd2|\n"
        f"14|FAILED|{name}|/r/wd4|odd comment;$(x)\nbad|x|y|z|w\n",
    )
    proc = run("find-jobs", name, env=env)
    assert proc.returncode == 0
    assert proc.stdout.decode().splitlines() == [
        f"11 COMPLETED {name} - /r/wd1",  # sacct keeps no comment on RCCS
        f"12 RUNNING {name} umrs-step:wf-abc:2 /r/wd2",
        f"14 FAILED {name} ? /r/wd4",
    ]
    squeue_call = next(c for c in calls() if c[0] == "squeue")
    assert squeue_call[2:] == ["--me", "-h", f"--name={name}", "-o", "%i|%T|%j|%Z|%k"]
    sacct_call = next(c for c in calls() if c[0] == "sacct")
    assert sacct_call[-1] == "JobID,State,JobName,WorkDir,Comment"


def test_find_jobs_reports_a_physical_workdir_under_the_named_root(tmp_path, slurm):
    env, answer, _calls = slurm
    (tmp_path / "dne1" / "steps").mkdir(parents=True)
    (tmp_path / "rccs-dmp").symlink_to(tmp_path / "dne1")
    root = tmp_path / "rccs-dmp" / "steps"
    physical = tmp_path / "dne1" / "steps" / "wf-abc-s1-1-abcdefgh"
    name = "umrs-step-wf-abc-s1"
    answer("squeue", out="")
    answer(
        "sacct",
        out=f"11|COMPLETED|{name}|{physical}|\n12|FAILED|{name}|{tmp_path}/other/x-abcdefgh|\n",
    )
    rows = run("find-jobs", "--root", root, name, env=env).stdout.decode().splitlines()
    assert rows == [
        f"11 COMPLETED {name} - {root}/wf-abc-s1-1-abcdefgh",
        f"12 FAILED {name} - {tmp_path}/other/x-abcdefgh",  # elsewhere: as reported
    ]
    unscoped = run("find-jobs", name, env=env).stdout.decode().splitlines()
    assert unscoped[0].endswith(str(physical))
    assert run("find-jobs", "--root", "rel/root", name, env=env).returncode == helper.EXIT_REFUSED


def test_find_jobs_fails_when_squeue_cannot_list_and_refuses_other_names(slurm):
    env, answer, _calls = slurm
    answer("squeue", rc=1, err="slurm_load_jobs error: Unable to contact slurm controller")
    # Slurm down is its own exit code: the server rides it out (peer review F3).
    assert run("find-jobs", "umrs-step-wf-s1", env=env).returncode == helper.EXIT_SCHEDULER
    assert run("find-jobs", "some-other-job", env=env).returncode == helper.EXIT_REFUSED


def test_a_missing_slurm_tool_is_a_plain_failure_not_an_outage(slurm, tmp_path):
    # Peer review (2026-09-26): rc 127 was retried for the whole budget.
    import shutil

    env, _answer, _calls = slurm
    (tmp_path / "bin" / "squeue").unlink()
    env = {**env, "PATH": f"{tmp_path / 'bin'}:/nonexistent"}
    if shutil.which("squeue", path=env["PATH"]):
        pytest.skip("a real squeue is on PATH")
    assert run("find-jobs", "umrs-step-wf-s1", env=env).returncode == helper.EXIT_FAILED


def test_find_jobs_fails_when_sacct_cannot_list(slurm):
    # A partial list hides a COMPLETED job whose kept workdir is the only
    # copy of its outputs: the server would submit the step again.
    env, answer, _calls = slurm
    answer("squeue", out="")
    answer("sacct", rc=1, err="sacct: error: Problem talking to the database")
    proc = run("find-jobs", "umrs-step-wf-s1", env=env)
    assert proc.returncode == helper.EXIT_SCHEDULER
    assert "sacct failed" in reason(proc) and proc.stdout == b""


# ── sweep ───────────────────────────────────────────────────────────────────


def _age(path: Path, seconds: float) -> None:
    when = time.time() - seconds
    for sub in ("", "log", "reply", "out"):
        target = path / sub if sub else path
        if target.exists():
            os.utime(target, (when, when), follow_symlinks=False)


@needs_gnu_rm
def test_sweep_removes_idle_orphans_and_keeps_live_and_recent_workdirs(root, wd, slurm):
    env, answer, calls = slurm
    live = Path(run("mkwd", root, "wf-abc-s2-1-").stdout.decode().strip())
    recent = Path(run("mkwd", root, "wf-abc-s3-1-").stdout.decode().strip())
    (root / "not-a-workdir").mkdir()
    (wd / "out" / "locked").mkdir(mode=0o000)
    for path in (wd, live, root / "not-a-workdir"):
        _age(path, 7200)
    _age(recent, 10)
    answer("squeue", out=f"41|RUNNING|umrs-step-wf-abc-s2|{live}|umrs-step:wf-abc:1\n")
    proc = run("sweep", root, "--older-than", 3600, env=env)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary == {"removed": [wd.name], "failed": [], "live": 1}
    assert not wd.exists() and live.exists() and recent.exists()
    assert (root / "not-a-workdir").exists()
    assert calls()[-1][2:] == ["--me", "-h", "-o", "%i|%T|%j|%Z|%k"]


def test_sweep_ignores_the_step_dating_out_and_reply_into_the_future(root, wd, slurm):
    # The step owns the mount roots of out/ and reply/ (keep-id): their
    # mtimes kept an orphaned workdir "active" for ever (review round 4).
    env, _answer, _calls = slurm
    _age(wd, 7200)
    for sub in ("out", "reply"):
        os.utime(wd / sub, (4e9, 4e9))
    summary = json.loads(run("sweep", root, "--older-than", 3600, env=env).stdout)
    assert summary["removed"] == [wd.name] and not wd.exists()


def test_sweep_keeps_a_live_job_recorded_under_the_physical_root(tmp_path, slurm):
    env, answer, _calls = slurm
    (tmp_path / "dne1").mkdir()
    (tmp_path / "rccs-dmp").symlink_to(tmp_path / "dne1")
    root = tmp_path / "rccs-dmp" / "steps"
    wd = Path(run("mkwd", root, "wf-abc-s1-1-").stdout.decode().strip())
    _age(wd, 7200)
    answer("squeue", out=f"41|PENDING|umrs-step-wf-abc-s1|{os.path.realpath(wd)}|\n")
    assert json.loads(run("sweep", root, "--older-than", 3600, env=env).stdout)["removed"] == []
    assert wd.exists()


def test_sweep_removes_nothing_when_squeue_fails_or_the_age_is_short(root, wd, slurm):
    env, answer, _calls = slurm
    _age(wd, 7200)
    assert run("sweep", root, "--older-than", 60, env=env).returncode == helper.EXIT_REFUSED
    answer("squeue", rc=1, err="slurm_load_jobs error: Unable to contact slurm controller")
    assert run("sweep", root, "--older-than", 3600, env=env).returncode == helper.EXIT_SCHEDULER
    assert wd.exists()


def test_an_unknown_verb_is_a_usage_error():
    assert run("exec", "sh").returncode == helper.EXIT_USAGE
