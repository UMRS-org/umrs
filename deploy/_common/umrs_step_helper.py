#!/usr/bin/env python3
"""Login-node helper for the connected-HPC step lane (ADR-074).

The UMRS server runs this over its SSH path as
``python3 -I umrs_step_helper.py <verb> ...`` (build spec
docs/development/rccs-step-lane-spec-2026-09-26.md §4 in the superproject).
It is the only thing the server executes on the cluster for a step, so the
verb set below is also the future forced-command allow-list.

A step's workdir holds files the step wrote (``out/``, ``reply/``), and the
account running this helper can read far more than the step may (SSH keys,
serving secrets). So every path the helper opens under the workdir root is
opened relative to its parent directory with ``O_NOFOLLOW`` (no symlink is
followed at any level below the root), only regular singly-linked files are
read, and every read is capped. Standard library only; Python >= 3.9.

Exit codes: 0 done, 1 failed, 2 usage, 3 absent, 4 refused. A failure or a
refusal prints one JSON object ``{"error": ..., "detail": ...}`` on stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import time

EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ABSENT = 3
EXIT_REFUSED = 4

WORKDIR_DIRS = ("in", "out", "reply", "src", "log")
STAGE_TOPS = ("in", "src")
#: Files the server may write into a workdir, with their size caps.
PUT_CAPS = {"input.pkl": 64 * 1024 * 1024, "job.sbatch": 1024 * 1024}
REPLY_PATH = ("reply", "output.json")
LOG_PATH = ("log", "slurm.err")
MAX_REPLY_CAP = 64 * 1024 * 1024
MAX_TAIL_BYTES = 1024 * 1024
MAX_NAME_BYTES = 255
MAX_PATH_BYTES = 4096
CHUNK = 1024 * 1024
SLURM_TIMEOUT = 120
#: Inside the server's 300 s transfer timeout for ``rmwd``.
REMOVE_TIMEOUT = 280
#: ``sweep`` never removes a workdir active this recently (staging still
#: streaming in, a collect still streaming out).
MIN_SWEEP_AGE = 3600
#: What moves while a workdir is in use: mkwd/put (the workdir) and the job's
#: log. Never ``out/`` or ``reply/``: the step owns their mount roots and can
#: date them into the future, which kept its workdir "active" for ever; while
#: it writes there its job is live in squeue anyway.
_ACTIVITY_PATHS = ((), ("log",), ("log", "slurm.err"))

_TEMP_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789_"
_TEMP_LEN = 8
#: ``<run>-<step>-<attempt>-<inputs tag>-``: two 40-character segments, the
#: attempt and a 12-digit tag fit.
PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
WORKDIR_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}[a-z0-9_]{8}")
JOB_ID_RE = re.compile(r"[0-9]{1,20}")
JOB_NAME_RE = re.compile(r"umrs-step-[A-Za-z0-9_-]{1,100}")
STATE_RE = re.compile(r"[A-Z_]{1,32}")
COMMENT_RE = re.compile(r"[A-Za-z0-9:._-]{1,200}")
NO_COMMENT = "-"
ODD_COMMENT = "?"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


class HelperError(Exception):
    """Ends the verb with ``code`` and a JSON reason on stderr."""

    code = EXIT_FAILED
    kind = "failed"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class Refused(HelperError):
    code = EXIT_REFUSED
    kind = "refused"


class Absent(HelperError):
    code = EXIT_ABSENT
    kind = "absent"


# ── paths ───────────────────────────────────────────────────────────────────


def check_root(root: str) -> str:
    """*root* as an absolute, normalised path (no ``..``, no NUL)."""
    if not root.startswith("/") or "\x00" in root:
        raise Refused(f"root {root!r} is not an absolute path")
    trimmed = root.rstrip("/") or "/"
    if os.path.normpath(trimmed) != trimmed or trimmed == "/":
        raise Refused(f"root {root!r} is not a normalised directory path")
    return trimmed


def open_root(root: str) -> int:
    """Descriptor of *root*: a real directory (the last component is never
    followed), owned by this account and not writable by anyone else."""
    fd = _open_dir(None, root)
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o022:
        os.close(fd)
        raise Refused(f"root {root} is not a private directory of this account")
    return fd


def workdir_name(root: str, workdir: str) -> str:
    """The workdir's name: a direct child of *root* made by ``mkwd``."""
    parent, name = os.path.split(workdir.rstrip("/"))
    if parent != root or not WORKDIR_NAME_RE.fullmatch(name):
        raise Refused(f"{workdir!r} is not a step workdir under {root}")
    return name


@contextlib.contextmanager
def step_owned(what: str):
    """``out/`` and ``reply/`` belong to the step (its uid owns their mount
    roots and everything in them): a permission it withheld there is its
    doing, a refusal the server removes the workdir for, never a helper
    failure the server keeps the workdir (and retries) for."""
    try:
        yield
    except PermissionError as exc:
        raise Refused(f"{what} is not readable: the step hid it ({exc.strerror})") from exc


def _open_dir(parent_fd: int | None, name: str) -> int:
    try:
        if parent_fd is None:
            return os.open(name, _DIR_FLAGS)
        return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise Absent(f"{name} does not exist") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise Refused(f"{name} is not a directory (links are refused)") from exc
        raise


def open_workdir(root: str, workdir: str) -> int:
    root = check_root(root)
    name = workdir_name(root, workdir)
    root_fd = open_root(root)
    try:
        return _open_dir(root_fd, name)
    finally:
        os.close(root_fd)


def open_path(base_fd: int, parts: list[str] | tuple[str, ...]) -> int:
    """Descriptor of the directory *parts* below *base_fd*, no link followed."""
    fd = os.dup(base_fd)
    for part in parts:
        try:
            nxt = _open_dir(fd, part)
        finally:
            os.close(fd)
        fd = nxt
    return fd


def open_plain(dir_fd: int, name: str) -> tuple[int, os.stat_result]:
    """A regular, singly-linked file opened once and checked on that
    descriptor: a symlink, hard link, FIFO, socket or device is refused.
    A step can bind a socket where a file is read (``--network=none``
    leaves AF_UNIX usable), and ``open`` fails on it with ENXIO: that is
    the step's shape, a refusal, never a helper failure the server keeps
    the workdir (and retries) for."""
    try:
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            kind = "a symlink" if stat.S_ISLNK(mode) else "not a regular file"
            raise Refused(f"{name} is {kind}")
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except FileNotFoundError as exc:
        raise Absent(f"{name} does not exist") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENXIO):
            raise Refused(f"{name} is a symlink or a socket") from exc
        raise
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise Refused(f"{name} is not a regular, singly-linked file")
    return fd, info


def read_capped(fd: int, cap: int) -> bytes:
    """At most *cap* bytes from *fd*; more than that is refused."""
    chunks, total = [], 0
    while total <= cap:
        chunk = os.read(fd, min(CHUNK, cap + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    if total > cap:
        raise Refused(f"file exceeds {cap} bytes")
    return b"".join(chunks)


def make_unique_dir(root_fd: int, prefix: str) -> str:
    for _ in range(100):
        suffix = "".join(secrets.choice(_TEMP_CHARS) for _ in range(_TEMP_LEN))
        name = prefix + suffix
        try:
            os.mkdir(name, 0o700, dir_fd=root_fd)
        except FileExistsError:
            continue
        return name
    raise HelperError(f"no free workdir name for prefix {prefix!r}")


def free_bytes(fd: int) -> int:
    info = os.fstatvfs(fd)
    return info.f_bavail * info.f_frsize


# ── mkwd / put / read-reply / tail-log / rmwd ────────────────────────────────


def mkwd(root: str, prefix: str) -> str:
    """Create ``<root>/<prefix>XXXXXXXX`` with its five subdirectories."""
    root = check_root(root)
    if not PREFIX_RE.fullmatch(prefix):
        raise Refused(f"workdir prefix {prefix!r} is not [A-Za-z0-9_-]")
    with contextlib.suppress(FileExistsError):
        os.mkdir(root, 0o700)
    root_fd = open_root(root)
    try:
        name = make_unique_dir(root_fd, prefix)
        wd_fd = _open_dir(root_fd, name)
    finally:
        os.close(root_fd)
    try:
        for sub in WORKDIR_DIRS:
            os.mkdir(sub, 0o700, dir_fd=wd_fd)
    finally:
        os.close(wd_fd)
    return os.path.join(root, name)


def put(root: str, workdir: str, name: str, source) -> int:
    """Write *source* to ``<wd>/<name>`` (a new file; nothing is replaced)."""
    cap = PUT_CAPS.get(name)
    if cap is None:
        raise Refused(f"{name!r} is not a file the server writes")
    wd_fd = open_workdir(root, workdir)
    try:
        try:
            fd = os.open(name, _CREATE_FLAGS, 0o600, dir_fd=wd_fd)
        except FileExistsError as exc:
            raise Refused(f"{name} exists already") from exc
        try:
            written = _copy_capped(source, fd, cap)
        except HelperError:
            os.unlink(name, dir_fd=wd_fd)
            raise
        finally:
            os.close(fd)
    finally:
        os.close(wd_fd)
    return written


def _copy_capped(source, fd: int, cap: int) -> int:
    total = 0
    while True:
        chunk = source.read(CHUNK)
        if not chunk:
            return total
        total += len(chunk)
        if total > cap:
            raise Refused(f"input exceeds {cap} bytes")
        _write_all(fd, chunk)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def read_reply(root: str, workdir: str, cap: int) -> bytes:
    if not 0 < cap <= MAX_REPLY_CAP:
        raise Refused(f"reply cap {cap} is outside 1..{MAX_REPLY_CAP}")
    wd_fd = open_workdir(root, workdir)
    try:
        with step_owned("reply/"):
            dir_fd = open_path(wd_fd, REPLY_PATH[:-1])
    finally:
        os.close(wd_fd)
    try:
        with step_owned("reply/output.json"):
            fd, _info = open_plain(dir_fd, REPLY_PATH[-1])
    finally:
        os.close(dir_fd)
    try:
        return read_capped(fd, cap)
    finally:
        os.close(fd)


def tail_log(root: str, workdir: str, nbytes: int) -> bytes:
    if not 0 < nbytes <= MAX_TAIL_BYTES:
        raise Refused(f"tail size {nbytes} is outside 1..{MAX_TAIL_BYTES}")
    wd_fd = open_workdir(root, workdir)
    try:
        dir_fd = open_path(wd_fd, LOG_PATH[:-1])
    finally:
        os.close(wd_fd)
    try:
        fd, info = open_plain(dir_fd, LOG_PATH[-1])
    finally:
        os.close(dir_fd)
    try:
        os.lseek(fd, max(0, info.st_size - nbytes), os.SEEK_SET)
        return read_capped(fd, nbytes)
    finally:
        os.close(fd)


def rmwd(root: str, workdir: str) -> bool:
    """Remove a workdir; False when it was already gone."""
    root = check_root(root)
    name = workdir_name(root, workdir)
    root_fd = open_root(root)
    try:
        try:
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(info.st_mode):
            raise Refused(f"{workdir} is not a directory")
        path = os.path.join(root, name)
        try:
            shutil.rmtree(path)
        except Exception:  # noqa: BLE001 — RecursionError too (python3.9 recurses per level)
            # The step shaped this tree: directories it made unreadable, a
            # tree deeper than Python recurses, more subdirectories than this
            # process may hold open. GNU fts has none of these limits and
            # never follows a link (chmod -R skips links it meets).
            _force_remove(path)
    finally:
        os.close(root_fd)
    return True


def _force_remove(path: str) -> None:
    for argv in (
        ["chmod", "-R", "u+rwX", "--", path],
        ["rm", "-rf", "--one-file-system", "--", path],
    ):
        proc = _slurm(argv, timeout=REMOVE_TIMEOUT)
    if os.path.lexists(path):
        raise HelperError(f"{path} could not be removed: {proc.stderr.strip()[-300:]}")


def _dir_entries(fd: int) -> list[str]:
    with os.scandir(fd) as it:
        return sorted(entry.name for entry in it)


# ── stage-in ─────────────────────────────────────────────────────────────────


def _clean_parts(name: str) -> list[str]:
    """Relative path segments of a member or file name; ``..``, absolute
    names, control characters and over-long names are refused."""
    if name.startswith("/") or _CONTROL_RE.search(name):
        raise Refused(f"name {name!r} is absolute or has control characters")
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise Refused(f"name {name!r} leaves its directory")
    sizes = [len(part.encode("utf-8", "surrogateescape")) for part in parts]
    if max(sizes, default=0) > MAX_NAME_BYTES or sum(sizes) + len(sizes) > MAX_PATH_BYTES:
        raise Refused(f"name {name[:80]!r} is too long")
    return parts


def stage_member_parts(name: str) -> tuple[str, list[str]]:
    parts = _clean_parts(name)
    if not parts or parts[0] not in STAGE_TOPS:
        raise Refused(f"member {name!r} is outside {' and '.join(STAGE_TOPS)}")
    return parts[0], parts[1:]


def _is_plain_member(member: tarfile.TarInfo) -> bool:
    return member.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and member.sparse is None


def make_dirs(base_fd: int, parts: list[str]) -> int:
    """Descriptor of *parts* below *base_fd*, creating directories 0700."""
    fd = os.dup(base_fd)
    for part in parts:
        try:
            with contextlib.suppress(FileExistsError):
                os.mkdir(part, 0o700, dir_fd=fd)
            nxt = _open_dir(fd, part)
        finally:
            os.close(fd)
        fd = nxt
    return fd


def _extract_file(dir_fd: int, name: str, source, size: int) -> None:
    try:
        fd = os.open(name, _CREATE_FLAGS, 0o600, dir_fd=dir_fd)
    except FileExistsError as exc:
        raise Refused(f"member {name!r} appears twice or collides") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise Refused(f"member {name!r} collides with a link") from exc
        raise
    try:
        written = _copy_capped(source, fd, size)
    finally:
        os.close(fd)
    if written != size:
        raise Refused(f"member {name!r} is shorter than its header says")


class _StageCounter:
    def __init__(self, max_files: int, max_bytes: int) -> None:
        self.max_files, self.max_bytes = max_files, max_bytes
        self.files = self.bytes = 0

    def add(self, member: tarfile.TarInfo) -> None:
        self.files += 1
        self.bytes += member.size
        if self.files > self.max_files:
            raise Refused(f"stage-in carries more than {self.max_files} files")
        if self.bytes > self.max_bytes:
            raise Refused(f"stage-in carries more than {self.max_bytes} bytes")


def _stage_member(tar, member, tops: dict[str, int], counter: _StageCounter) -> None:
    top, parts = stage_member_parts(member.name)
    if member.isdir():
        os.close(make_dirs(tops[top], parts))
        return
    if not _is_plain_member(member) or not parts:
        raise Refused(f"member {member.name!r} is not a regular file")
    counter.add(member)
    dir_fd = make_dirs(tops[top], parts[:-1])
    try:
        _extract_file(dir_fd, parts[-1], tar.extractfile(member), member.size)
    finally:
        os.close(dir_fd)


def stage_in(root: str, workdir: str, source, max_files: int, max_bytes: int) -> dict:
    """Extract a tar stream into ``<wd>/in`` and ``<wd>/src``."""
    wd_fd = open_workdir(root, workdir)
    tops: dict[str, int] = {}
    try:
        for top in STAGE_TOPS:
            tops[top] = _open_dir(wd_fd, top)
        counter = _StageCounter(max_files, max_bytes)
        try:
            with tarfile.open(fileobj=source, mode="r|") as tar:
                for member in tar:
                    _stage_member(tar, member, tops, counter)
        except (tarfile.TarError, EOFError) as exc:
            raise Refused(f"stage-in stream is not a valid tar: {exc}") from exc
        return {"files": counter.files, "bytes": counter.bytes, "free_bytes": free_bytes(wd_fd)}
    finally:
        for fd in tops.values():
            os.close(fd)
        os.close(wd_fd)


# ── collect-out ──────────────────────────────────────────────────────────────


class _OutScan:
    """What ``out/`` holds, checked before one byte of the stream is sent."""

    def __init__(self, max_files: int, max_bytes: int, max_file_bytes: int) -> None:
        self.caps = (max_files, max_bytes, max_file_bytes)
        self.files: list[tuple[str, int]] = []
        self.dirs = 0
        self.bytes = 0

    def add_dir(self) -> None:
        self.dirs += 1
        if self.dirs > self.caps[0]:
            raise Refused(f"out/ holds more than {self.caps[0]} directories")

    def add_file(self, rel: str, size: int) -> None:
        max_files, max_bytes, max_file_bytes = self.caps
        self.bytes += size
        self.files.append((rel, size))
        if len(self.files) > max_files:
            raise Refused(f"out/ holds more than {max_files} files")
        if size > max_file_bytes:
            raise Refused(f"out/{rel} is {size} bytes (cap {max_file_bytes})")
        if self.bytes > max_bytes:
            raise Refused(f"out/ holds more than {max_bytes} bytes")


def _entry_rel(parts: list[str], name: str) -> str:
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise Refused(f"out/ entry {name!r} is not a UTF-8 name") from exc
    rel = "/".join([*parts, name])
    _clean_parts(rel)
    return rel


def _check_readable(rel: str, info: os.stat_result, need: int) -> None:
    """Refuse before one byte is streamed what the step made unreadable to
    its own uid (*need*: the owner bits the collect needs); under keep-id
    everything the step writes is this account's."""
    if info.st_uid != os.geteuid():
        raise Refused(f"out/{rel} is not this account's (uid {info.st_uid})")
    if info.st_mode & need != need:
        raise Refused(
            f"out/{rel} is not readable: the step hid it (mode {info.st_mode & 0o7777:o})"
        )


def _scan_entry(fd: int, parts: list[str], name: str, scan: _OutScan, stack: list) -> None:
    rel = _entry_rel(parts, name)
    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    if stat.S_ISDIR(info.st_mode):
        _check_readable(rel, info, stat.S_IRUSR | stat.S_IXUSR)
        scan.add_dir()
        stack.append([*parts, name])
    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
        _check_readable(rel, info, stat.S_IRUSR)
        scan.add_file(rel, info.st_size)
    else:
        raise Refused(f"out/{rel} is not a regular, singly-linked file")


def scan_out(out_fd: int, max_files: int, max_bytes: int, max_file_bytes: int) -> _OutScan:
    scan = _OutScan(max_files, max_bytes, max_file_bytes)
    stack: list[list[str]] = [[]]
    while stack:
        parts = stack.pop()
        fd = open_path(out_fd, parts)
        try:
            for name in _dir_entries(fd):
                _scan_entry(fd, parts, name, scan, stack)
        finally:
            os.close(fd)
    scan.files.sort()
    return scan


def _add_out_file(tar: tarfile.TarFile, out_fd: int, rel: str, size: int) -> None:
    parts = rel.split("/")
    dir_fd = open_path(out_fd, parts[:-1])
    try:
        fd, info = open_plain(dir_fd, parts[-1])
    finally:
        os.close(dir_fd)
    with os.fdopen(fd, "rb") as handle:
        if info.st_size != size:
            raise Refused(f"out/{rel} changed while it was collected")
        member = tarfile.TarInfo(rel)
        member.size, member.mode, member.mtime = size, 0o600, 0
        tar.addfile(member, handle)


def collect_out(root, workdir, sink, max_files, max_bytes, max_file_bytes) -> dict:
    """Stream ``<wd>/out`` as a tar of its regular files."""
    wd_fd = open_workdir(root, workdir)
    try:
        with step_owned("out/"):
            out_fd = _open_dir(wd_fd, "out")
    finally:
        os.close(wd_fd)
    try:
        with step_owned("out/"):
            scan = scan_out(out_fd, max_files, max_bytes, max_file_bytes)
            with tarfile.open(fileobj=sink, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for rel, size in scan.files:
                    _add_out_file(tar, out_fd, rel, size)
    finally:
        os.close(out_fd)
    return {"files": len(scan.files), "bytes": scan.bytes}


# ── Slurm ────────────────────────────────────────────────────────────────────


def _slurm(
    argv: list[str], cwd: str | None = None, timeout: int = SLURM_TIMEOUT
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def _check_job_id(job_id: str) -> str:
    if not JOB_ID_RE.fullmatch(job_id):
        raise Refused(f"{job_id!r} is not a Slurm job id")
    return job_id


def _first_state(text: str) -> str:
    lines = (text or "").strip().splitlines()
    token = lines[0].split()[0].upper() if lines and lines[0].split() else ""
    return token if STATE_RE.fullmatch(token) else ""


def submit(root: str, workdir: str) -> str:
    """``sbatch --parsable --chdir=<workdir>``. Slurm keeps an absolute
    ``--chdir`` as given, so the WorkDir ``find-jobs`` reports is the path the
    server named even when the root sits behind a symlink (without it Slurm
    records ``getcwd()``, the physical path)."""
    wd_fd = open_workdir(root, workdir)
    try:
        fd, _info = open_plain(wd_fd, "job.sbatch")
        os.close(fd)
    finally:
        os.close(wd_fd)
    root = check_root(root)
    path = os.path.join(root, workdir_name(root, workdir))
    proc = _slurm(
        ["sbatch", "--parsable", f"--chdir={path}", os.path.join(path, "job.sbatch")],
        cwd=path,
    )
    if proc.returncode != 0:
        raise HelperError(f"sbatch failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")
    first = (proc.stdout or "").strip().splitlines()
    job_id = first[0].split(";")[0].strip() if first else ""
    if not JOB_ID_RE.fullmatch(job_id):
        raise HelperError(f"sbatch printed no job id: {(proc.stdout or '').strip()[-200:]}")
    return job_id


def status(job_id: str) -> str:
    """squeue's state while the job is queued or running, else sacct's."""
    _check_job_id(job_id)
    queued = _slurm(["squeue", "-h", "-j", job_id, "-o", "%T"])
    state = _first_state(queued.stdout) if queued.returncode == 0 else ""
    if state:
        return state
    final = _slurm(["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "State"])
    if final.returncode != 0:
        raise HelperError(f"sacct failed (rc={final.returncode}): {final.stderr.strip()[-300:]}")
    return _first_state(final.stdout) or "UNKNOWN"


def cancel(job_id: str) -> None:
    proc = _slurm(["scancel", _check_job_id(job_id)])
    if proc.returncode != 0:
        raise HelperError(f"scancel failed (rc={proc.returncode}): {proc.stderr.strip()[-300:]}")


def _comment(raw: str) -> str:
    raw = raw.strip()
    if not raw or raw == "(null)":
        return NO_COMMENT
    return raw if COMMENT_RE.fullmatch(raw) else ODD_COMMENT


def _job_workdir(workdir: str, root: str | None) -> str:
    """*workdir* as ``<root>/<name>`` when its parent resolves to *root*
    (Slurm may record the physical path of a symlinked root)."""
    workdir = workdir.strip()
    if root is None or not workdir.startswith("/"):
        return workdir
    parent, name = os.path.split(workdir.rstrip("/"))
    if name and os.path.realpath(parent) == os.path.realpath(root):
        return os.path.join(root, name)
    return workdir


def _job_rows(text: str, name: str, root: str | None) -> dict[str, tuple[str, str, str]]:
    rows: dict[str, tuple[str, str, str]] = {}
    for line in (text or "").splitlines():
        fields = line.strip().split("|")
        if len(fields) != 5 or not JOB_ID_RE.fullmatch(fields[0]) or fields[2] != name:
            continue
        state = _first_state(fields[1]) or "UNKNOWN"
        rows[fields[0]] = (state, _comment(fields[4]), _job_workdir(fields[3], root))
    return rows


def find_jobs(name: str, root: str | None = None) -> list[str]:
    """``job_id state name comment workdir`` of this account's jobs named
    *name* (``-`` for no comment: sacct keeps none unless Slurm stores job
    comments). With *root*, a workdir under it is reported as
    ``<root>/<name>``."""
    if not JOB_NAME_RE.fullmatch(name):
        raise Refused(f"{name!r} is not a step job name")
    root = check_root(root) if root is not None else None
    queued = _slurm(["squeue", "--me", "-h", f"--name={name}", "-o", "%i|%T|%j|%Z|%k"])
    if queued.returncode != 0:
        raise HelperError(f"squeue failed (rc={queued.returncode}): {queued.stderr.strip()[-300:]}")
    finished = _slurm(
        [
            "sacct",
            "-n",
            "-X",
            "-P",
            f"--name={name}",
            "-S",
            "now-7days",
            "-o",
            "JobID,State,JobName,WorkDir,Comment",
        ]
    )
    # A partial list would hide a COMPLETED job whose kept workdir holds the
    # only copy of its outputs: the server would then run the step again.
    if finished.returncode != 0:
        raise HelperError(
            f"sacct failed (rc={finished.returncode}): {finished.stderr.strip()[-300:]}"
        )
    rows = _job_rows(finished.stdout, name, root)
    rows.update(_job_rows(queued.stdout, name, root))
    return [
        f"{job} {state} {name} {comment} {wd}" for job, (state, comment, wd) in sorted(rows.items())
    ]


# ── sweep ────────────────────────────────────────────────────────────────────


def _live_workdirs(root: str) -> set[str]:
    """Workdirs of this account's queued or running jobs (any name)."""
    proc = _slurm(["squeue", "--me", "-h", "-o", "%i|%T|%j|%Z|%k"])
    if proc.returncode != 0:
        raise HelperError(f"squeue failed (rc={proc.returncode}): {proc.stderr.strip()[-300:]}")
    live = set()
    for line in proc.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) == 5 and JOB_ID_RE.fullmatch(fields[0]):
            live.add(_job_workdir(fields[3], root))
    return live


def _last_active(root_fd: int, name: str) -> float:
    newest = 0.0
    for parts in _ACTIVITY_PATHS:
        try:
            info = os.stat(os.path.join(name, *parts), dir_fd=root_fd, follow_symlinks=False)
        except OSError:
            continue
        newest = max(newest, info.st_mtime)
    return newest


def _is_stale(root_fd: int, root: str, name: str, live: set[str], cutoff: float) -> bool:
    if not WORKDIR_NAME_RE.fullmatch(name) or os.path.join(root, name) in live:
        return False
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and _last_active(root_fd, name) < cutoff


def sweep(root: str, older_than: int) -> dict:
    """Remove the workdirs no queued or running job uses that have been idle
    for *older_than* seconds: a server that died between ``mkwd`` and
    ``submit``, or before it removed a finished job's workdir, leaves them
    on a filesystem without quota."""
    if older_than < MIN_SWEEP_AGE:
        raise Refused(f"sweep age {older_than}s is under {MIN_SWEEP_AGE}s")
    root = check_root(root)
    root_fd = open_root(root)
    try:
        live = _live_workdirs(root)
        cutoff = time.time() - older_than
        stale = [n for n in _dir_entries(root_fd) if _is_stale(root_fd, root, n, live, cutoff)]
    finally:
        os.close(root_fd)
    removed, failed = [], []
    for name in stale:
        try:
            rmwd(root, os.path.join(root, name))
            removed.append(name)
        except (HelperError, OSError):
            failed.append(name)
    return {"removed": removed, "failed": failed, "live": len(live)}


# ── command line ─────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umrs_step_helper.py")
    verbs = parser.add_subparsers(dest="verb", required=True)
    for verb, args in (
        ("mkwd", ("root", "prefix")),
        ("stage-in", ("root", "workdir")),
        ("put", ("root", "workdir", "name")),
        ("read-reply", ("root", "workdir", "cap")),
        ("collect-out", ("root", "workdir")),
        ("tail-log", ("root", "workdir", "nbytes")),
        ("rmwd", ("root", "workdir")),
        ("submit", ("root", "workdir")),
        ("status", ("job_id",)),
        ("cancel", ("job_id",)),
        ("find-jobs", ("name",)),
        ("sweep", ("root",)),
    ):
        sub = verbs.add_parser(verb)
        for arg in args:
            sub.add_argument(arg, type=int if arg in ("cap", "nbytes") else str)
        if verb in ("stage-in", "collect-out"):
            sub.add_argument("--max-files", type=int, required=True)
            sub.add_argument("--max-bytes", type=int, required=True)
        if verb == "collect-out":
            sub.add_argument("--max-file-bytes", type=int, required=True)
        if verb == "find-jobs":
            sub.add_argument("--root", default=None)
        if verb == "sweep":
            sub.add_argument("--older-than", type=int, required=True)
    return parser


def _emit_bytes(data: bytes) -> None:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _emit_line(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _run_io_verb(ns: argparse.Namespace) -> int:
    if ns.verb == "stage-in":
        summary = stage_in(ns.root, ns.workdir, sys.stdin.buffer, ns.max_files, ns.max_bytes)
        _emit_line(json.dumps(summary, sort_keys=True))
    elif ns.verb == "collect-out":
        summary = collect_out(
            ns.root,
            ns.workdir,
            sys.stdout.buffer,
            ns.max_files,
            ns.max_bytes,
            ns.max_file_bytes,
        )
        sys.stdout.buffer.flush()
        sys.stderr.write(json.dumps(summary, sort_keys=True) + "\n")
    elif ns.verb == "put":
        put(ns.root, ns.workdir, ns.name, sys.stdin.buffer)
    elif ns.verb == "read-reply":
        _emit_bytes(read_reply(ns.root, ns.workdir, ns.cap))
    else:
        _emit_bytes(tail_log(ns.root, ns.workdir, ns.nbytes))
    return 0


def _run_verb(ns: argparse.Namespace) -> int:
    if ns.verb in ("stage-in", "collect-out", "put", "read-reply", "tail-log"):
        return _run_io_verb(ns)
    if ns.verb == "find-jobs":
        for row in find_jobs(ns.name, ns.root):
            _emit_line(row)
        return 0
    verbs = {
        "mkwd": lambda: _emit_line(mkwd(ns.root, ns.prefix)),
        "rmwd": lambda: rmwd(ns.root, ns.workdir),
        "submit": lambda: _emit_line(submit(ns.root, ns.workdir)),
        "status": lambda: _emit_line(status(ns.job_id)),
        "cancel": lambda: cancel(ns.job_id),
        "sweep": lambda: _emit_line(json.dumps(sweep(ns.root, ns.older_than), sort_keys=True)),
    }
    verbs[ns.verb]()
    return 0


def _fail(code: int, kind: str, detail: str) -> int:
    sys.stderr.write(json.dumps({"error": kind, "detail": detail}) + "\n")
    return code


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    ns = _parser().parse_args(argv)
    try:
        return _run_verb(ns)
    except HelperError as exc:
        return _fail(exc.code, exc.kind, exc.detail)
    except OSError as exc:
        return _fail(EXIT_FAILED, "failed", str(exc))


if __name__ == "__main__":
    sys.exit(main())
