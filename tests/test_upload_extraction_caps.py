"""An uploaded workflow's zip is sized before anything is extracted (review round 3).

Normal users upload by design (ADR-074 Amendment 1). A small zip of
compressible members expands to whatever it declares in the server's
workflow state dir, which every server on the host shares: the members'
declared sizes and count must fit the source caps the step lane stages
(``StageCaps.max_src_bytes`` / ``max_src_files``), and a link member is
refused, before one file is written.
"""

from __future__ import annotations

import dataclasses
import json
import stat
import zipfile
from pathlib import Path

import pytest
from umrs.server import step_staging
from umrs.server.file_handling import handle_upload_workflow
from umrs.server.step_executor import StepExecutionError
from umrs.server.step_staging import DEFAULT_CAPS

_WORKFLOW = {
    "name": "wf",
    "version": "1.0",
    "description": "One-step upload fixture.",
    "source": {"type": "local_path", "path": "."},
    "steps": [{"name": "s1", "entrypoint": "a:B", "depends_on": []}],
}


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


def _upload(tmp_path: Path, add) -> tuple[bool, str | None, str | None]:
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", json.dumps(_WORKFLOW))
        add(zf)
    return handle_upload_workflow("alice", str(path), "wf")


def _nothing_extracted(state: Path) -> bool:
    return not (state / "extracted" / "alice" / "wf").exists()


def test_a_zip_that_expands_past_the_source_cap_is_refused_unextracted(
    tmp_path, state, monkeypatch
):
    small = dataclasses.replace(DEFAULT_CAPS, max_src_bytes=64 * 1024)
    monkeypatch.setattr(step_staging, "DEFAULT_CAPS", small)
    ok, dest, message = _upload(tmp_path, lambda zf: zf.writestr("bomb.bin", b"\0" * 100_000))
    assert not ok and dest is None
    assert "expands to" in message and str(64 * 1024) in message
    assert _nothing_extracted(state)


def test_a_zip_with_more_files_than_the_source_cap_is_refused(tmp_path, state, monkeypatch):
    monkeypatch.setattr(step_staging, "DEFAULT_CAPS",
                        dataclasses.replace(DEFAULT_CAPS, max_src_files=5))  # fmt: skip

    def many(zf):
        for i in range(6):
            zf.writestr(f"f{i}.py", "")

    ok, _dest, message = _upload(tmp_path, many)
    assert not ok and "max 5 files" in message
    assert _nothing_extracted(state)


def test_a_link_member_is_refused(tmp_path, state):
    def link(zf):
        info = zipfile.ZipInfo("key")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/home/user/.ssh/id_rsa")

    ok, _dest, message = _upload(tmp_path, link)
    assert not ok and "'key' is not a regular file or directory" in message
    assert _nothing_extracted(state)


def test_a_zip_within_the_caps_extracts(tmp_path, state):
    def tree(zf):
        zf.writestr("pkg/", "")
        zf.writestr("pkg/a.py", "X = 1\n")

    ok, dest, _message = _upload(tmp_path, tree)
    assert ok and (Path(dest) / "pkg" / "a.py").read_text() == "X = 1\n"


def test_the_stager_stops_walking_a_source_tree_at_its_cap(tmp_path, monkeypatch):
    for i in range(50):
        (tmp_path / f"f{i:02d}.py").write_text("")
    seen = []
    real = step_staging.os.lstat
    monkeypatch.setattr(step_staging.os, "lstat", lambda p: seen.append(p) or real(p))
    caps = dataclasses.replace(DEFAULT_CAPS, max_src_files=3)
    with pytest.raises(StepExecutionError, match="passes its caps"):
        step_staging._source_files(str(tmp_path), caps)
    assert len(seen) == 4


def test_a_central_directory_past_the_entry_cap_is_refused_before_zipfile_parses_it(
    tmp_path, state, monkeypatch
):
    # ZipFile builds one object per central directory entry, walking to the
    # directory's declared size whatever the end record's count says: a
    # 500 MB zip of minimal entries was ~6M objects before any cap (round 4).
    monkeypatch.setattr(step_staging, "DEFAULT_CAPS",
                        dataclasses.replace(DEFAULT_CAPS, max_src_files=5))  # fmt: skip
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("workflow.json", json.dumps(_WORKFLOW))
        for i in range(11):
            zf.writestr(f"d{i}/", "")
    raw = bytearray(path.read_bytes())
    raw[-22 + 8 : -22 + 12] = b"\x01\x00\x01\x00"  # the end record claims one entry
    path.write_bytes(bytes(raw))

    built = []

    class _Counted(zipfile.ZipInfo):
        def __init__(self, *args, **kwargs):
            built.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(zipfile, "ZipInfo", _Counted)
    ok, _dest, message = handle_upload_workflow("alice", str(path), "wf")
    assert not ok and "more than 10 entries" in message
    assert built == []  # refused before ZipFile built one entry object
    assert _nothing_extracted(state)


@pytest.mark.parametrize("entries", [5, 11])
def test_the_entry_count_reads_a_zip64_end_record(tmp_path, state, monkeypatch, entries):
    monkeypatch.setattr(step_staging, "DEFAULT_CAPS",
                        dataclasses.replace(DEFAULT_CAPS, max_src_files=5))  # fmt: skip
    path = tmp_path / "upload.zip"
    with monkeypatch.context() as m:
        m.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 0)  # every archive ends in zip64
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("workflow.json", json.dumps(_WORKFLOW))
            for i in range(entries):
                zf.writestr(f"d{i}/", "")
    assert zipfile.stringEndArchive64 in path.read_bytes()[-200:]
    ok, _dest, message = handle_upload_workflow("alice", str(path), "wf")
    if entries < 10:
        assert ok, message
    else:  # counted before ZipFile, not by the infolist check after it
        assert not ok and "more than 10 entries" in message


def test_an_oversized_workflow_json_is_refused_before_it_is_parsed(tmp_path, state):
    # json.loads of a compressible workflow.json up to max_src_bytes was
    # ~20 GiB of Python objects on a 15 GiB front (review round 4).
    from umrs.server.file_handling import MAX_WORKFLOW_JSON_BYTES

    path = tmp_path / "upload.zip"
    padded = json.dumps(_WORKFLOW) + " " * MAX_WORKFLOW_JSON_BYTES
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", padded)
    ok, _dest, message = handle_upload_workflow("alice", str(path), "wf")
    assert not ok and f"(max {MAX_WORKFLOW_JSON_BYTES})" in message
    assert _nothing_extracted(state)
