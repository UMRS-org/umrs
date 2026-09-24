"""D-40 transport: .kb.tar.gz pack / hostile-input-safe extract."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from umrs.knowledge import bundle_archive as ba


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "src" / "proj-a"
    (root / "tables").mkdir(parents=True)
    (root / "manifest.json").write_text('{"project_id": "proj-a"}')
    (root / "tables" / "nodes.jsonl").write_text('{"id": 1}\n')
    return root


def _tar_bytes(add) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        add(tar)
    buf.seek(0)
    return buf


def _file(tar, name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_pack_then_extract_round_trips(tmp_path):
    archive = ba.pack_bundle(_bundle(tmp_path), tmp_path / "b.kb.tar.gz")
    dest = tmp_path / "out"
    dest.mkdir()
    with open(archive, "rb") as handle:
        root = ba.safe_extract(handle, dest)
    assert root == dest / "bundle"
    assert (root / "tables" / "nodes.jsonl").read_text() == '{"id": 1}\n'


def test_top_level_manifest_is_the_root(tmp_path):
    buf = _tar_bytes(lambda t: (_file(t, "manifest.json", b"{}"), _file(t, "tables/a", b"1")))
    assert ba.safe_extract(buf, tmp_path) == tmp_path


@pytest.mark.parametrize(
    "name", ["/etc/cron.d/evil", "../escape", "bundle/../../escape", "bundle/../../../x"]
)
def test_unsafe_paths_refused_before_any_write(tmp_path, name):
    buf = _tar_bytes(lambda t: (_file(t, "manifest.json", b"{}"), _file(t, name)))
    with pytest.raises(ba.BundleArchiveError, match="unsafe member path"):
        ba.safe_extract(buf, tmp_path / "d")
    assert not (tmp_path / "d").exists()  # validation precedes every write


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_links_and_devices_refused(tmp_path, kind):
    def add(tar):
        _file(tar, "manifest.json", b"{}")
        info = tarfile.TarInfo("bundle/link")
        info.type = kind
        info.linkname = "/etc/passwd"
        tar.addfile(info)

    with pytest.raises(ba.BundleArchiveError, match="unsupported member type"):
        ba.safe_extract(_tar_bytes(add), tmp_path)


def test_member_and_size_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "MAX_MEMBERS", 2)
    buf = _tar_bytes(lambda t: [_file(t, f"f{i}") for i in range(3)])
    with pytest.raises(ba.BundleArchiveError, match="members"):
        ba.safe_extract(buf, tmp_path)
    monkeypatch.setattr(ba, "MAX_MEMBERS", 100)
    monkeypatch.setattr(ba, "MAX_UNPACKED_BYTES", 4)
    buf = _tar_bytes(lambda t: _file(t, "manifest.json", b"0123456789"))
    with pytest.raises(ba.BundleArchiveError, match="size cap"):
        ba.safe_extract(buf, tmp_path)


def test_no_manifest_and_not_gzip(tmp_path):
    with pytest.raises(ba.BundleArchiveError, match="manifest.json not found"):
        ba.safe_extract(_tar_bytes(lambda t: _file(t, "readme.txt")), tmp_path / "a")
    with pytest.raises(ba.BundleArchiveError, match="gzip'd tar"):
        ba.safe_extract(io.BytesIO(b"not an archive"), tmp_path)


def test_pack_refuses_a_non_bundle(tmp_path):
    with pytest.raises(ba.BundleArchiveError, match="no manifest.json"):
        ba.pack_bundle(tmp_path, tmp_path / "x.kb.tar.gz")


def test_export_key_is_project_scoped():
    assert ba.export_archive_key("proj-a", "job-1") == "proj-a/knowledge-exports/job-1.kb.tar.gz"
