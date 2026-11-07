"""``umrs workflow export`` client half (2026-11-07): fetch through the
gateway, verify against EXPORT.json, then rename into place."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import zipfile
from types import SimpleNamespace

import pytest
from umrs.client import workflow_export as cx


def _archive(files: dict[str, bytes], *, tamper: str | None = None) -> bytes:
    rows = [
        {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in files.items()
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, b"tampered" if name == tamper else data)
        zf.writestr("EXPORT.json", json.dumps({"files": rows, "excluded": [".env"]}))
    return buffer.getvalue()


GOOD = {"workflow.json": b'{"name": "wf"}', "steps/a.py": b"print(1)\n"}


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


@pytest.fixture
def served(monkeypatch):
    seen: list = []

    def serve(body: bytes | Exception):
        def fake_urlopen(req, timeout=None):  # noqa: ARG001 — urlopen signature
            seen.append(req)
            if isinstance(body, Exception):
                raise body
            return _Response(body)

        monkeypatch.setattr(cx.urllib.request, "urlopen", fake_urlopen)
        return seen

    return serve


def test_verify_accepts_a_matching_archive(tmp_path):
    path = tmp_path / "wf.zip"
    path.write_bytes(_archive(GOOD))
    assert len(cx.verify_export_archive(str(path))["files"]) == 2


def test_verify_rejects_a_tampered_member(tmp_path):
    path = tmp_path / "wf.zip"
    path.write_bytes(_archive(GOOD, tamper="steps/a.py"))
    with pytest.raises(cx.ExportDownloadError, match="steps/a.py"):
        cx.verify_export_archive(str(path))


def test_destination_defaults_and_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cx.export_destination("wf", None) == str(tmp_path / "wf.zip")
    assert cx.export_destination("wf", str(tmp_path)) == str(tmp_path / "wf.zip")
    assert cx.export_destination("wf", "out.zip") == str(tmp_path / "out.zip")


def test_trailing_separator_names_a_new_directory(tmp_path, monkeypatch, served):
    """``-o ./exports/`` with no exports dir used to write a FILE named
    ``exports``; a trailing separator now means a directory (2026-11-07)."""
    monkeypatch.chdir(tmp_path)
    dest = cx.export_destination("wf", "./exports/nested/")
    assert dest == str(tmp_path / "exports" / "nested" / "wf.zip")
    assert (tmp_path / "exports" / "nested").is_dir()
    served(_archive(GOOD))
    cx.download_workflow_export("https://gw", "wf", dest)
    assert zipfile.is_zipfile(tmp_path / "exports" / "nested" / "wf.zip")


def test_download_saves_after_verifying(tmp_path, served):
    seen = served(_archive(GOOD))
    dest = tmp_path / "wf.zip"
    manifest = cx.download_workflow_export(
        "https://gw.example/", "wf", str(dest), api_key="tok", project="opossum"
    )
    assert dest.is_file() and len(manifest["files"]) == 2
    req = seen[0]
    assert req.full_url == "https://gw.example/gw/workflows/wf/export"
    assert req.get_header("X-project-id") == "opossum"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["wf.zip"]


def test_download_refuses_to_overwrite_without_force(tmp_path, served):
    served(_archive(GOOD))
    dest = tmp_path / "wf.zip"
    dest.write_bytes(b"mine")
    with pytest.raises(cx.ExportDownloadError, match="exists"):
        cx.download_workflow_export("https://gw", "wf", str(dest))
    assert dest.read_bytes() == b"mine"
    cx.download_workflow_export("https://gw", "wf", str(dest), overwrite=True)
    assert zipfile.is_zipfile(dest)


def test_a_bad_archive_leaves_nothing_behind(tmp_path, served):
    served(_archive(GOOD, tamper="workflow.json"))
    with pytest.raises(cx.ExportDownloadError):
        cx.download_workflow_export("https://gw", "wf", str(tmp_path / "wf.zip"))
    assert list(tmp_path.iterdir()) == []


def test_http_errors_carry_the_server_detail(tmp_path, served):
    body = io.BytesIO(json.dumps({"detail": "Workflow 'wf' not found"}).encode())
    served(urllib.error.HTTPError("https://gw", 404, "Not Found", {}, body))
    with pytest.raises(cx.ExportDownloadError, match="HTTP 404: Workflow 'wf' not found"):
        cx.download_workflow_export("https://gw", "wf", str(tmp_path / "wf.zip"))
    assert list(tmp_path.iterdir()) == []


def test_cli_action_prints_path_and_counts(tmp_path, served, monkeypatch, capsys):
    import client_umrs

    served(_archive(GOOD))
    monkeypatch.setattr(client_umrs, "_get_api_key_from_profile", lambda _ctx: None)
    args = argparse.Namespace(
        name="wf", dest=str(tmp_path), project=None, force=False, output="text"
    )
    ctx = SimpleNamespace(
        args=args, control_mq=None, connection=SimpleNamespace(gateway_url="https://gw")
    )
    assert client_umrs._action_workflow_export(ctx) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "wf.zip") in out and "2 files, 1 excluded" in out
