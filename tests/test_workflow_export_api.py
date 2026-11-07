"""GET /umrs/workflows/{name}/export (2026-11-07).

Collaborators who see a workflow through an ADR-080 ``project:`` share must be
able to take its code home. The export resolves exactly like ``/package``
(owner, system, share for the VERIFIED project, admin) and ships the
registered source tree, a relative ``workflow.json`` and an ``EXPORT.json``
whose hashes match; secrets-shaped paths and symlinks stay out and are listed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import audit_events
from umrs.api import workflow_export as wx
from umrs.api.workflows import router
from umrs.server import workflow_registry_handler as wrh

SECRET = "test-secret"
NAME = "ct-analysis"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)
    catalog_file = str(tmp_path / "workflow_catalog.json")
    monkeypatch.setattr(wrh, "_catalog_path", lambda: catalog_file)
    wrh.set_workflow_circles_resolver(lambda _user: [])
    yield
    wrh.set_workflow_circles_resolver(lambda _user: [])


@pytest.fixture
def audit(monkeypatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(audit_events, "emit_audit", lambda event, **kw: events.append((event, kw)))
    return events


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _definition(source_path: str = ".") -> dict:
    return {
        "name": NAME,
        "version": "1.0",
        "description": "export test",
        "source": {"type": "uploaded", "path": source_path},
        "steps": [
            {"name": "segment", "entrypoint": "steps.segment:Seg", "depends_on": []},
            {"name": "stats", "entrypoint": "steps.stats:Stats", "depends_on": ["segment"]},
        ],
    }


def _write(path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _register(tmp_path, owner: str = "alice", access: list[str] | None = None):
    source = tmp_path / f"{owner}-src"
    source.mkdir()
    definition = _definition(str(source.resolve()))
    _write(source / "workflow.json", json.dumps(definition))
    _write(source / "steps" / "segment.py", "class Seg:\n    pass\n")
    _write(source / "steps" / "stats.py", "class Stats:\n    pass\n")
    wrh.register_workflow(
        workflow_name=NAME,
        version="1.0",
        user_id=owner,
        source_path=str(source),
        workflow_json=definition,
    )
    if access:
        wrh.set_workflow_access(NAME, owner, access)
    return source


def _export(client, headers, name: str = NAME):
    return client.get(f"/umrs/workflows/{name}/export", headers=headers)


def _zip(response) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(response.content))


def _manifest(zf: zipfile.ZipFile) -> dict:
    return json.loads(zf.read("EXPORT.json"))


# ── exclusion rules (unit) ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rel",
    [
        ".env",
        ".env.local",
        "sub/.envrc",
        ".git/config",
        "sub/.GIT/HEAD",
        "pkg/__pycache__/m.cpython-311.pyc",
        "tls/server.pem",
        "server.KEY",
        "id_rsa",
        "keys/id_ed25519.pub",
        "cert.p12",
        "cert.pfx",
        "credentials",
        "conf/credentials.json",
        ".netrc",
        ".pgpass",
        "secrets.yaml",
        "deploy/secrets.yml",
    ],
)
def test_secret_shaped_paths_are_excluded(rel):
    assert wx.export_excluded(rel)


@pytest.mark.parametrize(
    "rel",
    [
        "workflow.json",
        "steps/segment.py",
        "README.md",
        "environment.yml",
        "keys.py",
        "my_credentials.txt",
        ".gitignore",
        ".github/x",
        ".gitattributes",
    ],
)
def test_ordinary_paths_are_kept(rel):
    assert not wx.export_excluded(rel)


def test_relative_source_inverts_the_upload_rewrite(tmp_path):
    root = tmp_path / "wf"
    (root / "src").mkdir(parents=True)
    inside = {"source": {"type": "uploaded", "path": str(root / "src")}}
    at_root = {"source": {"type": "uploaded", "path": str(root)}}
    outside = {"source": {"type": "local_path", "path": "/opt/elsewhere"}}
    relative = {"source": {"type": "local_path", "path": "src"}}
    assert wx.relative_source(inside, str(root))["source"]["path"] == "src"
    assert wx.relative_source(at_root, str(root))["source"]["path"] == "."
    assert wx.relative_source(outside, str(root))["source"]["path"] == "."
    assert wx.relative_source(relative, str(root)) is relative


# ── route: who may export ───────────────────────────────────────────────────


def test_owner_exports_the_registered_tree(client, tmp_path):
    _register(tmp_path)
    response = _export(client, _headers("alice", "proj-a"))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == f'attachment; filename="{NAME}.zip"'
    names = set(_zip(response).namelist())
    assert names == {"workflow.json", "EXPORT.json", "steps/segment.py", "steps/stats.py"}


def test_project_member_exports_a_project_shared_workflow(client, tmp_path):
    _register(tmp_path, access=["project:opossum"])
    response = _export(client, _headers("ryoshimi2", "opossum"))
    assert response.status_code == 200, response.text
    assert "steps/segment.py" in _zip(response).namelist()


def test_non_member_gets_the_package_404(client, tmp_path):
    _register(tmp_path, access=["project:opossum"])
    export = _export(client, _headers("ryoshimi2", "other-proj"))
    package = client.get(
        f"/umrs/workflows/{NAME}/package", headers=_headers("ryoshimi2", "other-proj")
    )
    assert export.status_code == package.status_code == 404
    assert export.json() == package.json()
    assert export.headers["x-platform-error-code"] == "workflows.not_found"


def test_unsigned_identity_is_refused(client, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    _register(tmp_path)
    headers = _headers("alice", "proj-a")
    headers["X-Verified-HMAC"] = "0" * 64
    assert _export(client, headers).status_code == 401


def test_admin_exports_another_owners_workflow(client, tmp_path):
    _register(tmp_path, owner="carol")
    response = _export(client, _headers("root", "ops", admin=True))
    assert response.status_code == 200, response.text


# ── archive content ─────────────────────────────────────────────────────────


def test_secret_shaped_files_are_absent_and_listed(client, tmp_path):
    source = _register(tmp_path)
    for rel in (
        ".env",
        ".git/config",
        "steps/__pycache__/s.pyc",
        "tls/server.pem",
        "id_rsa",
        ".netrc",
    ):
        _write(source / rel)
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    excluded = set(_manifest(zf)["excluded"])
    assert {".env", ".git", "steps/__pycache__", "tls/server.pem", "id_rsa", ".netrc"} <= excluded
    assert not excluded & set(zf.namelist())
    assert "tls" not in {n.split("/")[0] for n in zf.namelist()}


def test_symlinks_are_skipped_never_followed(client, tmp_path):
    source = _register(tmp_path)
    outside = tmp_path / "outside"
    _write(outside / "secret.txt", "do not ship\n")
    os.symlink(outside / "secret.txt", source / "link.txt")
    os.symlink(outside, source / "linkdir")
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    assert {"link.txt", "linkdir"} <= set(_manifest(zf)["excluded"])
    assert not any("secret" in n or n.startswith("link") for n in zf.namelist())


def test_workflow_json_source_path_is_relative(client, tmp_path):
    _register(tmp_path)
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    exported = json.loads(zf.read("workflow.json"))
    assert exported["source"]["path"] == "."
    assert str(tmp_path) not in zf.read("workflow.json").decode()
    assert str(tmp_path) not in zf.read("EXPORT.json").decode()


def _member_rows(zf: zipfile.ZipFile) -> dict[str, dict]:
    """What EXPORT.json should list: every member but itself, hashed."""
    return {
        name: {
            "path": name,
            "sha256": hashlib.sha256(zf.read(name)).hexdigest(),
            "size": len(zf.read(name)),
        }
        for name in zf.namelist()
        if name != "EXPORT.json"
    }


def test_export_json_hashes_match_the_members(client, tmp_path):
    _register(tmp_path)
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    manifest = _manifest(zf)
    header = {k: manifest[k] for k in ("workflow", "images", "workflow_json")}
    assert header == {"workflow": NAME, "images": "not included", "workflow_json": "source"}
    assert manifest["exported_at"].endswith("Z")
    assert manifest["umrs_version"]
    assert {row["path"]: row for row in manifest["files"]} == _member_rows(zf)


def test_size_cap_is_a_typed_413(client, tmp_path, monkeypatch):
    _register(tmp_path)
    monkeypatch.setattr(wx, "MAX_EXPORT_BYTES", 64)
    response = _export(client, _headers("alice", "proj-a"))
    assert response.status_code == 413
    assert response.headers["x-platform-error-code"] == "workflows.export_too_large"
    assert os.listdir(tmp_path / "tmp") == []


def test_missing_source_dir_ships_only_the_definition(client, tmp_path):
    source = _register(tmp_path)
    for child in sorted(source.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    source.rmdir()
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    assert sorted(zf.namelist()) == ["EXPORT.json", "workflow.json"]
    assert _manifest(zf)["workflow_json"] == "catalog"
    assert [s["name"] for s in json.loads(zf.read("workflow.json"))["steps"]] == [
        "segment",
        "stats",
    ]


def test_export_is_audited_and_the_temp_file_removed(client, tmp_path, audit):
    _register(tmp_path, access=["project:opossum"])
    response = _export(client, _headers("ryoshimi2", "opossum"))
    assert response.status_code == 200
    event, kw = audit[-1]
    assert (event, kw["actor"], kw["project_id"]) == ("workflow.exported", "ryoshimi2", "opossum")
    details = dict(kw["details"])
    assert details.pop("bytes") > 0
    assert details == {
        "workflow": NAME,
        "owner": "alice",
        "caller": "ryoshimi2",
        "project": "opossum",
        "files": 3,
        "excluded": 0,
    }
    assert os.listdir(tmp_path / "tmp") == []


def _code_hashes(zf: zipfile.ZipFile) -> dict[str, str]:
    rows = _manifest(zf)["files"]
    return {r["path"]: r["sha256"] for r in rows if r["path"] != "workflow.json"}


def test_round_trip_export_upload_keeps_the_definition(client, tmp_path):
    _register(tmp_path, access=["project:opossum"])
    exported = _export(client, _headers("ryoshimi2", "opossum"))
    uploaded = client.post(
        "/umrs/workflows/upload",
        headers=_headers("ryoshimi2", "opossum"),
        data={"workflow_name": NAME},
        files={"file": (f"{NAME}.zip", exported.content, "application/zip")},
    )
    assert uploaded.status_code == 202, uploaded.text
    mine = wrh.resolve_workflow_for_caller(NAME, "ryoshimi2")
    assert mine["user_id"] == "ryoshimi2"
    with open(os.path.join(mine["source_path"], "workflow.json"), encoding="utf-8") as fh:
        assert json.load(fh)["steps"] == _definition()["steps"]
    again = _zip(_export(client, _headers("ryoshimi2", "opossum")))
    assert json.loads(again.read("workflow.json"))["source"]["path"] == "."
    assert _code_hashes(again) == _code_hashes(_zip(exported))


# ── review fixes (2026-11-07) ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "extra",
    [{}, {"Range": "bytes=abc"}, {"Range": "bytes=999999-"}],
    ids=["plain", "malformed-range", "unsatisfiable-range"],
)
def test_no_temp_archive_is_left_behind(client, tmp_path, extra):
    """FileResponse answered 400/416 on these Range headers before its
    BackgroundTask ran, leaving umrs-export-*.zip in TMPDIR; the archive is
    now unlinked before the response starts and always sent whole."""
    _register(tmp_path)
    response = client.get(
        f"/umrs/workflows/{NAME}/export", headers={**_headers("alice", "proj-a"), **extra}
    )
    assert [n for n in os.listdir(tmp_path / "tmp") if n.startswith("umrs-export-")] == []
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert int(response.headers["content-length"]) == len(response.content)
    assert "steps/segment.py" in _zip(response).namelist()


@pytest.mark.parametrize("body", ["{not json", "\xff\xfe garbage"])
def test_corrupt_source_workflow_json_falls_back_to_the_catalog(client, tmp_path, body):
    source = _register(tmp_path)
    (source / "workflow.json").write_bytes(body.encode("latin-1"))
    response = _export(client, _headers("alice", "proj-a"))
    assert response.status_code == 200, response.text
    zf = _zip(response)
    assert _manifest(zf)["workflow_json"] == "catalog"
    assert json.loads(zf.read("workflow.json"))["name"] == NAME
    assert "steps/segment.py" in zf.namelist()


def test_hard_linked_files_are_skipped_and_listed(client, tmp_path):
    source = _register(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    _write(outside, "do not ship\n")
    os.link(outside, source / "hardlink.txt")
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    assert "hardlink.txt" in _manifest(zf)["excluded"]
    assert "hardlink.txt" not in zf.namelist()
    assert "steps/segment.py" in zf.namelist()


def test_dot_git_dir_is_excluded_but_gitignore_and_github_ship(client, tmp_path):
    source = _register(tmp_path)
    _write(source / ".git" / "config")
    _write(source / ".gitignore")
    _write(source / ".github" / "workflows" / "ci.yml")
    zf = _zip(_export(client, _headers("alice", "proj-a")))
    assert ".git" in _manifest(zf)["excluded"]
    assert {".gitignore", ".github/workflows/ci.yml"} <= set(zf.namelist())
    assert not any(n.startswith(".git/") for n in zf.namelist())
