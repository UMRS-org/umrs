"""Workflow versions (ADR-113 WV1/WV2): SemVer, versioned storage, the
default version, access per name, delete per version, one trigger per name,
run pinning, the first-start migration and adopt.

Every test runs against a scratch ``UMRS_WORKFLOW_STATE_DIR`` (catalog and
extracted trees under ``tmp_path``)."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest
from umrs.server import file_handling as fh
from umrs.server import workflow_registry_handler as wrh
from umrs.server.workflow_execution_registry import reset_execution_registry_for_tests
from umrs.server.workflow_versions import (
    WorkflowVersionError,
    default_entry,
    is_semver,
    name_suffix_warning,
    requested_version,
    sort_versions_desc,
    tree_digest,
    version_sort_key,
)


@pytest.fixture(autouse=True)
def state(tmp_path, monkeypatch):
    root = tmp_path / "state"
    root.mkdir()
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(root))
    wrh.set_workflow_circles_resolver(lambda _user: [])
    reset_execution_registry_for_tests()
    monkeypatch.setattr("umrs.server.step_reaper.durable_job_store", lambda: None)
    yield root
    reset_execution_registry_for_tests()


def _definition(version: str, **extra) -> dict:
    return {
        "name": "wf",
        "version": version,
        "description": "versions test",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "s1", "entrypoint": "steps:S1", "depends_on": []}],
        **extra,
    }


def _zip(tmp_path: Path, members: dict[str, str]) -> str:
    path = tmp_path / f"upload-{len(list(tmp_path.glob('upload-*')))}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return str(path)


def _bundle(tmp_path: Path, version: str, code: str = "X = 1\n", **extra) -> str:
    return _zip(
        tmp_path,
        {
            "workflow.json": json.dumps(_definition(version, **extra)),
            "steps.py": f"class S1:\n    pass\n{code}",
        },
    )


def _upload(tmp_path: Path, version: str, *, owner="alice", name="wf", **kw):
    """Upload + register, as the HTTP route does."""
    outcome = fh.upload_workflow_bundle(owner, _bundle(tmp_path, version, **kw), name)
    assert outcome.ok, outcome.message
    with open(os.path.join(outcome.dest, "workflow.json")) as handle:
        definition = json.load(handle)
    result = wrh.register_workflow(
        name,
        outcome.version,
        owner,
        outcome.dest,
        definition,
        source_sha256=outcome.source_sha256,
    )
    assert result["status"] == "ok", result
    return outcome


# ── SemVer ────────────────────────────────────────────────────────────────


def test_semver_grammar():
    for good in ("0.0.1", "1.2.3", "10.20.30", "1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-0.3.7"):
        assert is_semver(good), good
    for bad in ("1.0", "01.0.0", "1.0.0+build", "1.0.0-", "v1.0.0", "", None, "1.0.0-01"):
        assert not is_semver(bad), bad


def test_semver_precedence_follows_the_spec():
    ordered = [
        "1.0.0-alpha",
        "1.0.0-alpha.1",
        "1.0.0-alpha.beta",
        "1.0.0-beta",
        "1.0.0-beta.2",
        "1.0.0-beta.11",
        "1.0.0-rc.1",
        "1.0.0",
        "1.0.1",
        "1.10.0",
        "2.0.0",
    ]
    assert sorted(ordered[5:] + ordered[:5], key=version_sort_key) == ordered
    assert sort_versions_desc(["1.0.0", "2.0.0", "1.0.0"]) == ["2.0.0", "1.0.0"]


def test_legacy_versions_sort_numerically_below_their_semver():
    assert sort_versions_desc(["0.1", "0.2", "0.1.5", "junk"]) == ["0.2", "0.1.5", "0.1", "junk"]


def test_default_prefers_releases_and_skips_rows_without_code():
    rows = [
        {"version": "2.0.0-rc.1"},
        {"version": "1.4.0"},
        {"version": "1.9.0", "code": "unavailable"},
    ]
    assert default_entry(rows)["version"] == "1.4.0"
    assert (
        default_entry([{"version": "2.0.0-rc.1"}, {"version": "2.0.0-rc.2"}])["version"]
        == "2.0.0-rc.2"
    )


def test_version_refs():
    assert requested_version("wf@1.2.3", None) == ("wf", "1.2.3")
    assert requested_version("wf", "1.2.3") == ("wf", "1.2.3")
    with pytest.raises(WorkflowVersionError):
        requested_version("wf@1.0.0", "2.0.0")
    assert name_suffix_warning("opossum-cnn-track-v0.2.9")
    assert name_suffix_warning("cnn_track") is None


# ── storage (D-1, D-2) ────────────────────────────────────────────────────


def test_each_version_has_its_own_directory_and_old_files_stay(tmp_path, state):
    """Before: every version extracted over one directory, so 1.0.0's
    catalog row ran 2.0.0's code plus leftover files."""
    one = _upload(tmp_path, "1.0.0", code="OLD = 1\n")
    two = _upload(tmp_path, "2.0.0", code="NEW = 2\n")
    assert one.dest == str(state / "extracted" / "alice" / "wf" / "1.0.0")
    assert two.dest == str(state / "extracted" / "alice" / "wf" / "2.0.0")
    assert "OLD = 1" in Path(one.dest, "steps.py").read_text()
    assert "NEW = 2" in Path(two.dest, "steps.py").read_text()
    source = json.loads(Path(two.dest, "workflow.json").read_text())["source"]
    assert source == {"type": "uploaded", "path": two.dest}
    assert not list((state / "extracted" / "alice" / ".incoming").iterdir())


def test_same_bundle_again_is_unchanged_other_content_is_refused(tmp_path):
    first = _upload(tmp_path, "1.0.0")
    again = fh.upload_workflow_bundle("alice", _bundle(tmp_path, "1.0.0"), "wf")
    assert (again.ok, again.unchanged, again.source_sha256) == (True, True, first.source_sha256)
    other = fh.upload_workflow_bundle("alice", _bundle(tmp_path, "1.0.0", code="Y = 2\n"), "wf")
    assert (other.ok, other.code) == (False, "workflows.version_exists")
    assert "1.0.0" in other.message and "bump" in other.message
    assert "Y = 2" not in Path(first.dest, "steps.py").read_text()


def test_a_bundle_without_a_semver_version_is_refused_by_name(tmp_path):
    for version in (None, "1.0", "latest"):
        definition = _definition("x")
        definition.pop("version")
        if version:
            definition["version"] = version
        outcome = fh.upload_workflow_bundle(
            "alice", _zip(tmp_path, {"workflow.json": json.dumps(definition)}), "wf"
        )
        assert (outcome.ok, outcome.code) == (False, "workflows.invalid_version"), version


def test_the_digest_survives_the_source_rewrite_and_tracks_content(tmp_path):
    """The digest of the staged bundle equals the digest of the stored
    version, whose workflow.json ``source`` the upload rewrote."""
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "workflow.json").write_text(json.dumps(_definition("1.0.0")))
    (staged / "steps.py").write_text("A = 1\n")
    before = tree_digest(str(staged))
    final, digest, unchanged = fh._place_version(str(staged), str(tmp_path / "wf"), "1.0.0")
    assert (digest, unchanged) == (before, False)
    assert json.loads(Path(final, "workflow.json").read_text())["source"]["path"] == final
    assert tree_digest(final) == before
    Path(final, "steps.py").write_text("A = 2\n")
    assert tree_digest(final) != before


def test_a_nested_bundle_whose_folder_repeats_its_name_flattens(tmp_path):
    """Before: ``wf/wf/x.py`` moved ``wf/wf`` into itself and the upload
    failed ("Failed to extract workflow")."""
    members = {
        "wf/workflow.json": json.dumps(_definition("1.0.0")),
        "wf/steps.py": "class S1:\n    pass\n",
        "wf/wf/helper.py": "H = 1\n",
    }
    outcome = fh.upload_workflow_bundle("alice", _zip(tmp_path, members), "wf")
    assert outcome.ok, outcome.message
    assert Path(outcome.dest, "wf", "helper.py").read_text() == "H = 1\n"
    assert Path(outcome.dest, "workflow.json").is_file()


def test_a_version_like_name_gets_a_warning_not_a_refusal(tmp_path):
    outcome = fh.upload_workflow_bundle(
        "alice", _bundle(tmp_path, "0.2.9"), "opossum-cnn-track-v0.2.9"
    )
    assert outcome.ok
    assert outcome.warnings and "version-like suffix" in outcome.warnings[0]


# ── resolution (D-3) and access (D-5) ─────────────────────────────────────


def test_the_default_is_the_highest_semver_not_the_first_registered(tmp_path):
    """Before: lookup took the first matching row in file order (the oldest)."""
    _upload(tmp_path, "1.2.0")
    _upload(tmp_path, "1.10.0")
    _upload(tmp_path, "2.0.0-rc.1")
    resolved = wrh.resolve_workflow_version("wf", "alice")
    assert resolved.entry["version"] == "1.10.0"
    assert resolved.versions == ("2.0.0-rc.1", "1.10.0", "1.2.0")
    assert wrh.resolve_workflow_version("wf@1.2.0", "alice").entry["version"] == "1.2.0"
    assert (
        wrh.resolve_workflow_version("wf", "alice", version="2.0.0-rc.1").entry["version"]
        == "2.0.0-rc.1"
    )


def test_an_unknown_version_is_refused_listing_the_versions(tmp_path):
    _upload(tmp_path, "1.0.0")
    with pytest.raises(WorkflowVersionError) as refused:
        wrh.resolve_workflow_version("wf", "alice", version="9.9.9")
    assert refused.value.code == "workflows.version_not_found"
    assert "1.0.0" in str(refused.value)
    assert wrh.resolve_workflow_version("nope", "alice", version="1.0.0") is None


def test_the_ladder_restricted_to_a_version(tmp_path):
    """bob owns 1.0.0, a shared alice row is 2.0.0: bob's default stays his
    own rung; asking for 2.0.0 finds alice's shared one."""
    _upload(tmp_path, "2.0.0")
    wrh.set_workflow_access("wf", "alice", ["user:bob"])
    _upload(tmp_path, "1.0.0", owner="bob")
    assert wrh.resolve_workflow_version("wf", "bob").entry["user_id"] == "bob"
    pinned = wrh.resolve_workflow_version("wf", "bob", version="2.0.0").entry
    assert (pinned["user_id"], pinned["version"]) == ("alice", "2.0.0")


def test_a_new_version_inherits_the_name_access(tmp_path):
    """Before: a new version started private (access []) while sharees kept
    resolving the old one."""
    _upload(tmp_path, "1.0.0")
    wrh.set_workflow_access("wf", "alice", ["user:bob"])
    _upload(tmp_path, "1.1.0")
    resolved = wrh.resolve_workflow_version("wf", "bob")
    assert resolved.entry["version"] == "1.1.0"
    assert resolved.entry["access"] == ["user:bob"]


def test_list_groups_by_name_unless_all_versions(tmp_path):
    _upload(tmp_path, "1.0.0")
    _upload(tmp_path, "1.1.0")
    rows = json.loads(wrh._handle_list_workflows("alice", '{"user_id": "alice"}'))["workflows"]
    assert [(r["version"], r["versions"]) for r in rows] == [("1.1.0", ["1.1.0", "1.0.0"])]
    assert rows[0]["source_sha256"]
    every = json.loads(
        wrh._handle_list_workflows("alice", '{"user_id": "alice", "all_versions": true}')
    )["workflows"]
    assert [r["version"] for r in every] == ["1.1.0", "1.0.0"]
    assert every[0]["source_sha256"] != every[1]["source_sha256"]


# ── delete (D-6) ──────────────────────────────────────────────────────────


def test_delete_one_version_removes_only_its_files(tmp_path):
    one = _upload(tmp_path, "1.0.0")
    two = _upload(tmp_path, "2.0.0")
    result = wrh.unregister_workflow("wf", user_id="alice", version="1.0.0")
    assert (result["removed"], result["versions"], result["deferred"]) == (1, ["1.0.0"], 0)
    assert not os.path.exists(one.dest)
    assert Path(two.dest, "workflow.json").is_file()
    assert wrh.resolve_workflow_version("wf", "alice").versions == ("2.0.0",)


def test_files_of_a_version_a_run_uses_wait_for_the_run(tmp_path):
    from umrs.server.workflow_execution_registry import get_execution_registry

    one = _upload(tmp_path, "1.0.0")
    record = get_execution_registry().create(
        workflow_name="wf", user_id="alice", workflow_version="1.0.0"
    )
    record.update(status="running")
    result = wrh.unregister_workflow("wf", user_id="alice")
    assert (result["removed"], result["deferred"]) == (1, 1)
    assert os.path.isdir(one.dest)
    record.update(status="succeeded")
    wrh.sweep_removed_versions()
    assert not os.path.exists(one.dest)


def test_delete_never_removes_files_outside_the_extracted_tree(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "workflow.json").write_text(json.dumps(_definition("1.0.0")))
    wrh.register_workflow("wf", "1.0.0", "alice", str(outside), _definition("1.0.0"))
    assert wrh.unregister_workflow("wf", user_id="alice")["removed"] == 1
    assert (outside / "workflow.json").is_file()


# ── triggers (D-7) ────────────────────────────────────────────────────────


def _capture_trigger_runs(monkeypatch) -> list[dict]:
    from umrs.server import workflow_triggers

    fired: list[dict] = []

    def run(name, owner, params, version=None):
        fired.append({"name": name, "owner": owner, "version": version})
        return json.dumps({"status": "ok", "execution_id": f"wf-{len(fired)}"})

    monkeypatch.setattr(workflow_triggers, "_run_triggered", run)
    return fired


def _dispatch():
    from umrs.server import workflow_triggers

    return workflow_triggers.dispatch_catalog_event(
        project_id="p", bucket="up", s3_key="a/b.mp4", owner="uploader"
    )


_INGEST = [{"on": "ingest", "bucket": "up"}]


def test_one_event_fires_one_run_per_name(tmp_path, monkeypatch):
    """Before: every registered version fired its own run."""
    fired = _capture_trigger_runs(monkeypatch)
    _upload(tmp_path, "1.0.0", triggers=_INGEST)
    _upload(tmp_path, "1.1.0", triggers=_INGEST)
    assert len(_dispatch()) == 1
    assert fired == [{"name": "wf", "owner": "alice", "version": "1.1.0"}]


def test_triggers_come_from_the_firing_version_and_follow_the_pin(tmp_path, monkeypatch):
    fired = _capture_trigger_runs(monkeypatch)
    _upload(tmp_path, "1.0.0", triggers=_INGEST)
    _upload(tmp_path, "2.0.0")  # the default declares no triggers
    assert _dispatch() == []
    wrh.set_trigger_version("wf", "alice", "1.0.0")
    assert len(_dispatch()) == 1
    assert fired[-1]["version"] == "1.0.0"
    with pytest.raises(WorkflowVersionError):
        wrh.set_trigger_version("wf", "alice", "3.0.0")


# ── runs (D-4) ────────────────────────────────────────────────────────────


class _Backend:
    store_pool = None

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def submit_workflow_run(self, _client_id, payload_str):
        payload = json.loads(payload_str)
        self.payloads.append(payload)
        return json.dumps({"status": "ok", "job": {"job_id": "job-1", "spec": payload["spec"]}})


def _run(backend, payload: dict, client="alice") -> dict:
    return json.loads(wrh._handle_run_workflow(client, json.dumps(payload), backend))


def test_a_durable_run_pins_version_digest_and_directory(tmp_path):
    one = _upload(tmp_path, "1.0.0")
    two = _upload(tmp_path, "2.0.0")
    backend = _Backend()
    reply = _run(backend, {"workflow_name": "wf", "version": "1.0.0", "params": {}})
    assert (reply["workflow_version"], reply["source_sha256"]) == ("1.0.0", one.source_sha256)
    spec = backend.payloads[-1]["spec"]
    assert (spec["workflow_version"], spec["source_sha256"]) == ("1.0.0", one.source_sha256)
    assert spec["source_path"] == one.dest
    default = _run(backend, {"workflow_name": "wf@2.0.0", "params": {}})
    assert default["source_sha256"] == two.source_sha256
    missing = _run(backend, {"workflow_name": "wf", "version": "3.0.0", "params": {}})
    assert missing["code"] == "workflows.version_not_found"


def test_a_dispatch_of_changed_files_fails_by_name(tmp_path, monkeypatch):
    """A retry / re-dispatch re-checks the digest the trigger pinned."""
    one = _upload(tmp_path, "1.0.0")
    Path(one.dest, "steps.py").write_text("TAMPERED = 1\n")
    with pytest.raises(WorkflowVersionError) as refused:
        wrh._launch_workflow_from_catalog(
            workflow_name="wf",
            client_id="alice",
            params={},
            source_path=one.dest,
            workflow_version="1.0.0",
            source_sha256=one.source_sha256,
        )
    assert refused.value.code == "workflows.source_mismatch"


def test_the_durable_backend_passes_the_pin_on(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from umrs.jobs.workflow_run import WorkflowRunBackend
    from umrs.shared import workflow_execution_access as access

    seen: dict = {}
    monkeypatch.setattr(access, "_handler", lambda **kw: seen.update(kw))
    job = SimpleNamespace(
        job_id="job-1",
        user_id="alice",
        project_id="p",
        spec={
            "workflow_name": "wf",
            "execution_id": "wf-1",
            "source_path": "/nonexistent",
            "workflow_version": "1.0.0",
            "source_sha256": "abc",
        },
    )
    asyncio.run(WorkflowRunBackend().submit_job(job, SimpleNamespace(attempt_id="a")))
    assert (seen["workflow_version"], seen["source_sha256"]) == ("1.0.0", "abc")
    assert seen["source_path"] is None  # re-resolved: that exact version


def test_the_run_record_carries_the_version(tmp_path, monkeypatch):
    from umrs.server import workflow_runner

    one = _upload(tmp_path, "1.0.0")
    monkeypatch.setattr(
        workflow_runner,
        "run_workflow_record",
        lambda record, **_kw: record.update(status="succeeded"),
    )
    reply = _run(None, {"workflow_name": "wf", "params": {}})
    from umrs.server.workflow_execution_registry import get_execution_registry

    record = get_execution_registry().get(reply["execution_id"])
    assert record is not None
    data = record.to_dict()
    assert (data["workflow_version"], data["source_sha256"]) == ("1.0.0", one.source_sha256)


def test_the_manifest_records_the_pinned_version():
    from umrs.jobs.manifest import execution_manifest
    from umrs.jobs.store import _manifest_identity

    identity = _manifest_identity({"workflow_name": "wf", "workflow_version": "1.2.0"})
    manifest = execution_manifest(provenance={}, **identity)
    assert (manifest["pipeline_name"], manifest["pipeline_version"]) == ("wf", "1.2.0")


# ── migration (D-8) ───────────────────────────────────────────────────────


def _legacy_catalog(state: Path, rows: list[dict]) -> None:
    (state / "workflow_catalog.json").write_text(json.dumps({"workflows": rows}))


def _legacy_tree(state: Path, version: str) -> Path:
    tree = state / "extracted" / "alice" / "wf"
    tree.mkdir(parents=True)
    definition = {**_definition(version), "source": {"type": "uploaded", "path": str(tree)}}
    (tree / "workflow.json").write_text(json.dumps(definition))
    (tree / "steps.py").write_text("class S1:\n    pass\n")
    return tree


def test_migration_moves_the_shared_tree_and_marks_the_rest(state, caplog):
    tree = _legacy_tree(state, "0.2")
    rows = [
        {
            "name": "wf",
            "version": "0.1",
            "user_id": "alice",
            "source_path": str(tree),
            "access": ["user:bob"],
        },
        {"name": "wf", "version": "0.2", "user_id": "alice", "source_path": str(tree)},
    ]
    _legacy_catalog(state, rows)
    caplog.set_level("INFO")
    resolved = wrh.resolve_workflow_version("wf", "alice")
    target = tree / "0.2"
    assert (resolved.entry["version"], resolved.entry["source_path"]) == ("0.2", str(target))
    assert resolved.entry["source_sha256"] == tree_digest(str(target))
    moved = json.loads((target / "workflow.json").read_text())
    assert moved["source"]["path"] == str(target)
    old = wrh.resolve_workflow_version("wf", "alice", version="0.1").entry
    assert old["code"] == "unavailable"
    assert "per-version directories" in caplog.text
    stored = json.loads((state / "workflow_catalog.json").read_text())
    assert stored["layout"] == 2
    # idempotent: a second load changes nothing
    wrh._load_catalog()
    assert json.loads((state / "workflow_catalog.json").read_text()) == stored


def test_the_declared_version_keeps_the_files_even_if_lower(state):
    """The shared tree holds the last upload: when that was the lower
    version, the lower row keeps the files, not the highest SemVer."""
    tree = _legacy_tree(state, "1.0.0")
    _legacy_catalog(
        state,
        [
            {"name": "wf", "version": "2.0.0", "user_id": "alice", "source_path": str(tree)},
            {"name": "wf", "version": "1.0.0", "user_id": "alice", "source_path": str(tree)},
        ],
    )
    assert wrh.resolve_workflow_version("wf", "alice").entry["version"] == "1.0.0"


def test_running_a_version_without_code_fails_by_name(state):
    tree = _legacy_tree(state, "0.2")
    _legacy_catalog(
        state,
        [
            {"name": "wf", "version": "0.1", "user_id": "alice", "source_path": str(tree)},
            {"name": "wf", "version": "0.2", "user_id": "alice", "source_path": str(tree)},
        ],
    )
    reply = _run(_Backend(), {"workflow_name": "wf", "version": "0.1", "params": {}})
    assert reply["code"] == "workflows.code_unavailable"


# ── adopt (D-8) ───────────────────────────────────────────────────────────


def test_adopt_registers_the_files_under_a_stable_name(tmp_path, monkeypatch):
    fired = _capture_trigger_runs(monkeypatch)
    _upload(tmp_path, "0.1.0", name="cnn-v0.2.9", triggers=_INGEST)
    wrh.set_workflow_access("cnn-v0.2.9", "alice", ["user:bob"])
    result = wrh.adopt_workflow("cnn-v0.2.9", "cnn", "0.2.9", caller="alice", is_admin=False)
    assert (result["name"], result["version"], result["from_name"]) == (
        "cnn",
        "0.2.9",
        "cnn-v0.2.9",
    )
    adopted = wrh.resolve_workflow_version("cnn", "bob").entry  # access carried
    assert (adopted["version"], adopted["source_sha256"]) == ("0.2.9", result["source_sha256"])
    definition = json.loads(Path(adopted["source_path"], "workflow.json").read_text())
    assert definition["version"] == "0.2.9"
    _dispatch()
    assert [f["name"] for f in fired] == ["cnn"]  # the old name's triggers are off
    with pytest.raises(PermissionError):
        wrh.adopt_workflow("cnn-v0.2.9", "x", "1.0.0", caller="bob", is_admin=False)
