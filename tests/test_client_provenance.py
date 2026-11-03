"""ADR-108 D-6: where an ingested file came from on the user's workstation.

Only basenames reached UMRS before, so nothing tied an artifact to the commit
a result was produced from. The client now reports path, repository, branch,
commit and dirty; the server keeps a bounded copy, marked client-reported.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

from umrs.api import knowledge_upload
from umrs.jobs import knowledge_adapters
from umrs.knowledge.client_provenance import attach_provenance, clean_provenance

GOOD = {
    "path": "analysis/fit.py",
    "repo": "github.com/lab/mof",
    "branch": "r4/feat",
    "commit": "0123456789abcdef0123456789abcdef01234567",
    "dirty": False,
}


def test_known_fields_are_kept_and_marked_client_reported() -> None:
    assert clean_provenance(GOOD) == {**GOOD, "client_reported": True}


def test_unknown_fields_bad_commit_and_bad_types_are_dropped() -> None:
    raw = {**GOOD, "commit": "not-a-sha", "dirty": "yes", "token": "s3cret", "path": "a\x00b"}
    assert clean_provenance(raw) == {
        "repo": GOOD["repo"],
        "branch": GOOD["branch"],
        "client_reported": True,
    }


def test_lengths_are_bounded_and_nothing_usable_is_none() -> None:
    long = clean_provenance({"path": "x" * 5000})
    assert len(long["path"]) == 1024
    for raw in (None, "path", [], {}, {"commit": "zz"}):
        assert clean_provenance(raw) is None


class _Storage:
    @staticmethod
    def key(project_id: str, content_hash: str) -> str:
        return f"projects/{project_id}/{content_hash}"

    def put_artifact(self, _project_id: str, content: bytes):
        return hashlib.sha256(content).hexdigest(), len(content), None


def test_an_inline_source_carries_its_provenance_onto_the_artifact(monkeypatch) -> None:
    monkeypatch.setattr(knowledge_adapters, "_bridge_a2l_ref", lambda *_a: None)
    monkeypatch.setattr(
        "umrs.knowledge.artifact_storage.ArtifactStorage", lambda *_a: _Storage()
    )
    descs: list[dict] = []
    src = {"kind": "inline_text", "uri": "fit.py", "content": "x = 1\n", "provenance": GOOD}
    knowledge_adapters._ingest_source(
        SimpleNamespace(job_id="j"),
        SimpleNamespace(attempt_id="a"),
        None,
        "bucket",
        "p",
        None,
        src,
        knowledge_adapters.KnowledgeIngestResult(),
        [],
        descs,
    )
    attach_provenance(descs, src)
    meta = descs[0]["metadata"]
    assert (meta["source_name"], meta["provenance"]["commit"]) == ("fit.py", GOOD["commit"])


def test_no_provenance_leaves_the_artifact_as_before() -> None:
    descs = [{"metadata": {"source_name": "a.txt"}}]
    attach_provenance(descs, {"kind": "inline_text"})
    assert descs == [{"metadata": {"source_name": "a.txt"}}]


def test_the_ingest_loop_attaches_per_source() -> None:
    """Wiring check: provenance is attached to the slice each source produced."""
    body = inspect.getsource(knowledge_adapters._run_ingest_sync)
    assert "attach_provenance(success_descs[start:], src_spec)" in body


def test_the_upload_form_field_is_parsed_and_bounded() -> None:
    assert knowledge_upload._form_provenance(json.dumps(GOOD))["commit"] == GOOD["commit"]
    assert knowledge_upload._form_provenance("{not json") is None
    assert knowledge_upload._form_provenance(None) is None
