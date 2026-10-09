"""Provenance lists the artifacts a job produced.

Plan §3c.2 [REMOTE TOO], measured live on the stand-alone stack 2026-10-09:
``GET /gw/provenance/{job}`` answered ``artifacts: []`` for an ingest that
stored an artifact. The reader took ``metrics["artifacts"]`` entries only with
a ``uri`` key; the workflow-run producer writes ``source_ref``/``content_hash``
(so every workflow artifact was dropped), and knowledge ingest wrote none.
"""

from __future__ import annotations

from types import SimpleNamespace

from umrs.jobs.provenance import _artifacts_from_attempts


def _attempt(metrics, attempt_id="att-1"):
    return SimpleNamespace(attempt_id=attempt_id, log_uri=None, metrics=metrics)


def test_workflow_run_descriptors_are_read():
    entry = {
        "artifact_id": "wfart-1",
        "content_hash": "ab" * 32,
        "kind": "s3_uri",
        "source_ref": "s3://r4-data/wf/att-1/out.csv",
        "name": "out.csv",
        "size_bytes": 12,
    }
    refs = _artifacts_from_attempts(
        SimpleNamespace(commit_attempt_id="att-1"), [_attempt({"artifacts": [entry]})]
    )
    assert [(r.uri, r.sha256, r.size_bytes) for r in refs] == [
        ("s3://r4-data/wf/att-1/out.csv", "ab" * 32, 12)
    ]


def test_the_canonical_uri_shape_still_reads():
    entry = {"uri": "s3://b/k", "media_type": "text/csv", "sha256": "cd" * 32}
    refs = _artifacts_from_attempts(
        SimpleNamespace(commit_attempt_id=None), [_attempt({"artifacts": [entry]})]
    )
    assert [(r.uri, r.media_type) for r in refs] == [("s3://b/k", "text/csv")]


def test_an_entry_with_no_location_is_skipped():
    refs = _artifacts_from_attempts(
        SimpleNamespace(commit_attempt_id=None), [_attempt({"artifacts": [{"name": "x"}]})]
    )
    assert refs == []


def test_knowledge_ingest_records_what_it_stored(monkeypatch):
    """The ingest backend writes the stored descriptors where provenance reads."""
    import asyncio

    from umrs.jobs import knowledge_adapters as ka

    descs = [
        {
            "artifact_id": "art-1",
            "content_hash": "ef" * 32,
            "kind": "document",
            "source_ref": "projects/p/knowledge/artifacts/" + "ef" * 32,
            "metadata": {"source_name": "note.md"},
        }
    ]
    monkeypatch.setattr(ka, "_run_ingest_sync", lambda *_a: ("knowledge_ingest|artifacts=1|failed=0", descs))
    monkeypatch.setattr(ka, "_enqueue_understand_job", lambda *_a, **_k: None)
    backend = ka.KnowledgeIngestBackend()
    backend._s3_client = object()
    monkeypatch.setattr(backend, "_init_s3", lambda _spec: None)
    attempt = SimpleNamespace(attempt_id="att-1", metrics={}, log_uri=None)
    job = SimpleNamespace(job_id="j", spec={"sources": [{"kind": "inline_text"}]}, project_id="p", user_id="u")
    asyncio.run(backend.submit_job(job, attempt))

    refs = _artifacts_from_attempts(SimpleNamespace(commit_attempt_id="att-1"), [attempt])
    assert [(r.uri, r.sha256) for r in refs] == [
        ("projects/p/knowledge/artifacts/" + "ef" * 32, "ef" * 32)
    ]
