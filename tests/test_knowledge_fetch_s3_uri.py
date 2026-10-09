"""Artifact fetch accepts both bare keys and s3://bucket/key URIs.

The upload facade records ``s3://<bucket>/<key>``; the fetch helper used it
as the object key, so every uploaded document failed with NoSuchKey.
"""

from __future__ import annotations

import io

from umrs.jobs import knowledge_adapters as ka


class _S3:
    def __init__(self):
        self.calls = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.calls.append((Bucket, Key))
        return {"Body": io.BytesIO(b"bytes")}


def test_bare_key_uses_the_configured_bucket():
    s3 = _S3()
    assert ka._fetch_artifact_bytes(s3, "cfg-bucket", "knowledge-uploads/p/u/a.md") == b"bytes"
    assert s3.calls == [("cfg-bucket", "knowledge-uploads/p/u/a.md")]


def test_s3_uri_is_split_into_bucket_and_key():
    s3 = _S3()
    assert ka._fetch_artifact_bytes(s3, "cfg-bucket", "s3://r4-data/knowledge-uploads/p/u/a.md")
    assert s3.calls == [("r4-data", "knowledge-uploads/p/u/a.md")]


def test_empty_ref_and_no_client():
    assert ka._fetch_artifact_bytes(_S3(), "b", "") is None
    assert ka._fetch_artifact_bytes(None, "b", "s3://b/k") is None


def test_understand_records_a_source_without_content_hash(monkeypatch):
    """`r4 knowledge upload` submits an s3_uri source registered with content_hash "".

    _run_understand used to `continue` past it silently: ir_ids=0, fail=0 and a
    clean result for an upload that produced nothing.
    """
    import asyncio
    from types import SimpleNamespace

    async def _passthrough(sources, _store, _project):
        return sources

    monkeypatch.setattr(ka, "_artifacts_to_process", _passthrough)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", lambda _pool: object())
    job = SimpleNamespace(spec={}, job_id="job-1", user_id="u")
    attempt = SimpleNamespace(attempt_id="att-1")
    src = {
        "artifact_id": "art-1",
        "content_hash": "",
        "kind": "s3_uri",
        "source_ref": "s3://r4-data/knowledge-uploads/p/u/a.md",
    }
    result = asyncio.run(ka._run_understand(job, attempt, "p", [src], object(), None, "b"))
    assert result.ir_ids == []
    assert result.failed_items and result.failed_items[0]["artifact_id"] == "art-1"
    assert "no content_hash" in result.failed_items[0]["error"]
