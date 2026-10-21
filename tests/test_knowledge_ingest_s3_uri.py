"""An s3_uri ingest source is copied into content-addressed artifact storage.

`r4 knowledge upload` stages the bytes at ``s3://<bucket>/knowledge-uploads/...``
and submits that URI as an ``s3_uri`` source. Ingest used to register it
"without copy" — ``content_hash: ""`` and ``kind: "s3_uri"`` — so understand
recorded "no content_hash" (or, before that, skipped it silently), nothing was
embedded, and a bundle could not carry the bytes (raw objects are read only
from ``projects/<p>/knowledge/artifacts/<hash>``). Measured on the stand-alone
stack 2026-10-09.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from tests.s3_ranged import ranged
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import canonical_identity

_BODY = b"# Graphene note\n\nGraphene oxide membranes reject salt ions.\n"
_URI = "s3://r4-data/knowledge-uploads/proj-1/0b1c/graphene-note.md"


class _ClientError(Exception):
    pass


class _S3:
    """The calls ArtifactStorage + the fetch helper make, in memory."""

    exceptions = SimpleNamespace(ClientError=_ClientError)

    def __init__(self, objects: dict[tuple[str, str], bytes]):
        self.objects = dict(objects)
        self.puts: list[tuple[str, str]] = []
        self.copies: list[tuple] = []

    def get_object(self, Bucket, Key, Range=""):  # noqa: N803 — boto3 signature
        if (Bucket, Key) not in self.objects:
            raise KeyError(f"NoSuchKey {Bucket}/{Key}")
        return {**ranged(self.objects[(Bucket, Key)], Range), "ETag": '"e1"'}

    def copy_object(self, Bucket, Key, CopySource, **kw):  # noqa: N803
        self.copies.append(((CopySource["Bucket"], CopySource["Key"]), (Bucket, Key), kw))
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]

    def head_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise _ClientError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **_kw):  # noqa: N803
        self.puts.append((Bucket, Key))
        self.objects[(Bucket, Key)] = Body


def _ingest(s3: _S3, uri: str = _URI):
    job = SimpleNamespace(job_id="job-1", spec={})
    attempt = SimpleNamespace(attempt_id="att-1")
    result = ka.KnowledgeIngestResult()
    ids: list[str] = []
    descs: list[dict] = []
    ka._ingest_one_source(
        job, attempt, s3, "kb-bucket", "proj-1", "s3_uri", uri, [], result, ids, descs
    )
    return result, ids, descs


def _no_bridge(monkeypatch):
    monkeypatch.setattr(ka, "_bridge_a2l_ref", lambda *_a, **_k: None)


def test_s3_uri_source_gets_a_real_hash_and_a_document_kind(monkeypatch):
    _no_bridge(monkeypatch)
    s3 = _S3({("r4-data", "knowledge-uploads/proj-1/0b1c/graphene-note.md"): _BODY})
    result, ids, descs = _ingest(s3)

    digest = hashlib.sha256(_BODY).hexdigest()
    assert result.failed_items == []
    assert len(ids) == 1 and len(descs) == 1
    desc = descs[0]
    assert desc["content_hash"] == digest
    # A document kind, so understand runs semantic extraction (_PROSE_KINDS).
    assert desc["kind"] == "document"


def test_s3_uri_bytes_land_at_the_content_addressed_key(monkeypatch):
    """Bundles and understand read the canonical key, not the staged upload key."""
    _no_bridge(monkeypatch)
    s3 = _S3({("r4-data", "knowledge-uploads/proj-1/0b1c/graphene-note.md"): _BODY})
    _result, _ids, descs = _ingest(s3)

    canonical = f"projects/proj-1/knowledge/artifacts/{hashlib.sha256(_BODY).hexdigest()}"
    assert descs[0]["source_ref"] == canonical
    assert canonical_identity.stored_object(descs[0]["source_ref"])
    assert s3.objects[("kb-bucket", canonical)] == _BODY


def test_s3_uri_descriptor_is_stamped_and_keeps_the_staged_uri(monkeypatch):
    _no_bridge(monkeypatch)
    s3 = _S3({("r4-data", "knowledge-uploads/proj-1/0b1c/graphene-note.md"): _BODY})
    _result, _ids, descs = _ingest(s3)

    meta = descs[0]["metadata"]
    assert canonical_identity.carried(meta) is not None
    assert meta["staged_from"] == _URI
    # The artifact row has no title column: the name rides in the metadata.
    assert meta["source_name"] == "graphene-note.md"


def test_kind_follows_the_staged_filename(monkeypatch):
    _no_bridge(monkeypatch)
    uri = "s3://r4-data/knowledge-uploads/proj-1/0b1c/fit.py"
    s3 = _S3({("r4-data", "knowledge-uploads/proj-1/0b1c/fit.py"): b"def fit():\n    pass\n"})
    _result, _ids, descs = _ingest(s3, uri)
    assert descs[0]["kind"] == "code"


def test_unreadable_s3_object_is_a_failed_item_not_a_registration(monkeypatch):
    _no_bridge(monkeypatch)
    result, ids, descs = _ingest(_S3({}))
    assert ids == [] and descs == []
    assert result.artifact_ids == []
    assert len(result.failed_items) == 1
    failed = result.failed_items[0]
    assert failed["source"] == "s3_uri" and failed["uri"] == _URI
    assert "unreadable" in failed["error"]


def test_same_bytes_twice_dedup_to_one_object(monkeypatch):
    _no_bridge(monkeypatch)
    s3 = _S3(
        {
            ("r4-data", "knowledge-uploads/proj-1/a/n.md"): _BODY,
            ("r4-data", "knowledge-uploads/proj-1/b/n.md"): _BODY,
        }
    )
    _ingest(s3, "s3://r4-data/knowledge-uploads/proj-1/a/n.md")
    _ingest(s3, "s3://r4-data/knowledge-uploads/proj-1/b/n.md")
    # Copied server side (cycle-3 P1), once: the second finds the hash's key.
    assert s3.puts == [] and len(s3.copies) == 1
