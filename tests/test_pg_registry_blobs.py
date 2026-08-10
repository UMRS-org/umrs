"""PG registry blob round-trip tests (Phase 2A).

Guarantees the checker can resolve workflow output bytes from a PG-hydrated
record exactly as from the in-memory record:

    runner writes blobs → persist serializes blobs (base64)
    → PG record_json → hydration decodes blobs → _persist_run_artifacts
    produces metrics["artifacts"].
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from umrs.jobs import workflow_run as wr
from umrs.workflow.pg_registry import (
    DurableExecutionRecord,
    _decode_blobs,
    _encode_blobs,
    _record_from_payload,
    _record_json,
)


def _record(results: dict) -> DurableExecutionRecord:
    rec = DurableExecutionRecord(
        execution_id="wf-test-1",
        workflow_name="demo08_yield_eval",
        user_id="alice",
        params={"project_id": "p"},
    )
    rec.results = results
    rec._persist_hook = None  # no PG mirror needed for serializer tests
    return rec


# ── Test A: exact bytes round trip ───────────────────────────────────────────


def test_blobs_bytes_round_trip() -> None:
    results = {
        "blobs": {
            "output.txt": b"hello world",
            "data.bin": b"\x00\x01\x02\xff",
            "nested/name.json": b'{"a": 1}',
        }
    }
    payload = json.loads(_record_json(_record(results)))
    hydrated = _record_from_payload(payload)

    assert hydrated.results["blobs"] == results["blobs"]
    assert hydrated.results["blobs"]["data.bin"] == b"\x00\x01\x02\xff"
    assert isinstance(hydrated.results["blobs"]["output.txt"], bytes)


def test_blob_encoding_is_json_safe() -> None:
    encoded = _encode_blobs({"k": b"\x00\xff binary"})
    assert encoded["k"]["_blob_encoding"] == "base64"
    json.dumps(encoded)  # must be JSON-serializable
    assert _decode_blobs(encoded) == {"k": b"\x00\xff binary"}
    # Idempotent encode (already-wrapped values pass through).
    assert _encode_blobs(encoded) == encoded


# ── Test B: empty / no blobs stay valid ──────────────────────────────────────


def test_empty_results_round_trip() -> None:
    hydrated = _record_from_payload(json.loads(_record_json(_record({}))))
    assert hydrated.results == {}


def test_empty_blobs_round_trip() -> None:
    rec = _record({"blobs": {}})
    hydrated = _record_from_payload(json.loads(_record_json(rec)))
    assert hydrated.results == {"blobs": {}}


def test_other_results_keys_preserved() -> None:
    rec = _record({"final": {"x": 1}, "blobs": {"k": b"v"}})
    hydrated = _record_from_payload(json.loads(_record_json(rec)))
    assert hydrated.results["final"] == {"x": 1}
    assert hydrated.results["blobs"] == {"k": b"v"}


# ── Test C: backward compatibility with pre-fix payloads ─────────────────────


def test_legacy_payload_without_results() -> None:
    legacy = {
        "execution_id": "wf-old",
        "workflow_name": "w",
        "user_id": "u",
        "params": {},
        "status": "succeeded",
        "steps": [],
        "artifacts": [],
    }
    hydrated = _record_from_payload(legacy)
    assert hydrated.results == {}  # exact pre-fix behaviour
    assert hydrated.execution_id == "wf-old"


def test_legacy_plain_blob_values_pass_through() -> None:
    plain = {"blobs": {"note": "already-a-string"}}
    assert _encode_blobs(plain) == plain
    assert _decode_blobs(plain) == plain
    assert _decode_blobs(None) == {}
    assert _encode_blobs(None) == {}


# ── Test D: hydrated record unblocks artifact persistence ────────────────────


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> dict:
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": Body})
        return {"ETag": '"fake"'}


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.created: list[object] = []

    def create(self, artifact: object) -> SimpleNamespace:
        self.created.append(artifact)
        return SimpleNamespace(id=uuid4())


def test_hydrated_record_unblocks_persist_run_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PG-hydrated record must feed _persist_run_artifacts like the live one."""
    monkeypatch.setenv("S3_BUCKET", "r4-data")
    blob = b'{"metrics": {"yield_pct": 85.0}}'
    live = _record({"blobs": {"wf-runner-out/out.json": blob}})

    # Serialize then hydrate — the exact path the checker takes on failover.
    hydrated = _record_from_payload(json.loads(_record_json(live)))

    s3 = _FakeS3()
    store = _FakeArtifactStore()
    job = SimpleNamespace(job_id="job-pg2a", user_id="alice", project_id="p", priority=0, spec={})
    attempt = SimpleNamespace(
        attempt_id="att-pg2a",
        job_id="job-pg2a",
        attempt_no=1,
        output_prefix="jobs/job-pg2a/attempt_1",
    )
    artifacts = wr._persist_run_artifacts(
        record=hydrated, job=job, attempt=attempt, s3_client=s3, artifact_store=store
    )

    assert artifacts, "hydrated record must unblock metrics['artifacts']"
    assert s3.puts[0]["Key"] == "jobs/job-pg2a/attempt_1/out.json"
    assert s3.puts[0]["Body"] == blob
    assert artifacts[0]["kind"] == "s3_uri"
    assert len(store.created) == 1
