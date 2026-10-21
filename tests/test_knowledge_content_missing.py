"""A row whose content is not in artifact storage reads as "document content
missing", typed — never an S3 GET for a reference that names no object, a
warning per row, or an understand failure.

ops-watch #522/#523 (2026-10-21, umrs-server-octopus): 211 "S3 fetch failed
for rag://... NoSuchKey" warnings in one window. RAG corpus rows
(``rag://<document_id>``) and loop findings (``loop:...``) are minted without
bytes, yet understand and embed sent their reference to GetObject as a key,
and understand failed each such row (retried, so an ingest of a RAG-only
project failed its understand job).
"""

from __future__ import annotations

import asyncio
import io
import logging
from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka


class _S3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    """Records every call; GET raises ``get_error`` when set."""

    def __init__(self, get_error: Exception | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._get_error = get_error

    def get_object(self, Bucket, Key):  # noqa: N803, ARG002 — boto3 signature
        self.calls.append(("get", Key))
        if self._get_error is not None:
            raise self._get_error
        return {"Body": io.BytesIO(b"text")}

    def head_object(self, Bucket, Key):  # noqa: N803, ARG002 — boto3 signature
        self.calls.append(("head", Key))
        raise _S3Error("NoSuchKey")


_NO_OBJECT = [
    "rag://legacy-c265705db359abbcbee7a4d38394fba3",
    "rag://doc-50c4e6-b34e48",
    "loop:l1#iteration-2",
]


@pytest.mark.parametrize("ref", _NO_OBJECT)
def test_a_reference_that_names_no_object_is_never_fetched(ref, caplog):
    s3 = _S3()
    with caplog.at_level(logging.INFO, logger=ka.logger.name):
        assert ka._fetch_artifact_bytes(s3, "b", ref) is None
    assert s3.calls == []
    assert not caplog.records


def test_a_gone_object_is_content_missing_not_a_store_warning(caplog):
    s3 = _S3(get_error=_S3Error("NoSuchKey"))
    with caplog.at_level(logging.INFO, logger=ka.logger.name):
        assert ka._fetch_artifact_bytes(s3, "b", "projects/p/knowledge/artifacts/abc") is None
    assert [r.levelno for r in caplog.records] == [logging.INFO]
    assert "document content missing" in caplog.records[0].getMessage()


def test_a_store_fault_still_warns(caplog):
    s3 = _S3(get_error=_S3Error("AccessDenied"))
    with caplog.at_level(logging.INFO, logger=ka.logger.name):
        assert ka._fetch_artifact_bytes(s3, "b", "projects/p/knowledge/artifacts/abc") is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


@pytest.mark.parametrize("ref", _NO_OBJECT)
def test_embed_types_a_row_without_bytes_as_permanently_missing(ref):
    s3 = _S3()
    result = ka.KnowledgeEmbedResult()
    src = {"artifact_id": "art-1", "kind": "document", "source_ref": ref}
    assert ka._artifact_text(src, s3, "b", result) is None
    (failure,) = result.failed_items
    assert failure["code"] == "content_missing"
    assert failure["permanent"] is True  # the reindex pass passes over it
    assert failure["error"].startswith("document content missing: ")
    assert s3.calls == []


def _understand(monkeypatch, src):
    async def _no(*_a):
        return ""

    async def _none(*_a):
        return None

    monkeypatch.setattr(ka, "_artifact_row_absent", _no)
    monkeypatch.setattr(ka, "_existing_current_ir", _none)
    job = SimpleNamespace(spec={"project_id": "p"}, job_id="job-1", project_id="p")
    attempt = SimpleNamespace(attempt_id="att-1")
    result = ka.KnowledgeUnderstandResult()
    s3 = _S3(get_error=_S3Error("NoSuchKey"))
    out = asyncio.run(ka._understand_input(job, attempt, object(), s3, "b", src, result))
    assert out is None
    return result, s3


@pytest.mark.parametrize("ref", _NO_OBJECT)
def test_understand_counts_a_row_without_bytes_instead_of_failing_it(monkeypatch, ref):
    result, s3 = _understand(monkeypatch, {"artifact_id": "a1", "source_ref": ref})
    assert result.no_object == 1
    assert result.failed_items == []
    assert s3.calls == []


def test_understand_types_a_gone_object(monkeypatch):
    ref = "projects/p/knowledge/artifacts/abc"
    result, _s3 = _understand(monkeypatch, {"artifact_id": "a1", "source_ref": ref})
    (failure,) = result.failed_items
    assert failure["code"] == "content_missing"
    assert failure["error"] == f"document content missing: the stored object {ref} is gone"


def test_an_empty_reference_is_a_broken_row_not_a_skip(monkeypatch):
    result, _s3 = _understand(monkeypatch, {"artifact_id": "a1", "source_ref": ""})
    assert result.no_object == 0
    (failure,) = result.failed_items
    assert failure["error"] == "document content missing: the row names no source_ref"


def test_an_understand_pass_over_rows_without_bytes_succeeds():
    checker = ka.KnowledgeUnderstandStatusChecker()
    attempt = SimpleNamespace(
        backend_ref="knowledge_understand|project=p|ir_ids=0|already=0|fail=0|no_object=211",
        metrics={},
    )
    outcome = asyncio.run(checker.check_status(attempt, SimpleNamespace()))
    assert outcome.status == "succeeded"
