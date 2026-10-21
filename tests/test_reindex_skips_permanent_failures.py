"""A reindex pass passes over artifacts the embed job failed for good.

ADR-093 D-11 (2026-10-11, eval Phase 13): every KNOWLEDGE_REINDEX pass on
the stand-alone stack re-sent the same 9 never-embeddable artifacts (images
on a text-only site, loop artifacts whose objects are gone, a note with no
file node), and their embed job failed by name each time, so "0 stale" was
never reached. The embed job now records a permanent failure on the
artifact row under the model and scheme that failed it; the sweep skips it
while those still hold, counts it as ``failed_skipped``, and a successful
stamp or ``retry_failed`` brings it back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs import reindex
from umrs.knowledge import embedding_metadata as em

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 64
SKIP = "NOT (COALESCE(metadata->'embed_failed'->>'model', '') = %s"


class _Cur:
    def __init__(self, rows=(), one=(0,)):
        self.statements: list[tuple[str, Any]] = []
        self._rows, self._one = list(rows), one

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._one


def _conn(cur):
    return SimpleNamespace(cursor=lambda: cur)


# ── the sweep ───────────────────────────────────────────────────────────────


def test_the_sweep_skips_what_this_model_and_scheme_failed():
    cur = _Cur([("art-2",)])
    ids = em.select_stale_artifacts(
        _conn(cur), "m", limit=9, chunk_scheme="sent-v1", project_id="p1", skip_failed=True
    )
    ((sql, params),) = cur.statements
    assert ids == ["art-2"]
    assert SKIP in sql
    assert params[-3:] == ("m", "sent-v1", 9)


def test_an_unschemed_sweep_matches_failures_recorded_under_fixed_v0():
    cur = _Cur()
    em.select_stale_artifacts(_conn(cur), "m", skip_failed=True)
    assert cur.statements[0][1][-3:] == ("m", "fixed-v0", 500)


def test_without_skip_failed_the_sweep_is_unchanged():
    cur = _Cur()
    em.select_stale_artifacts(_conn(cur), "m", chunk_scheme="sent-v1", project_id="p1")
    assert SKIP not in cur.statements[0][0]


def test_the_skipped_ones_are_counted_not_dropped():
    cur = _Cur(one=(9,))
    n = em.count_failed_artifacts(_conn(cur), "m", chunk_scheme="fixed-v0", project_id="p1")
    ((sql, params),) = cur.statements
    assert n == 9
    assert sql.startswith("SELECT count(*) FROM knowledge_artifacts WHERE status = 'active'")
    assert "AND (COALESCE(metadata->'embed_failed'->>'model', '') = %s" in sql
    assert SKIP not in sql
    assert params[-2:] == ("m", "fixed-v0") and "p1" in params


def test_a_successful_stamp_clears_the_record():
    assert "- 'embed_failed'" in em._STAMP_SQL
    assert "- 'embed_failed'" in em._STAMP_CHUNKS_SQL


# ── the reindex job ─────────────────────────────────────────────────────────


def _spec(**extra) -> dict:
    base = {"target_model": "m", "index_name": "i", "database": "d", "table": "t"}
    return reindex.validate_knowledge_reindex_spec({**base, **extra})


def test_retry_failed_defaults_to_false_and_must_be_a_bool():
    assert _spec()["retry_failed"] is False
    assert _spec(retry_failed=True)["retry_failed"] is True
    with pytest.raises(em.EmbeddingMetadataError, match="retry_failed"):
        _spec(retry_failed="yes")


@pytest.mark.parametrize(("retry", "skipped"), [(False, True), (True, False)])
def test_the_pass_skips_failures_unless_asked_to_retry(retry, skipped):
    cur = _Cur()
    spec = {**_spec(retry_failed=retry), "chunk_scheme": "fixed-v0", "project_id": "p1"}
    reindex._sweep(_conn(cur), spec, None)
    assert (SKIP in cur.statements[0][0]) is skipped


def test_the_pass_reports_what_it_skipped():
    spec = {**_spec(), "chunk_scheme": "fixed-v0", "project_id": "p1"}
    assert reindex._failed_skipped(_conn(_Cur(one=(9,))), spec) == 9
    assert reindex._failed_skipped(_conn(_Cur(one=(9,))), {**spec, "retry_failed": True}) == 0
    ref = reindex._backend_ref(spec, reindex._PassTotals(failed_skipped=9))
    assert ref.endswith("|failed_skipped=9")


# ── the embed job classifies its failures ───────────────────────────────────


class _S3Error(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class _S3:
    def __init__(self, head_error: Exception | None) -> None:
        self._head_error = head_error

    def head_object(self, **_kw):
        if self._head_error is not None:
            raise self._head_error
        return {}


def _text(monkeypatch, content, s3=None, src=None):
    read = None if content is None else ka._Bounded(content, len(content))
    monkeypatch.setattr(ka, "_read_bounded", lambda *_a, **_k: read)
    result = ka.KnowledgeEmbedResult()
    src = src or {"artifact_id": "art-1", "kind": "document", "source_ref": "k/a.md"}
    assert ka._artifact_text(src, s3, "b", result) is None
    return result.failed_items[0]


def test_content_with_no_embedder_is_a_permanent_failure(monkeypatch):
    failure = _text(monkeypatch, _PNG, src={"artifact_id": "art-1", "kind": "image"})
    assert failure["permanent"] is True


@pytest.mark.parametrize("code", ["NoSuchKey", "404", "NotFound", "NoSuchBucket"])
def test_an_object_that_is_gone_is_a_permanent_failure(monkeypatch, code):
    failure = _text(monkeypatch, None, s3=_S3(_S3Error(code)))
    assert failure == {
        "artifact_id": "art-1",
        "error": "document content missing: the stored object k/a.md is gone",
        "code": "content_missing",
        "permanent": True,
    }


@pytest.mark.parametrize(
    "s3",
    [None, _S3(None), _S3(_S3Error("AccessDenied")), _S3(ConnectionError("down"))],
    ids=["no-client", "object-exists", "denied", "unreachable"],
)
def test_an_unreadable_object_that_may_come_back_is_retried(monkeypatch, s3):
    failure = _text(monkeypatch, None, s3=s3)
    assert failure["error"] == "artifact content unavailable"
    assert failure["permanent"] is False


def test_the_gone_probe_reads_an_s3_uri_reference():
    seen = {}

    class _Probe(_S3):
        def head_object(self, **kw):
            seen.update(kw)
            raise _S3Error("NoSuchKey")

    assert ka._object_gone(_Probe(None), "default-bucket", "s3://other/k/a.md")
    assert seen == {"Bucket": "other", "Key": "k/a.md"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "permanent"),
    [
        (ka._NoFileNode("no single file node", permanent=True), True),
        (ka._NoFileNode("file node unreadable"), False),
        (RuntimeError("pool exhausted"), False),
    ],
)
async def test_a_write_failure_is_permanent_only_when_reingest_repairs_it(
    monkeypatch, error, permanent
):
    async def fail(*_a, **_k):
        raise error

    monkeypatch.setattr(ka, "_write_embedded_chunks", fail)
    result = ka.KnowledgeEmbedResult()
    await ka._write_isolated(None, {"artifact_id": "art-1"}, "p", "g", {}, result)
    assert result.failed_items == [
        {"artifact_id": "art-1", "error": str(error), "permanent": permanent}
    ]


# ── the record ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_only_permanent_failures_are_recorded_under_the_site_model_and_scheme(
    monkeypatch,
):
    seen: dict[str, Any] = {}

    async def record(pool, failures, *, model, scheme):
        seen.update(pool=pool, failures=failures, model=model, scheme=scheme)
        return len(failures)

    monkeypatch.setattr(em, "record_embed_failures", record)
    monkeypatch.setattr(ka, "configured_text_embedder", lambda: "/opt/models/e5-base-v2")
    result = ka.KnowledgeEmbedResult(
        failed_items=[
            {"artifact_id": "art-1", "error": "gone", "permanent": True},
            {"artifact_id": "art-2", "error": "pool exhausted", "permanent": False},
            {"artifact_id": "art-3", "error": "batch failed"},
        ]
    )
    await ka._record_permanent_failures("pool", result, SimpleNamespace(scheme="sent-v1"))
    assert seen == {
        "pool": "pool",
        "failures": [("art-1", "gone")],
        "model": "/opt/models/e5-base-v2",
        "scheme": "sent-v1",
    }


@pytest.mark.asyncio
async def test_a_failed_record_never_fails_the_job(monkeypatch):
    async def record(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(em, "record_embed_failures", record)
    result = ka.KnowledgeEmbedResult(
        failed_items=[{"artifact_id": "art-1", "error": "gone", "permanent": True}]
    )
    await ka._record_permanent_failures("pool", result, SimpleNamespace(scheme="fixed-v0"))


class _Pool:
    def __init__(self):
        self.calls: list[tuple[str, list]] = []

    def acquire(self):
        pool = self

        class _Ctx:
            async def __aenter__(self):
                return SimpleNamespace(executemany=pool._executemany)

            async def __aexit__(self, *_a):
                return False

        return _Ctx()

    async def _executemany(self, sql, rows):
        self.calls.append((sql, rows))


@pytest.mark.asyncio
async def test_the_record_names_model_scheme_and_a_bounded_reason():
    pool = _Pool()
    n = await em.record_embed_failures(
        pool, [("art-1", "x" * 1000), ("", "no id")], model="m", scheme="fixed-v0"
    )
    ((sql, rows),) = pool.calls
    assert n == 1 and "'embed_failed'" in sql
    assert rows == [("m", "fixed-v0", "x" * em._FAILED_REASON_MAX, "art-1")]


@pytest.mark.asyncio
async def test_nothing_to_record_touches_no_connection():
    pool = _Pool()
    assert await em.record_embed_failures(pool, [], model="m", scheme="fixed-v0") == 0
    assert pool.calls == []


class _Lookup:
    """The file-node lookup: ``own`` answers the writer's own id, ``rows``
    the artifact's file nodes, ``error`` makes the read fail."""

    def __init__(self, *, rows=(), error: Exception | None = None) -> None:
        self._rows, self._error = list(rows), error

    async def fetchval(self, *_a):
        if self._error is not None:
            raise self._error
        return None

    async def fetch(self, *_a):
        return self._rows


def _target() -> Any:
    return ka._ChunkTarget("art-1", "p", "g-1", "g-1-", "fixed-v0")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows",
    [[], [("node-file-a-art-1",), ("node-file-b-art-1",)]],
    ids=["none", "several"],
)
async def test_no_single_file_node_is_permanent(rows):
    with pytest.raises(ka._NoFileNode) as caught:
        await ka._resolve_target(_Lookup(rows=rows), _target())
    assert caught.value.permanent is True


@pytest.mark.asyncio
async def test_an_unreadable_file_node_is_transient():
    with pytest.raises(ka._NoFileNode) as caught:
        await ka._resolve_target(_Lookup(error=RuntimeError("conn reset")), _target())
    assert caught.value.permanent is False
