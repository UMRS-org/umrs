"""Cycle-3 mem lane, P1 (2026-10-22): the knowledge chain never reads a whole
stored object it does not use.

Found live (lane stack, ``.stab-cycle3/mem/repro/p1_ingest_rss.py``): a 400
MiB text through POST /umrs/knowledge/upload took UMRS peak RSS from 180 MB
to 2783 MB in the ingest and understand jobs. ``_fetch_artifact_bytes`` read
every object whole (``get_object()["Body"].read()``) for each consumer:

- ingest held the staged upload to hash it, stamp its canonical identity and
  ``put_object`` it again under the hash;
- understand parsed all of it, though a prose artifact's excerpt and LLM
  extraction read 9000 characters at most;
- embed chunked and embedded every chunk and kept the first 128 (8 MiB of
  text embedded 8656 chunks);
- retrieval read the artifact whole to slice one passage.

Now ingest streams it once in parts (hash, ``IdentityStream``, the 8 KiB head)
and copies it server side, conditional on the ETag it hashed; understand reads
whole up to ``READ_CAP`` and a longer text's first MiB (saying so); embed and
retrieval read a text's first ``_embed_read_bytes()``.
"""

from __future__ import annotations

import hashlib
import io
import random
from types import SimpleNamespace
from typing import Any

import pytest
from botocore.exceptions import ClientError
from tests.s3_ranged import check_copy_precondition, etag_of, ranged
from umrs.api import rag_spine
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import canonical_identity

_STAGED = ("r4-data", "knowledge-uploads/p/u/big.md")
_URI = f"s3://{_STAGED[0]}/{_STAGED[1]}"


class _Body:
    def __init__(self, data: bytes, reads: list[int]) -> None:
        self._buf = io.BytesIO(data)
        self._reads = reads

    def read(self, size: int = -1) -> bytes:
        out = self._buf.read(size)
        self._reads.append(len(out))
        return out


class _S3:
    """An object store that records every read, ranged GETs and copies."""

    exceptions = SimpleNamespace(ClientError=ClientError)

    def __init__(self, objects: dict[tuple[str, str], bytes]) -> None:
        self.objects = dict(objects)
        self.reads: list[int] = []
        self.copies: list[dict[str, Any]] = []
        self.puts: list[str] = []
        self.copy_error: Exception | None = None
        self.replaced_after_read: bytes | None = None  # another upload lands

    def get_object(self, Bucket: str, Key: str, Range: str = "") -> dict[str, Any]:  # noqa: N803
        answer = ranged(self.objects[(Bucket, Key)], Range)
        answer["Body"] = _Body(answer["Body"].read(), self.reads)
        if self.replaced_after_read is not None and not Range:
            self.objects[(Bucket, Key)] = self.replaced_after_read
        return answer

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def copy_object(self, Bucket: str, Key: str, CopySource: dict, **kw: Any) -> None:  # noqa: N803
        if self.copy_error is not None:
            raise self.copy_error
        source = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.copies.append({"to": (Bucket, Key), "from": CopySource, **kw})
        check_copy_precondition(source, kw)
        self.objects[(Bucket, Key)] = source

    def put_object(self, Bucket: str, Key: str, Body: bytes, **_kw: Any) -> None:  # noqa: N803
        self.puts.append(Key)
        self.objects[(Bucket, Key)] = Body


def _big_text(parts: float) -> bytes:
    line = "Graphene oxide membranes reject salt ions, café.  \r\n".encode()
    return line * int(parts * ka._STREAM_PART_BYTES / len(line))


def _ingest(s3: _S3, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict]]:
    monkeypatch.setattr(ka, "_bridge_a2l_ref", lambda *_a, **_k: None)
    result = ka.KnowledgeIngestResult()
    descs: list[dict] = []
    ka._ingest_one_source(
        SimpleNamespace(job_id="j", spec={}),
        SimpleNamespace(attempt_id="a"),
        s3,
        "kb",
        "p",
        "s3_uri",
        _URI,
        [],
        result,
        [],
        descs,
    )
    return result, descs


# ── ingest ──────────────────────────────────────────────────────────────────


def _ingested_big(monkeypatch: pytest.MonkeyPatch) -> tuple[_S3, bytes, Any, list[dict]]:
    body = _big_text(3.5)
    s3 = _S3({_STAGED: body})
    result, descs = _ingest(s3, monkeypatch)
    return s3, body, result, descs


def test_ingest_reads_a_staged_upload_in_parts_never_whole(monkeypatch):
    s3, body, result, descs = _ingested_big(monkeypatch)
    assert result.failed_items == []
    assert descs[0]["content_hash"] == hashlib.sha256(body).hexdigest()
    assert max(s3.reads) <= ka._STREAM_PART_BYTES < len(body)


def test_ingest_copies_the_staged_upload_server_side_on_its_etag(monkeypatch):
    s3, body, _result, _descs = _ingested_big(monkeypatch)
    digest = hashlib.sha256(body).hexdigest()
    assert s3.puts == []
    (copy,) = s3.copies
    assert copy["to"] == ("kb", f"projects/p/knowledge/artifacts/{digest}")
    assert copy["from"] == {"Bucket": _STAGED[0], "Key": _STAGED[1]}
    assert copy["CopySourceIfMatch"] == etag_of(body)
    assert s3.objects[copy["to"]] == body


def test_ingest_stamps_the_identity_of_every_byte(monkeypatch):
    _s3, body, _result, descs = _ingested_big(monkeypatch)
    meta = descs[0]["metadata"]
    whole = canonical_identity.identity_of(body, hashlib.sha256(body).hexdigest())
    assert meta["canonical_identity"] == whole.to_dict()
    assert meta["source_name"] == "big.md"
    assert meta["staged_from"] == _URI
    assert descs[0]["kind"] == "document"


def test_an_upload_replaced_while_it_was_ingested_is_not_stored(monkeypatch):
    """The store refuses the conditional copy itself (412), as SeaweedFS
    does (test_s3_fake_parity.py): nothing is injected."""
    s3 = _S3({_STAGED: b"# note\n"})
    s3.replaced_after_read = b"# another note\n"
    result, descs = _ingest(s3, monkeypatch)
    assert descs == []
    (failed,) = result.failed_items
    assert "changed while it was ingested" in failed["error"]
    assert set(s3.objects) == {_STAGED}  # nothing stored under the hash


def test_a_ranged_read_of_an_empty_object_is_empty_not_unreadable():
    """S3 answers a range over an empty object 416 InvalidRange."""
    s3, src = _stored(b"")
    assert ka._fetch_artifact_prefix(s3, "kb", src["source_ref"], 1024) == (b"", 0)
    content, result = _understand_read(s3, src)
    assert content == b"" and result.failed_items == []


def test_any_other_copy_fault_still_raises(monkeypatch):
    s3 = _S3({_STAGED: b"# note\n"})
    s3.copy_error = ClientError({"Error": {"Code": "SlowDown"}}, "CopyObject")
    with pytest.raises(ClientError):
        _ingest(s3, monkeypatch)


# ── the streamed canonical identity ────────────────────────────────────────


_PIECES = [
    "a",
    " ",
    "\t",
    "\r",
    "\n",
    "\r\n",
    "e\u0301",
    "\u0301",
    "\u00e9",
    "\ufeff",
    "\u1100",
    "\u1161",
    "\u11a8",
    "\u65e5",
    "\U0001f600",
    "xyzzy",
    "  \n",
    "\u212b",
]
_RAW = [b"\x00", b"\xff", b"\xef\xbb\xbf", b"\xc3", b"\xa9", b"\xe6\x97"]


def _sample(rng: random.Random) -> bytes:
    parts = [
        rng.choice(_PIECES).encode() if rng.random() < 0.93 else rng.choice(_RAW)
        for _ in range(rng.randint(0, 30))
    ]
    lead = [b"\xef\xbb\xbf"] if rng.random() < 0.3 else []
    return b"".join(lead + parts)


def _streamed(raw: bytes, rng: random.Random) -> canonical_identity.CanonicalIdentity:
    stream = canonical_identity.IdentityStream()
    i = 0
    while i < len(raw):
        size = rng.randint(1, 7)
        stream.feed(raw[i : i + size])
        i += size
    return stream.identity()


def test_the_streamed_identity_is_the_identity(monkeypatch):
    """Parts split CRLF pairs, combining marks, BOMs and UTF-8 sequences;
    a 4-character flush exercises the long-line path."""
    monkeypatch.setattr(canonical_identity, "_FLUSH_CHARS", 4)
    rng = random.Random(20261022)
    for _ in range(3000):
        raw = _sample(rng)
        assert _streamed(raw, rng) == canonical_identity.identity_of(raw, "p"), raw


@pytest.mark.parametrize(
    "raw",
    [b"", b"\xef\xbb", b"\xef\xbb\xbf", b"\n\n", b"a  \n\n", b"x\r", b"\x00", b"a" * 50],
)
def test_the_streamed_identity_of_edge_inputs(raw, monkeypatch):
    monkeypatch.setattr(canonical_identity, "_FLUSH_CHARS", 4)
    assert _streamed(raw, random.Random(1)) == canonical_identity.identity_of(raw, "p")


# ── understand ──────────────────────────────────────────────────────────────


def _understand_read(s3: _S3, src: dict[str, Any]) -> tuple[bytes | None, Any]:
    result = ka.KnowledgeUnderstandResult()
    return ka._understand_bytes(s3, "kb", src, result), result


def _stored(data: bytes) -> tuple[_S3, dict[str, Any]]:
    key = ("kb", "projects/p/knowledge/artifacts/h")
    src = {"artifact_id": "art-1", "source_ref": key[1], "metadata": {"source_name": "n.md"}}
    return _S3({key: data}), src


def test_understand_reads_a_text_within_read_cap_whole(monkeypatch):
    monkeypatch.setattr(ka, "_UNDERSTAND_PREFIX_BYTES", 1024)
    data = b"line\n" * 1000
    s3, src = _stored(data)
    content, result = _understand_read(s3, src)
    assert content == data and not result.degraded


def test_understand_reads_a_text_past_read_cap_only_to_its_prefix(monkeypatch):
    monkeypatch.setattr(ka, "_UNDERSTAND_PREFIX_BYTES", 1024)
    monkeypatch.setattr(canonical_identity, "READ_CAP", 4096)
    data = "café line\n".encode() * 1000
    s3, src = _stored(data)
    content, result = _understand_read(s3, src)
    assert content is not None and data.startswith(content)
    assert len(content) <= 1024 and sum(s3.reads) <= 1025
    content.decode("utf-8")  # the cut never splits a character
    assert not result.degraded  # per artifact (cycle-3 memfix #5)
    (partial,) = result.partial_reads
    assert partial["artifact_id"] == "art-1"
    assert "understood from its first" in partial["reason"]


def test_understand_refuses_a_pdf_past_the_spool_cap_by_name(monkeypatch):
    """Past READ_CAP a document is extracted from disk (memfix #2,
    test_memfix_large_documents.py); past the spool cap it fails by name,
    after a ranged read only."""
    monkeypatch.setattr(ka, "_UNDERSTAND_PREFIX_BYTES", 1024)
    monkeypatch.setattr(canonical_identity, "READ_CAP", 4096)
    monkeypatch.setattr(ka, "_DOCUMENT_SPOOL_MAX", 8192)
    s3, src = _stored(b"%PDF-1.7\n" + b"0" * 9000)
    content, result = _understand_read(s3, src)
    assert content is None and sum(s3.reads) <= 1025
    assert "too large to extract" in result.failed_items[0]["error"]


# ── embed and retrieval ─────────────────────────────────────────────────────


def test_embed_reads_a_text_only_to_the_embed_budget(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS", "1")
    budget = ka._embed_read_bytes()
    data = b"word " * (budget // 2)
    s3, src = _stored(data)
    result = ka.KnowledgeEmbedResult()
    text = ka._artifact_text(src, s3, "kb", result)
    assert text is not None and len(text) <= budget < len(data)
    assert sum(s3.reads) <= budget + 1
    assert result.cut_texts == {"art-1"}


def test_a_cut_text_never_writes_its_last_chunks_and_reports_the_truncation():
    chunks, vectors = list(range(10)), list(range(10))
    pairs, total = ka._kept_chunks(chunks, vectors, cap=128, cut=True)
    assert [c for c, _ in pairs] == list(range(8))
    assert total > len(pairs)
    pairs, total = ka._kept_chunks(chunks, vectors, cap=128, cut=False)
    assert len(pairs) == 10 and total == 10


def test_retrieval_reads_an_artifact_only_to_the_embed_budget(monkeypatch):
    budget = ka._embed_read_bytes()
    data = b"word " * budget
    s3, src = _stored(data)
    monkeypatch.setattr(ka, "_build_s3_client", lambda _spec: (s3, "kb"))
    got = rag_spine._artifact_fetcher()(src["source_ref"])
    assert got is not None and data.startswith(got) and len(got) == budget
    assert sum(s3.reads) <= budget + 1
