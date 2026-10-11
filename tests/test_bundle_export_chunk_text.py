"""ADR-093 D-12 (C14): portable chunk text by explicit opt-in, r4-kb v1.3.

Export: without ``chunk_text`` the bundle keeps the v1.2 shape (excerpts
only); with it every chunk of an active artifact carries its md5-verified
body (stored ``search_text``, else the artifact slice; at most 4,096 chars)
and the manifest becomes v1.3. A project whose class is not
``non_sensitive`` (unset included) exports it only on its owner's flag, and
the refusal happens before anything is written (MD-5).

Import: ``search_text`` is filled from the node text when its md5 matches,
else from an accepted raw object's verified slice, else left NULL with the
backfill named. API: the flag is gated (owner only on a sensitive project),
audited and forwarded to the job.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _extracted_documents import chunk_content, embed_text, pdf_bytes
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.knowledge import bundle_raw
from umrs.knowledge.bundle_export import (
    ChunkTextRefused,
    KnowledgeExportEngine,
    verify_bundle_integrity,
)
from umrs.knowledge.bundle_import import BundleFormatError, KnowledgeImportEngine
from umrs.knowledge.bundle_reader import CHUNK_TEXT_MAX_CHARS, chunk_passage
from umrs.knowledge.store import KnowledgeArtifact, KnowledgeNode

from r4_contracts.errors import ERROR_CODE_HEADER

_DOC_A = "Interleukin six rises after exercise. It falls during rest periods."
_DOC_B = "TiO2 photocatalysis degrades dyes. Rates double under UV light."
_REF_A = "projects/p/knowledge/artifacts/aaa"
_REF_B = "projects/p/knowledge/artifacts/bbb"
_OWNER = ("alice", "non_sensitive")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _chunk(node_id: str, artifact: str, doc: str, start: int, end: int, **extra):
    body = doc[start:end]
    content = {
        "kind": "chunk",
        "artifact_id": artifact,
        "chunk_index": 0,
        "chunk_start": start,
        "chunk_end": end,
        "chunk_md5": _md5(body),
        "excerpt": body[:10],
        **extra,
    }
    return KnowledgeNode(
        node_id=node_id,
        project_id="p",
        label=node_id,
        node_type="entity",
        content=content,
        content_hash=_md5(body),
    )


def _artifact(artifact_id: str, ref: str, doc: str) -> KnowledgeArtifact:
    raw = doc.encode()
    return KnowledgeArtifact(
        artifact_id=artifact_id,
        content_hash=_sha(raw),
        kind="document",
        project_id="p",
        source_ref=ref,
        size_bytes=len(raw),
        mime_type="text/plain",
    )


# ── export fakes ────────────────────────────────────────────────────────────


class _ExportPool:
    """asyncpg-shaped reads of the chunk-text inputs and the project facts.
    ``refs``: artifact_id -> object key, or (object key, source name)."""

    def __init__(self, search_texts, refs, facts):
        self.search_texts, self.refs, self.facts = search_texts, refs, facts
        self.queries: list[str] = []

    async def fetch(self, query, *_args):
        self.queries.append(query)
        if "search_text IS NOT NULL" in query:
            return list(self.search_texts.items())
        if "SELECT artifact_id, source_ref" in query:
            return [(a, *(r if isinstance(r, tuple) else (r, ""))) for a, r in self.refs.items()]
        return []

    async def fetchrow(self, query, *_args):
        self.queries.append(query)
        if "FROM projects" not in query:
            return None
        if isinstance(self.facts, Exception):
            raise self.facts
        return (
            None
            if self.facts is None
            else dict(zip(("owner_id", "data_class"), self.facts, strict=True))
        )

    def acquire(self):
        raise RuntimeError("no connection in this fake")

    def close(self):
        return None

    async def wait_closed(self):
        return None


class _ExportStore:
    def __init__(self, pool, nodes, artifacts):
        self._pool, self._nodes, self._artifacts = pool, nodes, artifacts

    async def list_artifacts(self, **_kw):
        return self._artifacts, len(self._artifacts)

    async def list_irs(self, **_kw):
        return []

    async def list_nodes(self, **_kw):
        return self._nodes

    async def list_edges(self, **_kw):
        return []


def _corpus():
    """Chunks: a1 stored text verifies; a2 stored text is stale (slice
    verifies); b1 has neither; z1's artifact is not active; one plain node."""
    nodes = [
        _chunk("n-a1", "art-a", _DOC_A, 0, 37),
        _chunk("n-a2", "art-a", _DOC_A, 38, len(_DOC_A)),
        _chunk("n-b1", "art-b", _DOC_B, 0, 34),
        _chunk("n-z1", "art-gone", _DOC_B, 0, 10),
        KnowledgeNode(node_id="n-x", project_id="p", label="X", node_type="class"),
    ]
    search_texts = {"n-a1": _DOC_A[0:37], "n-a2": "stale text", "n-z1": _DOC_B[0:10]}
    refs = {"art-a": _REF_A, "art-b": _REF_B}
    return nodes, search_texts, refs


async def _export(
    tmp_path: Path,
    monkeypatch,
    *,
    chunk_text: bool,
    facts=_OWNER,
    submitter="",
    blobs=None,
    corpus=None,
    secret="",
):
    nodes, search_texts, refs = corpus or _corpus()
    monkeypatch.setattr(KnowledgeExportEngine, "_resolve_export_snapshot", lambda *_a: None)
    pool = _ExportPool(search_texts, refs, facts)
    artifacts = [_artifact("art-a", _REF_A, _DOC_A), _artifact("art-b", _REF_B, _DOC_B)]
    engine = KnowledgeExportEngine(
        store=_ExportStore(pool, nodes, artifacts),
        output_dir=tmp_path,
        secret_key=secret,
        allow_unsigned=not secret,
        artifact_reader=(blobs if blobs is not None else {_REF_A: _DOC_A.encode()}).get,
        chunk_text=chunk_text,
        chunk_text_submitter=submitter,
    )
    await engine.export_project("p")
    return tmp_path / "p", pool


def _rows(bundle: Path, name: str = "nodes.jsonl") -> dict[str, dict]:
    lines = (bundle / name).read_text().splitlines()
    return {row["node_id"]: row for row in map(json.loads, lines)}


def _manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text())


# ── export ──────────────────────────────────────────────────────────────────


def _carried_keys(rows: dict[str, dict]) -> set[str]:
    return {key for row in rows.values() for key in row} & {"text", "text_header"}


async def test_without_the_flag_the_bundle_keeps_the_v1_2_shape(tmp_path, monkeypatch):
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=False)
    rows, manifest = _rows(bundle), _manifest(bundle)
    assert _carried_keys(rows) == set()
    assert (manifest["version"], "chunk_text" in manifest) == ("1.2", False)
    assert rows["n-a1"]["content"]["excerpt"] == _DOC_A[:10]


async def test_without_the_flag_no_gate_or_text_read_runs(tmp_path, monkeypatch):
    _bundle_dir, pool = await _export(tmp_path, monkeypatch, chunk_text=False)
    touched = {q for q in pool.queries if "FROM projects" in q or "search_text" in q}
    assert touched == set()


async def test_chunk_text_carries_verified_bodies_and_v1_3(tmp_path, monkeypatch):
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True)
    rows, manifest = _rows(bundle), _manifest(bundle)
    assert rows["n-a1"]["text"] == _DOC_A[0:37]  # stored search_text, md5 ok
    assert rows["n-a2"]["text"] == _DOC_A[38:]  # stale search_text -> slice
    # b1: artifact unreadable (excerpt only); z1: artifact not active; x: no chunk
    assert [n for n in ("n-b1", "n-z1", "n-x") if "text" in rows[n]] == []
    assert (manifest["version"], manifest["chunk_text"]) == ("1.3", True)
    summary = manifest["chunk_text_summary"]
    assert (summary["chunks"], summary["search_text"], summary["artifact"]) == (4, 1, 1)
    assert (summary["missing"], summary["no_artifact"], summary["max_chars"]) == (1, 1, 4096)


async def test_long_bodies_are_cut_and_flagged(tmp_path, monkeypatch):
    doc = "x" * (CHUNK_TEXT_MAX_CHARS + 100)
    corpus = (
        [_chunk("n-long", "art-a", doc, 0, len(doc))],
        {"n-long": doc},
        {"art-a": _REF_A},
    )
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True, corpus=corpus)
    row = _rows(bundle)["n-long"]
    assert (len(row["text"]), row["text_truncated"]) == (CHUNK_TEXT_MAX_CHARS, True)
    assert _manifest(bundle)["chunk_text_summary"]["truncated"] == 1


class _CountingBlobs(dict):
    """An artifact store that counts reads per object key."""

    def __init__(self, *args):
        super().__init__(*args)
        self.reads: dict[str, int] = {}

    def get(self, key, default=None):
        self.reads[key] = self.reads.get(key, 0) + 1
        return super().get(key, default)


def _interleaved_corpus(n: int):
    """``n`` artifacts (more than the slice cache holds), two chunks each,
    written first-chunks-then-second-chunks so artifacts interleave."""
    docs = {f"art-{i}": f"Document {i:02d} opens. It closes {i:02d}." for i in range(n)}
    firsts = [_chunk(f"n-{a}-0", a, d, 0, 16) for a, d in docs.items()]
    seconds = [_chunk(f"n-{a}-1", a, d, 17, len(d)) for a, d in docs.items()]
    refs = {a: f"refs/{a}" for a in docs}
    blobs = _CountingBlobs({refs[a]: d.encode() for a, d in docs.items()})
    return (firsts + seconds, {}, refs), blobs


async def test_each_artifact_is_read_once_however_its_chunks_interleave(tmp_path, monkeypatch):
    corpus, blobs = _interleaved_corpus(10)
    bundle, _pool = await _export(
        tmp_path, monkeypatch, chunk_text=True, corpus=corpus, blobs=blobs
    )
    assert set(blobs.reads.values()) == {1}
    assert _manifest(bundle)["chunk_text_summary"]["artifact"] == 20
    order = list(_rows(bundle))
    assert order[:2] == ["n-art-0-0", "n-art-1-0"]  # nodes.jsonl order kept


async def test_a_headed_chunk_carries_its_body_and_exact_header(tmp_path, monkeypatch):
    header = "Paper A › Results\n"
    node = _chunk("n-h", "art-a", _DOC_A, 0, 37, header_chars=len(header))
    corpus = ([node], {"n-h": header + _DOC_A[0:37]}, {"art-a": _REF_A})
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True, corpus=corpus)
    row = _rows(bundle)["n-h"]
    assert (row["text"], row["text_header"]) == (_DOC_A[0:37], header)


@pytest.mark.parametrize(
    "facts",
    [("alice", "sensitive"), ("alice", ""), None, RuntimeError("registry down")],
)
async def test_sensitive_unset_or_unknown_class_refuses_before_writing(
    tmp_path, monkeypatch, facts
):
    with pytest.raises(ChunkTextRefused):
        await _export(tmp_path, monkeypatch, chunk_text=True, facts=facts)
    assert not (tmp_path / "p").exists()


@pytest.mark.parametrize("submitter", ["bob", ""])
async def test_a_non_owner_submitter_does_not_unlock_a_sensitive_project(
    tmp_path, monkeypatch, submitter
):
    with pytest.raises(ChunkTextRefused, match="owner"):
        await _export(
            tmp_path,
            monkeypatch,
            chunk_text=True,
            facts=("alice", "sensitive"),
            submitter=submitter,
        )
    assert not (tmp_path / "p").exists()


async def test_the_owner_flag_exports_a_sensitive_project_and_is_recorded(tmp_path, monkeypatch):
    bundle, _pool = await _export(
        tmp_path, monkeypatch, chunk_text=True, facts=("alice", "sensitive"), submitter="alice"
    )
    assert _rows(bundle)["n-a1"]["text"] == _DOC_A[0:37]
    ack = _manifest(bundle)["chunk_text_summary"]["acknowledgement"]
    assert (ack["by"], ack["data_class"]) == ("alice", "sensitive")
    assert ack["verified"] == "job submitter is the project owner"


async def test_a_non_sensitive_project_records_no_acknowledgement(tmp_path, monkeypatch):
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True, submitter="bob")
    assert "acknowledgement" not in _manifest(bundle)["chunk_text_summary"]


# ── the job: the verified submitter, never the spec (review chunk-data-2) ──


def _export_job(user: str) -> SimpleNamespace:
    forged = {"by": "alice", "data_class": "sensitive", "at": "2026-10-10T00:00:00+00:00"}
    spec = {
        "project_id": "p",
        "allow_unsigned": True,
        "chunk_text": True,
        "chunk_text_acknowledgement": forged,
    }
    return SimpleNamespace(job_id="j-x", project_id="p", user_id=user, spec=spec)


async def _run_export_job(tmp_path, monkeypatch, user: str):
    from umrs.jobs import knowledge_adapters as ka

    nodes, search_texts, refs = _corpus()
    pool = _ExportPool(search_texts, refs, ("alice", "sensitive"))
    artifacts = [_artifact("art-a", _REF_A, _DOC_A), _artifact("art-b", _REF_B, _DOC_B)]
    async def created():
        return pool

    monkeypatch.setattr(ka, "_created_pool", created)
    monkeypatch.setattr(ka, "_lazy_artifact_reader", lambda: {_REF_A: _DOC_A.encode()}.get)
    monkeypatch.setattr(
        ka.KnowledgeExportBackend, "_publish_archive", staticmethod(lambda *_a: None)
    )
    monkeypatch.setattr(
        "umrs.knowledge.store.KnowledgeStore", lambda p: _ExportStore(p, nodes, artifacts)
    )
    monkeypatch.setattr(KnowledgeExportEngine, "_resolve_export_snapshot", lambda *_a: None)
    monkeypatch.setenv("UMRS_KB_EXPORT_DIR", str(tmp_path))
    monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    attempt = SimpleNamespace(attempt_id="a-x", metrics={})
    return await ka.KnowledgeExportBackend().submit_job(_export_job(user), attempt)


async def test_a_member_cannot_forge_the_owners_acknowledgement_on_post_jobs(tmp_path, monkeypatch):
    """Mallory (a member, not the owner) put ``{"by": "alice"}`` in the spec
    of a POST /jobs knowledge_export; the engine trusted it and published
    every chunk body of a sensitive project."""
    with pytest.raises(ChunkTextRefused, match="'mallory'"):
        await _run_export_job(tmp_path, monkeypatch, "mallory")
    assert not (tmp_path / "p").exists()


async def test_the_owner_submitting_the_job_exports_the_chunk_text(tmp_path, monkeypatch):
    ref = await _run_export_job(tmp_path, monkeypatch, "alice")
    assert ref.startswith("knowledge_export|project=p|")
    assert _rows(tmp_path / "p")["n-a1"]["text"] == _DOC_A[0:37]


async def test_the_signature_covers_the_chunk_text(tmp_path, monkeypatch):
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True, secret="k")
    assert verify_bundle_integrity(bundle, "k")
    nodes = bundle / "nodes.jsonl"
    nodes.write_text(nodes.read_text().replace("Interleukin", "Interleukon"))
    assert not verify_bundle_integrity(bundle, "k")


# ── reader ──────────────────────────────────────────────────────────────────


def _row(text=None, **extra) -> dict:
    node = _chunk("n", "art-a", _DOC_A, 0, 37)
    row = {"node_id": "n", "content": dict(node.content), **extra}
    if text is not None:
        row["text"] = text
    return row


def test_reader_prefers_the_node_text_then_raw_then_cut_text_then_excerpt():
    def raw(_row):
        return "from raw"

    assert chunk_passage(_row(_DOC_A[0:37]), raw) == (_DOC_A[0:37], "node")
    assert chunk_passage(_row("tampered"), raw) == ("from raw", "raw")
    assert chunk_passage(_row("cut body", text_truncated=True)) == ("cut body", "node_truncated")
    assert chunk_passage(_row()) == (_DOC_A[:10], "excerpt")
    assert chunk_passage({"node_id": "x", "content": {}}) == ("", "")


# ── import ──────────────────────────────────────────────────────────────────


class _Conn:
    def __init__(self, column: bool):
        self._column = column

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def fetchval(self, _sql, *_a):
        return self._column

    async def fetchrow(self, _sql, *_a):
        return None  # no identity collision

    async def execute(self, *_a):
        return "INSERT 0 1"


class _ImportPool:
    def __init__(self, column: bool):
        self._column = column

    def acquire(self):
        return _Conn(self._column)


class _ImportStore:
    def __init__(self, column: bool = True):
        self._pool = _ImportPool(column)
        self.nodes: dict[str, KnowledgeNode] = {}

    async def insert_artifact(self, **kw):
        return kw["artifact_id"]

    async def insert_ir(self, **_kw):
        return True

    async def upsert_node(self, node):
        self.nodes[node.node_id] = node

    async def upsert_edge(self, _edge):
        return None

    async def rebind_artifact_source(self, *_a):
        return True


def _bundle(root: Path, node_rows: list[dict], *, version="1.3", raw: bytes | None = None, name=""):
    bundle = root / "bundle"
    bundle.mkdir()
    art = _artifact("art-a", _REF_A, raw.decode() if raw else _DOC_A)
    art_row = {
        "artifact_id": art.artifact_id,
        "content_hash": art.content_hash,
        "kind": "document",
        "source_ref": art.source_ref,
        "size_bytes": art.size_bytes,
        "metadata": {"source_name": name} if name else {},
    }
    (bundle / "artifacts.jsonl").write_text(json.dumps(art_row) + "\n")
    (bundle / "nodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in node_rows))
    if raw is not None:
        caps = bundle_raw.RawCaps(object_max=1 << 20, total_max=1 << 20)
        bundle_raw.write_raw(bundle, [art_row], {_REF_A: raw}.get, caps, {})
    manifest = {"format": "r4-kb", "version": version, "project_id": "src", "tables": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


async def _import(bundle: Path, store: _ImportStore, *, raw_writer=None):
    engine = KnowledgeImportEngine(store=store, raw_writer=raw_writer)
    return await engine.import_project(
        bundle, target_project_id="dst", verify=False, allow_unsigned=True
    )


async def test_import_fills_search_text_from_the_verified_node_text(tmp_path):
    store = _ImportStore()
    result = await _import(_bundle(tmp_path, [_row(_DOC_A[0:37])]), store)
    assert store.nodes["n"].search_text == _DOC_A[0:37]
    assert (result.search_text_from_node, result.search_text_pending) == (1, 0)
    assert result.to_dict()["search_text_backfill"] == ""


async def test_import_falls_back_to_an_accepted_raw_object(tmp_path):
    store = _ImportStore()
    bundle = _bundle(tmp_path, [_row("tampered")], raw=_DOC_A.encode())
    result = await _import(bundle, store, raw_writer=lambda _p, c: f"k/{_sha(c)}")
    assert store.nodes["n"].search_text == _DOC_A[0:37]
    assert (result.search_text_from_raw, result.raw_restored) == (1, 1)


async def test_declined_raw_objects_are_not_sliced(tmp_path):
    store = _ImportStore()
    bundle = _bundle(tmp_path, [_row("tampered")], raw=_DOC_A.encode())
    result = await _import(bundle, store)  # no raw writer: declined
    assert (store.nodes["n"].search_text, result.raw_declined) == (None, 1)
    assert result.search_text_pending == 1


async def test_a_v1_2_bundle_without_raw_objects_names_the_remedy_that_works(tmp_path):
    """Review chunk-data-14: the backfill reads each artifact's object, and
    an artifact imported without its raw object keeps the source
    installation's key — the backfill alone cannot fill it here."""
    store = _ImportStore()
    result = await _import(_bundle(tmp_path, [_row()], version="1.2"), store)
    assert store.nodes["n"].search_text is None
    assert (result.search_text_pending, result.search_text_pending_no_raw) == (1, 1)
    advice = result.search_text_backfill
    assert advice.startswith("1 of 1 pending chunk(s) have no raw object restored here")
    assert "re-export the source with --chunk-text" in advice
    assert result.to_dict()["search_text_pending_no_raw"] == 1


async def test_a_pending_chunk_whose_raw_object_was_restored_gets_the_backfill(tmp_path):
    store = _ImportStore()
    other = b"Another document entirely, whose slice never verifies."
    bundle = _bundle(tmp_path, [_row("tampered")], raw=other)
    result = await _import(bundle, store, raw_writer=lambda _p, c: f"k/{_sha(c)}")
    assert (result.search_text_pending, result.search_text_pending_no_raw) == (1, 0)
    assert result.search_text_backfill == (
        "python -m umrs.knowledge.retrieval.backfill --project dst"
    )


async def test_without_migration_0120_nothing_is_written_and_it_says_so(tmp_path):
    store = _ImportStore(column=False)
    result = await _import(_bundle(tmp_path, [_row(_DOC_A[0:37])]), store)
    assert store.nodes["n"].search_text is None
    assert result.search_text_pending == 1
    assert result.search_text_backfill.startswith("apply migration 0120")


async def test_a_headed_chunk_gets_header_plus_body(tmp_path):
    header = "Paper A › Results\n"
    row = _row(_DOC_A[0:37], text_header=header)
    row["content"]["header_chars"] = len(header)
    store = _ImportStore()
    await _import(_bundle(tmp_path, [row]), store)
    assert store.nodes["n"].search_text == header + _DOC_A[0:37]


async def test_an_unknown_header_leaves_the_chunk_pending(tmp_path):
    row = _row(_DOC_A[0:37])
    row["content"]["header_chars"] = 12
    store = _ImportStore()
    result = await _import(_bundle(tmp_path, [row]), store)
    assert (store.nodes["n"].search_text, result.search_text_pending) == (None, 1)


async def test_unsupported_versions_are_still_refused(tmp_path):
    with pytest.raises(BundleFormatError, match="version"):
        await _import(_bundle(tmp_path, [], version="1.4"), _ImportStore())


def _row_with(node_id: str, text: str) -> dict:
    """A chunk row whose node text verifies (its md5 is the text's)."""
    row = _row(text)
    row["node_id"] = node_id
    row["content"].update(chunk_end=len(text), chunk_md5=_md5(text))
    return row


# ── PDF chunks verify through the extracted text (review chunk-data-12) ────

_PAPER = "Interleukin six rises after exercise. It falls during rest periods."


def _pdf_chunk(node_id: str = "n-pdf") -> tuple[bytes, KnowledgeNode]:
    raw = pdf_bytes(_PAPER)
    document = embed_text(raw, "paper.pdf")
    content = {**chunk_content(document, 0, len(document)), "artifact_id": "art-a"}
    node = KnowledgeNode(
        node_id=node_id, project_id="p", label=node_id, node_type="entity", content=content
    )
    return raw, node


async def test_a_pdf_chunk_exports_its_extracted_body(tmp_path, monkeypatch):
    """The export's slice decoded the PDF's raw bytes, so a PDF chunk with no
    stored text never verified and travelled as its excerpt only."""
    raw, node = _pdf_chunk()
    corpus = ([node], {}, {"art-a": (_REF_A, "paper.pdf")})
    bundle, _pool = await _export(
        tmp_path, monkeypatch, chunk_text=True, corpus=corpus, blobs={_REF_A: raw}
    )
    assert _rows(bundle)["n-pdf"]["text"] == _PAPER
    assert _manifest(bundle)["chunk_text_summary"]["artifact"] == 1


async def test_an_imported_pdf_raw_object_fills_its_chunks(tmp_path):
    raw, node = _pdf_chunk()
    row = {"node_id": node.node_id, "content": dict(node.content)}
    store = _ImportStore()
    bundle = _bundle(tmp_path, [row], raw=raw, name="paper.pdf")
    result = await _import(bundle, store, raw_writer=lambda _p, c: f"k/{_sha(c)}")
    assert store.nodes["n-pdf"].search_text == _PAPER
    assert result.search_text_from_raw == 1


async def test_a_node_text_holding_a_nul_is_left_pending(tmp_path):
    """Review chunk-data-10: Postgres TEXT refuses NUL."""
    store = _ImportStore()
    result = await _import(_bundle(tmp_path, [_row_with("n", "ab\x00cd")]), store)
    assert (store.nodes["n"].search_text, result.search_text_pending) == (None, 1)


class _RefusingStore(_ImportStore):
    """Postgres refusing one row's data (a NUL in a JSONB value, say)."""

    async def upsert_node(self, node):
        if node.node_id == "n-bad":
            from asyncpg.exceptions import UntranslatableCharacterError

            raise UntranslatableCharacterError("unsupported Unicode escape sequence")
        await super().upsert_node(node)


async def test_a_row_postgres_refuses_is_skipped_and_the_rest_restored(tmp_path):
    """Review chunk-data-10: asyncpg's DataError is no ValueError, so one
    refused row aborted the import mid-table."""
    rows = [_row_with("n-bad", "first"), _row_with("n-ok", "second")]
    store = _RefusingStore()
    result = await _import(_bundle(tmp_path, rows), store)
    assert list(store.nodes) == ["n-ok"] and result.node_count == 1


class _HeldConn(_Conn):
    """Every incoming node row already present, identical, in ``holder``."""

    def __init__(self, holder: str, hashes: dict[str, str]):
        super().__init__(True)
        self._holder, self._hashes = holder, hashes

    async def fetchrow(self, sql, *args):
        if "FROM knowledge_nodes" not in sql or args[0] not in self._hashes:
            return None
        return {"content_hash": self._hashes[args[0]], "project_id": self._holder}


async def test_rows_held_by_another_project_are_reported_not_silently_dropped(tmp_path):
    """Review chunk-data-13: node ids are installation-global, so importing
    into a second project of the same installation restored nothing and
    reported 0/0."""
    row = _row(_DOC_A[0:37])
    row["content_hash"] = "h-a1"
    store = _ImportStore()
    store._pool.acquire = lambda: _HeldConn("default", {"n": "h-a1"})
    result = await _import(_bundle(tmp_path, [row]), store)
    assert (result.node_count, result.nodes_skipped_identical) == (0, 1)
    assert result.to_dict()["nodes_skipped_other_project"] == 1


async def test_export_then_import_restores_every_stored_body(tmp_path, monkeypatch):
    """S8 receipt: 100% md5 agreement between the source's search_text and
    the imported one for every chunk that carried text."""
    bundle, _pool = await _export(tmp_path, monkeypatch, chunk_text=True)
    store = _ImportStore()
    result = await _import(bundle, store)
    restored = {nid: n.search_text for nid, n in store.nodes.items() if n.search_text}
    assert restored == {"n-a1": _DOC_A[0:37], "n-a2": _DOC_A[38:]}
    assert (result.search_text_from_node, result.search_text_pending) == (2, 2)


# ── api + job wiring ────────────────────────────────────────────────────────


class _FactsPool:
    """psycopg-shaped registry read of the api gate: (owner, class, licence)."""

    def __init__(self, row):
        self._row = row

    def getconn(self):
        cur = SimpleNamespace(execute=lambda *_a: None, fetchone=lambda: self._row)
        return SimpleNamespace(cursor=lambda: cur)

    def putconn(self, _conn):
        pass


def _post_export(monkeypatch, facts, user, **body):
    from umrs.api import knowledge as kmod

    runtime = SimpleNamespace(submit_job_sync=AsyncMock(return_value="job-9"))
    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: runtime)
    app = FastAPI()
    app.include_router(kmod.router)
    app.state.umrs_db_pool = _FactsPool(facts) if facts is not None else None
    headers = {"X-Project-ID": "caller-proj"}
    if user:
        headers.update({"X-User-ID": user, "X-Verified-Project-ID": "caller-proj"})
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/umrs/knowledge/export", json={"project_id": "caller-proj", **body}, headers=headers
    )
    return resp, runtime


def _spec(runtime) -> dict:
    return runtime.submit_job_sync.call_args.kwargs["spec"]


def test_chunk_text_on_a_non_sensitive_project_is_forwarded(monkeypatch):
    resp, runtime = _post_export(
        monkeypatch, ("alice", "non_sensitive", ""), "bob", chunk_text=True
    )
    assert resp.status_code == 200, resp.text
    assert (_spec(runtime)["chunk_text"], _spec(runtime)["chunk_text_acknowledgement"]) == (
        True,
        {},
    )


def test_chunk_text_of_a_sensitive_project_by_a_non_owner_is_refused(monkeypatch):
    resp, runtime = _post_export(monkeypatch, ("alice", "sensitive", ""), "bob", chunk_text=True)
    assert (resp.status_code, resp.headers.get(ERROR_CODE_HEADER)) == (
        403,
        "knowledge.raw_forbidden",
    )
    runtime.submit_job_sync.assert_not_called()


def test_the_owner_flag_on_a_sensitive_project_carries_the_acknowledgement(monkeypatch):
    resp, runtime = _post_export(monkeypatch, ("alice", "", ""), "alice", chunk_text=True)
    assert resp.status_code == 200, resp.text
    ack = _spec(runtime)["chunk_text_acknowledgement"]
    assert (ack["by"], ack["data_class"]) == ("alice", "sensitive")


def test_an_unreadable_registry_refuses_chunk_text(monkeypatch):
    resp, runtime = _post_export(monkeypatch, None, "alice", chunk_text=True)
    assert resp.status_code == 503
    runtime.submit_job_sync.assert_not_called()


def test_a_plain_export_is_submitted_under_its_caller(monkeypatch):
    """Submitted under "" its job was unreadable to the user who started it."""
    resp, runtime = _post_export(monkeypatch, None, "bob")
    assert resp.status_code == 200, resp.text
    assert runtime.submit_job_sync.call_args.kwargs["user_id"] == "bob"


def test_a_plain_export_carries_no_chunk_text(monkeypatch):
    resp, runtime = _post_export(monkeypatch, None, None)
    assert resp.status_code == 200, resp.text
    assert (_spec(runtime)["chunk_text"], _spec(runtime)["chunk_text_acknowledgement"]) == (
        False,
        {},
    )


async def test_the_job_hands_the_opt_in_to_the_engine(monkeypatch):
    from umrs.jobs.knowledge_adapters import KnowledgeExportBackend

    seen: dict = {}

    class _Engine:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def export_project(self, project_id):
            return SimpleNamespace(degraded=True, bundle_path="", project_id=project_id)

    spec = {"chunk_text": True, "chunk_text_acknowledgement": {"by": "alice"}}
    await KnowledgeExportBackend()._export_with_integrity(
        object(), "p", "", lambda _pool: None, _Engine, None, chunk_text_spec=spec, submitter="bob"
    )
    assert (seen["chunk_text"], seen["chunk_text_submitter"]) == (True, "bob")
    assert "chunk_text_acknowledgement" not in seen  # the spec's claim never reaches it


def test_a_chunk_text_export_job_is_submitted_under_the_verified_caller(monkeypatch):
    """The engine checks the job's submitter against the owner, so the route
    submits the job as the caller it verified (it submitted it as nobody)."""
    resp, runtime = _post_export(monkeypatch, ("alice", "sensitive", ""), "alice", chunk_text=True)
    assert resp.status_code == 200, resp.text
    assert runtime.submit_job_sync.call_args.kwargs["user_id"] == "alice"


# ── POST /jobs: releases go through POST /knowledge/export only ─────────────


class _JobStore:
    def __init__(self):
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    def create_fan_out_graph(self, **kwargs):
        self.created.append(kwargs)
        return "job-fan"

    def enqueue_job(self, _job_id):
        return None

    def _get_conn(self):
        raise RuntimeError("no catalog in this fake")


def _post_job(monkeypatch, spec: dict, fan_out=None):
    from umrs.api import jobs as jobs_api

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
    store = _JobStore()
    app = FastAPI()
    app.include_router(jobs_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    body = {"project_id": "p", "user_id": "mallory", "job_type": "knowledge_export", "spec": spec}
    if fan_out is not None:
        body.update(spec={"project_id": "p"}, fan_out_jobs=fan_out)
    resp = TestClient(app, raise_server_exceptions=False).post("/umrs/jobs", json=body)
    return resp, store


@pytest.mark.parametrize("release", ["chunk_text", "include_raw"])
def test_post_jobs_refuses_a_knowledge_export_asking_for_a_release(monkeypatch, release):
    """chunk-data-2: POST /jobs took any spec, so a member skipped the
    /knowledge/export gate (owner, data class, audit) entirely."""
    forged = {"by": "alice", "data_class": "sensitive"}
    resp, store = _post_job(monkeypatch, {release: True, "chunk_text_acknowledgement": forged})
    assert resp.status_code == 400, resp.text
    assert "POST /knowledge/export" in resp.text and store.created == []


@pytest.mark.parametrize(("release", "status"), [(True, 400), (False, 201)])
def test_post_jobs_polices_the_release_in_a_fan_out_child(monkeypatch, release, status):
    child = {"job_type": "knowledge_export", "spec": {"chunk_text": release}}
    resp, store = _post_job(monkeypatch, {}, fan_out=[child])
    assert resp.status_code == status, resp.text
    assert len(store.created) == (status == 201)


def test_post_jobs_still_takes_a_plain_knowledge_export(monkeypatch):
    resp, store = _post_job(monkeypatch, {"project_id": "p", "chunk_text": False})
    assert resp.status_code == 201, resp.text
    assert store.created[0]["user_id"] == "mallory"
