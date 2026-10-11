"""ADR-032 precondition #1 (contract cid-1): canonical content identity.

The same logical text converges on one document_id across serializations;
binary content keeps its byte identity; unreadable content keeps the physical
digest; the exporter maps every artifact row and the manifest counts the
normalizations; import carries only a well-formed block and reports (never
merges) documents already present under another id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from umrs.jobs import knowledge_adapters
from umrs.knowledge import canonical_identity as cid
from umrs.knowledge.bundle_export import KnowledgeExportEngine
from umrs.knowledge.bundle_import import KnowledgeImportEngine, _metadata_with_identity

# ── the identity contract ─────────────────────────────────────────────────────


def test_text_serializations_converge_on_one_document_id():
    variants = [
        "Café result\nline two\n".encode(),
        b"\xef\xbb\xbf" + "Café result\r\nline two\r\n\r\n".encode(),
        "Café result  \nline two\t\n".encode(),  # decomposed é, trailing ws
        "Café result\rline two".encode(),
    ]
    ids = {cid.identity_of(v, "phys").document_id for v in variants}
    assert len(ids) == 1
    assert cid.identity_of(variants[0], "phys").normalization == cid.TEXT_V1


def test_different_text_different_identity():
    a = cid.identity_of(b"alpha", "p")
    b = cid.identity_of(b"beta", "p")
    assert a.document_id != b.document_id


def test_binary_content_keeps_the_byte_identity():
    raw = b"%PDF-1.7\x00\xff\xfe binary"
    ident = cid.identity_of(raw, "ignored")
    assert ident.normalization == cid.BYTES
    assert ident.identity == hashlib.sha256(raw).hexdigest()


def test_invalid_utf8_is_bytes_not_text():
    assert cid.identity_of(b"\xff\xfeabc", "p").normalization == cid.BYTES


def test_unreadable_content_keeps_the_physical_digest():
    digest = "a" * 64
    ident = cid.identity_of(None, digest)
    assert (ident.identity, ident.normalization) == (digest, cid.PHYSICAL)
    assert ident.document_id == "kdoc-" + "a" * 32


def test_from_row_accepts_only_a_well_formed_block():
    good = cid.identity_of(b"x", "p").to_dict()
    assert cid.from_row({"canonical_identity": good}) == good
    for bad in (
        None,
        "string",
        {**good, "contract": "cid-0"},
        {**good, "normalization": "fuzzy"},
        {**good, "identity": "short"},
    ):
        assert cid.from_row({"canonical_identity": bad}) is None


def test_from_row_recomputes_the_document_id():
    forged = {**cid.identity_of(b"x", "p").to_dict(), "document_id": "kdoc-forged"}
    assert cid.from_row({"canonical_identity": forged})["document_id"] != "kdoc-forged"


# ── export maps every artifact row ───────────────────────────────────────────


def _engine(tmp_path: Path, reader=None) -> KnowledgeExportEngine:
    return KnowledgeExportEngine(
        store=None, output_dir=tmp_path, allow_unsigned=True, artifact_reader=reader
    )


def test_export_stamps_rows_and_counts_normalizations(tmp_path):
    blobs = {
        "projects/p/knowledge/artifacts/text": b"hello\r\n",
        "projects/p/knowledge/artifacts/bin": b"\x00\x01",
    }
    engine = _engine(tmp_path, reader=blobs.get)
    rows = [
        {
            "source_ref": "projects/p/knowledge/artifacts/text",
            "content_hash": "h1",
            "size_bytes": 7,
        },
        {"source_ref": "projects/p/knowledge/artifacts/bin", "content_hash": "h2", "size_bytes": 2},
        {
            "source_ref": "projects/p/knowledge/artifacts/gone",
            "content_hash": "c" * 64,
            "size_bytes": 5,
        },
    ]
    for row in rows:
        engine._stamp_identity(row)
    assert [r["canonical_identity"]["normalization"] for r in rows] == [
        cid.TEXT_V1,
        cid.BYTES,
        cid.PHYSICAL,
    ]
    assert rows[0]["canonical_identity"]["document_id"] == (
        cid.identity_of(b"hello", "").document_id
    )
    manifest = engine._build_manifest("p", {}, {})
    assert manifest["identity"] == {
        "contract": cid.CONTRACT,
        "normalizations": {"bytes": 1, "physical": 1, "text-v1": 1},
    }


def test_export_without_a_reader_keeps_the_physical_identity(tmp_path):
    engine = _engine(tmp_path)
    row = {
        "source_ref": "projects/p/knowledge/artifacts/k",
        "content_hash": "d" * 64,
        "size_bytes": 3,
    }
    engine._stamp_identity(row)
    assert row["canonical_identity"]["normalization"] == cid.PHYSICAL


def test_oversized_artifacts_are_not_read(tmp_path):
    calls = []
    engine = _engine(tmp_path, reader=lambda ref: calls.append(ref) or b"x")
    row = {
        "source_ref": "projects/p/knowledge/artifacts/big",
        "content_hash": "e" * 64,
        "size_bytes": 1 << 40,
    }
    engine._stamp_identity(row)
    assert calls == []
    assert row["canonical_identity"]["normalization"] == cid.PHYSICAL


def test_export_reader_builds_s3_only_on_first_read(monkeypatch):
    built = []

    def _build(spec):
        built.append(spec)
        return object(), "bucket"

    monkeypatch.setattr(knowledge_adapters, "_build_s3_client", _build)
    monkeypatch.setattr(
        knowledge_adapters,
        "_fetch_artifact_bytes",
        lambda _c, bucket, ref: f"{bucket}:{ref}".encode(),
    )
    read = knowledge_adapters._lazy_artifact_reader()
    assert built == []
    assert read("a") == b"bucket:a"
    assert read("b") == b"bucket:b"
    assert len(built) == 1


# ── import carries the block and reports matches ──────────────────────────────


def test_import_metadata_drops_a_malformed_block():
    row = {"metadata": {"n": 1, "canonical_identity": {"contract": "evil"}}}
    assert _metadata_with_identity(row) == {"n": 1}
    good = cid.identity_of(b"t", "").to_dict()
    assert _metadata_with_identity({"canonical_identity": good})["canonical_identity"] == good


class _Store:
    def __init__(self, local=None, fail=False, lookup=True):
        self.artifacts: list[dict] = []
        self._local = local or {}
        self._fail = fail
        if not lookup:
            self.artifacts_by_canonical_document = None

    async def artifacts_by_canonical_document(self, project_id, document_ids):
        assert project_id == "proj-b"
        if self._fail:
            raise RuntimeError("db down")
        return {d: ids for d, ids in self._local.items() if d in document_ids}

    async def insert_artifact(self, **kwargs):
        self.artifacts.append(kwargs)
        return kwargs["artifact_id"]


def _bundle(root: Path, texts: dict[str, bytes]) -> Path:
    bundle = root / "b"
    bundle.mkdir()
    rows = [
        {
            "artifact_id": art,
            "content_hash": hashlib.sha256(raw).hexdigest(),
            "kind": "document",
            "canonical_identity": cid.identity_of(raw, "").to_dict(),
        }
        for art, raw in texts.items()
    ]
    (bundle / "artifacts.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    manifest = {"format": "r4-kb", "version": "1.2", "project_id": "proj-a", "tables": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


async def test_import_reports_same_document_under_another_id(tmp_path):
    doc = cid.identity_of(b"same text", "").document_id
    store = _Store(local={doc: ["local-1", "art-1"]})
    bundle = _bundle(tmp_path, {"art-1": b"same text\r\n", "art-2": b"other"})
    result = await KnowledgeImportEngine(store=store).import_project(
        bundle, target_project_id="proj-b", verify=False, allow_unsigned=True
    )
    assert result.canonical_matches == [
        {"document_id": doc, "incoming_artifact_id": "art-1", "local_artifact_id": "local-1"}
    ]
    assert result.to_dict()["canonical_matches"] == result.canonical_matches
    assert result.artifact_count == 2  # reported, never merged
    stamped = store.artifacts[0]["metadata"]["canonical_identity"]
    assert stamped["document_id"] == doc


async def test_import_without_the_lookup_reports_nothing(tmp_path):
    bundle = _bundle(tmp_path, {"art-1": b"x"})
    result = await KnowledgeImportEngine(store=_Store(lookup=False)).import_project(
        bundle, target_project_id="proj-b", verify=False, allow_unsigned=True
    )
    assert result.canonical_matches == []
    assert result.collision_checks_degraded == 0


async def test_failed_lookup_is_degraded_not_fatal(tmp_path):
    bundle = _bundle(tmp_path, {"art-1": b"x"})
    result = await KnowledgeImportEngine(store=_Store(fail=True)).import_project(
        bundle, target_project_id="proj-b", verify=False, allow_unsigned=True
    )
    assert result.canonical_matches == []
    assert result.collision_checks_degraded == 1
    assert result.artifact_count == 1


# ── addendum 7: stamped at ingest, reused at export, backfilled ─────────────


def test_carried_decodes_codecless_jsonb_text():
    block = cid.identity_of(b"t", "").to_dict()
    assert cid.carried(json.dumps({"canonical_identity": block})) == block
    assert cid.carried("not json") is None
    assert cid.carried(None) is None
    assert cid.metadata_dict("[1, 2]") == {}


def test_import_metadata_tolerates_text_metadata():
    assert _metadata_with_identity({"metadata": '{"n": 1}'}) == {"n": 1}


def test_export_reuses_a_carried_stamp_without_reading(tmp_path):
    calls = []
    engine = _engine(tmp_path, reader=lambda ref: calls.append(ref) or b"other")
    stamped = cid.stamp(b"original text", "h")
    row = {"source_ref": "k", "content_hash": "h", "size_bytes": 1, "metadata": stamped}
    engine._stamp_identity(row)
    assert calls == []
    assert row["canonical_identity"] == stamped["canonical_identity"]


class _Storage:
    @staticmethod
    def key(project_id, content_hash):
        return f"projects/{project_id}/{content_hash}"

    def put_artifact(self, _project_id, content):
        return hashlib.sha256(content).hexdigest(), len(content), None


def test_ingest_stamps_the_descriptor(monkeypatch):
    monkeypatch.setattr(knowledge_adapters, "_bridge_a2l_ref", lambda *_a: None)
    descs: list[dict] = []
    result = knowledge_adapters.KnowledgeIngestResult()
    knowledge_adapters._put_inline_text_sync(
        "j",
        "a",
        _Storage(),
        "p",
        {"uri": "n.md", "content": "hello\r\n"},
        "art-1",
        result,
        [],
        descs,
    )
    assert descs[0]["metadata"]["canonical_identity"]["document_id"] == (
        cid.identity_of(b"hello", "").document_id
    )


class _BackfillStore:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r["artifact_id"])
        self.stamped: dict[str, dict] = {}
        self.pages = 0

    async def artifacts_missing_identity(self, project_id, after, limit):
        assert project_id == "p"
        self.pages += 1
        todo = [
            r
            for r in self.rows
            if r["artifact_id"] > after and r["artifact_id"] not in self.stamped
        ]
        return todo[:limit]

    async def stamp_canonical_identity(self, artifact_id, block):
        self.stamped[artifact_id] = block
        return True


def _rows(n):
    return [
        {
            "artifact_id": f"a{i:02d}",
            "content_hash": "h",
            "source_ref": f"projects/p/knowledge/artifacts/k{i}",
            "size_bytes": 1,
        }
        for i in range(n)
    ]


async def test_backfill_dry_run_writes_nothing():
    from umrs.knowledge.identity_backfill import backfill

    store = _BackfillStore(_rows(3))
    outcome = await backfill(store, lambda ref: ref.encode(), project_id="p", batch=2)
    assert (outcome.scanned, outcome.stamped) == (3, 0)
    assert outcome.by_normalization == {"text-v1": 3}
    assert store.stamped == {}


async def test_backfill_apply_stamps_readable_rows_and_terminates():
    from umrs.knowledge.identity_backfill import backfill

    store = _BackfillStore(_rows(5))
    unreadable = {"projects/p/knowledge/artifacts/k1", "projects/p/knowledge/artifacts/k3"}
    outcome = await backfill(
        store,
        lambda ref: None if ref in unreadable else b"\x00bin" + ref.encode(),
        project_id="p",
        apply=True,
        batch=2,
    )
    assert outcome.to_dict() == {
        "scanned": 5,
        "stamped": 3,
        "unreadable": 2,
        "no_object": 0,
        "by_normalization": {"bytes": 3},
    }
    assert sorted(store.stamped) == ["a00", "a02", "a04"]
    assert store.pages <= 4  # keyset paging: unreadable rows are never re-scanned


async def test_backfill_skips_oversized_artifacts():
    from umrs.knowledge.identity_backfill import backfill

    rows = [
        {
            "artifact_id": "big",
            "content_hash": "h",
            "source_ref": "projects/p/knowledge/artifacts/k",
            "size_bytes": 1 << 40,
        }
    ]
    store = _BackfillStore(rows)
    outcome = await backfill(store, lambda _r: b"x", project_id="p", apply=True)
    assert (outcome.unreadable, outcome.stamped) == (1, 0)


async def test_backfill_skips_rows_without_a_stored_object():
    from umrs.knowledge.identity_backfill import backfill

    rows = [
        {"artifact_id": "a", "content_hash": "h", "source_ref": "rag://legacy-1", "size_bytes": 1},
        {
            "artifact_id": "b",
            "content_hash": "h",
            "source_ref": "loop:L#iteration-1",
            "size_bytes": 1,
        },
    ]
    reads = []
    store = _BackfillStore(rows)
    outcome = await backfill(
        store, lambda ref: reads.append(ref) or b"x", project_id="p", apply=True
    )
    assert (outcome.no_object, outcome.unreadable, outcome.stamped) == (2, 0, 0)
    assert reads == []  # nothing fetched


def test_stored_object_recognises_artifact_keys():
    assert cid.stored_object("projects/p/knowledge/artifacts/abc")
    assert not cid.stored_object("rag://legacy-1")
    assert not cid.stored_object("loop:L#iteration-1")
