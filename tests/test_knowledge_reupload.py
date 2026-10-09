"""A re-upload of bytes the project already holds continues on the stored artifact.

SL-CONF T2 (2026-10-09) uploaded a note whose bytes matched an earlier upload.
Ingest minted a fresh artifact id, the per-project dedup (audit B8) skipped its
row, and the chain ran under an id with no row: understand "completed" with
every node failing its FK, and embed failed on the missing file node.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from umrs.jobs import knowledge_adapters as ka


class _Store:
    def __init__(self, stored: dict[tuple[str, str], str]) -> None:
        self.stored = stored
        self.inserted: list[str] = []

    async def active_artifact_for(self, content_hash, project_id):
        return self.stored.get((content_hash, project_id))

    async def insert_artifact(self, **kwargs):
        self.inserted.append(kwargs["artifact_id"])
        return True


def _persist(monkeypatch, store, descs):
    @asynccontextmanager
    async def pool():
        yield object()

    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgres://db/umrs")
    monkeypatch.setattr(ka, "_open_pool", pool)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", lambda _pool: store)
    return ka._store_artifacts_in_db_sync("p", descs)


def test_known_bytes_continue_on_the_stored_artifact(monkeypatch):
    store = _Store({("h1", "p"): "art-stored"})
    descs = [{"artifact_id": "art-fresh", "content_hash": "h1", "source_ref": "k"}]
    assert _persist(monkeypatch, store, descs) == {"art-fresh": "art-stored"}
    assert descs[0]["artifact_id"] == "art-stored" and store.inserted == []


def test_new_bytes_get_their_row(monkeypatch):
    store = _Store({("h1", "other-project"): "art-elsewhere"})
    descs = [{"artifact_id": "art-fresh", "content_hash": "h1", "source_ref": "k"}]
    assert _persist(monkeypatch, store, descs) == {}
    assert store.inserted == ["art-fresh"]


def test_bytes_without_a_hash_are_never_matched(monkeypatch):
    store = _Store({("", "p"): "art-hashless"})
    descs = [{"artifact_id": "art-fresh", "content_hash": "", "source_ref": "k"}]
    assert _persist(monkeypatch, store, descs) == {}
    assert store.inserted == ["art-fresh"]
