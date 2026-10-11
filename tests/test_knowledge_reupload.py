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
        return kwargs["artifact_id"]


def _persist(monkeypatch, store, descs):
    @asynccontextmanager
    async def pool():
        yield object()

    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgres://db/umrs")
    monkeypatch.setattr(ka, "_open_pool", pool)
    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", lambda _pool: store)
    return ka._store_artifacts_in_db_sync("p", descs).deduped


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


# ── wave F (ADR-093 D-11): the real store, and a row that is not written ────


class _Conn:
    """An asyncpg connection stand-in: ``probe`` answers the dedup SELECT;
    ``insert`` is the INSERT's status, or an exception to raise."""

    def __init__(self, probe=None, insert="INSERT 0 1", after_race=None):
        self.probe, self.insert, self.after_race = probe, insert, after_race
        self.probes: list[tuple] = []
        self.inserts: list[tuple] = []

    async def fetchval(self, _sql, *args):
        self.probes.append(args)
        return self.after_race if self.inserts else self.probe

    async def execute(self, _sql, *args):
        self.inserts.append(args)
        if isinstance(self.insert, BaseException):
            raise self.insert
        return self.insert


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class _UniqueViolation(Exception):
    sqlstate = "23505"


class _ForeignKeyViolation(Exception):
    sqlstate = "23503"


def _insert(conn, content_hash="h1", artifact_id="art-fresh"):
    import asyncio

    from umrs.knowledge.store import KnowledgeStore

    return asyncio.run(
        KnowledgeStore(_Pool(conn)).insert_artifact(
            artifact_id=artifact_id, content_hash=content_hash, kind="document", project_id="p"
        )
    )


def test_the_real_store_never_dedups_a_hashless_artifact():
    """'' matched the first hashless upload (art-49fb58a2aea0): the second went
    down the chain with no row (2026-10-08). The probe is not even asked."""
    conn = _Conn(probe="art-hashless")
    assert _insert(conn, content_hash="") == "art-fresh"
    assert conn.probes == [] and len(conn.inserts) == 1


def test_known_bytes_return_the_stored_id_and_write_nothing():
    conn = _Conn(probe="art-stored")
    assert _insert(conn) == "art-stored"
    assert conn.inserts == []


def test_a_race_on_the_unique_hash_index_repoints():
    """Two uploads of the same bytes race past the probe: the loser's insert
    hits idx_artifacts_project_hash_active and resolves to the winner's id."""
    conn = _Conn(insert=_UniqueViolation("dup"), after_race="art-winner")
    assert _insert(conn) == "art-winner"


def test_a_unique_violation_with_no_stored_row_fails_by_name():
    import pytest
    from umrs.knowledge.store import ArtifactWriteError

    conn = _Conn(insert=_UniqueViolation("dup"), after_race=None)
    with pytest.raises(ArtifactWriteError, match="art-fresh"):
        _insert(conn)


def test_an_id_held_by_another_project_fails_by_name():
    """ON CONFLICT (artifact_id) reactivates only the project's own row: an id
    held elsewhere updates nothing ('INSERT 0 0') and is refused, by name."""
    import pytest
    from umrs.knowledge.store import ArtifactWriteError

    with pytest.raises(ArtifactWriteError, match="another project"):
        _insert(_Conn(insert="INSERT 0 0"))


def test_reactivation_stays_scoped_to_the_project():
    from umrs.knowledge.store import _INSERT_ARTIFACT_SQL

    assert "SET updated_at = $10, status = 'active'" in _INSERT_ARTIFACT_SQL
    assert "WHERE knowledge_artifacts.project_id = EXCLUDED.project_id" in _INSERT_ARTIFACT_SQL


def test_any_other_insert_error_is_raised():
    import pytest

    with pytest.raises(_ForeignKeyViolation):
        _insert(_Conn(insert=_ForeignKeyViolation("fk")), content_hash="")


class _FailingStore(_Store):
    def __init__(self, fail: set[str]):
        super().__init__({})
        self.fail = fail

    async def insert_artifact(self, **kwargs):
        if kwargs["artifact_id"] in self.fail:
            raise RuntimeError("disk full")
        return await super().insert_artifact(**kwargs)


def test_a_failed_insert_is_named_and_the_rest_are_written(monkeypatch):
    store = _FailingStore({"art-2"})
    descs = [
        {"artifact_id": f"art-{i}", "content_hash": f"h{i}", "source_ref": "k"} for i in (1, 2, 3)
    ]
    _persist(monkeypatch, store, descs)
    stored = ka._store_artifacts_in_db_sync("p", descs)
    assert [f["artifact_id"] for f in stored.failed] == ["art-2"]
    assert "artifact row not written: disk full" in stored.failed[0]["error"]


def test_an_unreachable_pool_names_every_descriptor(monkeypatch):
    @asynccontextmanager
    async def no_pool():
        yield None

    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgres://db/umrs")
    monkeypatch.setattr(ka, "_open_pool", no_pool)
    descs = [{"artifact_id": "art-1", "content_hash": "h1", "source_ref": "k"}]
    stored = ka._store_artifacts_in_db_sync("p", descs)
    assert [f["artifact_id"] for f in stored.failed] == ["art-1"]
    assert "knowledge pool unavailable" in stored.failed[0]["error"]


def test_no_configured_database_writes_and_fails_nothing(monkeypatch):
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "")
    stored = ka._store_artifacts_in_db_sync("p", [{"artifact_id": "art-1"}])
    assert (stored.deduped, stored.failed) == ({}, [])


def test_a_failed_insert_drops_the_descriptor_from_the_chain():
    """The descriptor whose row was not written leaves success_descs and the
    result, into failed_items by name: understand and embed never see it."""
    result = ka.KnowledgeIngestResult(artifact_ids=["art-1", "art-2"])
    success_ids = ["art-1", "art-2"]
    descs = [{"artifact_id": "art-1"}, {"artifact_id": "art-2"}]
    stored = ka._StoredArtifacts(
        deduped={}, failed=[{"artifact_id": "art-2", "error": "artifact row not written: x"}]
    )
    ka._drop_rowless(stored, result, success_ids, descs)
    assert descs == [{"artifact_id": "art-1"}]
    assert success_ids == ["art-1"] and result.artifact_ids == ["art-1"]
    assert result.failed_items == stored.failed


def _insert_ir(error):
    import asyncio

    from umrs.knowledge.store import KnowledgeStore

    return asyncio.run(
        KnowledgeStore(_Pool(_Conn(insert=error))).insert_ir(
            ir_id="ir-1",
            artifact_id="art-rowless",
            ir_type="document",
            content={},
            content_hash="h",
        )
    )


def test_an_ir_insert_on_a_missing_artifact_raises():
    """insert_ir logged the knowledge_ir FK violation as "likely duplicate" and
    understand went on to build nodes on an IR that did not exist."""
    import pytest

    with pytest.raises(_ForeignKeyViolation):
        _insert_ir(_ForeignKeyViolation("knowledge_ir_artifact_id_fkey"))


def test_a_duplicate_ir_is_still_reported_not_raised():
    assert _insert_ir(_UniqueViolation("dup")) is False
