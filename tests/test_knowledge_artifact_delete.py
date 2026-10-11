"""ADR-093 MD-16: the knowledge artifact delete (decided 2026-10-11).

``DELETE /umrs/knowledge/artifacts/{id}`` and the rag facade's
``DELETE /umrs/rag/documents/{id}`` over a fake knowledge database that
applies each tagged statement (``/* artifact_delete:<step> */``) the way its
SQL says, with a transaction that restores every table on error. The SQL
itself runs on a real Postgres in the integration test at the bottom
(skipped without ``--db-url`` / ``UMRS_TEST_DB_URL``).
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import knowledge_artifact_delete as route
from umrs.api import rag as rag_api
from umrs.knowledge import artifact_delete as core

P = "p1"
OTHER = "p2"
BUCKET = "umrs-files"
CONTENT_KEY = "projects/p1/knowledge/artifacts/h-a"
STAGED = f"s3://{BUCKET}/knowledge-uploads/p1/u-1/a.pdf"
HEADERS = {"x-user-id": "u1", "x-verified-project-id": P}

_TAG = re.compile(r"/\* artifact_delete:(\w+) \*/")
#: tag -> the module's $n SQL (the psycopg2 driver recovers $n arguments).
_SQL_BY_TAG = {
    _TAG.search(sql).group(1): sql
    for sql in (
        core.LOCK_SQL,
        core.NODES_SQL,
        core.DELETE_ARTIFACT_SQL,
        core.OBJECT_REFS_SQL,
        core.RESOLVE_SQL,
        *(s for _k, s, _p in core.NODE_STEPS),
        *(s for _k, s in core.ARTIFACT_STEPS),
    )
}


# ── the fake knowledge database ──────────────────────────────────────────────


def _artifact(project: str, content_hash: str, source_ref: str, **metadata: Any) -> dict:
    return {
        "project": project,
        "hash": content_hash,
        "source_ref": source_ref,
        "metadata": metadata,
    }


def _node(project: str, node_type: str = "entity", label: str = "", source_ir=None, **content):
    return {
        "project": project,
        "type": node_type,
        "label": label,
        "source_ir": source_ir,
        "content": content,
    }


class FakeDb:
    def __init__(self) -> None:
        self.t: dict[str, dict] = {
            "artifacts": {
                "art-a": _artifact(P, "h-a", CONTENT_KEY, staged_from=STAGED),
                "art-b": _artifact(P, "h-b", "projects/p1/knowledge/artifacts/h-b"),
                "artifact-rag-d1": _artifact(
                    P, "h-r", "rag://d1", origin="rag.upload", document_id="d1"
                ),
                "art-x": _artifact(OTHER, "h-x", "projects/p2/knowledge/artifacts/h-x"),
            },
            "ir": {"ir-a": "art-a", "ir-b": "art-b", "ir-x": "art-x"},
            "nodes": {
                "node-file-art-a": _node(P, "file", artifact_id="art-a", ir_id="ir-a"),
                "node-class-K-art-a": _node(P, "class", source_ir="ir-a", artifact_id="art-a"),
                "node-sem-foo-art-a": _node(P, "concept", artifact_id="art-a", semantic_type="x"),
                "chunk-a-0": _node(P, kind="chunk", artifact_id="art-a", chunk_index=0),
                "node-ir-only": _node(P, "concept", source_ir="ir-a"),
                "art-a": _node(P, label="artifact:art-a", kind="document"),
                "prod-job-j1": _node(P, label="job:j1", job_id="j1"),
                "node-file-art-b": _node(P, "file", artifact_id="art-b"),
                "chunk-b-0": _node(P, kind="chunk", artifact_id="art-b", chunk_index=0),
                "ragdoc-d1": _node(P, "file", artifact_id="artifact-rag-d1", origin="rag.upload"),
                "ragdoc-d1-chunk0": _node(P, kind="chunk", document_id="d1", origin="rag.upload"),
                "ragdoc-d1-chunk1": _node(P, kind="chunk", document_id="d1", origin="rag.upload"),
                "node-file-art-x": _node(OTHER, "file", artifact_id="art-x"),
                # A foreign node naming art-a must never be touched.
                "foreign-claims-a": _node(OTHER, "concept", artifact_id="art-a"),
            },
            "edges": {
                "e-chunk-a": ("chunk-a-0", "node-file-art-a", "part_of"),
                "e-mentions": ("node-file-art-a", "node-sem-foo-art-a", "mentions"),
                "e-produced": ("art-a", "prod-job-j1", "produced_by"),
                "e-cross-out": ("node-sem-foo-art-a", "node-file-art-b", "related"),
                "e-cross-in": ("node-file-art-b", "node-class-K-art-a", "cites"),
                "e-chunk-b": ("chunk-b-0", "node-file-art-b", "part_of"),
                "e-rag-0": ("ragdoc-d1-chunk0", "ragdoc-d1", "part_of"),
                "e-rag-1": ("ragdoc-d1-chunk1", "ragdoc-d1", "part_of"),
                "e-x": ("node-file-art-x", "foreign-claims-a", "related"),
            },
            "embeddings": {
                ("chunk-a-0", "m"),
                ("node-sem-foo-art-a", "m"),
                ("chunk-b-0", "m"),
                ("ragdoc-d1-chunk0", "m"),
                ("ragdoc-d1-chunk1", "m"),
                ("foreign-claims-a", "m"),
            },
            "findings": {"f-a": "art-a", "f-b": "art-b", "f-x": "art-x"},
            "events": {"ev-1": "art-a", "ev-2": "art-a", "ev-3": "art-b"},
        }
        self.commits = 0
        self.rollbacks = 0
        self.fail_on: str | None = None
        self.statements: list[str] = []
        self.in_tx = False

    def snapshot(self) -> dict:
        return copy.deepcopy(self.t)

    # each tagged statement, applied as its SQL says ------------------------

    def run(self, tag: str, args: tuple) -> Any:
        self.statements.append(tag)
        if tag == self.fail_on:
            raise RuntimeError(f"injected failure at {tag}")
        return getattr(self, f"_{tag}")(*args)

    def _lock_artifact(self, aid, pid):
        row = self.t["artifacts"].get(aid)
        if row is None or row["project"] != pid:
            return []
        return [(aid, row["hash"], row["source_ref"], json.dumps(row["metadata"]))]

    def _own(self, aid, pid) -> set[str]:
        irs = {i for i, a in self.t["ir"].items() if a == aid}
        return {
            nid
            for nid, n in self.t["nodes"].items()
            if n["project"] == pid
            and (
                n["content"].get("artifact_id") == aid
                or n["source_ir"] in irs
                or (nid == aid and n["label"] == f"artifact:{aid}")
            )
        }

    def _select_nodes(self, aid, pid):
        own = self._own(aid, pid)
        files = {n for n in own if self.t["nodes"][n]["type"] == "file"}
        rag = {
            src
            for src, tgt, rel in self.t["edges"].values()
            if rel == "part_of"
            and tgt in files
            and self.t["nodes"].get(src, {}).get("project") == pid
            and self.t["nodes"][src]["content"].get("kind") == "chunk"
        }
        return [(n,) for n in sorted(own | rag)]

    def _drop(self, table: str, keep) -> int:
        before = len(self.t[table])
        if isinstance(self.t[table], set):
            self.t[table] = {r for r in self.t[table] if keep(r)}
        else:
            self.t[table] = {k: v for k, v in self.t[table].items() if keep((k, v))}
        return before - len(self.t[table])

    def _delete_embeddings(self, ids):
        return self._drop("embeddings", lambda r: r[0] not in ids)

    def _delete_edges(self, ids):
        return self._drop("edges", lambda kv: kv[1][0] not in ids and kv[1][1] not in ids)

    def _delete_nodes(self, ids, pid):
        return self._drop("nodes", lambda kv: not (kv[0] in ids and kv[1]["project"] == pid))

    def _delete_findings(self, aid):
        return self._drop("findings", lambda kv: kv[1] != aid)

    def _delete_ir(self, aid):
        return self._drop("ir", lambda kv: kv[1] != aid)

    def _detach_events(self, aid):
        hit = [e for e, a in self.t["events"].items() if a == aid]
        for e in hit:
            self.t["events"][e] = None
        return len(hit)

    def _delete_artifact(self, aid, pid):
        if aid in self.t["findings"].values():  # FK without ON DELETE: restrict
            raise RuntimeError("verification_findings_artifact_id_fkey")
        return self._drop("artifacts", lambda kv: not (kv[0] == aid and kv[1]["project"] == pid))

    def _object_refs(self, ref, content_hash, pid):
        return [
            (
                sum(
                    1
                    for a in self.t["artifacts"].values()
                    if a["source_ref"] == ref
                    or a["metadata"].get("staged_from") == ref
                    or (content_hash and a["project"] == pid and a["hash"] == content_hash)
                ),
            )
        ]

    def _resolve_document(self, doc, pid):
        hits = [
            aid
            for aid, a in self.t["artifacts"].items()
            if a["project"] == pid
            and (
                aid == doc
                or aid == f"artifact-rag-{doc}"
                or a["metadata"].get("document_id") == doc
            )
        ]
        hits.sort(key=lambda a: (a != doc, a != f"artifact-rag-{doc}", a))
        return [(a,) for a in hits[:2]]


class _Tx:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    async def __aenter__(self):
        self.saved = self.db.snapshot()
        self.db.in_tx = True

    async def __aexit__(self, exc_type, *_a):
        self.db.in_tx = False
        if exc_type is not None:
            self.db.t = self.saved
            self.db.rollbacks += 1
        else:
            self.db.commits += 1
        return False


class FakeConn:
    """asyncpg-shaped: fetch/execute with ``$n`` arguments, transaction()."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db

    def transaction(self) -> _Tx:
        return _Tx(self.db)

    async def fetch(self, sql: str, *args: Any):
        return self.db.run(_TAG.search(sql).group(1), args)

    async def execute(self, sql: str, *args: Any) -> str:
        return f"DELETE {self.db.run(_TAG.search(sql).group(1), args)}"


class FakePool:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return FakeConn(pool.db)

            async def __aexit__(self, *_a):
                return False

        return _Acquire()


class FakeStore:
    content_bucket = BUCKET
    buckets = frozenset({BUCKET})

    def __init__(self, fail: set[str] | None = None) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.fail = fail or set()

    def delete(self, bucket: str, key: str) -> None:
        if key in self.fail:
            raise OSError(f"S3 refused {key}")
        self.deleted.append((bucket, key))


@pytest.fixture()
def db() -> FakeDb:
    return FakeDb()


@pytest.fixture()
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture()
def audit(monkeypatch) -> list[tuple[str, dict]]:
    from umrs.api import audit_events

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(audit_events, "emit_audit", lambda event, **kw: events.append((event, kw)))
    return events


@pytest.fixture()
def client(monkeypatch, db, store, audit) -> TestClient:
    async def _pool(_request):
        return FakePool(db)

    monkeypatch.setattr(route, "knowledge_asyncpg_pool", _pool)
    monkeypatch.setattr(route, "object_store", lambda: store)
    app = FastAPI()
    app.include_router(route.router)
    app.include_router(rag_api.router)
    return TestClient(app, raise_server_exceptions=False)


def _delete(client: TestClient, artifact_id: str, headers=None, **params):
    return client.delete(
        f"/umrs/knowledge/artifacts/{artifact_id}", headers=headers or HEADERS, params=params
    )


def _code(response) -> str:
    return response.headers.get("x-platform-error-code", "")


# ── the knowledge artifact delete ────────────────────────────────────────────


def test_delete_removes_the_artifact_and_everything_derived_from_it(client, db, audit):
    response = _delete(client, "art-a")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["deleted"] is True
    assert body["counts"] == {
        "artifacts": 1,
        "nodes": 6,  # file, class, semantic, chunk, source_ir-only, provenance
        "edges": 5,  # part_of, mentions, produced_by, and both cross edges
        "embeddings": 2,
        "findings": 1,
        "ir": 1,
        "events_detached": 2,
    }
    assert set(db.t["artifacts"]) == {"art-b", "artifact-rag-d1", "art-x"}
    assert "ir-a" not in db.t["ir"]
    nodes = set(db.t["nodes"])
    assert not nodes & {
        "node-file-art-a",
        "node-class-K-art-a",
        "node-sem-foo-art-a",
        "chunk-a-0",
        "node-ir-only",
        "art-a",
    }
    # The job's provenance node is shared by the job's other outputs.
    assert {"prod-job-j1", "node-file-art-b", "chunk-b-0"} <= nodes
    assert set(db.t["edges"]) == {"e-chunk-b", "e-rag-0", "e-rag-1", "e-x"}
    assert ("chunk-a-0", "m") not in db.t["embeddings"]
    assert ("chunk-b-0", "m") in db.t["embeddings"]
    assert db.t["findings"] == {"f-b": "art-b", "f-x": "art-x"}
    assert db.commits == 1 and db.rollbacks == 0
    assert audit[0][0] == "knowledge.artifact_deleted"
    assert audit[0][1]["details"]["artifact_id"] == "art-a"


def test_events_are_detached_not_deleted(client, db):
    assert _delete(client, "art-a").status_code == 200

    assert db.t["events"] == {"ev-1": None, "ev-2": None, "ev-3": "art-b"}


def test_another_projects_rows_are_never_touched(client, db):
    before = {k: v for k, v in db.snapshot()["nodes"].items() if v["project"] == OTHER}

    assert _delete(client, "art-a").status_code == 200

    after = {k: v for k, v in db.t["nodes"].items() if v["project"] == OTHER}
    assert after == before  # incl. a foreign node whose content names art-a
    assert ("foreign-claims-a", "m") in db.t["embeddings"]
    assert "e-x" in db.t["edges"]


@pytest.mark.parametrize(
    "step",
    [
        "select_nodes",
        "delete_embeddings",
        "delete_edges",
        "delete_nodes",
        "delete_findings",
        "delete_ir",
        "detach_events",
        "delete_artifact",
    ],
)
def test_a_failure_mid_way_commits_nothing(client, db, store, audit, step):
    before = db.snapshot()
    db.fail_on = step

    response = _delete(client, "art-a")

    assert response.status_code == 500
    assert "nothing was deleted" in response.json()["detail"]
    assert db.t == before
    assert db.commits == 0 and db.rollbacks == 1
    assert store.deleted == []
    assert audit == []


def test_the_steps_run_inside_one_transaction(client, db, monkeypatch):
    seen: list[tuple[str, bool]] = []
    run = db.run

    def _spy(tag, args):
        seen.append((tag, db.in_tx))
        return run(tag, args)

    monkeypatch.setattr(db, "run", _spy)
    assert _delete(client, "art-a").status_code == 200

    in_tx = [tag for tag, inside in seen if inside]
    assert in_tx[0] == "lock_artifact" and in_tx[-1] == "delete_artifact"
    assert [tag for tag, inside in seen if not inside] == ["object_refs", "object_refs"]
    assert db.commits == 1


def test_another_projects_artifact_is_404_by_name(client, db, store):
    before = db.snapshot()

    response = _delete(client, "art-x")

    assert response.status_code == 404
    assert _code(response) == "knowledge.not_found"
    assert "art-x" in response.json()["detail"]
    assert db.t == before
    assert store.deleted == []


def test_an_unknown_id_is_404_by_name(client, db):
    response = _delete(client, "art-nope")

    assert response.status_code == 404
    assert _code(response) == "knowledge.not_found"
    assert "'art-nope'" in response.json()["detail"]
    assert db.statements == ["lock_artifact"]


def test_deleting_twice_is_404_the_second_time(client, db, store):
    assert _delete(client, "art-a").status_code == 200
    after_first = db.snapshot()

    again = _delete(client, "art-a")

    assert again.status_code == 404
    assert _code(again) == "knowledge.not_found"
    assert db.t == after_first
    assert len(store.deleted) == 2  # only the first delete touched objects


def test_a_project_the_caller_cannot_write_is_refused(client, db):
    before = db.snapshot()

    response = _delete(client, "art-x", project_id=OTHER)

    assert response.status_code == 403
    assert _code(response) == "knowledge_guard.forbidden"
    assert db.t == before and db.statements == []


def test_an_admin_names_the_project_and_deletes_there(client, db):
    admin = {**HEADERS, "x-verified-is-admin": "true"}

    response = _delete(client, "art-x", headers=admin, project_id=OTHER)

    assert response.status_code == 200
    assert response.json()["project_id"] == OTHER
    assert "art-x" not in db.t["artifacts"]
    assert "foreign-claims-a" in db.t["nodes"]  # names art-a, not art-x


def test_agents_may_not_delete(client, db):
    response = _delete(client, "art-a", headers={**HEADERS, "x-r4-actor": "agent"})

    assert response.status_code == 403
    assert _code(response) == "knowledge_guard.forbidden"
    assert response.json()["detail"]["error"] == "delete_forbidden_for_agent"
    assert db.statements == []


def test_the_objects_go_after_the_commit(client, store):
    body = _delete(client, "art-a").json()

    assert store.deleted == [(BUCKET, CONTENT_KEY), (BUCKET, "knowledge-uploads/p1/u-1/a.pdf")]
    assert [(o["key"], o["outcome"]) for o in body["objects"]] == [
        (CONTENT_KEY, "deleted"),
        ("knowledge-uploads/p1/u-1/a.pdf", "deleted"),
    ]


def test_an_object_another_row_references_is_kept(client, db, store):
    # Same bytes in the project under an archived row: the content key is shared.
    db.t["artifacts"]["art-a-old"] = _artifact(P, "h-a", CONTENT_KEY)

    body = _delete(client, "art-a").json()

    assert (BUCKET, CONTENT_KEY) not in store.deleted
    shared = body["objects"][0]
    assert shared["outcome"] == "kept_shared"
    assert "1 other artifact row" in shared["detail"]
    assert body["objects"][1]["outcome"] == "deleted"


def test_a_failed_object_delete_is_named_and_the_db_delete_stands(client, db, store):
    store.fail = {CONTENT_KEY}

    response = _delete(client, "art-a")

    assert response.status_code == 200
    failed = response.json()["objects"][0]
    assert failed["outcome"] == "failed"
    assert CONTENT_KEY in failed["detail"]
    assert "art-a" not in db.t["artifacts"]
    assert db.commits == 1


def test_an_unreachable_object_store_is_reported_not_attempted(client, db, monkeypatch):
    monkeypatch.setattr(route, "object_store", lambda: None)

    body = _delete(client, "art-a").json()

    assert {o["outcome"] for o in body["objects"]} == {"not_attempted"}
    assert "art-a" not in db.t["artifacts"]


def test_rag_minted_chunks_go_through_their_file_nodes_part_of(client, db, store):
    body = _delete(client, "artifact-rag-d1").json()

    assert body["counts"]["nodes"] == 3
    assert not set(db.t["nodes"]) & {"ragdoc-d1", "ragdoc-d1-chunk0", "ragdoc-d1-chunk1"}
    assert not {e for e in db.t["edges"] if e.startswith("e-rag")}
    assert body["objects"] == []  # rag://d1 is no object
    assert store.deleted == []
    assert "endpoint corpus" in body["notes"][0]


# ── the rag facade ───────────────────────────────────────────────────────────


def _rag_delete(client: TestClient, document_id: str, headers=None):
    return client.delete(
        f"/umrs/rag/documents/{document_id}",
        headers=headers or {"x-user-id": "u1", "x-project-id": P},
    )


@pytest.fixture()
def recorded(monkeypatch) -> list[tuple]:
    from umrs.api import governance

    calls: list[tuple] = []
    monkeypatch.setattr(governance, "record_resource_deleted", lambda *a, **k: calls.append((a, k)))
    return calls


def test_a_rag_document_id_resolves_to_its_minted_artifact(client, db, recorded):
    response = _rag_delete(client, "d1")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == "d1"
    assert body["artifact_id"] == "artifact-rag-d1"
    assert body["deleted"] is True and body["counts"]["nodes"] == 3
    assert "artifact-rag-d1" not in db.t["artifacts"]
    assert recorded == [(("rag_documents", "d1"), {"project_id": P})]


def test_an_unresolvable_rag_document_is_404_and_records_nothing(client, db, recorded, audit):
    before = db.snapshot()

    response = _rag_delete(client, "doc-unknown")

    assert response.status_code == 404
    assert _code(response) == "rag.not_found"
    assert "doc-unknown" in response.json()["detail"]
    assert "nothing was deleted" in response.json()["detail"]
    assert db.t == before and recorded == [] and audit == []


def test_a_foreign_projects_rag_document_does_not_resolve(client, db, recorded):
    response = _rag_delete(client, "art-x")

    assert response.status_code == 404
    assert "art-x" in db.t["artifacts"]


def test_a_central_upload_job_id_resolves_to_the_artifact_it_stored(
    client, db, monkeypatch, recorded
):
    monkeypatch.setattr(
        route, "ingest_job_artifacts", lambda job, _project: ["art-a"] if job == "job-7" else []
    )

    response = _rag_delete(client, "job-7")

    assert response.status_code == 200, response.text
    assert response.json()["artifact_id"] == "art-a"
    assert "art-a" not in db.t["artifacts"]


def test_a_job_with_several_artifacts_is_404_naming_them(client, db, monkeypatch):
    monkeypatch.setattr(route, "ingest_job_artifacts", lambda *_a: ["art-a", "art-b"])
    before = db.snapshot()

    response = _rag_delete(client, "job-8")

    assert response.status_code == 404
    assert "art-a, art-b" in response.json()["detail"]
    assert db.t == before


def test_agents_may_not_delete_rag_documents(client, db):
    response = _rag_delete(
        client, "d1", headers={"x-user-id": "u1", "x-project-id": P, "x-r4-actor": "agent"}
    )

    assert response.status_code == 403
    assert db.statements == []


def test_ingest_job_artifacts_reads_the_committed_attempt(monkeypatch):
    from types import SimpleNamespace

    from umrs.api import jobs
    from umrs.jobs.models import JobType

    job = SimpleNamespace(project_id=P, job_type=JobType.KNOWLEDGE_INGEST, commit_attempt_id="at-2")
    attempts = [
        SimpleNamespace(attempt_id="at-1", metrics={"artifacts": [{"artifact_id": "art-old"}]}),
        SimpleNamespace(attempt_id="at-2", metrics={"artifacts": [{"artifact_id": "art-a"}]}),
    ]
    fake = SimpleNamespace(get_job=lambda _id: job, get_attempts_for_job=lambda _id: attempts)
    monkeypatch.setattr(jobs, "get_optional_job_store", lambda: fake)

    assert route.ingest_job_artifacts("job-7", P) == ["art-a"]
    assert route.ingest_job_artifacts("job-7", OTHER) == []


# ── the stand-alone RAG service (psycopg2 driver) ────────────────────────────


class _SyncCursor:
    """psycopg2-shaped cursor over the fake: recovers the ``$n`` arguments."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.rows: list = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, sql: str, params: tuple) -> None:
        assert "$" not in sql
        tag = _TAG.search(sql).group(1)
        order = [int(n) - 1 for n in re.findall(r"\$(\d+)", _SQL_BY_TAG[tag])]
        args = tuple(params[order.index(i)] for i in range(max(order) + 1))
        result = self.db.run(tag, args)
        self.rows, self.rowcount = (
            (result, len(result)) if isinstance(result, list) else ([], result)
        )

    def fetchall(self) -> list:
        return self.rows


class _SyncConn:
    def __init__(self, db: FakeDb) -> None:
        self.db = db

    def __enter__(self):
        self.saved = self.db.snapshot()
        self.db.in_tx = True
        return self

    def __exit__(self, exc_type, *_a):
        self.db.in_tx = False
        if exc_type is not None:
            self.db.t = self.saved
            self.db.rollbacks += 1
        else:
            self.db.commits += 1
        return False

    def cursor(self) -> _SyncCursor:
        return _SyncCursor(self.db)


class _SyncPool:
    def __init__(self, db: FakeDb) -> None:
        self.db = db
        self.out = 0

    def getconn(self):
        self.out += 1
        return _SyncConn(self.db)

    def putconn(self, _conn) -> None:
        self.out -= 1


@pytest.fixture()
def rag_service(monkeypatch, db, store):
    from umrs.api import rag_service as service
    from umrs.shared import pg

    pool = _SyncPool(db)
    monkeypatch.setattr(pg, "pg_configured_from_env", lambda: True)
    monkeypatch.setattr(pg, "get_default_sync_pool", lambda: pool)
    monkeypatch.setattr(route, "object_store", lambda: store)
    return TestClient(service.app, raise_server_exceptions=False), pool


def test_the_rag_service_deletes_the_documents_artifact(rag_service, db):
    client, pool = rag_service

    response = client.delete("/rag/documents/d1", params={"project_id": P})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == "d1" and body["artifact_id"] == "artifact-rag-d1"
    assert body["counts"]["nodes"] == 3 and body["counts"]["artifacts"] == 1
    assert "artifact-rag-d1" not in db.t["artifacts"]
    assert pool.out == 0


def test_the_rag_service_answers_404_for_an_unresolvable_document(rag_service, db):
    client, _pool = rag_service
    before = db.snapshot()

    response = client.delete("/rag/documents/doc-123", params={"project_id": P})

    assert response.status_code == 404
    assert "doc-123" in response.json()["detail"]
    assert db.t == before


def test_the_rag_service_rolls_back_a_failed_delete(rag_service, db, store):
    client, pool = rag_service
    db.fail_on = "delete_nodes"
    before = db.snapshot()

    response = client.delete("/rag/documents/art-a", params={"project_id": P})

    assert response.status_code == 500
    assert "nothing was deleted" in response.json()["detail"]
    assert db.t == before and store.deleted == [] and pool.out == 0


def test_the_rag_service_without_a_database_deletes_nothing(monkeypatch):
    from umrs.api import rag_service as service
    from umrs.shared import pg

    monkeypatch.setattr(pg, "pg_configured_from_env", lambda: False)

    response = TestClient(service.app, raise_server_exceptions=False).delete(
        "/rag/documents/doc-123", params={"project_id": "p"}
    )

    assert response.status_code == 500
    assert "doc-123" in response.json()["detail"]
    assert "not deleted" in response.json()["detail"]


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_to_pyformat_orders_the_arguments_by_placeholder():
    sql, params = core.to_pyformat("a = $2 AND b = $1 AND c = ANY($1::text[])", ["x", "y"])

    assert sql == "a = %s AND b = %s AND c = ANY(%s::text[])"
    assert params == ("y", "x", "x")


@pytest.mark.parametrize(
    ("source_ref", "staged", "keys"),
    [
        (CONTENT_KEY, STAGED, [CONTENT_KEY, "knowledge-uploads/p1/u-1/a.pdf"]),
        ("rag://d1", "", []),
        ("https://example.org/a.pdf", "", []),
        ("projects/p2/knowledge/artifacts/h-a", "", []),  # another project's key
        ("", f"s3://{BUCKET}/knowledge-uploads/p2/u/a.pdf", []),  # another project's stage
        ("", f"s3://{BUCKET}/elsewhere/a.pdf", []),
    ],
)
def test_only_the_projects_own_objects_are_targets(source_ref, staged, keys):
    row = ("art-a", "h-a", source_ref, {"staged_from": staged} if staged else {})

    assert [t.key for t in core.object_targets(row, P)] == keys


def test_a_staged_object_outside_the_platforms_buckets_is_kept():
    target = core.ObjectTarget(ref="s3://other/knowledge-uploads/p1/x", key="x", bucket="other")
    store = FakeStore()

    outcome = asyncio.run(core._delete_object(None, target, P, store))

    assert outcome["outcome"] == "kept" and store.deleted == []


# ── integration: a real Postgres (skipped without a database URL) ────────────


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("no database URL (--db-url / UMRS_TEST_DB_URL)")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.rollback()  # the test's own transaction: nothing is ever committed
        conn.close()


def _pg_seed(cur: Any) -> None:
    def artifact(aid: str, project: str, h: str, ref: str, meta: dict) -> None:
        cur.execute(
            "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, project_id, "
            "source_ref, status, metadata) VALUES (%s, %s, 'document', %s, %s, 'active', "
            "%s::jsonb)",
            (aid, h, project, ref, json.dumps(meta)),
        )

    def node(nid: str, project: str, node_type: str, content: dict, label: str = "n") -> None:
        cur.execute(
            "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
            "content_hash) VALUES (%s, %s, %s, %s, %s::jsonb, 'h')",
            (nid, project, label, node_type, json.dumps(content)),
        )

    def edge(eid: str, src: str, tgt: str, rel: str) -> None:
        cur.execute(
            "INSERT INTO knowledge_edges (edge_id, project_id, source_id, target_id, "
            "relation_type) VALUES (%s, 'md16-it', %s, %s, %s)",
            (eid, src, tgt, rel),
        )

    artifact("md16-a", "md16-it", "md16-h-a", "projects/md16-it/knowledge/artifacts/md16-h-a", {})
    artifact("md16-b", "md16-it", "md16-h-b", "", {})
    artifact(
        "artifact-rag-md16-d", "md16-it", "md16-h-r", "rag://md16-d", {"document_id": "md16-d"}
    )
    artifact("md16-x", "md16-it-other", "md16-h-x", "", {})
    cur.execute(
        "INSERT INTO knowledge_ir (ir_id, artifact_id, ir_type, content_hash) "
        "VALUES ('md16-ir', 'md16-a', 'document', 'h')"
    )
    node("md16-file", "md16-it", "file", {"artifact_id": "md16-a"})
    node("md16-chunk", "md16-it", "entity", {"kind": "chunk", "artifact_id": "md16-a"})
    node("md16-a", "md16-it", "entity", {}, label="artifact:md16-a")
    node("md16-file-b", "md16-it", "file", {"artifact_id": "md16-b"})
    node("md16-ragdoc", "md16-it", "file", {"artifact_id": "artifact-rag-md16-d"})
    node("md16-ragchunk", "md16-it", "entity", {"kind": "chunk", "document_id": "md16-d"})
    node("md16-foreign", "md16-it-other", "concept", {"artifact_id": "md16-a"})
    cur.execute("UPDATE knowledge_nodes SET source_ir = 'md16-ir' WHERE node_id = 'md16-file'")
    edge("md16-e1", "md16-chunk", "md16-file", "part_of")
    edge("md16-e2", "md16-file-b", "md16-file", "cites")
    edge("md16-e3", "md16-ragchunk", "md16-ragdoc", "part_of")
    cur.execute(
        "INSERT INTO knowledge_events (event_id, event_type, job_id, attempt_id, artifact_id) "
        "VALUES ('md16-ev', 'indexed', 'j', 'a', 'md16-a')"
    )
    cur.execute(
        "INSERT INTO verification_reports (report_id, project_id, version, spec_name) "
        "VALUES ('md16-rep', 'md16-it', 1, 's')"
    )
    cur.execute(
        "INSERT INTO verification_findings (finding_key, report_id, artifact_id, category, "
        "severity, title) VALUES ('md16-f', 'md16-rep', 'md16-a', 'c', 'info', 't')"
    )


def _pg(coro_fn, cur, *args):
    return asyncio.run(coro_fn(core.Psycopg2Tx(cur), *args))


@pytest.mark.integration
def test_the_sql_deletes_one_artifact_on_postgres(pg_conn: Any) -> None:
    with pg_conn.cursor() as cur:
        _pg_seed(cur)
        report, targets = _pg(core.delete_rows, cur, "md16-a", "md16-it")
        assert report.counts == {
            "embeddings": 0,
            "edges": 2,
            "nodes": 3,
            "findings": 1,
            "ir": 1,
            "events_detached": 1,
            "artifacts": 1,
        }
        assert [t.key for t in targets] == ["projects/md16-it/knowledge/artifacts/md16-h-a"]
        cur.execute("SELECT node_id FROM knowledge_nodes WHERE node_id LIKE 'md16-%' ORDER BY 1")
        assert [r[0] for r in cur.fetchall()] == [
            "md16-file-b",
            "md16-foreign",
            "md16-ragchunk",
            "md16-ragdoc",
        ]
        cur.execute("SELECT artifact_id FROM knowledge_events WHERE event_id = 'md16-ev'")
        assert cur.fetchall() == [(None,)]
        refs = _pg(core._references, cur, targets[0], "md16-it")
        assert refs == 0


@pytest.mark.integration
def test_the_sql_reaches_rag_chunks_and_refuses_other_projects_on_postgres(pg_conn) -> None:
    with pg_conn.cursor() as cur:
        _pg_seed(cur)
        assert _pg(core.resolve_document, cur, "md16-d", "md16-it") == "artifact-rag-md16-d"
        assert _pg(core.resolve_document, cur, "md16-x", "md16-it") is None
        report, _t = _pg(core.delete_rows, cur, "artifact-rag-md16-d", "md16-it")
        assert report.counts["nodes"] == 2
        with pytest.raises(core.ArtifactNotFound):
            _pg(core.delete_rows, cur, "md16-x", "md16-it")
