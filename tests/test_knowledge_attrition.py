"""WP-15 dwell sweep pins: corpus attrition report read-only semantics."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.files import get_project_id
from umrs.api.knowledge_attrition import _bucket, _report_for_project, router


def _app(monkeypatch, pool, project_id: str = "proj-9"):
    async def _pool(_request):
        return pool

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_project_id] = lambda: project_id
    monkeypatch.setattr("umrs.api.knowledge_attrition.knowledge_asyncpg_pool", _pool)
    return TestClient(app, raise_server_exceptions=False)


class _Conn:
    def __init__(self, rows: list[tuple]):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def fetch(self, sql, *args):
        return self._rows


class TestBucketMapping:
    def test_bucket_edges(self):
        assert _bucket(1) == "lt_30d"
        assert _bucket(29) == "lt_30d"
        assert _bucket(30) == "d30_to_90"
        assert _bucket(89) == "d30_to_90"
        assert _bucket(90) == "gt_90d"
        assert _bucket(None) == "unknown"


class TestReportShape:
    def test_sweep_counts_buckets_and_venues(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        rows = [
            {
                "id": 1,
                "corpus_venue": "endpoint",
                "document_id": "doc-1",
                "created_at": now - timedelta(days=10),
            },
            {
                "id": 2,
                "corpus_venue": "endpoint",
                "document_id": "doc-1",
                "created_at": now - timedelta(days=10),
            },
            {
                "id": 3,
                "corpus_venue": "endpoint",
                "document_id": "doc-2",
                "created_at": now - timedelta(days=45),
            },
            {
                "id": 4,
                "corpus_venue": "central-pg",
                "document_id": "doc-3",
                "created_at": now - timedelta(days=200),
            },
            {
                "id": 5,
                "corpus_venue": None,
                "document_id": "doc-4",
                "created_at": now - timedelta(days=400),
            },
        ]
        pool = MagicMock()
        pool.acquire = lambda: _Conn(rows)

        report = asyncio.run(_report_for_project(pool, "proj-9"))
        assert report["rows_total"] == 5
        assert report["documents"] == 5 - 1  # doc-1 shared two vectors
        assert report["buckets"]["lt_30d"]["documents"] == 1
        assert report["buckets"]["d30_to_90"]["documents"] == 1
        assert report["buckets"]["gt_90d"]["documents"] == 2
        assert report["venues"]["endpoint"] == 3
        assert report["venues"]["central-pg"] == 1
        assert report["venues"]["unset"] == 1


class TestRouteReadOnly:
    def test_route_never_mutates(self):
        import inspect

        from umrs.api.knowledge_attrition import corpus_attrition_report as route

        src = inspect.getsource(route) + inspect.getsource(_report_for_project)
        forbidden = ("DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "TRUNCATE")
        for token in forbidden:
            assert (
                f'"{token}' not in src
                and f"('{token}'" not in src
                and f"conn.execute({token}" not in src
            )

    def test_project_id_required(self, monkeypatch):
        app = _app(monkeypatch, pool=MagicMock(), project_id="")
        r = app.post("/umrs/knowledge/corpus-attrition-report")
        assert r.status_code == 400

    def test_all_projects_requires_admin_true_gate(self, monkeypatch):
        pool = MagicMock()
        pool.acquire = lambda: _Conn(rows=[])

        app = _app(monkeypatch, pool=pool)
        r = app.post(
            "/umrs/knowledge/corpus-attrition-report?all_projects=true",
            headers={"x-verified-is-admin": "false", "x-user-id": "u-1"},
        )
        assert r.status_code == 403
