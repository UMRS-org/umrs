"""Project-binding for the UMRS tool-cache (wave-2 closure of open item).

Pre-fix audit: session_id alone formed the cache key across every route;
knowing one cross-tenant session_id granted read/wipe rights on its cached
tool outputs. Fix namespaced session keys by the authenticated project and
gated the cross-project cleanup to admin.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI

from umrs.api import tool_cache
from umrs.api.files import get_project_id


class _FakeStore:
    def __init__(self):
        self.put_calls = []
        self.list_calls = []
        self.get_calls = []
        self.invalidate_calls = []
        self.cleanup_calls = 0

    def put(self, **kw):
        self.put_calls.append(kw)

        class _Entry:
            def to_dict(self):
                return {"session_id": kw["session_id"], "tool_name": kw["tool_name"]}

        return _Entry()

    def list_session(self, session_id):
        self.list_calls.append(session_id)
        return []

    def get(self, session_id, tool_name, inputs):
        self.get_calls.append((session_id, tool_name, inputs))
        return None

    def invalidate(self, session_id, tool_name):
        self.invalidate_calls.append((session_id, tool_name))
        return 1

    def cleanup_expired(self):
        self.cleanup_calls += 1
        return 0


@pytest.fixture
def app():
    store = _FakeStore()
    tool_cache.set_store(store)
    app = FastAPI()
    app.dependency_overrides[get_project_id] = lambda: "proj-a"
    app.include_router(tool_cache.router)
    return app, store


def test_put_is_project_namespaced(app):
    app_instance, store = app
    client = TestClient(app_instance)
    resp = client.post(
        "/umrs/tool-cache",
        json={
            "session_id": "s1",
            "tool_name": "t",
            "inputs": {},
            "output": {},
        },
    )
    assert resp.status_code == 200
    assert store.put_calls[0]["session_id"] == "proj-a\x1fs1"


def test_list_is_project_namespaced(app):
    app_instance, store = app
    client = TestClient(app_instance)
    client.get("/umrs/tool-cache/session/s1")
    assert store.list_calls == ["proj-a\x1fs1"]


def test_invalidate_is_project_namespaced(app):
    app_instance, store = app
    client = TestClient(app_instance)
    client.delete("/umrs/tool-cache/session/s1")
    assert store.invalidate_calls[0][0] == "proj-a\x1fs1"


def test_cleanup_requires_admin_403():
    with pytest.raises(HTTPException) as exc:
        # Call the route function directly with non-admin
        import asyncio

        asyncio.run(tool_cache.cleanup_expired(is_admin=False))
    assert exc.value.status_code == 403
