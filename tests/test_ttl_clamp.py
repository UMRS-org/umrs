"""WP-16 round 2: TTL<=0 rejected at every request boundary (owner decision).

Policy: reject TTL<=0 at input validation (422 pydantic / 400 route-level)
and at the generation layer (sidecar). jobs.py's existing validator set the
pattern (None/-1/positive allowed); workload_identity, traces, tool_cache,
and the s3 presign generator get the same clamp now.
"""

from __future__ import annotations

import pytest


# -- sidecar generator (generation-layer rejection) --------------------------


def test_generate_presigned_url_ttl_clamp():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # ai4s-platform
    sidecar = str(root / "gateway" / "sidecar")
    if sidecar not in sys.path:
        sys.path.insert(0, sidecar)
    import importlib

    prev = sys.modules.get("main")
    main = importlib.import_module("main")
    try:
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            main.generate_presigned_url("bucket", "key", ttl_seconds=0)
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            main.generate_presigned_url("bucket", "key", ttl_seconds=-300)
        # preservation: a valid TTL still mints the full URL
        url = main.generate_presigned_url("bucket", "key", ttl_seconds=60)
        assert url.startswith(f"{main.S3_ENDPOINT}/bucket/key?")
        assert "Expires=" in url
    finally:
        if prev is not None:
            sys.modules["main"] = prev


# -- umrs request-model boundaries (422) -------------------------------------


def test_workload_identity_issue_request_ttl_422():
    from pydantic import ValidationError
    from umrs.api.workload_identity import IssueRequest

    base = {"job_id": "j", "step_name": "s", "scope": "x"}
    with pytest.raises(ValidationError):
        IssueRequest(**base, ttl_seconds=0)
    with pytest.raises(ValidationError):
        IssueRequest(**base, ttl_seconds=-5)
    ok = IssueRequest(**base)  # default 900
    assert ok.ttl_seconds == 900


def test_trace_request_ttl_422():
    from pydantic import ValidationError
    from umrs.api.traces import TraceCreateRequest

    fields = {
        "job_id": "j", "step_name": "n", "step_type": "generic",
        "inputs": None, "model_ref": None,
    }
    with pytest.raises(ValidationError):
        TraceCreateRequest(**fields, ttl_seconds=0)
    with pytest.raises(ValidationError):
        TraceCreateRequest(**fields, ttl_seconds=-900)
    assert TraceCreateRequest(**fields).ttl_seconds is None


def test_tool_cache_route_rejects_nonpositive_ttl():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import files, tool_cache

    class _Store:
        def put(self, **kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(to_dict=lambda: dict(kwargs))

    app = FastAPI()
    app.include_router(tool_cache.router)
    app.dependency_overrides[files.get_project_id] = lambda: "proj-a"
    client = TestClient(app)
    tool_cache.set_store(_Store())
    try:
        body = {
            "session_id": "s1", "tool_name": "probe",
            "inputs": {}, "output": "x", "ttl_seconds": 0,
        }
        r = client.post("/umrs/tool-cache", json=body)
        assert r.status_code == 400
        body["ttl_seconds"] = -10
        r = client.post("/umrs/tool-cache", json=body)
        assert r.status_code == 400
        body["ttl_seconds"] = True  # bool is not an int here
        r = client.post("/umrs/tool-cache", json=body)
        assert r.status_code == 400
        body["ttl_seconds"] = 120
        r = client.post("/umrs/tool-cache", json=body)
        assert r.status_code == 200
        assert r.json()["ttl_seconds"] == 120
    finally:
        tool_cache.set_store(None)
