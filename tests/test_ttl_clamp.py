"""WP-16 round 2: TTL<=0 rejected at every request boundary (owner decision).

Policy: reject TTL<=0 at input validation (422 pydantic / 400 route-level)
and at the generation layer (sidecar). jobs.py's existing validator set the
pattern (None/-1/positive allowed); workload_identity, traces,
(The tool-cache pin retired with the unmounted /umrs/tool-cache router —
ADR-053; the jobs-side tool cache store is covered by test_batch5_*.)
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
    from umrs.jobs.workload_identity import IssueRequest

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
        "job_id": "j",
        "step_name": "n",
        "step_type": "generic",
        "inputs": None,
        "model_ref": None,
    }
    with pytest.raises(ValidationError):
        TraceCreateRequest(**fields, ttl_seconds=0)
    with pytest.raises(ValidationError):
        TraceCreateRequest(**fields, ttl_seconds=-900)
    assert TraceCreateRequest(**fields).ttl_seconds is None


