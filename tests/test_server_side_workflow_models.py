"""``user_script://`` / ``user_workflow://`` bucket models (ADR-074 D-2).

Both schemes build a ``WorkflowPipeline`` inside ``load_model``, which imports
the workflow's step modules and runs their ``setup()`` in the UMRS server —
any authenticated user's config (``/ops/reload``, the ZMQ FILE command)
reaches it. They load only where the site chose the recorded legacy
transition (``step_executor.uploaded_code: legacy_in_process``).
"""

from __future__ import annotations

import pytest

loading = pytest.importorskip("umrs.inference.loading")


@pytest.fixture
def pipelines(monkeypatch):
    """Record every WorkflowPipeline built (i.e. every uploaded tree imported)."""
    import umrs.pipeline_sdk.workflow as workflow

    built: list[str] = []

    class _Recorded:
        def __init__(self, source_uri, *_args, **_kwargs):
            built.append(source_uri)
            raise RuntimeError("imported")

    monkeypatch.setattr(workflow, "WorkflowPipeline", _Recorded)
    monkeypatch.setattr("umrs.server.step_trust._load_site_env", lambda: None)
    return built


def _cfg(scheme: str) -> dict:
    return {
        "model_name": "demo",
        "vector_size": "0",
        "model_type": f"{scheme}://evil/workflow.json",
        "return_type": "file",
    }


@pytest.mark.parametrize("scheme", ["user_script", "user_workflow"])
@pytest.mark.parametrize("policy", ["", "refuse", "something-else"])
def test_server_side_workflow_models_are_refused_before_any_import(
    pipelines, monkeypatch, scheme, policy
):
    monkeypatch.setenv("UMRS_STEP_UPLOADED_CODE", policy)
    with pytest.raises(ValueError, match=rf"{scheme}:// imports uploaded workflow code"):
        loading.load_model(_cfg(scheme), "image")
    assert pipelines == []


def test_the_legacy_transition_still_loads_them(pipelines, monkeypatch):
    # user_workflow:// gets past the check too, then fails on its own
    # long-broken import (umrs.inference.plugins does not exist).
    monkeypatch.setenv("UMRS_STEP_UPLOADED_CODE", "legacy_in_process")
    with pytest.raises(RuntimeError, match="imported"):
        loading.load_model(_cfg("user_script"), "image")
    assert pipelines == ["evil/workflow.json"]
