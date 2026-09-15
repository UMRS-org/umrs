"""Tests for the Assistant Service removal from the compute node.

Validates that ``serve-model.yaml`` no longer includes the Assistant
Service as a co-located service inside mini-RiVault on the compute
node. The Assistant Service (ADR-012) has been moved to the frontend
server (umrs-front) as a persistent CPU-only service.

See:
- docs/adr/012-assistant-service-and-conversational-surfaces.md
- deploy/ansible/roles/front_end/templates/frontend-assistant-service.yaml.j2
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SERVE_MODEL = (
    Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm" / "serve-model.yaml"
)


@pytest.fixture
def serve_model_services() -> dict:
    """Load serve-model.yaml and return the services dict."""
    if yaml is None:
        pytest.skip("PyYAML not installed")
    if not SERVE_MODEL.exists():
        pytest.skip(f"{SERVE_MODEL} not found")
    with SERVE_MODEL.open() as f:
        doc = yaml.safe_load(f)
    return doc.get("services", {})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssistantServiceRemovedFromComputeNode:
    """Verify the Assistant Service has been removed from the compute node."""

    def test_no_assistant_service(self, serve_model_services: dict) -> None:
        """Assistant Service must not be defined in serve-model.yaml."""
        assert "assistant" not in serve_model_services, (
            "assistant service should have been removed from serve-model.yaml "
            "(moved to frontend server per ADR-012 amendment 2026-07-05)"
        )
