"""Tests for the Assistant Service in the production Slurm launch path.

Validates that ``serve-model.yaml`` includes the Assistant Service with
the correct production configuration:

- Uses a pre-built image (not runtime pip install)
- Has ``profiles: [ assistant ]`` (opt-in)
- Sets offline env vars (``TRANSFORMERS_OFFLINE=1``, ``HF_HUB_OFFLINE=1``)
- Connects to LiteLLM as upstream
- Has a healthcheck
- Uses the ``genai-private-net`` network
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


@pytest.fixture
def assistant_service(serve_model_services: dict) -> dict:
    """Return the assistant service definition."""
    svc = serve_model_services.get("assistant")
    if svc is None:
        pytest.fail("assistant service not defined in serve-model.yaml")
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssistantServiceProdConfig:
    """Verify the assistant service is correctly configured for production."""

    def test_service_exists(self, serve_model_services: dict) -> None:
        assert "assistant" in serve_model_services

    def test_uses_prebuilt_image(self, assistant_service: dict) -> None:
        """Must use a pre-built image, not python:3.12-slim with runtime pip."""
        image = assistant_service.get("image", "")
        assert "umrs-assistant" in image, f"Expected pre-built umrs-assistant image, got: {image}"

    def test_no_runtime_pip(self, assistant_service: dict) -> None:
        """Command must not include pip install (offline production constraint)."""
        command = str(assistant_service.get("command", ""))
        assert "pip install" not in command, (
            "Runtime pip install forbidden in production (offline constraint)"
        )
        assert "uvicorn" in command, "Command must start uvicorn"

    def test_has_assistant_profile(self, assistant_service: dict) -> None:
        """Service should be opt-in via profiles."""
        profiles = assistant_service.get("profiles", [])
        assert "assistant" in profiles, f"Expected 'assistant' in profiles, got: {profiles}"

    def test_sets_offline_env(self, assistant_service: dict) -> None:
        """Must set TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1."""
        env = assistant_service.get("environment", {})
        assert str(env.get("TRANSFORMERS_OFFLINE", "")) == "1", (
            "TRANSFORMERS_OFFLINE=1 required for offline production"
        )
        assert str(env.get("HF_HUB_OFFLINE", "")) == "1", (
            "HF_HUB_OFFLINE=1 required for offline production"
        )

    def test_connects_to_litellm_upstream(self, assistant_service: dict) -> None:
        """Upstream URL must point to litellm."""
        env = assistant_service.get("environment", {})
        upstream = str(env.get("ASSISTANT_UPSTREAM_BASE_URL", ""))
        assert "litellm" in upstream, f"Expected upstream to reference litellm, got: {upstream}"

    def test_has_healthcheck(self, assistant_service: dict) -> None:
        """Service should have a healthcheck for readiness probing."""
        healthcheck = assistant_service.get("healthcheck")
        assert healthcheck is not None, "Missing healthcheck"
        assert "8090" in str(healthcheck.get("test", "")), "Healthcheck should probe port 8090"

    def test_uses_genai_private_net(self, assistant_service: dict) -> None:
        """Service should be on the genai-private-net network."""
        networks = assistant_service.get("networks", {})
        assert "genai-private-net" in networks, (
            f"Expected genai-private-net in networks, got: {list(networks)}"
        )

    def test_has_cap_drop_all(self, assistant_service: dict) -> None:
        """Service should drop all Linux capabilities."""
        cap_drop = assistant_service.get("cap_drop", [])
        assert "ALL" in cap_drop, f"Expected ALL in cap_drop, got: {cap_drop}"

    def test_extra_hosts_include_litellm(self, assistant_service: dict) -> None:
        """Service should resolve litellm hostname via extra_hosts."""
        extra_hosts = assistant_service.get("extra_hosts", [])
        host_entries = [str(entry) for entry in extra_hosts if isinstance(entry, str)]
        # extra_hosts can be list of "name:addr" or dict
        if isinstance(extra_hosts, list):
            assert any("litellm" in entry for entry in host_entries), (
                f"Expected litellm in extra_hosts, got: {extra_hosts}"
            )
        elif isinstance(extra_hosts, dict):
            assert "litellm" in extra_hosts, f"Expected litellm in extra_hosts, got: {extra_hosts}"
