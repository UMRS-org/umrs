"""U3 S6c: the steps' platform LLM gateway has a neutral name.

``UMRS_PLATFORM_GATEWAY_URL`` (``UMRS_GATEWAY_URL`` is already the UMRS API
base) is read before the legacy ``R4_GATEWAY_URL``. Both names count as the
platform's own base, so the platform key rides to either and nowhere else.
"""

from __future__ import annotations

import pytest
from umrs.shared.compat_names import platform_gateway_url


def test_neutral_name_wins_then_legacy():
    assert (
        platform_gateway_url(
            {"UMRS_PLATFORM_GATEWAY_URL": "https://a", "R4_GATEWAY_URL": "https://b"}
        )
        == "https://a"
    )
    assert platform_gateway_url({"R4_GATEWAY_URL": "https://b"}) == "https://b"
    assert platform_gateway_url({"UMRS_GATEWAY_URL": "https://umrs-api"}) == ""


@pytest.fixture
def router(monkeypatch):
    from umrs.server.step_router import StepRouter

    for name in ("OPENAI_BASE_URL", "UMRS_PLATFORM_GATEWAY_URL", "R4_GATEWAY_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    return StepRouter.__new__(StepRouter)


def _bare(router):
    router.default_api_base = None
    router.default_api_key = None
    return router


def test_the_platform_key_rides_to_a_neutral_named_base(router, monkeypatch):
    # Fails on the old router: only R4_GATEWAY_URL counted as the platform base.
    monkeypatch.setenv("UMRS_PLATFORM_GATEWAY_URL", "https://gw.example/v1/")
    r = _bare(router)
    assert r._is_platform_base("https://gw.example/v1")
    assert r._explicit_key({}, "https://gw.example/v1") == "sk-platform"
    assert r._explicit_key({}, "https://elsewhere.example/v1") is None


def test_the_legacy_name_still_counts(router, monkeypatch):
    monkeypatch.setenv("R4_GATEWAY_URL", "https://legacy.example")
    assert _bare(router)._is_platform_base("https://legacy.example/")
