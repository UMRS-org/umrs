"""frontend_session lane registration (ADR-046 A2 P10): factory + backoff + build gate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from umrs.jobs.adapters import create_backend_factory
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType


def _factory(**kw):
    return create_backend_factory(
        orchestrator=MagicMock(), registry=MagicMock(), port_manager=MagicMock(), **kw
    )


def test_factory_refuses_loudly_when_the_lane_is_absent():
    with pytest.raises(RuntimeError, match="frontend_session lane"):
        _factory()(JobType.FRONTEND_SESSION)


def test_factory_returns_the_configured_lane():
    lane = object()
    assert _factory(frontend_session_backend=lane)(JobType.FRONTEND_SESSION) is lane


def test_backoff_entry_is_single_attempt():
    assert DEFAULT_BACKOFFS[JobType.FRONTEND_SESSION]["max_attempts"] == 1


def test_build_lane_is_none_without_venue_or_pool():
    from umrs.jobs.frontend_session_backend import build_frontend_session_lane
    from umrs.venues.frontend_config import FrontendVenueConfig

    assert build_frontend_session_lane(MagicMock(), FrontendVenueConfig.from_env({})) is None
    assert build_frontend_session_lane(None) is None
