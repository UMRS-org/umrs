"""Regression: roster-defaults fallback must be observable, not silent.

Previously ``_resolve_roster_defaults`` returned ``None`` on *any* exception
with no log line at all, so a broken roster was indistinguishable from
"model has no recipe" in production logs.
"""

from __future__ import annotations

import logging


def test_resolve_roster_defaults_logs_on_failure(monkeypatch, caplog):
    from umrs.jobs import adapters

    def _boom(model_name):  # pragma: no cover - trivial
        raise RuntimeError("roster yaml unreadable")

    monkeypatch.setattr(
        "umrs.orchestrator.provision_roster.resolve_model_provision_defaults",
        _boom,
    )

    with caplog.at_level(logging.WARNING, logger="umrs.jobs.adapters"):
        result = adapters._resolve_roster_defaults("any-model")

    assert result is None, "fallback contract: None on failure"
    assert any("Roster defaults resolution failed" in r.getMessage() for r in caplog.records), (
        "failure must be observable at WARNING level"
    )


def test_resolve_roster_defaults_no_log_on_success(monkeypatch, caplog):
    from umrs.jobs import adapters

    monkeypatch.setattr(
        "umrs.orchestrator.provision_roster.resolve_model_provision_defaults",
        lambda _model: {"partition": "ng-dgx-m1"},
    )

    with caplog.at_level(logging.WARNING, logger="umrs.jobs.adapters"):
        result = adapters._resolve_roster_defaults("some-model")

    assert result == {"partition": "ng-dgx-m1"}
    assert not [
        r for r in caplog.records if "Roster defaults resolution failed" in r.getMessage()
    ], "success path must stay quiet"
