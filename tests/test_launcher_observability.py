"""Regression: launcher/site-config degradation paths must be observable.

Three failure shaping functions in the launcher path degraded silently:
site-config unreadable while resolving profiles, site-config unreadable while
selecting the slurm executor backend, and site-config unreadable while
resolving the rivault remote port. After audit-wave fixes, each emits a
WARNING with exc_info so operators can tell "feature degraded" from
"feature absent".
"""

from __future__ import annotations

import logging


def test_profiles_degradation_logs(monkeypatch, caplog):
    import umrs.server.rivault_launcher as rl

    def _boom():
        raise FileNotFoundError("/etc/umrs/site.yaml unreadable")

    import umrs.orchestrator.site_config as sc

    monkeypatch.setattr(sc, "get_site_config", _boom)

    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        result = rl._read_configured_profiles()

    assert result is None, "fallback contract preserved"
    assert any("Site config unreadable" in r.getMessage() for r in caplog.records), (
        "profiles degradation must be observable"
    )


def test_site_config_backend_degradation_logs(monkeypatch, caplog):
    import umrs.server.rivault_launcher as rl
    import umrs.orchestrator.site_config as sc

    monkeypatch.setattr(sc, "get_site_config", lambda: (_ for _ in ()).throw(RuntimeError("yaml")))

    with caplog.at_level(logging.WARNING, logger=rl.logger.name):
        result = rl._get_site_config_if_available()

    assert result is None
    assert any("Site config" in r.getMessage() for r in caplog.records)


def test_remote_port_degradation_logs(monkeypatch, caplog):
    import umrs.orchestrator.site_config as sc
    import umrs.server.config as server_config

    monkeypatch.setattr(sc, "get_site_config", lambda: (_ for _ in ()).throw(RuntimeError("yaml")))

    logger_name = server_config.__name__
    with caplog.at_level(logging.WARNING, logger=logger_name):
        port = server_config.get_rivault_remote_port()

    assert port <= 0, "fallback contract preserved (RIVAULT_REMOTE_PORT)"
    assert any("rivault remote port" in r.getMessage() for r in caplog.records), (
        "port fallback must be observable"
    )
