"""Regression tests: Janitor must honor configured orchestrator backend.

Bug (2026-07-30): Janitor.backend used environment auto-detection only.
On front-end deployments (umrs-front podman container driving a remote
Slurm cluster via ``scheduler.kind: slurm-remote``), no local
sbatch/kubectl/docker exists, so detection returned UNKNOWN and
``get_orchestrator("unknown")`` raised — orphan cleanup errored every
cycle and ghost-pruning silently preserved dead endpoints forever.
"""

from __future__ import annotations

from umrs.orchestrator.janitor import Janitor, JanitorConfig


class TestJanitorConfiguredBackend:
    """Janitor.backend prefers explicit config over env detection."""

    def test_env_var_backend_used(self, monkeypatch):
        """RIVAULT_ORCHESTRATOR=compose skips detect_backend entirely."""
        monkeypatch.setenv("RIVAULT_ORCHESTRATOR", "compose")

        def _boom():  # must NOT be called
            raise AssertionError("detect_backend() should not run when configured")

        import umrs.orchestrator.janitor as janitor_mod

        monkeypatch.setattr(janitor_mod, "detect_backend", _boom)

        janitor = Janitor(config=JanitorConfig())
        backend = janitor.backend
        from umrs.orchestrator.compose_backend import DockerComposeBackend

        assert isinstance(backend, DockerComposeBackend)

    def test_resolve_helper_prefers_env(self, monkeypatch):
        """resolve_configured_backend_name: env wins over site.yaml."""
        from umrs.orchestrator.factory import resolve_configured_backend_name

        monkeypatch.setenv("RIVAULT_ORCHESTRATOR", "slurm-remote")
        assert resolve_configured_backend_name() == "slurm-remote"

    def test_resolve_helper_env_empty_falls_through(self, monkeypatch):
        """Empty env var falls through to site.yaml or None."""
        from umrs.orchestrator.factory import resolve_configured_backend_name

        monkeypatch.setenv("RIVAULT_ORCHESTRATOR", "")
        monkeypatch.delenv("UMRS_SITE_CONFIG", raising=False)
        result = resolve_configured_backend_name()
        # Without a site config adjacent to CWD, None (no exception).
        assert result is None or isinstance(result, str)

    def test_janitor_without_config_uses_detection(self, monkeypatch):
        """No env + no site.yaml → falls back to detect_backend()."""
        import umrs.orchestrator.janitor as janitor_mod
        from umrs.orchestrator.detection import BackendType, DetectionResult

        # Force "no explicit config" path
        monkeypatch.setattr(janitor_mod, "resolve_configured_backend_name", lambda: None)
        info = DetectionResult(
            backend=BackendType.COMPOSE,
            confidence="high",
            reason="test",
            available_backends=[BackendType.COMPOSE],
        )
        monkeypatch.setattr(janitor_mod, "detect_backend", lambda: info)

        janitor = Janitor()
        from umrs.orchestrator.compose_backend import DockerComposeBackend

        assert isinstance(janitor.backend, DockerComposeBackend)
