"""Pins for ADR-058 D-1 (WP-22 Leg 1): backend_job_id set-once contract."""

from __future__ import annotations

import pytest


def _registry():
    from umrs.orchestrator.endpoint_registry import EndpointRegistry

    return EndpointRegistry(default_ttl_sec=3600)


def _register(registry, job_id="5434"):
    return registry.register(
        project_id="default",
        model="Bonsai-8B",
        job_id=job_id,
        host="pending",
        port=13062,
        backend="slurm",
        ttl_sec=3600,
    )


class TestBackendJobIdSetOnce:
    def test_new_row_starts_identityless(self):
        registry = _registry()
        endpoint = _register(registry)
        assert endpoint.backend_job_id is None

    def test_set_once_success(self):
        registry = _registry()
        _register(registry)
        updated = registry.set_backend_job_id("5434", "384897")
        assert updated is not None
        assert updated.backend_job_id == "384897"
        assert registry.get_by_job_id("5434").backend_job_id == "384897"

    def test_same_value_is_idempotent(self):
        registry = _registry()
        _register(registry)
        registry.set_backend_job_id("5434", "384897")
        again = registry.set_backend_job_id("5434", "384897")
        assert again.backend_job_id == "384897"

    def test_different_value_is_refused_not_overwritten(self):
        registry = _registry()
        _register(registry)
        registry.set_backend_job_id("5434", "384897")
        kept = registry.set_backend_job_id("5434", "999999")
        assert kept is not None
        assert kept.backend_job_id == "384897"

    def test_absent_row_returns_none(self):
        registry = _registry()
        assert registry.set_backend_job_id("nope", "384897") is None

    def test_empty_identity_rejected(self):
        registry = _registry()
        _register(registry)
        with pytest.raises(ValueError, match="non-empty"):
            registry.set_backend_job_id("5434", "")

    def test_to_dict_surfaces_identity_honestly(self):
        registry = _registry()
        endpoint = _register(registry)
        d = endpoint.to_dict()
        assert d["backend_job_id"] is None
        registry.set_backend_job_id("5434", "384897")
        assert registry.get_by_job_id("5434").to_dict()["backend_job_id"] == "384897"


class TestPersistenceWireShape:
    def test_pg_row_encoder_carries_column_order(self):
        from umrs.orchestrator.registry_backend import PostgresBackend

        cols = PostgresBackend._select_columns()
        assert cols[-1] == "backend_job_id"
        assert cols.index("backend_job_id") == 15


class TestSSHControlCrossLoop:
    def test_per_loop_locks_allow_foreign_loop_operations(self):
        """Cross-loop family pin (surfaced live 2026-09-18 on bdrt): two
        different event loops must each obtain a lock without the
        'bound to a different event loop' deadlock."""
        import asyncio

        from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager

        manager = SSHControlManager(
            SSHControlConfig(ssh_target="dummy.invalid", ssh_user="u")
        )

        loops = []

        async def grab_and_release():
            loop = asyncio.get_running_loop()
            loops.append(loop)
            async with manager._loop_lock():
                return manager._locks[id(loop)]

        first = asyncio.new_event_loop()
        second = asyncio.new_event_loop()
        try:
            lock_a = first.run_until_complete(grab_and_release())
            lock_b = second.run_until_complete(grab_and_release())
            # Each loop gets its own lock; none of them is the other loop's object.
            assert lock_a is not lock_b
            assert manager._locks[id(loops[0])] is lock_a
            assert manager._locks[id(loops[1])] is lock_b
        finally:
            first.close()
            second.close()

    def test_same_loop_reuses_same_lock(self):
        import asyncio

        from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager

        manager = SSHControlManager(
            SSHControlConfig(ssh_target="dummy.invalid", ssh_user="u")
        )

        async def grab():
            async with manager._loop_lock():
                return 1

        loop = asyncio.new_event_loop()
        try:
            assert loop.run_until_complete(grab()) == 1
            assert loop.run_until_complete(grab()) == 1
            assert len(manager._locks) == 1
        finally:
            loop.close()
