"""Unit tests for registry backend auto-detection."""

from unittest.mock import patch

from umrs.orchestrator.registry_backend import InMemoryBackend, get_backend


def test_get_backend_auto_detects_postgres_from_dsn(monkeypatch):
    monkeypatch.delenv("UMRS_REGISTRY_BACKEND", raising=False)
    monkeypatch.setenv(
        "UMRS_REGISTRY_POSTGRES_DSN",
        "postgres://umrs:umrs_rccs@localhost:5432/umrs_db",
    )

    with patch("umrs.orchestrator.registry_backend.PostgresBackend") as mock_backend:
        get_backend()

    mock_backend.assert_called_once_with("postgres://umrs:umrs_rccs@localhost:5432/umrs_db")


def test_get_backend_explicit_memory_overrides_dsn(monkeypatch):
    monkeypatch.setenv("UMRS_REGISTRY_BACKEND", "memory")
    monkeypatch.setenv(
        "UMRS_REGISTRY_POSTGRES_DSN",
        "postgres://umrs:umrs_rccs@localhost:5432/umrs_db",
    )

    backend = get_backend()

    assert isinstance(backend, InMemoryBackend)
