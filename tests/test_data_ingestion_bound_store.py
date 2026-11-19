"""DATA_INGESTION reads the object store bound to the job's project (ADR-077)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import umrs.data_catalog.ingest as ingest
import umrs.shared.catalog_store_access as access
from umrs.jobs import data_ingestion
from umrs.shared.storage import project_storage as ps


class _FakeResolver(ps.ProjectStorageResolver):
    def __init__(
        self,
        bindings: dict[str, ps.StorageBinding] | None = None,
        failure: str = "",
    ) -> None:
        super().__init__(pool_getter=lambda: None)
        self._bindings = bindings or {}
        self._failure = failure

    def binding(self, project_id: str, *, pool: Any = None) -> ps.StorageBinding | None:  # noqa: ARG002
        if self._failure:
            raise ps.ProjectStorageUnavailable(self._failure)
        return self._bindings.get(project_id)


def _job(project_id: str) -> Any:
    return SimpleNamespace(
        job_id="job-x",
        project_id=project_id,
        user_id="alice",
        priority=0,
        spec={
            "bucket": "project-data",
            "s3_key": "incoming/document.txt",
            "object_etag": "etag-1",
        },
    )


def _attempt() -> Any:
    return SimpleNamespace(attempt_id="att", metrics={})


def _submit(project_id: str) -> str:
    return asyncio.run(
        data_ingestion.DataIngestionBackend().submit_job(_job(project_id), _attempt())
    )


@pytest.fixture
def _wiring(monkeypatch: pytest.MonkeyPatch) -> Any:
    events: list[tuple[Any, ...]] = []
    meta_store = object()

    def ingest_object(client: Any, bucket: str, **kwargs: Any) -> dict[str, str]:
        events.append(("ingest", client, bucket, kwargs))
        return {"action": "created"}

    ps.set_project_storage_resolver(None)
    monkeypatch.setattr(ingest, "ingest_object", ingest_object)
    monkeypatch.setattr(access, "get_catalog_meta_store", lambda: meta_store)
    yield events
    ps.set_project_storage_resolver(None)


def _forbidden_default(events: list[tuple[Any, ...]]) -> Any:
    def build() -> Any:
        events.append(("default-client",))
        raise AssertionError("the environment-default store must not be used")

    return build


def _default_client(events: list[tuple[Any, ...]], client: Any) -> Any:
    def build() -> Any:
        events.append(("default-client", client))
        return client

    return build


def test_bound_project_ingests_from_its_bound_store(_wiring, monkeypatch) -> None:
    locator = ps.S3Locator(
        endpoint_url="http://bound-store.invalid",
        host="bound-store.invalid",
        port=80,
        access_key="",
        secret_key="",
    )
    ps.set_project_storage_resolver(
        _FakeResolver({"bound-project": ps.StorageBinding("bound-project", s3=locator)})
    )
    bound_client = object()
    monkeypatch.setattr(ps, "s3_client_for", lambda _actual, **_kw: bound_client)
    monkeypatch.setattr(data_ingestion, "_build_s3_client", _forbidden_default(_wiring))

    ref = _submit("bound-project")

    assert [event[:3] for event in _wiring] == [
        ("ingest", bound_client, "project-data")
    ]
    assert _wiring[0][3] == {
        "s3_key": "incoming/document.txt",
        "object_etag": "etag-1",
        "project_id": "bound-project",
        "owner": "alice",
        "acl_scope": "bound-project",
        "meta_store": _wiring[0][3]["meta_store"],
    }
    assert ref == (
        "data_ingestion|bucket=project-data|key=incoming/document.txt|action=created"
    )


def test_unbound_project_keeps_the_env_client(_wiring, monkeypatch) -> None:
    ps.set_project_storage_resolver(_FakeResolver())
    default_client = object()
    monkeypatch.setattr(
        data_ingestion,
        "_build_s3_client",
        _default_client(_wiring, default_client),
    )

    _submit("default")

    assert [event[:3] for event in _wiring] == [
        ("default-client", default_client),
        ("ingest", default_client, "project-data"),
    ]


@pytest.mark.parametrize("failure", ["registry unreadable", "sealed_value_invalid"])
def test_failed_binding_never_falls_back(_wiring, monkeypatch, failure: str) -> None:
    ps.set_project_storage_resolver(_FakeResolver(failure=failure))
    monkeypatch.setattr(data_ingestion, "_build_s3_client", _forbidden_default(_wiring))

    with pytest.raises(RuntimeError, match="project storage binding unreadable"):
        _submit("bound-project")

    assert _wiring == []


def test_a_job_without_a_project_uses_the_env_client(_wiring, monkeypatch) -> None:
    ps.set_project_storage_resolver(_FakeResolver(failure="must not be consulted"))
    default_client = object()
    monkeypatch.setattr(
        data_ingestion,
        "_build_s3_client",
        _default_client(_wiring, default_client),
    )

    _submit("")

    assert [event[:3] for event in _wiring] == [
        ("default-client", default_client),
        ("ingest", default_client, "project-data"),
    ]
