"""connect-info advertises the knowledge ingest models (ADR-116 amendment 2026-11-13).

Docling and GLiNER are on by default, served by the ``umrs-docmodels``
service when the front has one. connect-info carries a ``knowledge_models``
block so ``r4 doctor`` can show the state. connect-info is readable without
signing in, so the block never carries the service URL or a model path.
The service's ``/info`` is asked at most once a minute, with a 2 s timeout.

The settings and staging checks of ``umrs.knowledge.ingest_models`` are
stubbed; the service's ``/info`` goes through the real ``docmodels_client``
over an in-memory httpx transport.
"""

from __future__ import annotations

import json

import httpx
import pytest
from umrs.api import connect_info_knowledge as cik
from umrs.api import ops
from umrs.knowledge import docmodels_client
from umrs.knowledge import ingest_models as im

BUNDLED_DOCLING = "/opt/umrs/models/docling"
BUNDLED_GLINER = "/opt/umrs/models/gliner_medium-v2.1"
SERVICE_URL = "http://umrs-docmodels.internal.example:8091"
REVISION = "a" * 40

INFO_OK = {
    "service": "umrs-docmodels",
    "ready": True,
    "loading": False,
    "stuck": False,
    "docling": {
        "version": "2.150.0",
        "artifacts": BUNDLED_DOCLING,
        "staged": True,
        "loaded": True,
    },
    "gliner": {
        "repo": "urchade/gliner_medium-v2.1",
        "revision": REVISION,
        "dir": BUNDLED_GLINER,
        "staged": True,
        "loaded": True,
    },
    "limits": {"max_bytes": 67108864, "max_pages": 80, "concurrency": 1, "max_run_s": 300.0},
}
READY = {"url_set": True, "ready": True, "local_fallback": "sandbox"}


class _Settings:
    def __init__(self) -> None:
        self.parser = "docling"
        self.docling = BUNDLED_DOCLING
        self.gliner = BUNDLED_GLINER
        self.url = SERVICE_URL
        self.staged: set[str] = {BUNDLED_DOCLING, BUNDLED_GLINER}


@pytest.fixture
def settings(monkeypatch) -> _Settings:
    state = _Settings()
    stubs = {
        "configured_doc_parser": lambda: state.parser,
        "configured_docling_artifacts": lambda: state.docling,
        "configured_gliner_dir": lambda: state.gliner,
        "configured_docmodels_url": lambda: state.url,
        "is_docling_staged": lambda path: path in state.staged,
        "is_gliner_staged": lambda path: path in state.staged,
        "BUNDLED_DOCLING_DIR": BUNDLED_DOCLING,
        "BUNDLED_GLINER_DIR": BUNDLED_GLINER,
        "GLINER_REPO": "urchade/gliner_medium-v2.1",
        "GLINER_REVISION": REVISION,
    }
    for name, value in stubs.items():
        monkeypatch.setattr(im, name, value)
    monkeypatch.setattr(cik, "local_fallback", lambda: "sandbox")
    cik.clear_cache()
    yield state
    cik.clear_cache()


class _Fetch:
    def __init__(self, answer: dict | None) -> None:
        self.answer = answer
        self.timeouts: list[float] = []

    def __call__(self, timeout: float) -> dict | None:
        self.timeouts.append(timeout)
        return self.answer


def test_defaults_on_with_a_ready_service(settings) -> None:
    block = cik.build_knowledge_models(_Fetch(INFO_OK))
    assert block == {
        "doc_parser": "docling",
        "docling": {"enabled": True, "source": "bundled", "staged": True, "version": "2.150.0"},
        "gliner": {
            "enabled": True,
            "source": "bundled",
            "staged": True,
            "repo": "urchade/gliner_medium-v2.1",
            "revision": REVISION,
        },
        "service": READY,
    }


def test_block_names_no_address_and_no_path(settings) -> None:
    settings.docling = "/srv/site/models/docling"
    settings.gliner = "/srv/site/models/gliner"
    text = json.dumps(cik.build_knowledge_models(_Fetch(INFO_OK)))
    assert "umrs-docmodels.internal" not in text
    assert "8091" not in text
    assert "/srv/site" not in text
    assert "/opt/umrs" not in text


def test_custom_directories_are_named_custom(settings) -> None:
    settings.docling = "/srv/site/models/docling"
    settings.gliner = "/srv/site/models/gliner"
    block = cik.build_knowledge_models(_Fetch(None))
    assert block["docling"]["source"] == "custom"
    assert block["gliner"]["source"] == "custom"
    assert block["gliner"]["revision"] == ""
    assert block["docling"]["staged"] is False


def test_service_down_is_not_ready(settings) -> None:
    fetch = _Fetch(None)
    block = cik.build_knowledge_models(fetch)
    assert block["service"] == {**READY, "ready": False}
    assert fetch.timeouts == [2.0]


def test_service_missing_a_switched_on_model_is_not_ready(settings) -> None:
    info = {**INFO_OK, "gliner": {**INFO_OK["gliner"], "staged": False, "loaded": False}}
    assert cik.build_knowledge_models(_Fetch(info))["service"]["ready"] is False


@pytest.mark.parametrize(
    "info",
    [
        # Still loading: the directories are staged, nothing is loaded yet.
        {
            **INFO_OK,
            "ready": False,
            "loading": True,
            "docling": {**INFO_OK["docling"], "loaded": False},
            "gliner": {**INFO_OK["gliner"], "loaded": False},
        },
        # Docling failed to load (a broken wheel): staged but not loaded.
        {**INFO_OK, "docling": {**INFO_OK["docling"], "loaded": False}},
        # A request overran its deadline; the service restarts.
        {**INFO_OK, "stuck": True},
        # An older service that reports only "staged".
        {key: value for key, value in INFO_OK.items() if key not in ("ready", "loading")},
    ],
)
def test_staged_is_not_ready_until_the_service_loaded_it(settings, info) -> None:
    """service.ready came from the staged directories: connect-info said
    ready while the service was loading or had failed to load a model."""
    assert cik.build_knowledge_models(_Fetch(info))["service"]["ready"] is False


def test_a_front_with_its_own_model_is_not_served_by_the_service(settings) -> None:
    settings.gliner = "/srv/site/models/gliner"
    settings.staged.add(settings.gliner)
    assert cik.build_knowledge_models(_Fetch(INFO_OK))["service"]["ready"] is False


@pytest.mark.parametrize(
    ("mode", "sandbox", "effective"),
    [
        ("sandbox", "", "off"),
        ("sandbox", "podman", "sandbox"),
        ("inprocess", "", "inprocess"),
        ("off", "podman", "off"),
    ],
)
def test_the_local_fallback_in_effect(monkeypatch, mode, sandbox, effective) -> None:
    from umrs.knowledge import embed_sandbox

    monkeypatch.setenv("UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK", mode)
    monkeypatch.setattr(embed_sandbox, "worker_sandbox_config", lambda: sandbox or None)
    assert cik.local_fallback() == effective


def test_concurrent_misses_ask_the_service_once(settings) -> None:
    import threading

    started, release = threading.Event(), threading.Event()
    calls: list[float] = []

    def slow(timeout: float) -> dict:
        calls.append(timeout)
        started.set()
        release.wait(5)
        return INFO_OK

    threads = [
        threading.Thread(target=lambda: cik.connect_info_knowledge_models(fetch=slow))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    assert started.wait(5)
    release.set()
    for thread in threads:
        thread.join(5)
    assert calls == [2.0]


def test_no_service_url_never_asks_and_ready_is_unknown(settings) -> None:
    settings.url = ""
    fetch = _Fetch(INFO_OK)
    block = cik.build_knowledge_models(fetch)
    assert block["service"] == {"url_set": False, "ready": None}
    assert fetch.timeouts == []


def test_switched_off(settings) -> None:
    settings.parser = "pypdf"
    settings.gliner = ""
    block = cik.build_knowledge_models(_Fetch(INFO_OK))
    assert block["doc_parser"] == "pypdf"
    assert block["docling"]["enabled"] is False
    assert block["gliner"] == {
        "enabled": False,
        "source": "off",
        "staged": False,
        "repo": "urchade/gliner_medium-v2.1",
        "revision": REVISION,
    }


def test_answer_is_cached_for_a_minute(settings) -> None:
    fetch = _Fetch(INFO_OK)
    clock = [100.0]
    for _ in range(3):
        cik.connect_info_knowledge_models(fetch=fetch, now=lambda: clock[0])
    assert len(fetch.timeouts) == 1
    clock[0] += 61
    cik.connect_info_knowledge_models(fetch=fetch, now=lambda: clock[0])
    assert len(fetch.timeouts) == 2


def test_a_settings_change_is_not_served_from_the_cache(settings) -> None:
    fetch = _Fetch(INFO_OK)
    cik.connect_info_knowledge_models(fetch=fetch, now=lambda: 0.0)
    settings.gliner = ""
    block = cik.connect_info_knowledge_models(fetch=fetch, now=lambda: 1.0)
    assert block["gliner"]["enabled"] is False


def _serve_info(monkeypatch, handler) -> list[httpx.Request]:
    """The real docmodels client, over an in-memory transport."""
    seen: list[httpx.Request] = []

    def transport(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def new_client(timeout: float) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(transport),
            timeout=docmodels_client._timeout(timeout),
        )

    monkeypatch.setattr(docmodels_client, "_new_client", new_client)
    return seen


def test_fetch_asks_the_service_info_with_a_short_timeout(settings, monkeypatch) -> None:
    seen = _serve_info(monkeypatch, lambda _req: httpx.Response(200, json=INFO_OK))
    assert cik.fetch_service_info() == INFO_OK
    assert [str(req.url) for req in seen] == [f"{SERVICE_URL}/info"]
    assert seen[0].extensions["timeout"]["read"] == 2.0


def test_fetch_failure_reads_as_no_answer(settings, monkeypatch) -> None:
    def refuse(_req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _serve_info(monkeypatch, refuse)
    assert cik.fetch_service_info() is None


def test_fetch_of_a_loading_service_reads_as_no_answer(settings, monkeypatch) -> None:
    _serve_info(monkeypatch, lambda _req: httpx.Response(503, json={"error": "loading"}))
    assert cik.fetch_service_info() is None


def test_fetch_without_a_url_asks_nothing(settings, monkeypatch) -> None:
    settings.url = ""
    seen = _serve_info(monkeypatch, lambda _req: pytest.fail("must not ask"))
    assert cik.fetch_service_info() is None
    assert seen == []


def _route(monkeypatch) -> dict:
    monkeypatch.setattr(ops, "connect_info_default_models", lambda: {})
    result = ops.ops_connect_info(
        x_user_id=None,
        x_verified_project_id=None,
        x_verified_is_admin=None,
        x_verified_hmac=None,
    )
    return ops.ConnectInfoResponse(**result).model_dump()


def test_route_carries_the_block(settings, monkeypatch) -> None:
    monkeypatch.setattr(cik, "fetch_service_info", _Fetch(INFO_OK))
    body = _route(monkeypatch)
    assert body["knowledge_models"]["doc_parser"] == "docling"
    assert body["knowledge_models"]["service"] == READY


def test_route_survives_a_failing_block(settings, monkeypatch) -> None:
    def boom() -> dict:
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(cik, "connect_info_knowledge_models", boom)
    body = _route(monkeypatch)
    assert body["knowledge_models"] is None
    assert body["zmq"]["control_port"] > 0
