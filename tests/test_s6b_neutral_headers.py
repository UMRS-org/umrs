"""U3 S6b: the session-id and source headers are read under either name.

Core read ``x-r4-session-id`` and ``x-r4-source`` only; a sender on the
neutral ``x-umrs-*`` names went unheard. ``x-umrs-*`` wins when both are sent.
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from umrs.api.request_headers import session_id_header


def _request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw})


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, None),
        ({"x-r4-session-id": "s-r4"}, "s-r4"),
        ({"x-umrs-session-id": "s-umrs"}, "s-umrs"),
        ({"x-umrs-session-id": "s-umrs", "x-r4-session-id": "s-r4"}, "s-umrs"),
    ],
)
def test_session_id_header(headers, expected):
    assert session_id_header(_request(headers)) == expected


def test_routes_receive_the_neutral_session_id():
    app = FastAPI()

    @app.get("/probe")
    def probe(x_r4_session_id: str | None = Depends(session_id_header)) -> dict:
        return {"sid": x_r4_session_id}

    client = TestClient(app)
    assert client.get("/probe", headers={"x-umrs-session-id": "s-1"}).json() == {"sid": "s-1"}
    assert client.get("/probe", headers={"x-r4-session-id": "s-2"}).json() == {"sid": "s-2"}
    assert client.get("/probe").json() == {"sid": None}


def test_the_loop_reads_the_neutral_session_id():
    from umrs.api.loop import _companion_session

    assert _companion_session(_request({"x-umrs-session-id": "s-1"})) == "s-1"
    assert _companion_session(_request({"x-umrs-session-id": "bad id!"})) == ""


def test_the_infer_proxy_reads_the_neutral_source():
    from umrs.infer_proxy.app import _source_header

    assert _source_header(_request({"x-umrs-source": "site"})) == "site"
    assert _source_header(_request({"x-r4-source": "rikyu"})) == "rikyu"
    assert _source_header(_request({})) is None
