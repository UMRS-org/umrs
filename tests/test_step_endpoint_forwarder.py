"""The per-step endpoint forwarder and the step SDK over its socket
(ADR-109 D-5).

A fake upstream (a real TCP server on 127.0.0.1) records what reaches it;
the forwarder listens on a real Unix socket; requests come raw (asyncio) or
through the step SDK (``umrs.pipeline_sdk.endpoint``, stdlib http.client
over AF_UNIX). Pinned: an allowed path passes with the credential added and
the client's own Authorization dropped; a disallowed method or path is 403,
an oversize request 413, a spent step budget 413, all by name; one upstream
only (an absolute-form target or CONNECT never opens another connection);
an answer that carries the credential back is withheld; the log carries
method, path, status and bytes, never a header, body or query.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import pickle
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from umrs.pipeline_sdk import endpoint as sdk
from umrs.pipeline_sdk.step import StepContext
from umrs.server import step_endpoint_forwarder as fwd

SECRET = "sk-endpoint-secret-0123456789"
CREDENTIAL = {"Authorization": f"Bearer {SECRET}", "X-Api-Tenant": "tenant-7"}


@dataclass
class Upstream:
    """A one-request-per-connection HTTP server that records each request."""

    status: int = 200
    body: bytes = b'{"ok": true}'
    content_type: str = "application/json"
    echo_headers: bool = False
    extra_head: str = ""
    #: Seconds the upstream keeps the connection open after its answer.
    linger: float = 0.0
    requests: list[dict] = field(default_factory=list)
    connections: int = 0
    port: int = 0

    async def handle(self, reader, writer):
        self.connections += 1
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
        lines = head.split("\r\n")
        headers = dict(line.split(": ", 1) for line in lines[1:] if line)
        length = int(headers.get("Content-Length", "0"))
        body = await reader.readexactly(length) if length else b""
        self.requests.append({"line": lines[0], "headers": headers, "body": body})
        out = head.encode() if self.echo_headers else self.body
        writer.write(
            (
                f"HTTP/1.1 {self.status} X\r\nContent-Type: {self.content_type}\r\n"
                f"{self.extra_head}Content-Length: {len(out)}\r\nConnection: close\r\n\r\n"
            ).encode()
            + out
        )
        await writer.drain()
        await asyncio.sleep(self.linger)
        writer.close()


@pytest.fixture
async def chain(tmp_path):
    """(upstream, forwarder, socket path, log stream), the forwarder serving."""
    upstream = Upstream()
    server = await asyncio.start_server(upstream.handle, "127.0.0.1", 0)
    upstream.port = server.sockets[0].getsockname()[1]
    log = io.StringIO()
    forwarder = fwd.Forwarder(
        "127.0.0.1",
        upstream.port,
        "vllm",
        dict(CREDENTIAL),
        fwd.Limits(max_request=4096, max_response=64 * 1024, max_total=256 * 1024),
        fwd.Log(log),
    )
    path = str(tmp_path / "endpoint.sock")
    stop = asyncio.Event()
    task = asyncio.create_task(fwd.serve(forwarder, path, stop))
    for _ in range(100):
        if os.path.exists(path):
            break
        await asyncio.sleep(0.01)
    yield upstream, forwarder, path, log
    stop.set()
    await task
    server.close()
    await server.wait_closed()


async def _raw(path: str, data: bytes) -> tuple[int, bytes]:
    reader, writer = await asyncio.open_unix_connection(path)
    writer.write(data)
    await writer.drain()
    answer = await reader.read()
    writer.close()
    status = int(answer.split(b" ", 2)[1]) if answer.startswith(b"HTTP/1.1 ") else 0
    return status, answer


def _post(target: str, body: bytes = b"{}", extra: str = "", method: str = "POST") -> bytes:
    return (
        f"{method} {target} HTTP/1.1\r\nHost: endpoint\r\n{extra}"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode() + body


# ── allowed traffic ─────────────────────────────────────────────────────────


async def test_an_allowed_path_passes_with_the_credential_added(chain):
    upstream, _forwarder, path, _log = chain
    status, answer = await _raw(
        path, _post("/v1/chat/completions", b'{"m": 1}', "Authorization: Bearer mine\r\n")
    )
    assert status == 200 and answer.endswith(b'{"ok": true}')
    [request] = upstream.requests
    assert request["line"] == "POST /v1/chat/completions HTTP/1.1"
    assert request["body"] == b'{"m": 1}'
    # the forwarder's credential, never the step's own; one Host: the upstream's
    assert request["headers"]["Authorization"] == f"Bearer {SECRET}"
    assert request["headers"]["X-Api-Tenant"] == "tenant-7"
    assert request["headers"]["Host"] == f"127.0.0.1:{upstream.port}"
    assert request["headers"]["Connection"] == "close"


async def test_a_stated_length_ends_the_answer_even_if_the_upstream_lingers(chain):
    upstream, _forwarder, path, _log = chain
    upstream.linger = 5.0  # ignores Connection: close
    status, answer = await asyncio.wait_for(_raw(path, _post("/v1/x")), 2.0)
    assert status == 200 and answer.endswith(b'{"ok": true}')


async def test_the_sdk_reaches_the_endpoint_through_the_socket(chain, monkeypatch):
    upstream, _forwarder, path, _log = chain
    monkeypatch.setenv(sdk.ENDPOINT_SOCKET_ENV, path)
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="o")
    http = ctx.endpoint_http()
    reply = await asyncio.to_thread(http.post, "/v1/chat/completions", json={"q": "hi"})
    assert reply.status == 200 and reply.json() == {"ok": True}
    assert json.loads(upstream.requests[0]["body"]) == {"q": "hi"}
    assert upstream.requests[0]["headers"]["Authorization"] == f"Bearer {SECRET}"


async def test_the_sdk_sees_a_refusal_by_name(chain):
    _upstream, _forwarder, path, _log = chain
    http = sdk.endpoint_http(path)
    reply = await asyncio.to_thread(http.post, "/admin/shutdown", json={})
    assert reply.status == 403
    with pytest.raises(sdk.EndpointHTTPError, match="path is not allowed for a vllm endpoint"):
        reply.raise_for_status()


async def test_the_httpx_client_rides_the_same_socket(chain):
    pytest.importorskip("httpx")
    upstream, _forwarder, path, _log = chain
    with sdk.endpoint_httpx(path) as client:
        reply = await asyncio.to_thread(client.get, "/v1/models")
    assert reply.status_code == 200
    assert upstream.requests[0]["line"] == "GET /v1/models HTTP/1.1"


def test_a_step_without_a_routed_endpoint_fails_by_name(monkeypatch):
    monkeypatch.delenv(sdk.ENDPOINT_SOCKET_ENV, raising=False)
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="o")
    with pytest.raises(sdk.NoRoutedEndpoint, match="this step has no routed endpoint"):
        ctx.endpoint_http()
    with pytest.raises(sdk.NoRoutedEndpoint):
        ctx.endpoint_vision()


async def test_the_socket_is_the_owners_only(chain):
    _upstream, _forwarder, path, _log = chain
    mode = os.stat(path).st_mode
    assert stat.S_ISSOCK(mode) and stat.S_IMODE(mode) == 0o600


# ── refusals, by name ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("data", "status", "why"),
    [
        (_post("/admin", method="DELETE"), 403, "method DELETE is not allowed"),
        (b"CONNECT evil.example:443 HTTP/1.1\r\nHost: x\r\n\r\n", 403, "method CONNECT"),
        (_post("/metrics"), 403, "path is not allowed for a vllm endpoint"),
        (_post("http://evil.example/v1/models"), 403, "path is not allowed"),
        (_post("/v1/../admin"), 403, "path is not allowed"),
        (_post("/v1/%2e%2e/admin"), 403, "path is not allowed"),
        (_post("/v1//x"), 403, "path is not allowed"),
        (_post("/v1/x", b"y" * 5000), 413, "request body exceeds 4096 bytes"),
        (
            b"POST /v1/x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
            411,
            "needs Content-Length",
        ),
        (b"GARBAGE\r\n\r\n", 400, "malformed request line"),
    ],
)
async def test_anything_else_is_refused_by_name_and_never_reaches_the_upstream(
    chain, data, status, why
):
    upstream, forwarder, path, _log = chain
    got, answer = await _raw(path, data)
    assert got == status
    assert json.loads(answer.split(b"\r\n\r\n", 1)[1])["error"].find(why) >= 0
    assert upstream.connections == 0  # one upstream, and not even that one
    assert forwarder.refused == 1


async def test_the_steps_byte_budget_is_spent_by_name(chain):
    upstream, forwarder, path, _log = chain
    forwarder.used = forwarder.limits.max_total - 10
    status, answer = await _raw(path, _post("/v1/x"))
    assert status == 413 and b"endpoint budget" in answer
    assert upstream.connections == 0


async def test_an_answer_past_the_response_cap_is_not_passed_on(chain):
    upstream, _forwarder, path, _log = chain
    upstream.body = b"z" * (70 * 1024)
    status, answer = await _raw(path, _post("/v1/x"))
    assert status == 502 and b"exceeds" in answer
    assert b"zzzz" not in answer


async def test_an_answer_that_echoes_the_credential_is_withheld(chain):
    upstream, _forwarder, path, log = chain
    upstream.echo_headers = True  # a debug endpoint echoing its request
    _status, answer = await _raw(path, _post("/v1/echo"))
    assert SECRET.encode() not in answer
    head, body = answer.split(b"\r\n\r\n", 1)
    stated = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
    assert len(body) < stated  # the body was cut before the credential
    assert "echoed its credential; the answer was cut" in log.getvalue()
    assert SECRET not in log.getvalue()


async def test_a_credential_in_the_response_head_is_withheld(chain):
    upstream, _forwarder, path, _log = chain
    upstream.extra_head = f"X-Debug: {SECRET}\r\n"
    status, answer = await _raw(path, _post("/v1/x"))
    assert status == 502 and SECRET.encode() not in answer
    assert b"echoed its credential" in answer


async def test_a_dead_upstream_is_a_502_by_name(chain):
    _upstream, forwarder, path, _log = chain
    forwarder.port = 1  # nothing listens there
    status, answer = await _raw(path, _post("/v1/x"))
    assert status == 502 and b"did not answer" in answer


async def test_the_log_names_method_path_status_and_bytes_only(chain):
    _upstream, _forwarder, path, log = chain
    await _raw(path, _post("/v1/chat/completions?api_key=leak", b'{"secret": "body"}'))
    await _raw(path, _post("/nope"))
    text = log.getvalue()
    assert "POST /v1/chat/completions -> 200 out=" in text
    assert "POST /nope -> 403" in text and "path is not allowed" in text
    for needle in ("leak", "secret", SECRET, "Bearer", "tenant-7"):
        assert needle not in text


def test_the_log_is_capped():
    out = io.StringIO()
    log = fwd.Log(out, max_lines=3)
    for i in range(10):
        log(f"line {i}")
    lines = out.getvalue().splitlines()
    assert len(lines) == 4 and "log cap reached" in lines[-1]


# ── the spec and the credential file ────────────────────────────────────────


@pytest.mark.parametrize(
    ("kind", "path", "allowed"),
    [
        ("vllm", "/v1/chat/completions", True),
        ("vllm", "/v1/models?x=1", True),
        ("vllm", "/v2/x", False),
        ("vllm", "/infer/submit", False),
        ("bentoml", "/infer/submit", True),
        ("bentoml", "/infer/status?task_id=1", True),
        ("bentoml", "/vision/infer/submit", True),
        ("bentoml", "/readyz", True),
        ("bentoml", "/readyzz", False),
        ("bentoml", "/probe", False),
        ("bentoml", "/v1/chat/completions", False),
        ("other", "/v1/x", False),
    ],
)
def test_each_kind_allows_only_its_paths(kind, path, allowed):
    assert fwd.path_allowed(kind, path) is allowed


@pytest.mark.parametrize(
    ("api_base", "expect"),
    [
        ("http://gb10-07:8401", ("gb10-07", 8401)),
        ("http://10.129.0.31:4000/v1", ("10.129.0.31", 4000)),
    ],
)
def test_the_upstream_is_the_decisions_host_and_port(api_base, expect):
    assert fwd.parse_upstream(api_base) == expect


@pytest.mark.parametrize(
    "api_base",
    ["https://x:1", "http://x", "http://u:p@x:1", "file:///etc/passwd", "http://x y:1", ""],
)
def test_an_upstream_that_is_not_http_host_port_is_refused(api_base):
    with pytest.raises(ValueError):
        fwd.parse_upstream(api_base)


def test_the_credential_is_the_decisions_headers_plus_its_key():
    assert fwd.credential_headers("k-1", {"X-A": "b"}) == {
        "X-A": "b",
        "Authorization": "Bearer k-1",
    }
    assert fwd.credential_headers("k-1", {"authorization": "Basic z"}) == {
        "authorization": "Basic z"
    }
    with pytest.raises(ValueError):
        fwd.credential_headers(None, {"X-A": "b\r\nX-Evil: 1"})


def test_the_forwarder_builds_from_its_arguments(tmp_path):
    cred = tmp_path / "endpoint.json"
    cred.write_text(json.dumps({"headers": CREDENTIAL}))
    ns = fwd._parse_args(
        ["--credential", str(cred), "--upstream", "gb10-07:8401", "--kind", "bentoml",
         "--max-total-bytes", "99"]
    )  # fmt: skip
    built = fwd.build(ns)
    assert (built.host, built.port, built.kind) == ("gb10-07", 8401, "bentoml")
    assert built.credential == CREDENTIAL and built.limits.max_total == 99
    assert ns.socket == "/run/umrs/endpoint.sock"


def test_a_bad_credential_file_stops_the_forwarder(tmp_path, capsys):
    cred = tmp_path / "endpoint.json"
    cred.write_text("[]")
    rc = fwd.main(["--credential", str(cred), "--upstream", "h:1", "--kind", "vllm",
                   "--socket", str(tmp_path / "s.sock")])  # fmt: skip
    assert rc == 78 and "no headers object" in capsys.readouterr().err


def test_a_non_socket_at_the_socket_path_is_never_removed(tmp_path):
    target = tmp_path / "endpoint.sock"
    target.write_text("step's file")
    with pytest.raises(SystemExit, match="not a socket"):
        fwd._clear_socket_path(str(target))
    assert target.read_text() == "step's file"


# ── the vision helper through a bentoml forwarder ───────────────────────────


@dataclass
class VisionUpstream(Upstream):
    """BentoML's task API: submit, status, get (a pickled dict)."""

    async def handle(self, reader, writer):
        self.connections += 1
        head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
        lines = head.split("\r\n")
        headers = dict(line.split(": ", 1) for line in lines[1:] if line)
        length = int(headers.get("Content-Length", "0"))
        body = await reader.readexactly(length) if length else b""
        self.requests.append({"line": lines[0], "headers": headers, "body": body})
        target = lines[0].split(" ")[1]
        if target.startswith("/infer/submit"):
            out, ctype = b'{"task_id": "t-1", "status": "in_progress"}', "application/json"
        elif target.startswith("/infer/status"):
            out, ctype = b'{"task_id": "t-1", "status": "success"}', "application/json"
        else:
            out, ctype = pickle.dumps({"masks": 3}), "application/octet-stream"
        writer.write(
            f"HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {len(out)}\r\n"
            "Connection: close\r\n\r\n".encode() + out
        )
        await writer.drain()
        writer.close()


async def test_the_vision_helper_submits_polls_and_gets_through_the_socket(tmp_path):
    upstream = VisionUpstream()
    server = await asyncio.start_server(upstream.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    forwarder = fwd.Forwarder("127.0.0.1", port, "bentoml", {"Authorization": "Bearer v-key-12345"},
                              log=fwd.Log(io.StringIO()))  # fmt: skip
    path = str(tmp_path / "v.sock")
    stop = asyncio.Event()
    task = asyncio.create_task(fwd.serve(forwarder, path, stop))
    while not os.path.exists(path):
        await asyncio.sleep(0.01)
    vision = sdk.VisionClient(sdk.endpoint_http(path), poll_interval=0.01, timeout=5)
    result = await asyncio.to_thread(
        vision.submit_and_wait, "sam3", "s3://b/frame.png", {"prompt": "cell"}
    )
    stop.set()
    await task
    server.close()
    assert result == {"masks": 3}
    lines = [r["line"] for r in upstream.requests]
    assert lines == [
        "POST /infer/submit HTTP/1.1",
        "GET /infer/status?task_id=t-1 HTTP/1.1",
        "GET /infer/get?task_id=t-1 HTTP/1.1",
    ]
    assert json.loads(upstream.requests[0]["body"]) == {
        "model_type": "sam3",
        "inference_config": {"prompt": "cell"},
        "image_url": "s3://b/frame.png",
    }
    assert all(r["headers"]["Authorization"] == "Bearer v-key-12345" for r in upstream.requests)


# ── the process, as the job script runs it ──────────────────────────────────


def _wait_for(path: Path, proc: subprocess.Popen) -> None:
    for _ in range(200):
        if path.exists() or proc.poll() is not None:
            return
        time.sleep(0.05)


def test_the_module_serves_the_socket_and_logs_a_summary_on_term(tmp_path):
    cred = tmp_path / "endpoint.json"
    cred.write_text(json.dumps({"headers": CREDENTIAL}))
    sock = tmp_path / "endpoint.sock"
    src = Path(__file__).resolve().parents[1] / "src"
    proc = subprocess.Popen(
        [sys.executable, "-m", "umrs.server.step_endpoint_forwarder", "--socket", str(sock),
         "--credential", str(cred), "--upstream", "127.0.0.1:1", "--kind", "vllm"],
        env={**os.environ, "PYTHONPATH": str(src)},
        stderr=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        _wait_for(sock, proc)
        assert sdk.endpoint_http(str(sock), timeout=10).get("/metrics").status == 403
    finally:
        proc.send_signal(signal.SIGTERM)
        _out, err = proc.communicate(timeout=10)
    assert proc.returncode == 0, err
    assert "listening for a vllm endpoint at 127.0.0.1:1" in err
    assert "GET /metrics -> 403" in err
    assert "stopped: 0 forwarded, 1 refused" in err
    assert SECRET not in err
