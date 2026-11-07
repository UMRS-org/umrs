"""Per-step forwarder from a Unix socket to the step's routed endpoint.

ADR-109 D-5 (ADR-074 D-5 phase 2). A staged step runs with
``--network=none`` and holds no credential. When its routing decision names
an endpoint, the job script starts this forwarder from the PLATFORM step
image (trusted code, outside the step container, before the step starts and
stopped after it ends). It listens on ``/run/umrs/endpoint.sock`` (the
step's ``reply/endpoint.sock`` on the node) and, for each HTTP/1.1 request:

- allows only GET and POST, and only the endpoint kind's paths
  (:data:`ALLOWED_PATHS`: ``/v1/*`` for vLLM; ``/infer/*``, ``/vision/*``
  and ``/readyz`` for the BentoML vision service);
- caps the request body, the response, and the bytes of the whole step;
- opens TCP to its ONE upstream ``host:port`` (no other destination, no
  CONNECT, no absolute-form target, no redirect followed), adds the
  endpoint's credential headers itself (read from a 0600 file the step pod
  never mounts), and streams the answer back;
- withholds any answer that carries the credential back;
- logs method, path, status and bytes (never a body, a header or a query),
  at most :data:`MAX_LOG_LINES` lines.

Anything else is refused by name: 400 malformed, 403 method or path, 411 a
body without Content-Length, 413 over a cap, 502 upstream failure.

Stdlib only: it runs as ``python -I -m umrs.server.step_endpoint_forwarder``
in the platform step image, whose closure is the standard library plus umrs.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import stat
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

#: The endpoint kinds and the request paths each one allows. An entry ending
#: in ``/`` is a prefix; any other entry is one exact path. The BentoML
#: vision service answers its task API under ``/infer/`` (submit, status,
#: get, cancel); ``/vision/`` covers a service mounted under that base path.
ALLOWED_PATHS: dict[str, tuple[str, ...]] = {
    "vllm": ("/v1/",),
    "bentoml": ("/infer/", "/vision/", "/readyz"),
}
ALLOWED_METHODS = ("GET", "POST")

#: Default caps (site override: ``step_executor.endpoint_max_*_bytes``).
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 4 * 1024**3
MAX_HEAD_BYTES = 64 * 1024
MAX_LOG_LINES = 10_000
MAX_CONNECTIONS = 16
CONNECT_TIMEOUT = 30.0
HEAD_TIMEOUT = 60.0
#: Idle read timeout on either side: a long generation or segmentation
#: still answers within it.
IDLE_TIMEOUT = 1800.0
CHUNK = 64 * 1024
#: After a refusal the forwarder keeps reading (and dropping) what the client
#: still sends, at most this long and this much, before it closes: a client
#: still writing its body otherwise sees a broken pipe instead of the refusal
#: ("lingering close"; 2026-11-07, flaky on a loaded CI runner).
LINGER_SECONDS = 5.0
LINGER_MAX_BYTES = 64 * 1024 * 1024
#: A credential value shorter than this is not scanned for in answers.
MIN_SECRET_LEN = 8

#: The step's socket inside the forwarder and the step pod (short: AF_UNIX
#: paths are limited to 107 bytes, a Lustre workdir path is not).
SOCKET_PATH = "/run/umrs/endpoint.sock"

_HOST_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?")
_TOKEN_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,64}")
_PATH_RE = re.compile(r"/[A-Za-z0-9._~!$&'()*+,;=:@/%-]*")
_VALUE_RE = re.compile(r"[\x20-\x7e\t]*")
_DOT_SEGMENTS = frozenset({".", ".."})
#: Client headers never forwarded: hop-by-hop, framing (the forwarder sets
#: its own), and anything that would name another destination or identity.
_DROPPED_HEADERS = frozenset(
    {
        "host",
        "connection",
        "keep-alive",
        "proxy-authorization",
        "proxy-authenticate",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "expect",
        "authorization",
        "cookie",
        "forwarded",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
    }
)
_REASONS = {
    100: "Continue",
    400: "Bad Request",
    403: "Forbidden",
    411: "Length Required",
    413: "Payload Too Large",
    502: "Bad Gateway",
}


class Refusal(Exception):
    """A request the forwarder answers itself, by name."""

    def __init__(self, status: int, why: str) -> None:
        super().__init__(why)
        self.status = status
        self.why = why


# ── the spec the server hands the job ───────────────────────────────────────


def parse_upstream(api_base: str) -> tuple[str, int]:
    """``(host, port)`` of a routed endpoint's ``http://host:port[/path]``;
    ValueError for anything else (another scheme, user info, no port)."""
    parts = urlsplit(str(api_base or "").strip())
    if parts.scheme != "http" or parts.username or parts.password:
        raise ValueError(f"api_base {str(api_base)[:120]!r} is not http://host:port")
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"api_base {str(api_base)[:120]!r}: {exc}") from exc
    if not _HOST_RE.fullmatch(host) or not port:
        raise ValueError(f"api_base {str(api_base)[:120]!r} has no host and port")
    return host, int(port)


def credential_headers(
    api_key: str | None, headers: dict[str, str] | None
) -> dict[str, str]:
    """The headers the forwarder adds: the decision's own, plus a Bearer
    ``Authorization`` from its key when the headers carry none."""
    out = {str(k): str(v) for k, v in (headers or {}).items()}
    if api_key and not any(k.lower() == "authorization" for k in out):
        out["Authorization"] = f"Bearer {api_key}"
    check_headers(out)
    return out


def check_headers(headers: dict[str, str]) -> None:
    """ValueError for a header name or value that cannot go on the wire."""
    for name, value in headers.items():
        if not _TOKEN_RE.fullmatch(name) or not _VALUE_RE.fullmatch(value):
            raise ValueError(f"credential header {name[:40]!r} cannot be forwarded")


def load_credential(path: str) -> dict[str, str]:
    """``{"headers": {...}}`` from the job's credential file (0600, never in
    the step pod)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    headers = raw.get("headers") if isinstance(raw, dict) else None
    if not isinstance(headers, dict):
        raise TypeError(f"{path}: no headers object")
    out = {str(k): str(v) for k, v in headers.items()}
    check_headers(out)
    return out


def secrets_of(headers: dict[str, str]) -> tuple[bytes, ...]:
    """The credential values an answer must never carry (the whole value
    and, for ``<scheme> <token>``, the token)."""
    found: set[bytes] = set()
    for value in headers.values():
        for part in (value, value.partition(" ")[2]):
            if len(part) >= MIN_SECRET_LEN:
                found.add(part.encode("latin-1"))
    return tuple(sorted(found, key=len, reverse=True))


# ── request checks ──────────────────────────────────────────────────────────


def _plain_path(path: str) -> bool:
    """An absolute path with no empty, dot or encoded dot/slash segment."""
    if not _PATH_RE.fullmatch(path) or "//" in path:
        return False
    lowered = path.lower()
    if any(code in lowered for code in ("%2e", "%2f", "%5c")):
        return False
    return not any(seg in _DOT_SEGMENTS for seg in path.split("/"))


def path_allowed(kind: str, target: str) -> bool:
    """True when *target* (origin form) is one of the kind's paths."""
    path = target.partition("?")[0]
    if not _plain_path(path):
        return False
    return any(
        path.startswith(entry) if entry.endswith("/") else path == entry
        for entry in ALLOWED_PATHS.get(kind, ())
    )


@dataclass(frozen=True)
class Request:
    method: str
    target: str
    headers: list[tuple[str, str]]

    def header(self, name: str) -> list[str]:
        return [v for k, v in self.headers if k.lower() == name]

    @property
    def path(self) -> str:
        """The path for the log: no query, printable, bounded."""
        return _printable(self.target.partition("?")[0])[:200]


def _printable(text: str) -> str:
    return "".join(c if 0x20 < ord(c) < 0x7F else "?" for c in text)


def parse_head(raw: bytes) -> Request:
    """The request line and headers; Refusal(400) when malformed."""
    try:
        text = raw.decode("latin-1")
    except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes all
        raise Refusal(400, "the request head is not text") from exc
    lines = text.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] not in ("HTTP/1.1", "HTTP/1.0"):
        raise Refusal(400, "malformed request line")
    headers = [_parse_header(line) for line in lines[1:] if line]
    return Request(parts[0], parts[1], headers)


def _parse_header(line: str) -> tuple[str, str]:
    name, sep, value = line.partition(":")
    if not sep or not _TOKEN_RE.fullmatch(name) or not _VALUE_RE.fullmatch(value):
        raise Refusal(400, "malformed header")
    return name, value.strip()


def body_length(request: Request, max_request: int) -> int:
    """The request body's length; refusals for chunked or oversized bodies."""
    if request.header("transfer-encoding"):
        raise Refusal(411, "a request body needs Content-Length (no chunked upload)")
    values = request.header("content-length")
    if not values:
        return 0
    if len(set(values)) != 1 or not values[0].isdigit():
        raise Refusal(400, "bad Content-Length")
    length = int(values[0])
    if length > max_request:
        raise Refusal(413, f"request body exceeds {max_request} bytes")
    return length


def check_request(request: Request, kind: str) -> None:
    """Refusal(403) for a method or path the endpoint kind does not allow."""
    if request.method not in ALLOWED_METHODS:
        raise Refusal(403, f"method {_printable(request.method)[:16]} is not allowed")
    if not path_allowed(kind, request.target):
        raise Refusal(403, f"path is not allowed for a {kind} endpoint")


def upstream_head(
    request: Request, host: str, port: int, credential: dict[str, str], length: int
) -> bytes:
    """The request head sent upstream: the client's own headers minus the
    dropped set and any credential name, then the credential, framing and
    ``Connection: close`` (one request per connection)."""
    own = {k.lower() for k in credential}
    lines = [f"{request.method} {request.target} HTTP/1.1", f"Host: {host}:{port}"]
    lines += [
        f"{k}: {v}"
        for k, v in request.headers
        if k.lower() not in _DROPPED_HEADERS and k.lower() not in own
    ]
    lines += [f"{k}: {v}" for k, v in credential.items()]
    if length or request.method == "POST":
        lines.append(f"Content-Length: {length}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def response_status(head: bytes) -> int:
    """The status code of an upstream response head (0: unreadable)."""
    parts = head.split(b"\r\n", 1)[0].split(b" ", 2)
    if len(parts) < 2 or not parts[0].startswith(b"HTTP/1.") or not parts[1].isdigit():
        return 0
    return int(parts[1])


def response_length(head: bytes) -> int | None:
    """The upstream's Content-Length, when it states one."""
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length" and value.strip().isdigit():
            return int(value.strip())
    return None


def refusal_bytes(status: int, why: str) -> bytes:
    body = json.dumps({"error": why, "forwarder": "umrs-step-endpoint"}).encode()
    head = (
        f"HTTP/1.1 {status} {_REASONS.get(status, 'Error')}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return head.encode("latin-1") + body


# ── the forwarder ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Limits:
    max_request: int = DEFAULT_MAX_REQUEST_BYTES
    max_response: int = DEFAULT_MAX_RESPONSE_BYTES
    max_total: int = DEFAULT_MAX_TOTAL_BYTES


@dataclass
class Log:
    """At most ``max_lines`` lines to *stream*, then one notice."""

    stream: Any = None
    max_lines: int = MAX_LOG_LINES
    lines: int = 0

    def __call__(self, message: str) -> None:
        self.lines += 1
        if self.lines > self.max_lines + 1:
            return
        if self.lines > self.max_lines:
            message = "log cap reached; further requests are not logged"
        out = self.stream or sys.stderr
        out.write(f"[umrs-endpoint] {message}\n")
        out.flush()


@dataclass
class _Exchange:
    """One request's facts for its log line."""

    request: Request | None = None
    status: int = 0
    sent: int = 0
    received: int = 0
    note: str = ""


class _Withheld(Exception):
    """The upstream answer carried the credential or passed a cap."""


@dataclass
class Forwarder:
    host: str
    port: int
    kind: str
    credential: dict[str, str]
    limits: Limits = field(default_factory=Limits)
    log: Log = field(default_factory=Log)
    used: int = 0
    requests: int = 0
    refused: int = 0
    _secrets: tuple[bytes, ...] = ()
    _slots: asyncio.Semaphore | None = None

    def __post_init__(self) -> None:
        if self.kind not in ALLOWED_PATHS:
            raise ValueError(
                f"endpoint kind {self.kind!r} is not {'|'.join(ALLOWED_PATHS)}"
            )
        check_headers(self.credential)
        self._secrets = secrets_of(self.credential)

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One request per connection; the connection closes after it."""
        if self._slots is None:
            self._slots = asyncio.Semaphore(MAX_CONNECTIONS)
        exchange = _Exchange()
        refused = False
        async with self._slots:
            try:
                await self._serve(reader, writer, exchange)
            except Refusal as exc:
                self.refused += 1
                refused = True
                exchange.status, exchange.note = exc.status, exc.why
                await _send(writer, refusal_bytes(exc.status, exc.why))
            except (
                _Withheld,
                OSError,
                asyncio.TimeoutError,
                asyncio.IncompleteReadError,
            ) as exc:
                exchange.note = exchange.note or f"aborted: {type(exc).__name__}"
            finally:
                self._log_exchange(exchange)
                if refused:
                    await _linger(reader, writer)
                await _close(writer)

    async def _serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        exchange: _Exchange,
    ) -> None:
        request = await self._read_request_head(reader)
        exchange.request = request
        check_request(request, self.kind)
        length = body_length(request, self.limits.max_request)
        if "100-continue" in [v.lower() for v in request.header("expect")]:
            await _send(writer, b"HTTP/1.1 100 Continue\r\n\r\n")
        body = await asyncio.wait_for(reader.readexactly(length), IDLE_TIMEOUT)
        head = upstream_head(request, self.host, self.port, self.credential, length)
        self._spend(len(head) + length, exchange)
        exchange.sent = len(head) + length
        self.requests += 1
        await self._relay(head + body, writer, exchange)

    async def _read_request_head(self, reader: asyncio.StreamReader) -> Request:
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT)
        except asyncio.LimitOverrunError as exc:
            raise Refusal(400, f"request head exceeds {MAX_HEAD_BYTES} bytes") from exc
        return parse_head(raw[:-4])

    def _spend(self, count: int, exchange: _Exchange) -> None:
        """Count *count* bytes against the step's total; Refusal(413) when
        the request itself would pass it."""
        if self.used + count > self.limits.max_total:
            raise Refusal(
                413,
                f"the step's endpoint budget of {self.limits.max_total} bytes is spent",
            )
        self.used += count

    async def _relay(
        self, payload: bytes, writer: asyncio.StreamWriter, exchange: _Exchange
    ) -> None:
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), CONNECT_TIMEOUT
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise Refusal(
                502, f"the endpoint did not answer: {type(exc).__name__}"
            ) from exc
        try:
            await _send(up_writer, payload)
            await self._answer(up_reader, writer, exchange)
        finally:
            await _close(up_writer)

    async def _answer(
        self,
        up_reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        exchange: _Exchange,
    ) -> None:
        try:
            head = await asyncio.wait_for(
                up_reader.readuntil(b"\r\n\r\n"), IDLE_TIMEOUT
            )
        except (asyncio.LimitOverrunError, asyncio.IncompleteReadError) as exc:
            raise Refusal(502, "the endpoint sent no readable response head") from exc
        self._check_head(head)
        exchange.status = response_status(head)
        await _send(writer, head)
        exchange.received = len(head)
        await self._stream_body(up_reader, writer, exchange, response_length(head))

    def _check_head(self, head: bytes) -> None:
        if response_status(head) == 0:
            raise Refusal(502, "the endpoint sent no readable response head")
        if self._carries_secret(head):
            raise Refusal(
                502, "the endpoint echoed its credential; the answer is withheld"
            )
        length = response_length(head)
        if length is not None and length + len(head) > self.limits.max_response:
            raise Refusal(
                502, f"the endpoint's answer exceeds {self.limits.max_response} bytes"
            )
        if self.used + len(head) + (length or 0) > self.limits.max_total:
            raise Refusal(
                413,
                f"the step's endpoint budget of {self.limits.max_total} bytes is spent",
            )
        self.used += len(head)

    async def _stream_body(
        self,
        up_reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        exchange: _Exchange,
        length: int | None,
    ) -> None:
        """Body bytes to the step as they come, holding back enough of the
        tail that a credential split across two chunks is still seen. A
        stated Content-Length ends the body; otherwise the upstream's close
        does (it was asked for ``Connection: close``)."""
        hold = max((len(s) for s in self._secrets), default=1) - 1
        pending = b""
        left = -1 if length is None else length
        while left != 0:
            want = CHUNK if left < 0 else min(CHUNK, left)
            chunk = await asyncio.wait_for(up_reader.read(want), IDLE_TIMEOUT)
            if not chunk:
                break
            left = left - len(chunk) if left > 0 else left
            self._count_answer(len(chunk), exchange)
            pending += chunk
            if self._carries_secret(pending):
                exchange.note = "the endpoint echoed its credential; the answer was cut"
                raise _Withheld()
            cut = len(pending) - hold
            if cut > 0:
                await _send(writer, pending[:cut])
                pending = pending[cut:]
        await _send(writer, pending)

    def _count_answer(self, count: int, exchange: _Exchange) -> None:
        exchange.received += count
        self.used += count
        if exchange.received > self.limits.max_response:
            exchange.note = f"answer passed {self.limits.max_response} bytes; cut"
            raise _Withheld()
        if self.used > self.limits.max_total:
            exchange.note = (
                f"the step's budget of {self.limits.max_total} bytes passed; cut"
            )
            raise _Withheld()

    def _carries_secret(self, data: bytes) -> bool:
        return any(secret in data for secret in self._secrets)

    def _log_exchange(self, exchange: _Exchange) -> None:
        request = exchange.request
        what = f"{_printable(request.method)[:16]} {request.path}" if request else "-"
        note = f" ({exchange.note})" if exchange.note else ""
        self.log(
            f"{what} -> {exchange.status or '-'} "
            f"out={exchange.sent} in={exchange.received}{note}"
        )

    def summary(self) -> str:
        return (
            f"stopped: {self.requests} forwarded, {self.refused} refused, "
            f"{self.used} bytes of {self.limits.max_total}"
        )


async def _send(writer: asyncio.StreamWriter, data: bytes) -> None:
    if data:
        writer.write(data)
        await asyncio.wait_for(writer.drain(), IDLE_TIMEOUT)


async def _linger(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Half-close our side (the client sees the end of the refusal and
    closes), then drop what the client still sends until it closes or
    LINGER_SECONDS / LINGER_MAX_BYTES run out."""
    with contextlib.suppress(OSError, RuntimeError):
        if writer.can_write_eof():
            writer.write_eof()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + LINGER_SECONDS
    dropped = 0
    with contextlib.suppress(asyncio.TimeoutError, OSError):
        while dropped < LINGER_MAX_BYTES:
            chunk = await asyncio.wait_for(
                reader.read(CHUNK), max(deadline - loop.time(), 0)
            )
            if not chunk:
                return
            dropped += len(chunk)


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


# ── the process ─────────────────────────────────────────────────────────────


def _clear_socket_path(path: str) -> None:
    """A stale socket at *path* is removed; anything else there refuses."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(st.st_mode):
        raise SystemExit(f"[umrs-endpoint] {path} exists and is not a socket")
    os.unlink(path)


async def serve(forwarder: Forwarder, path: str, stop: asyncio.Event) -> None:
    """Listen on *path* until *stop* is set."""
    _clear_socket_path(path)
    old = os.umask(0o077)
    try:
        server = await asyncio.start_unix_server(
            forwarder.handle, path=path, limit=MAX_HEAD_BYTES
        )
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    forwarder.log(
        f"listening for a {forwarder.kind} endpoint at {forwarder.host}:{forwarder.port}"
    )
    async with server:
        await stop.wait()
    forwarder.log(forwarder.summary())


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="umrs-step-endpoint")
    parser.add_argument("--socket", default=SOCKET_PATH)
    parser.add_argument("--credential", required=True)
    parser.add_argument("--upstream", required=True, help="host:port")
    parser.add_argument("--kind", required=True, choices=sorted(ALLOWED_PATHS))
    parser.add_argument(
        "--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES
    )
    parser.add_argument(
        "--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES
    )
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    return parser.parse_args(list(argv) if argv is not None else None)


def build(ns: argparse.Namespace) -> Forwarder:
    host, port = parse_upstream(f"http://{ns.upstream}")
    limits = Limits(ns.max_request_bytes, ns.max_response_bytes, ns.max_total_bytes)
    if min(limits.max_request, limits.max_response, limits.max_total) <= 0:
        raise ValueError("every byte cap must be positive")
    return Forwarder(host, port, ns.kind, load_credential(ns.credential), limits)


async def _main(ns: argparse.Namespace) -> None:
    forwarder = build(ns)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    await serve(forwarder, ns.socket, stop)


def main(argv: Iterable[str] | None = None) -> int:
    ns = _parse_args(argv)
    try:
        asyncio.run(_main(ns))
    except (OSError, ValueError, TypeError) as exc:
        print(f"[umrs-endpoint] FATAL: {exc}", file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":  # pragma: no cover - the job script's entry
    raise SystemExit(main())
