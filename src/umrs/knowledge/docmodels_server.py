"""The ``umrs-docmodels`` service: warm Docling and GLiNER for the understand job.

ADR-116 amendment (2026-11-13). One long-running process per front host, on
the internal network only, run from the umrs server image as uid 65534::

    python -m umrs.knowledge.docmodels_server

It loads the bundled (or configured) models once, in a background thread at
startup, and then serves every front's understand job:

- ``GET /livez``: 200 while the process serves (the container healthcheck);
  503 ``stuck`` once a request overran its deadline (see below);
- ``GET /readyz``: 200 ``{"ready": true, "docling": bool, "gliner": bool}``
  once loading finished with at least one model; 503 ``loading``,
  ``no_models`` or ``stuck`` otherwise;
- ``GET /proof`` (header ``X-Docmodels-Nonce``): ``{"proof": HMAC-SHA256(token,
  nonce)}``, so a client checks the listener holds the shared token before
  it sends anything (``docmodels_client``);
- ``GET /info``: versions, directories, staged and loaded state, limits;
- ``POST /parse`` ``{"filename", "data_b64", "max_pages"?, "artifacts"?}`` ->
  ``{"text", "parser": "docling@<version>", "pages"}``;
- ``POST /tag`` ``{"texts", "labels", "threshold"?, "model_dir"?}`` ->
  ``{"candidates": [[{"text", "label", "score"}, ...], ...]}`` (the ingest
  worker's ``tag`` reply).

With ``UMRS_DOCMODELS_TOKEN`` set, ``/info``, ``/parse`` and ``/tag`` need
``Authorization: Bearer <token>`` (401 ``unauthorized``, compared in
constant time); ``/livez`` and ``/readyz`` stay open for the healthcheck.

Every error is JSON ``{"error": "<code>", "detail": "..."}``: ``too_large``
(413, a document over ``UMRS_KNOWLEDGE_DOCLING_MAX_BYTES``, default 64 MiB,
the same setting the servers read), ``too_many_pages`` (422, a PDF over
``UMRS_KNOWLEDGE_DOCLING_MAX_PAGES``, default 80, also shared),
``model_mismatch`` (409, the caller is set to another model directory),
``busy`` (503: no slot of ``UMRS_DOCMODELS_CONCURRENCY``, default 1 per
operation, came free within the caller's budget capped at
``UMRS_DOCMODELS_QUEUE_WAIT_S``, default 30 s, or
``UMRS_DOCMODELS_MAX_REQUESTS``, default 4 per operation, are already
waiting), ``loading`` / ``docling_unavailable`` / ``gliner_unavailable``
(503), ``bad_request`` (400), ``parse_failed`` / ``tag_failed`` (500).

Deadlines. Parse and tag have their own slots. Each request runs for at most
its caller's budget (``X-Docmodels-Budget``, capped at
``UMRS_DOCMODELS_MAX_RUN_S``, default 300 s), counted from when it asked. A
thread cannot be stopped, so a watchdog that finds a request still running
``GRACE_S`` past its deadline frees its slot, turns ``/livez`` and
``/readyz`` to 503 ``stuck`` and ends the process with exit code
``STUCK_EXIT_CODE``: the container restarts it (``restart:``) and the models
load again, instead of one document holding the only slot while the
healthcheck stays green. Body decoding and the PDF page count run in a
worker thread inside the slot, never on the event loop.

The Hugging Face hub is forced offline (``HF_HUB_OFFLINE=1``,
``TRANSFORMERS_OFFLINE=1``): nothing is downloaded at runtime. SIGTERM is
uvicorn's graceful shutdown: requests in flight finish (up to
``GRACEFUL_S``) before the process exits.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from starlette.requests import Request

from . import ingest_models

logger = logging.getLogger(__name__)

SERVICE_NAME = "umrs-docmodels"
HOST_ENV = "UMRS_DOCMODELS_HOST"
PORT_ENV = "UMRS_DOCMODELS_PORT"
TOKEN_ENV = "UMRS_DOCMODELS_TOKEN"
CONCURRENCY_ENV = "UMRS_DOCMODELS_CONCURRENCY"
QUEUE_WAIT_ENV = "UMRS_DOCMODELS_QUEUE_WAIT_S"
MAX_RUN_ENV = "UMRS_DOCMODELS_MAX_RUN_S"
MAX_REQUESTS_ENV = "UMRS_DOCMODELS_MAX_REQUESTS"
DEFAULT_HOST = "0.0.0.0"  # the internal compose network only
DEFAULT_PORT = 8091
DEFAULT_CONCURRENCY = 1
DEFAULT_QUEUE_WAIT_S = 30.0
DEFAULT_MAX_RUN_S = 300.0
DEFAULT_MAX_REQUESTS = 4
GRACEFUL_S = 30
GRACE_S = 5.0
WATCH_INTERVAL_S = 1.0
STUCK_EXIT_CODE = 75
BUDGET_HEADER = "x-docmodels-budget"
NONCE_HEADER = "x-docmodels-nonce"
OPS = ("parse", "tag")
# base64 grows a document by 4/3; the JSON envelope adds a little.
_ENVELOPE_BYTES = 1024 * 1024


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Limits:
    max_bytes: int = ingest_models.DEFAULT_DOCLING_MAX_BYTES
    max_pages: int = ingest_models.DEFAULT_DOCLING_MAX_PAGES
    concurrency: int = DEFAULT_CONCURRENCY
    queue_wait_s: float = DEFAULT_QUEUE_WAIT_S
    max_run_s: float = DEFAULT_MAX_RUN_S
    max_requests: int = DEFAULT_MAX_REQUESTS

    @classmethod
    def from_env(cls) -> Limits:
        return cls(
            max_bytes=ingest_models.configured_docling_max_bytes(),
            max_pages=ingest_models.configured_docling_max_pages(),
            concurrency=int(_env_number(CONCURRENCY_ENV, DEFAULT_CONCURRENCY)),
            queue_wait_s=_env_number(QUEUE_WAIT_ENV, DEFAULT_QUEUE_WAIT_S),
            max_run_s=_env_number(MAX_RUN_ENV, DEFAULT_MAX_RUN_S),
            max_requests=int(_env_number(MAX_REQUESTS_ENV, DEFAULT_MAX_REQUESTS)),
        )

    @property
    def body_bytes(self) -> int:
        """The request body cap: a ``max_bytes`` document, base64-encoded, in JSON."""
        return self.max_bytes * 4 // 3 + _ENVELOPE_BYTES

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_bytes": self.max_bytes,
            "max_pages": self.max_pages,
            "concurrency": self.concurrency,
            "max_run_s": self.max_run_s,
        }


@dataclass(frozen=True)
class Loaders:
    """The model seams: the real Docling and GLiNER code, or fakes in tests."""

    warm_docling: Callable[[str], Any]
    load_gliner: Callable[[str], Any]
    convert: Callable[[bytes, str, str, int], tuple[str, str, int]]
    tag: Callable[[Any, list[str], list[str], float], list[list[dict[str, Any]]]]
    docling_version: Callable[[], str]


def default_loaders() -> Loaders:
    """``doc_parse`` and ``pretag``: the same code the local path runs."""
    from . import doc_parse, pretag

    return Loaders(
        warm_docling=doc_parse.warm_converter,
        load_gliner=pretag.load_model,
        convert=doc_parse.docling_parse,
        tag=pretag.tag_texts,
        docling_version=doc_parse.docling_version,
    )


@dataclass
class ModelState:
    """What loaded. ``docling_dir`` / ``gliner_dir`` are "" when switched off."""

    docling_dir: str
    gliner_dir: str
    loading: bool = True
    docling: bool = False
    gliner: Any = None
    errors: dict[str, str] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)

    @classmethod
    def from_env(cls) -> ModelState:
        parser = ingest_models.configured_doc_parser()
        docling_dir = (
            ingest_models.configured_docling_artifacts() if parser == "docling" else ""
        )
        return cls(
            docling_dir=docling_dir, gliner_dir=ingest_models.configured_gliner_dir()
        )

    def ready(self) -> bool:
        return not self.loading and (self.docling or self.gliner is not None)


def _load_one(
    state: ModelState,
    name: str,
    path: str,
    staged: Callable[[str], bool],
    load: Callable[[str], Any],
) -> tuple[bool, Any]:
    """``(loaded, model)``; a model off, unstaged or failing is named in ``errors``."""
    if not path:
        state.errors[name] = "off"
        return False, None
    if not staged(path):
        state.errors[name] = "not_staged"
        return False, None
    try:
        return True, load(path)
    except Exception as exc:  # noqa: BLE001 — named in /readyz and the log
        state.errors[name] = f"load_failed:{type(exc).__name__}"
        logger.warning("docmodels: %s failed to load from %s: %s", name, path, exc)
        return False, None


def load_models(state: ModelState, loaders: Loaders) -> None:
    """Load both models once (the warm part); never raises."""
    try:
        state.docling, _ = _load_one(
            state,
            "docling",
            state.docling_dir,
            ingest_models.is_docling_staged,
            loaders.warm_docling,
        )
        ok, model = _load_one(
            state,
            "gliner",
            state.gliner_dir,
            ingest_models.is_gliner_staged,
            loaders.load_gliner,
        )
        state.gliner = model if ok else None
    finally:
        state.loading = False
        state.done.set()
    logger.info("docmodels: loaded docling=%s gliner=%s", state.docling, ok)


class ServiceError(Exception):
    """An HTTP error with a JSON ``{"error": code}`` body."""

    def __init__(self, status: int, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass
class _Run:
    """One request holding a slot until ``deadline`` (``time.monotonic()``)."""

    op: str
    slots: threading.BoundedSemaphore
    deadline: float
    released: bool = False


def exit_for_restart(op: str) -> None:
    """The default stuck action: end the process so the container restarts it."""
    logger.critical(
        "docmodels: a %s request overran its deadline; exiting (%d) to restart",
        op,
        STUCK_EXIT_CODE,
    )
    os._exit(STUCK_EXIT_CODE)


@dataclass
class Service:
    state: ModelState
    loaders: Loaders
    limits: Limits
    warm: bool = True
    watch: bool = True
    token: str = ""
    on_stuck: Callable[[str], None] = exit_for_restart
    stuck: str = ""
    slots: dict[str, threading.BoundedSemaphore] = field(init=False)
    waiting: dict[str, int] = field(init=False)
    running: list[_Run] = field(init=False, default_factory=list)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    stop: threading.Event = field(init=False, default_factory=threading.Event)

    def __post_init__(self) -> None:
        size = max(1, self.limits.concurrency)
        self.slots = {op: threading.BoundedSemaphore(size) for op in OPS}
        self.waiting = dict.fromkeys(OPS, 0)

    def start_loading(self) -> threading.Thread:
        return _daemon(load_models, "docmodels-load", self.state, self.loaders)

    def start_watchdog(self) -> threading.Thread:
        return _daemon(self._watch, "docmodels-watchdog")

    def _watch(self) -> None:
        while not self.stop.wait(WATCH_INTERVAL_S):
            self.check_stuck()

    @contextmanager
    def admitted(self, op: str) -> Iterator[None]:
        """A place among the requests of ``op`` in flight; ``busy`` when full."""
        with self.lock:
            if self.waiting[op] >= max(1, self.limits.max_requests):
                raise ServiceError(
                    503, "busy", f"{self.waiting[op]} {op} requests in flight"
                )
            self.waiting[op] += 1
        try:
            yield
        finally:
            with self.lock:
                self.waiting[op] -= 1

    def begin(self, op: str, deadline: float) -> _Run:
        run = _Run(op=op, slots=self.slots[op], deadline=deadline)
        with self.lock:
            self.running.append(run)
        return run

    def end(self, run: _Run) -> None:
        with self.lock:
            if run in self.running:
                self.running.remove(run)
            _release(run)

    def check_stuck(self, now: float | None = None) -> bool:
        """Free the slots of requests past ``deadline + GRACE_S``, mark the
        service stuck and run ``on_stuck``; True when one was found."""
        now = time.monotonic() if now is None else now
        with self.lock:
            late = [r for r in self.running if now > r.deadline + GRACE_S]
            for run in late:
                self.running.remove(run)
                _release(run)
            if late:
                self.stuck = late[0].op
        if late:
            self.on_stuck(late[0].op)
        return bool(late)


def _release(run: _Run) -> None:
    if not run.released:
        run.released = True
        run.slots.release()


def _daemon(target: Callable[..., Any], name: str, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, name=name, daemon=True)
    thread.start()
    return thread


# ── request helpers ─────────────────────────────────────────────────────────


def _authorize(service: Service, request: Request) -> None:
    if not service.token:
        return
    given = request.headers.get("authorization", "")
    if not hmac.compare_digest(given.encode(), f"Bearer {service.token}".encode()):
        raise ServiceError(401, "unauthorized", "a valid service token is required")


def _budget(request: Request, limits: Limits) -> float:
    """The caller's budget in seconds (header), capped at ``max_run_s``."""
    try:
        asked = float(request.headers.get(BUDGET_HEADER, "") or limits.max_run_s)
    except ValueError:
        asked = limits.max_run_s
    return min(asked, limits.max_run_s) if asked > 0 else limits.max_run_s


async def _read_body(request: Request, max_bytes: int) -> bytes:
    """The raw body, refused past ``max_bytes`` before it is held whole."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise ServiceError(413, "too_large", f"body over {max_bytes} bytes")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > max_bytes:
            raise ServiceError(413, "too_large", f"body over {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _json_object(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw or b"null")
    except ValueError as exc:
        raise ServiceError(400, "bad_request", "body is not JSON") from exc
    if not isinstance(body, dict):
        raise ServiceError(400, "bad_request", "body is not a JSON object")
    return body


def _require(state: ModelState, name: str) -> None:
    loaded = state.docling if name == "docling" else state.gliner is not None
    if loaded:
        return
    if state.loading:
        raise ServiceError(503, "loading", "models are still loading")
    raise ServiceError(503, f"{name}_unavailable", state.errors.get(name, ""))


def _same_model(body: dict[str, Any], key: str, served: str) -> None:
    """409 when the caller names a model directory other than the one served."""
    asked = body.get(key)
    if isinstance(asked, str) and asked.rstrip("/") != served.rstrip("/"):
        raise ServiceError(409, "model_mismatch", f"this service serves another {key}")


def _acquire(service: Service, op: str, budget: float) -> float:
    """Take a slot of ``op`` within the budget; the seconds left to run."""
    started = time.monotonic()
    slots = service.slots[op]
    if not slots.acquire(timeout=min(service.limits.queue_wait_s, budget)):
        raise ServiceError(503, "busy", f"every {op} slot is taken")
    left = budget - (time.monotonic() - started)
    if left <= 0:
        slots.release()
        raise ServiceError(503, "busy", "no budget left after the wait")
    return left


def _run_slot(
    service: Service, op: str, budget: float, fn: Callable[..., Any], *args: Any
) -> Any:
    """``fn(*args)`` in one ``op`` slot, watched against the budget."""
    left = _acquire(service, op, budget)
    run = service.begin(op, time.monotonic() + left)
    try:
        return fn(*args)
    except ServiceError:
        raise
    except Exception as exc:  # answered as a typed 500
        logger.warning("docmodels: %s failed: %s", op, exc)
        raise ServiceError(500, f"{op}_failed", type(exc).__name__) from exc
    finally:
        service.end(run)


def _page_cap(body: dict[str, Any], limits: Limits) -> int:
    asked = body.get("max_pages")
    if isinstance(asked, int) and not isinstance(asked, bool) and asked > 0:
        return min(asked, limits.max_pages)
    return limits.max_pages


def _parse_input(body: dict[str, Any], limits: Limits) -> tuple[bytes, str, int]:
    """``(data, filename, page cap)``; a PDF over the cap is ``too_many_pages``."""
    from .doc_parse import pdf_page_count

    filename = body.get("filename")
    encoded = body.get("data_b64")
    if not isinstance(filename, str) or not isinstance(encoded, str):
        raise ServiceError(400, "bad_request", "filename and data_b64 are required")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError(400, "bad_request", "data_b64 is not base64") from exc
    if len(data) > limits.max_bytes:
        raise ServiceError(413, "too_large", f"document over {limits.max_bytes} bytes")
    cap = _page_cap(body, limits)
    pages = pdf_page_count(data, filename)
    if pages is not None and pages > cap:
        raise ServiceError(422, "too_many_pages", f"{pages} pages, cap {cap}")
    return data, os.path.basename(filename) or "document.pdf", cap


def _strings(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        return None
    return list(value)


def _tag_input(body: dict[str, Any]) -> tuple[list[str], list[str], float]:
    from .pretag import THRESHOLD

    texts, labels = _strings(body.get("texts")), _strings(body.get("labels"))
    if texts is None or not labels:
        raise ServiceError(400, "bad_request", "texts and labels are string lists")
    threshold = body.get("threshold", THRESHOLD)
    if not isinstance(threshold, (int, float)) or not 0 < float(threshold) <= 1:
        raise ServiceError(400, "bad_request", "threshold is in (0, 1]")
    return texts, labels, float(threshold)


def _parse_work(service: Service, raw: bytes) -> tuple[str, str, int]:
    """Decode, check and convert one document (a worker thread, in a slot)."""
    body = _json_object(raw)
    _same_model(body, "artifacts", service.state.docling_dir)
    data, filename, cap = _parse_input(body, service.limits)
    return service.loaders.convert(data, filename, service.state.docling_dir, cap)


def _tag_work(service: Service, raw: bytes) -> list[list[dict[str, Any]]]:
    body = _json_object(raw)
    _same_model(body, "model_dir", service.state.gliner_dir)
    texts, labels, threshold = _tag_input(body)
    return service.loaders.tag(service.state.gliner, texts, labels, threshold)


# ── endpoints ───────────────────────────────────────────────────────────────


def _service(request: Request) -> Service:
    return request.app.state.docmodels


async def livez(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    stuck = _service(request).stuck
    body = {"alive": not stuck, **({"error": "stuck"} if stuck else {})}
    return JSONResponse(body, status_code=503 if stuck else 200)


def _not_ready(service: Service) -> str:
    if service.stuck:
        return "stuck"
    if service.state.loading:
        return "loading"
    return "" if service.state.ready() else "no_models"


async def readyz(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    service = _service(request)
    error = _not_ready(service)
    body: dict[str, Any] = {
        "ready": not error,
        "docling": service.state.docling,
        "gliner": service.state.gliner is not None,
    }
    if error:
        body["error"] = error
    return JSONResponse(body, status_code=503 if error else 200)


async def proof(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    service = _service(request)
    nonce = request.headers.get(NONCE_HEADER, "")
    if not service.token:
        raise ServiceError(404, "no_token", "this service runs without a token")
    if not 16 <= len(nonce) <= 128:
        raise ServiceError(400, "bad_request", "a 16-128 character nonce is required")
    digest = hmac.new(service.token.encode(), nonce.encode(), hashlib.sha256)
    return JSONResponse({"proof": digest.hexdigest()})


def service_info(service: Service) -> dict[str, Any]:
    state = service.state
    return {
        "service": SERVICE_NAME,
        "ready": not _not_ready(service),
        "loading": state.loading,
        "stuck": bool(service.stuck),
        "docling": {
            "version": service.loaders.docling_version(),
            "artifacts": state.docling_dir,
            "staged": bool(state.docling_dir)
            and ingest_models.is_docling_staged(state.docling_dir),
            "loaded": state.docling,
        },
        "gliner": {
            "repo": ingest_models.GLINER_REPO,
            "revision": ingest_models.gliner_revision(state.gliner_dir),
            "dir": state.gliner_dir,
            "staged": bool(state.gliner_dir)
            and ingest_models.is_gliner_staged(state.gliner_dir),
            "loaded": state.gliner is not None,
        },
        "limits": service.limits.as_dict(),
    }


async def info(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    service = _service(request)
    _authorize(service, request)
    return JSONResponse(service_info(service))


async def _serve(
    request: Request, op: str, model: str, work: Callable[..., Any]
) -> Any:
    """Authorize, check the model, admit, read the body; ``work`` in a slot."""
    from starlette.concurrency import run_in_threadpool

    service = _service(request)
    _authorize(service, request)
    _require(service.state, model)
    budget = _budget(request, service.limits)
    with service.admitted(op):
        raw = await _read_body(request, service.limits.body_bytes)
        return await run_in_threadpool(
            _run_slot, service, op, budget, work, service, raw
        )


async def parse(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    text, version, pages = await _serve(request, "parse", "docling", _parse_work)
    return JSONResponse({"text": text, "parser": f"docling@{version}", "pages": pages})


async def tag(request: Request) -> Any:
    from fastapi.responses import JSONResponse

    candidates = await _serve(request, "tag", "gliner", _tag_work)
    return JSONResponse({"candidates": candidates})


async def _service_error(_request: Any, exc: Exception) -> Any:
    from fastapi.responses import JSONResponse

    err = exc if isinstance(exc, ServiceError) else ServiceError(500, "internal")
    return JSONResponse(
        {"error": err.code, "detail": err.detail}, status_code=err.status
    )


@asynccontextmanager
async def _lifespan(app: Any) -> Any:
    service: Service = app.state.docmodels
    if service.warm:
        service.start_loading()
    if service.watch:
        service.start_watchdog()
    try:
        yield
    finally:
        service.stop.set()


def create_app(
    *,
    state: ModelState | None = None,
    loaders: Loaders | None = None,
    limits: Limits | None = None,
    warm: bool = True,
    watch: bool = True,
    token: str | None = None,
    on_stuck: Callable[[str], None] | None = None,
) -> Any:
    """The FastAPI app; models start loading, and the watchdog watching, when
    the app starts (``warm`` / ``watch``). ``token`` None reads the env."""
    from fastapi import FastAPI

    app = FastAPI(title=SERVICE_NAME, lifespan=_lifespan, docs_url=None, redoc_url=None)
    app.state.docmodels = Service(
        state=state or ModelState.from_env(),
        loaders=loaders or default_loaders(),
        limits=limits or Limits.from_env(),
        warm=warm,
        watch=watch,
        token=os.environ.get(TOKEN_ENV, "").strip() if token is None else token,
        on_stuck=on_stuck or exit_for_restart,
    )
    for path, endpoint, method in (
        ("/livez", livez, "GET"),
        ("/readyz", readyz, "GET"),
        ("/proof", proof, "GET"),
        ("/info", info, "GET"),
        ("/parse", parse, "POST"),
        ("/tag", tag, "POST"),
    ):
        app.add_api_route(path, endpoint, methods=[method])
    app.add_exception_handler(ServiceError, _service_error)
    return app


def enforce_offline_env(environ: MutableMapping[str, str] | None = None) -> None:
    """No model download at runtime (offline rule): forced, not defaulted."""
    env = os.environ if environ is None else environ
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("HF_HOME", "/tmp/hf")  # the tmpfs of the read-only rootfs


def main() -> int:
    """Run the service under uvicorn (SIGTERM = graceful shutdown)."""
    import uvicorn

    enforce_offline_env()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    host = os.environ.get(HOST_ENV, "").strip() or DEFAULT_HOST
    port = int(_env_number(PORT_ENV, DEFAULT_PORT))
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=GRACEFUL_S,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
