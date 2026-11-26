"""Postgres ``NOTIFY`` fan-out of console view ops between UMRS processes.

Several UMRS processes serve one database: a front runs one UMRS server per
cluster it fronts, all on the shared ``umrs_db``, and a venue worker's rail
reaches one of them while the owner's browser stream may reach another
(UI plan §3.4, lane UX-V2). Each process keeps its own :class:`ViewRing`;
a rail batch goes into the local ring and out on ``NOTIFY``, and every
other process's listener puts it into its ring. Nothing is stored: a
process that was down misses ops, and its readers get the journal snapshot.

Payload limit: ``NOTIFY`` carries at most 8000 bytes, so a batch goes out in
chunks; one op too large for a chunk goes out as a gap marker
(``{"gap": seq}``), and the other processes start their ring over after it.
"""

from __future__ import annotations

import json
import logging
import select
import threading
import uuid
from typing import Any

from umrs_r4.api._view_ring import ViewRing

logger = logging.getLogger(__name__)

CHANNEL = "umrs_r4_view_ops"
# Postgres refuses a payload of 8000 bytes or more; keep room for the head.
MAX_PAYLOAD = 7800
_POLL_SECONDS = 5.0
_RETRY_SECONDS = 2.0


def chunk_payloads(head: dict[str, Any], ops: list[dict[str, Any]]) -> list[str]:
    """The ``NOTIFY`` payloads for one batch, each under ``MAX_PAYLOAD``."""
    out: list[str] = []
    chunk: list[str] = []
    base = len(json.dumps({**head, "ops": []}, separators=(",", ":")))
    used = base
    for op in ops:
        text = json.dumps(op, separators=(",", ":"))
        if base + len(text) + 1 > MAX_PAYLOAD:
            out.extend(_flush(head, chunk))
            chunk, used = [], base
            out.append(json.dumps({**head, "gap": int(op["seq"])}, separators=(",", ":")))
            continue
        if used + len(text) + 1 > MAX_PAYLOAD:
            out.extend(_flush(head, chunk))
            chunk, used = [], base
        chunk.append(text)
        used += len(text) + 1
    out.extend(_flush(head, chunk))
    return out


def _flush(head: dict[str, Any], chunk: list[str]) -> list[str]:
    if not chunk:
        return []
    prefix = json.dumps(head, separators=(",", ":"))[:-1]
    return [f'{prefix},"ops":[{",".join(chunk)}]}}']


def _message(origin: str, payload: str) -> dict[str, Any] | None:
    """A notification from another process, or None (ours, or malformed)."""
    try:
        message = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("o") in (None, origin):
        return None
    if not (message.get("u") and message.get("s")):
        return None
    return message


def apply_payload(ring: ViewRing, origin: str, payload: str) -> bool:
    """Put one notification into ``ring``; False when it is ours or malformed."""
    message = _message(origin, payload)
    if message is None:
        return False
    key = (str(message["u"]), str(message["s"]))
    epoch = str(message.get("e") or "")
    if isinstance(message.get("gap"), int):
        ring.mark_gap(key, epoch, message["gap"])
        return True
    ops = message.get("ops")
    if not isinstance(ops, list):
        return False
    ring.append(key, epoch, [op for op in ops if isinstance(op, dict) and "seq" in op])
    return True


def publish(pool: Any, payloads: list[str]) -> None:
    """Send ``payloads`` on the channel in one transaction (best effort)."""
    if pool is None or not payloads:
        return
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            for payload in payloads:
                cur.execute("SELECT pg_notify(%s, %s)", (CHANNEL, payload))
        conn.commit()
    except Exception:  # noqa: BLE001 — the local ring already holds the batch
        logger.warning("view-op fan-out publish failed", exc_info=True)
        _rollback(conn)
    finally:
        pool.putconn(conn)


def _rollback(conn: Any) -> None:
    try:
        conn.rollback()
    except Exception:  # noqa: BLE001 — a broken connection is the pool's to drop
        logger.debug("rollback after a failed publish failed", exc_info=True)


def dedicated_connection(pool: Any) -> Any:
    """A connection of its own for ``LISTEN``, opened with the pool's settings.

    A pool lease held for the life of the process would starve the routes;
    psycopg2 pools keep their connect arguments in ``_args``/``_kwargs``.
    """
    import psycopg2

    args = getattr(pool, "_args", ())
    kwargs = dict(getattr(pool, "_kwargs", {}) or {})
    for bound in ("minconn", "maxconn"):
        kwargs.pop(bound, None)
    conn = psycopg2.connect(*args, **kwargs)
    conn.autocommit = True
    return conn


class Listener:
    """One daemon thread per process: ``LISTEN`` and feed the ring."""

    def __init__(self, ring: ViewRing, origin: str, connect: Any) -> None:
        self.ring = ring
        self.origin = origin
        self._connect = connect
        self._stop = threading.Event()
        self.ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="umrs-view-ops-listen", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=_POLL_SECONDS + 1)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._listen_once()
            except Exception:  # noqa: BLE001 — reconnect; readers fall back to snapshots
                logger.warning("view-op fan-out listener lost its connection", exc_info=True)
                self.ready.clear()
                self._stop.wait(_RETRY_SECONDS)

    def _listen_once(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(f"LISTEN {CHANNEL}")
            self.ready.set()
            while not self._stop.is_set():
                self._drain(conn)
        finally:
            conn.close()

    def _drain(self, conn: Any) -> None:
        if select.select([conn], [], [], _POLL_SECONDS) == ([], [], []):
            return
        conn.poll()
        while conn.notifies:
            note = conn.notifies.pop(0)
            apply_payload(self.ring, self.origin, note.payload)


class ViewHub:
    """This process's ring, its fan-out identity and its listener."""

    def __init__(self, ring: ViewRing | None = None, pool: Any = None) -> None:
        self.ring = ring or ViewRing()
        self.pool = pool
        self.origin = uuid.uuid4().hex
        self._listener: Listener | None = None
        self._lock = threading.Lock()

    def ensure_listener(self) -> None:
        """Start the listener once, when a pool exists (no pool: one process)."""
        if self.pool is None or self._listener is not None:
            return
        with self._lock:
            if self._listener is None:
                self._listener = Listener(
                    self.ring, self.origin, lambda: dedicated_connection(self.pool)
                )
                self._listener.start()

    def wait_listening(self, timeout: float) -> bool:
        """True once the listener has issued ``LISTEN`` (tests, start-up)."""
        listener = self._listener
        return listener is not None and listener.ready.wait(timeout)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def post(self, owner: str, sid: str, epoch: str, ops: list[dict[str, Any]]) -> Any:
        """Local ring first, then the other processes."""
        outcome = self.ring.append((owner, sid), epoch, ops)
        head = {"o": self.origin, "u": owner, "s": sid, "e": epoch}
        publish(self.pool, chunk_payloads(head, ops))
        return outcome
