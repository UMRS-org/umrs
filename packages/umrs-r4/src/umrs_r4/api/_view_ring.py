"""The bounded, in-process ring of console view ops (UI plan §3.4, ADR-119).

A remote session's venue worker posts the view-contract ops it emits (UI
plan §3.2: ``{v: 1, sid, seq, at, op, ...}``) on the rail; this ring keeps
the newest of them per ``(owner, session)`` so a browser stream can resume
with ``Last-Event-ID``. It is memory only, never a table: the session's
journal row stays the durable truth (ADR-057 D-6), and a client the ring
cannot serve gets a ``session.snapshot`` built from that row
(``_view_snapshot``). AGENTS.md rule 10: not a second store.

Per session the ring holds the ops ``floor..head`` with no hole (a hole
empties it), the newest ``session.snapshot`` the emitter sent (``pinned``),
the emitter's ``epoch`` (a restarted worker counts ``seq`` from 1 again
under a new epoch) and ``gen``, a process-local generation that changes
whenever the ring starts over, so a reader knows its cursor no longer
applies. Bounds: ``max_ops`` and ``max_bytes`` per session, ``max_sessions``
sessions (least recently written first out), and ``max_total_bytes`` over
all sessions (least recently written first out). The emitter's snapshot is
held in the op sequence too, so a live or resuming reader gets it in order
(a client that misses it sees a seq gap and goes stale).

Thread-safe: rail posts arrive on the event loop, fan-out notifications on
a listener thread. A reader waits with :meth:`ViewRing.wait`, which any
writer wakes from any thread.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

# Defaults: a turn of a busy session emits a few hundred ops; 2,000 ops or
# 4 MiB keeps several turns of resume window per session.
DEFAULT_MAX_OPS = 2000
DEFAULT_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_SESSIONS = 512
# Over all sessions of one process: 512 sessions x 4 MiB would be 2 GiB.
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
# The client takes only a JavaScript safe integer as seq (``Number.isSafeInteger``).
MAX_SAFE_INT = 2**53 - 1

_GENERATIONS = itertools.count(1)

Key = tuple[str, str]  # (owner, session_id)


_EPOCH_RE = re.compile(r"[A-Za-z0-9._-]{0,64}")


def valid_epoch(value: Any) -> bool:
    """An emitter run id: up to 64 of ``A-Z a-z 0-9 . _ -`` (it goes on the
    SSE ``id:`` line, so never a line break)."""
    return isinstance(value, str) and _EPOCH_RE.fullmatch(value) is not None


def safe_seq(value: Any) -> bool:
    """A seq the client applies: a positive integer up to 2**53 - 1."""
    return type(value) is int and 0 < value <= MAX_SAFE_INT


@dataclass
class Entry:
    """One op as it goes on the wire."""

    seq: int
    data: str  # the op, JSON-encoded once


@dataclass
class SessionRing:
    """The ring of one session (see the module docstring)."""

    epoch: str
    gen: int = field(default_factory=lambda: next(_GENERATIONS))
    ops: deque[Entry] = field(default_factory=deque)
    head: int = 0
    floor: int = 1
    pinned: Entry | None = None
    size: int = 0
    native: bool = False

    def contiguous_from(self, seq: int) -> bool:
        """True when every op after ``seq`` up to ``head`` is held."""
        return self.floor <= seq + 1 and seq <= self.head

    def pinned_usable(self) -> bool:
        """The emitter's snapshot can start a stream: the ops after it are held."""
        return self.pinned is not None and self.contiguous_from(self.pinned.seq)

    def footprint(self) -> int:
        """Bytes held: the ops, plus the snapshot once it left the op sequence."""
        pinned = self.pinned
        if pinned is None or (self.ops and self.ops[0].seq <= pinned.seq):
            return self.size
        return self.size + len(pinned.data)


@dataclass(frozen=True)
class Plan:
    """What a reader at cursor ``(gen, seq)`` gets next.

    ``resync`` False: send ``ops`` (each after the cursor), keep reading.
    ``resync`` True: the cursor no longer applies; start over with a
    snapshot (``pinned`` when the ring holds one usable, else the journal's)
    followed by ``ops``.
    """

    resync: bool
    gen: int
    epoch: str
    head: int
    ops: tuple[Entry, ...] = ()
    pinned: Entry | None = None
    native: bool = False


@dataclass
class Appended:
    """The outcome of one batch."""

    accepted: int = 0
    ignored: int = 0
    reset: bool = False


class ViewRing:
    """All sessions' rings in this process (see the module docstring)."""

    def __init__(
        self,
        *,
        max_ops: int = DEFAULT_MAX_OPS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        self.max_ops = max_ops
        self.max_bytes = max_bytes
        self.max_sessions = max_sessions
        self.max_total_bytes = max_total_bytes
        self._lock = threading.Lock()
        self._rings: OrderedDict[Key, SessionRing] = OrderedDict()
        self._waiters: dict[Key, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]] = {}

    # ── writers ──────────────────────────────────────────────────────────

    def append(self, key: Key, epoch: str, ops: list[dict[str, Any]]) -> Appended:
        """Add one batch of ops (envelopes already checked by the caller)."""
        with self._lock:
            ring, reset = self._ring_for(key, epoch)
            outcome = Appended(reset=reset)
            for op in ops:
                if self._add(ring, op):
                    outcome.accepted += 1
                else:
                    outcome.ignored += 1
            self._keep_budget(key)
        self._wake(key)
        return outcome

    def mark_gap(self, key: Key, epoch: str, seq: int) -> None:
        """An op this process never saw (too large to fan out): start over at it."""
        with self._lock:
            ring, _ = self._ring_for(key, epoch)
            if seq > ring.head:
                self._restart_at(ring, seq + 1)
                ring.head = seq
        self._wake(key)

    def _ring_for(self, key: Key, epoch: str) -> tuple[SessionRing, bool]:
        ring = self._rings.get(key)
        reset = ring is not None and ring.epoch != epoch
        if ring is None or reset:
            ring = SessionRing(epoch=epoch)
            self._rings[key] = ring
        self._rings.move_to_end(key)
        while len(self._rings) > self.max_sessions:
            self._rings.popitem(last=False)
        return ring, reset

    def _keep_budget(self, keep: Key) -> None:
        """Drop the least recently written sessions (never ``keep``) while
        all rings together hold more than ``max_total_bytes``."""
        total = sum(ring.footprint() for ring in self._rings.values())
        for key in list(self._rings):
            if total <= self.max_total_bytes:
                return
            if key != keep:
                total -= self._rings.pop(key).footprint()

    def total_bytes(self) -> int:
        """Bytes held over all sessions (tests, diagnostics)."""
        with self._lock:
            return sum(ring.footprint() for ring in self._rings.values())

    def _add(self, ring: SessionRing, op: dict[str, Any]) -> bool:
        seq = op.get("seq")
        if not safe_seq(seq) or seq <= ring.head:
            return False  # replay-safe: a duplicate changes nothing
        if seq > ring.head + 1:
            self._restart_at(ring, seq)
        entry = Entry(seq=seq, data=json.dumps(op, separators=(",", ":")))
        ring.head = seq
        ring.native = True
        if op.get("op") == "session.snapshot":
            ring.pinned = entry
        ring.ops.append(entry)
        ring.size += len(entry.data)
        self._trim(ring)
        return True

    @staticmethod
    def _restart_at(ring: SessionRing, seq: int) -> None:
        """A hole before ``seq``: nothing older can be replayed across it."""
        ring.ops.clear()
        ring.size = 0
        ring.floor = seq
        ring.pinned = None

    def _trim(self, ring: SessionRing) -> None:
        while ring.ops and (len(ring.ops) > self.max_ops or ring.size > self.max_bytes):
            dropped = ring.ops.popleft()
            ring.size -= len(dropped.data)
            ring.floor = dropped.seq + 1

    # ── readers ──────────────────────────────────────────────────────────

    def plan(self, key: Key, gen: int | None, after: int) -> Plan:
        """What a reader at ``(gen, after)`` gets; ``gen`` None means a new
        reader, which always starts with a snapshot."""
        with self._lock:
            ring = self._rings.get(key)
            if ring is None:
                return Plan(resync=gen is None or gen != 0, gen=0, epoch="", head=0)
            if gen == ring.gen and ring.contiguous_from(after):
                return Plan(False, ring.gen, ring.epoch, ring.head, _after(ring, after))
            return _resync_plan(ring)

    def resume_plan(self, key: Key, epoch: str | None, after: int) -> Plan:
        """A reconnect that names ``(epoch, after)`` (its ``Last-Event-ID``);
        ``epoch`` None (a bare seq) means the ring's current one."""
        with self._lock:
            ring = self._rings.get(key)
            if ring is not None and epoch in (None, ring.epoch) and ring.contiguous_from(after):
                return Plan(False, ring.gen, ring.epoch, ring.head, _after(ring, after))
        return self.plan(key, None, after)

    def _moved(self, key: Key, gen: int, seq: int) -> bool:
        ring = self._rings.get(key)
        if ring is None:
            return gen != 0
        return ring.gen != gen or ring.head != seq

    async def wait(self, key: Key, gen: int, seq: int, timeout: float) -> None:
        """Return once the ring of ``key`` is not at cursor ``(gen, seq)``,
        or after ``timeout`` seconds. Any writer, from any thread, wakes it."""
        loop = asyncio.get_running_loop()
        event = asyncio.Event()
        token = (loop, event)
        with self._lock:
            if self._moved(key, gen, seq):
                return
            self._waiters.setdefault(key, set()).add(token)
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            with self._lock:
                waiters = self._waiters.get(key)
                if waiters is not None:
                    waiters.discard(token)
                    if not waiters:
                        self._waiters.pop(key, None)

    def _wake(self, key: Key) -> None:
        with self._lock:
            waiters = list(self._waiters.get(key, ()))
        for loop, event in waiters:
            if not loop.is_closed():
                loop.call_soon_threadsafe(event.set)


def _after(ring: SessionRing, seq: int) -> tuple[Entry, ...]:
    return tuple(entry for entry in ring.ops if entry.seq > seq)


def _resync_plan(ring: SessionRing) -> Plan:
    if ring.pinned_usable():
        pinned = ring.pinned
        assert pinned is not None
        ops = _after(ring, pinned.seq)
        return Plan(True, ring.gen, ring.epoch, ring.head, ops, pinned, ring.native)
    # The journal's snapshot stands at ``head``; only later ops follow it.
    return Plan(True, ring.gen, ring.epoch, ring.head, (), None, ring.native)
