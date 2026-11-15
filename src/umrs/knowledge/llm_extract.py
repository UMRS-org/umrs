"""Whole-document windowed extraction (ADR-116 D-2, D-3, D-6).

The understand job reads a document's whole text in windows instead of its
first 9 000 characters:

- windows are packed from ``sent-v1`` sentence spans
  (``chunking.sentence_spans``) up to ``UMRS_KNOWLEDGE_EXTRACT_WINDOW_CHARS``
  characters (default 8000), at most ``UMRS_KNOWLEDGE_EXTRACT_WINDOWS_MAX``
  (default 24) of them;
- the windows run with ``UMRS_KNOWLEDGE_EXTRACT_CONCURRENCY`` worker threads
  (default 4, clamped to 1..10) inside a per-document budget of
  ``UMRS_KNOWLEDGE_EXTRACT_BUDGET_S`` seconds (default 300), cut to what is
  left of the understand job's budget ``UMRS_KNOWLEDGE_EXTRACT_JOB_BUDGET_S``
  (default 300, ``ExtractRequest.deadline``): the scheduler awaits the job
  inline, so the job budget bounds how long one job holds it. A document
  that starts with less than ``LITE_FLOOR_S`` of the job budget left reads
  one window (lite, ``extract_job_budget_exhausted``); windows dropped by the
  cap or the budget are named in the notes;
- each request asks for JSON mode and retries once without it when the
  endpoint refuses it with a 4xx; the reply parser strips code fences and
  ``<think>`` blocks, takes the first JSON graph object and never raises; a
  reply without one (cut off, another shape, empty) is a failed window, named
  ``extract_window_unparsed:<n>``;
- window results merge by normalized name into one entity per document, with
  ``aliases``, at most five ``evidence`` quotes and the extractor id; types and
  relations map onto the graph's extraction vocabulary (unmatched types become
  ``unclassified`` with ``type_raw``, unmatched relations ``related_to`` with
  ``rel_raw``).

This module does no I/O of its own: the caller hands it a ``chat`` callable
(``chat(messages, response_format=...) -> str``) and runs it in a worker
thread, never on the server event loop.
"""

from __future__ import annotations

import functools
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from typing import Any

WINDOW_CHARS_ENV = "UMRS_KNOWLEDGE_EXTRACT_WINDOW_CHARS"
WINDOWS_MAX_ENV = "UMRS_KNOWLEDGE_EXTRACT_WINDOWS_MAX"
CONCURRENCY_ENV = "UMRS_KNOWLEDGE_EXTRACT_CONCURRENCY"
BUDGET_ENV = "UMRS_KNOWLEDGE_EXTRACT_BUDGET_S"
JOB_BUDGET_ENV = "UMRS_KNOWLEDGE_EXTRACT_JOB_BUDGET_S"
MAX_TOKENS_ENV = "UMRS_KNOWLEDGE_EXTRACT_MAX_TOKENS"

EXTRACTOR_ID = "llm-windowed@2"
JSON_MODE: dict[str, str] = {"type": "json_object"}
MODES = ("full", "lite", "off")

ENTITIES_PER_WINDOW = 40
RELATIONS_PER_WINDOW = 60
EVIDENCE_MAX = 5
QUOTE_MAX = 200
ALIASES_MAX = 20
NAME_MAX = 120
DOC_NODES_MAX = 400
DOC_EDGES_MAX = 800
# Below this much job budget a document reads one window (lite): about what
# one window call needs, so a full document is never started only to be cut.
LITE_FLOOR_S = 60.0

Chat = Callable[..., str]
Tagger = Callable[[Sequence[str]], "list[list[dict[str, Any]]]"]


# ── settings ────────────────────────────────────────────────────────────────


def _env_number(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, "") or default)
    except ValueError:
        value = default
    return min(high, max(low, value))


@dataclass(frozen=True)
class ExtractSettings:
    window_chars: int = 8000
    windows_max: int = 24
    concurrency: int = 4
    budget_s: float = 300.0
    max_tokens: int = 2500
    job_budget_s: float = 300.0

    @classmethod
    def from_env(cls) -> ExtractSettings:
        return cls(
            window_chars=int(_env_number(WINDOW_CHARS_ENV, 8000, 500, 200_000)),
            windows_max=int(_env_number(WINDOWS_MAX_ENV, 24, 1, 1000)),
            concurrency=int(_env_number(CONCURRENCY_ENV, 4, 1, 10)),
            budget_s=_env_number(BUDGET_ENV, 300.0, 1.0, 86_400.0),
            max_tokens=int(_env_number(MAX_TOKENS_ENV, 2500, 256, 32_000)),
            job_budget_s=_env_number(JOB_BUDGET_ENV, 300.0, 1.0, 86_400.0),
        )


def normalize_mode(value: Any, default: str = "full") -> str:
    """``full`` | ``lite`` | ``off``; anything else reads as ``default``."""
    mode = str(value or "").strip().lower()
    return mode if mode in MODES else default


# ── windows ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    index: int
    start: int
    end: int
    text: str


def _spans(text: str, limit: int) -> list[tuple[int, int]]:
    """Sentence spans, a sentence longer than ``limit`` cut into pieces."""
    from .chunking import sentence_spans

    out: list[tuple[int, int]] = []
    for start, end in sentence_spans(text):
        out += [(s, min(s + limit, end)) for s in range(start, end, limit)]
    return out


def pack_windows(text: str, window_chars: int) -> list[Window]:
    """Contiguous windows of whole sentences, each at most ``window_chars``."""
    bounds: list[tuple[int, int]] = []
    start = end = -1
    for a, b in _spans(text, window_chars):
        if start >= 0 and b - start > window_chars:
            bounds.append((start, end))
            start = -1
        if start < 0:
            start = a
        end = b
    if start >= 0:
        bounds.append((start, end))
    return [Window(i, s, e, text[s:e]) for i, (s, e) in enumerate(bounds)]


# ── vocabulary ──────────────────────────────────────────────────────────────


def _snake(value: str) -> str:
    return "_".join(re.sub(r"[^0-9a-z]+", " ", value.lower()).split())


class BuiltinVocabulary:
    """Today's 13 extraction types, used only when the ontology vocabulary
    module cannot be imported. Relations are free: any phrase maps to its
    snake_case form."""

    ontology_id = "r4-builtin"
    version = "1"
    _CLASSES = (
        "concept", "method", "technique", "property", "process", "model", "paper",
        "dataset", "software", "application", "person", "material", "finding",
    )  # fmt: skip
    _ENTITY = frozenset(
        {
            "paper", "dataset", "software", "application", "person", "institution",
            "model", "code", "notebook", "figure", "result", "material",
        }
    )  # fmt: skip

    def class_names(self) -> list[str]:
        return list(self._CLASSES)

    def relation_names(self) -> list[str]:
        return []

    def prompt_block(self) -> str:
        return (
            "Entity types (use one): " + ", ".join(self._CLASSES) + ".\n"
            "Relations: a short verb phrase in snake_case, for example uses, "
            "measures, part_of."
        )

    def match_class(self, raw: str) -> str | None:
        name = _snake(str(raw or ""))
        for candidate in (name, name.removesuffix("s")):
            if candidate in self._CLASSES:
                return candidate
        return None

    def match_relation(self, raw: str) -> str | None:
        return _snake(str(raw or "")) or None

    def bucket_for(self, cls: str) -> str:
        return "entity" if cls in self._ENTITY else "concept"

    def gliner_labels(self) -> list[str]:
        return [name.replace("_", " ") for name in self._CLASSES]


def default_vocabulary() -> Any:
    """The built-in ``r4-science@1`` vocabulary, else :class:`BuiltinVocabulary`."""
    try:
        from .ontology_vocab import default_vocabulary as builtin_science

        return builtin_science()
    except Exception:  # noqa: BLE001 — the module may be absent; fall back
        return BuiltinVocabulary()


def resolve_vocabulary(project_id: str, graph_id: str) -> tuple[Any, list[str]]:
    """The graph's extraction vocabulary and notes; never raises (SYNC: the
    caller runs it in a worker thread)."""
    try:
        from .ontology_vocab import resolve_graph_vocabulary
    except ImportError:
        return BuiltinVocabulary(), ["ontology_fallback:vocabulary_module_missing"]
    try:
        vocab, notes = resolve_graph_vocabulary(project_id, graph_id)
    except Exception as exc:  # noqa: BLE001 — never fail a job on vocabulary
        return default_vocabulary(), [f"ontology_fallback:{type(exc).__name__}"]
    return vocab, list(notes or [])


# ── prompts ─────────────────────────────────────────────────────────────────


def system_prompt(vocab: Any) -> str:
    """The extraction system prompt, listing the vocabulary's classes and relations."""
    return (
        "You extract a knowledge graph from one part of a scientific or "
        "technical document. Reply with a JSON object only, no prose outside "
        "the JSON. Schema:\n"
        '{"entities": [{"name": "short canonical name", "type": "one class '
        'from the list", "aliases": ["other names the text uses for it"], '
        '"quote": "a short exact quote from the text that mentions it"}], '
        '"relations": [{"source": "entity name", "target": "entity name", '
        '"relation": "one relation from the list"}]}\n'
        f"{vocab.prompt_block()}\n"
        f"Rules: at most {ENTITIES_PER_WINDOW} entities and "
        f"{RELATIONS_PER_WINDOW} relations; names concise (at most 6 words); "
        "only relations between listed entities; skip trivial or generic "
        "mentions. When no listed class or relation fits, give the closest "
        "plain word."
    )


def user_message(window: Window, total: int, hints: Sequence[dict[str, Any]]) -> str:
    """The window text, with tagger candidates appended as hints."""
    body = f"Document part {window.index + 1} of {total}.\n\n{window.text}"
    if not hints:
        return body
    listed = "\n".join(f"- {h.get('text', '')} ({h.get('label', '')})" for h in hints)
    return (
        f"{body}\n\nCandidate entities a tagger found in this part (hints only: "
        f"keep, correct or drop them):\n{listed}"
    )


# ── reply parsing ───────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```[a-zA-Z]*")
_DECODER = json.JSONDecoder()
_MAX_STARTS = 64


def _strip_reply(reply: str) -> str:
    text = _THINK.sub("", reply)
    if "</think>" in text:  # a reply that dropped the opening tag
        text = text.rsplit("</think>", 1)[1]
    return _FENCE.sub("", text)


_GRAPH_KEYS = ("entities", "nodes", "relations", "edges")


def is_graph(obj: Any) -> bool:
    """A dict with an ``entities`` / ``nodes`` / ``relations`` / ``edges`` list."""
    return isinstance(obj, dict) and any(
        isinstance(obj.get(key), list) for key in _GRAPH_KEYS
    )


def _decode_at(text: str, start: int) -> Any:
    try:
        return _DECODER.raw_decode(text, start)[0]
    except ValueError:
        return None


def parse_reply(reply: Any) -> dict[str, Any] | None:
    """The first JSON graph object (:func:`is_graph`) in a model reply, or
    None. Never raises.

    An object without a graph list is skipped and the scan goes on, so a
    wrapper (``{"result": {...}}``) yields the graph inside it, and a reply
    cut off at ``max_tokens`` (the outer object does not decode) yields None
    instead of its first nested entity object.
    """
    if not isinstance(reply, str) or not reply.strip():
        return None
    text = _strip_reply(reply)
    start = text.find("{")
    tries = 0
    while start >= 0 and tries < _MAX_STARTS:
        obj = _decode_at(text, start)
        if is_graph(obj):
            return obj
        start = text.find("{", start + 1)
        tries += 1
    return None


# ── one window ──────────────────────────────────────────────────────────────


def _json_mode_refused(exc: BaseException) -> bool:
    """A 4xx answer (not a timeout or rate limit), or a chat callable that
    takes no ``response_format``."""
    if isinstance(exc, TypeError):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", 0)
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


@dataclass
class _Caller:
    chat: Chat
    json_mode: bool = True

    def call(self, messages: list[dict[str, Any]]) -> str:
        if self.json_mode:
            try:
                return self.chat(messages, response_format=JSON_MODE)
            except Exception as exc:
                if not _json_mode_refused(exc):
                    raise
                self.json_mode = False  # one retry, and no JSON mode after it
        return self.chat(messages, response_format=None)


@dataclass(frozen=True)
class WindowResult:
    window: Window
    data: dict[str, Any] | None
    error: str = ""
    # The model answered, but with no JSON graph (cut off, another shape, or
    # nothing at all) -- as opposed to a failed call.
    unparsed: bool = False


def _extract_window(
    caller: _Caller,
    system: str,
    window: Window,
    total: int,
    hints: Sequence[dict[str, Any]],
    model: str,
) -> WindowResult:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(window, total, hints)},
    ]
    try:
        reply = caller.call(messages)
    except Exception as exc:  # noqa: BLE001 — one window's failure is named
        return WindowResult(window, None, f"{type(exc).__name__}: {exc}")
    data = parse_reply(reply)
    if data is None:
        return WindowResult(
            window,
            None,
            f"model {model!r} replied without a JSON graph (truncated or wrong shape?)",
            unparsed=True,
        )
    return WindowResult(window, data)


def run_bounded(
    jobs: Sequence[Callable[[], Any]], concurrency: int, timeout: float
) -> tuple[list[Any], int]:
    """Run ``jobs`` on at most ``concurrency`` threads for ``timeout`` seconds.

    Returns the results in job order (None for a job that did not finish)
    and how many did not finish. The jobs must not raise.
    """
    if not jobs:
        return [], 0
    pool = ThreadPoolExecutor(
        max_workers=max(1, min(concurrency, len(jobs))), thread_name_prefix="kg-extract"
    )
    try:
        futures = [pool.submit(job) for job in jobs]
        done, _pending = wait(futures, timeout=max(0.0, timeout))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    results = [f.result() if f in done else None for f in futures]
    return results, sum(r is None for r in results)


# ── merge ───────────────────────────────────────────────────────────────────

_NON_WORD = re.compile(r"[^\w]+")


def norm_name(value: Any) -> str:
    """Case-, punctuation- and space-insensitive key of a name."""
    return " ".join(_NON_WORD.sub(" ", str(value or "").lower()).split())


def _clean(value: Any, limit: int = NAME_MAX) -> str:
    return " ".join(str(value or "").split())[:limit] if isinstance(value, str) else ""


def _first_list(data: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


@dataclass
class _Entity:
    name: str
    type_raw: str
    cls: str | None
    aliases: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


class _Merger:
    """One entity per normalized name (or alias) per document."""

    def __init__(self, vocab: Any, parser_id: str) -> None:
        self.vocab = vocab
        self.parser_id = parser_id
        self.entities: dict[str, _Entity] = {}
        self.index: dict[str, str] = {}
        self.relations: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def _match_class(self, type_raw: str) -> str | None:
        return self.vocab.match_class(type_raw) if type_raw else None

    def add_entity(self, raw: Any, window: int) -> None:
        if not isinstance(raw, dict):
            return
        raw = _entity_fields(raw)
        name = _clean(_first_value(raw, ("name", "label", "id")))
        key = norm_name(name)
        if not key:
            return
        type_raw = _clean(raw.get("type"), 80)
        cls = self._match_class(type_raw)
        owner = self.index.get(key)
        if owner is None:
            owner = self.index[key] = key
            self.entities[key] = _Entity(name=name, type_raw=type_raw, cls=cls)
        entity = self.entities[owner]
        if entity.cls is None and cls is not None:  # a vocabulary match wins
            entity.cls, entity.type_raw = cls, type_raw
        self._add_aliases(owner, entity, [name, *_aliases_of(raw)])
        self._add_evidence(entity, window, _clean(raw.get("quote"), QUOTE_MAX))

    def _add_aliases(self, owner: str, entity: _Entity, names: list[str]) -> None:
        for alias in names:
            if not alias or alias == entity.name or alias in entity.aliases:
                continue
            if len(entity.aliases) < ALIASES_MAX:
                entity.aliases.append(alias)
            self.index.setdefault(norm_name(alias), owner)

    def _add_evidence(self, entity: _Entity, window: int, quote: str) -> None:
        if len(entity.evidence) >= EVIDENCE_MAX:
            return
        item = {"window": window, "quote": quote, "parser": self.parser_id}
        seen_window = any(e["window"] == window for e in entity.evidence)
        if item not in entity.evidence and (quote or not seen_window):
            entity.evidence.append(item)

    def _endpoint(self, raw: dict[str, Any], *keys: str) -> str | None:
        return self.index.get(norm_name(_first_value(raw, keys)))

    def add_relation(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        src = self._endpoint(raw, "source", "from")
        dst = self._endpoint(raw, "target", "to")
        if not src or not dst or src == dst:
            return
        rel_raw = _clean(_first_value(raw, ("relation", "type", "label")), 80)
        relation = self.vocab.match_relation(rel_raw) if rel_raw else None
        item: dict[str, Any] = {
            "source": self.entities[src].name,
            "target": self.entities[dst].name,
            "relation": relation or "related_to",
        }
        if relation is None:
            item["rel_raw"] = rel_raw
        self.relations.setdefault((src, item["relation"], rel_raw, dst), item)

    def _node(self, entity: _Entity) -> dict[str, Any]:
        cls = entity.cls or "unclassified"
        node: dict[str, Any] = {
            "name": entity.name,
            "type": cls,
            "bucket": self.vocab.bucket_for(cls),
            "aliases": list(entity.aliases),
            "evidence": list(entity.evidence),
        }
        if entity.cls is None:
            node["type_raw"] = entity.type_raw
        return node

    def payload(self) -> dict[str, Any]:
        return {
            "nodes": [self._node(e) for e in self.entities.values()][:DOC_NODES_MAX],
            "edges": list(self.relations.values())[:DOC_EDGES_MAX],
            "extractor": EXTRACTOR_ID,
            "parser": self.parser_id,
        }


def _first_value(raw: dict[str, Any], keys: Sequence[str]) -> Any:
    """The first truthy value under ``keys``."""
    return next((raw[key] for key in keys if raw.get(key)), None)


_ENTITY_KEYS = frozenset({"name", "label", "id", "type"})


def _entity_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """The entity as ``{name, type, ...}``; a single-key map ``{"GLiNER":
    "Model"}`` (seen from qwen2.5:14b) reads as name -> type."""
    if len(raw) != 1 or _ENTITY_KEYS & raw.keys():
        return raw
    ((name, type_raw),) = raw.items()
    return {"name": name, "type": type_raw if isinstance(type_raw, str) else ""}


def _aliases_of(raw: dict[str, Any]) -> list[str]:
    aliases = raw.get("aliases")
    if not isinstance(aliases, list):
        return []
    return [a for a in (_clean(v) for v in aliases) if a]


def merge_windows(
    results: Sequence[WindowResult], vocab: Any, parser_id: str
) -> dict[str, Any]:
    """Window replies (in window order) → one ``{nodes, edges}`` payload."""
    merger = _Merger(vocab, parser_id)
    for result in results:
        entities = _first_list(result.data or {}, "entities", "nodes")
        for raw in entities[:ENTITIES_PER_WINDOW]:
            merger.add_entity(raw, result.window.index)
    for result in results:
        relations = _first_list(result.data or {}, "relations", "edges")
        for raw in relations[:RELATIONS_PER_WINDOW]:
            merger.add_relation(raw)
    return merger.payload()


# ── session-history packages ────────────────────────────────────────────────


def _history_lines(record: dict[str, Any]) -> list[str]:
    lines = record.get("lines")
    if not isinstance(lines, list):
        return []
    return [
        f"{line.get('role') or 'turn'}: {line['text']}"
        for line in lines
        if isinstance(line, dict)
        and isinstance(line.get("text"), str)
        and line["text"].strip()
    ]


def _json_record(line: str) -> dict[str, Any] | None:
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def session_history_text(text: str) -> str | None:
    """The turn texts of an agent session-history package (``.jsonl`` with
    ``lane: "session-history"``), or None for any other text."""
    stripped = text.strip()
    if not stripped.startswith("{") or "session-history" not in stripped[:4096]:
        return None
    out: list[str] = []
    for line in stripped.splitlines():
        record = _json_record(line) if line.strip() else {"lane": "session-history"}
        if record is None or record.get("lane") != "session-history":
            return None
        out += _history_lines(record)
    return "\n".join(out)


# ── the driver ──────────────────────────────────────────────────────────────


def _selected(
    windows: list[Window], mode: str, settings: ExtractSettings
) -> list[Window]:
    cap = 1 if mode == "lite" else settings.windows_max
    return windows[:cap]


def _hints(
    tagger: Tagger | None, windows: Sequence[Window]
) -> list[list[dict[str, Any]]]:
    empty: list[list[dict[str, Any]]] = [[] for _ in windows]
    if tagger is None:
        return empty
    try:
        tagged = tagger([w.text for w in windows])
    except Exception:  # noqa: BLE001 — the tagger notes its own failures
        return empty
    return tagged if len(tagged) == len(windows) else empty


def _unparsed_notes(done: Sequence[WindowResult]) -> list[str]:
    unparsed = sum(r.unparsed for r in done)
    return [f"extract_window_unparsed:{unparsed}"] if unparsed else []


def _window_notes(skipped: int, late: int, failed: int) -> list[str]:
    notes = [f"extract_windows_skipped:{skipped + late}"] if skipped + late else []
    if late:
        notes.append("extract_budget_exhausted")
    if failed:
        notes.append(f"extract_windows_failed:{failed}")
    return notes


@dataclass
class ExtractRequest:
    """What one document's extraction needs besides its text."""

    chat: Chat
    model: str
    vocab: Any = None
    mode: str = "full"
    parser_id: str = "raw"
    tagger: Tagger | None = None
    settings: ExtractSettings = field(default_factory=ExtractSettings.from_env)
    # ``time.monotonic()`` by which the whole understand job must be done
    # (its budget is per job, not per document); None = only ``budget_s``.
    deadline: float | None = None

    def remaining(self, started: float) -> float:
        """Seconds left: the per-document budget, cut to the job deadline."""
        now = time.monotonic()
        left = self.settings.budget_s - (now - started)
        return left if self.deadline is None else min(left, self.deadline - now)

    def job_spent(self) -> bool:
        """Less than ``LITE_FLOOR_S`` of the job budget is left."""
        return self.deadline is not None and (
            self.deadline - time.monotonic() < LITE_FLOOR_S
        )


def _within_job_budget(request: ExtractRequest, notes: list[str]) -> ExtractRequest:
    """The request as it runs: past the job budget a document still reads
    one window (lite, outside the deadline, so it costs one call as before
    ADR-116) instead of being cut to nothing, named in the notes."""
    if not request.job_spent():
        return request
    notes.append("extract_job_budget_exhausted")
    return replace(request, mode="lite", deadline=None)


def _after_hints(
    request: ExtractRequest,
    windows: list[Window],
    run: list[Window],
    hints: list[list[dict[str, Any]]],
    notes: list[str],
) -> tuple[ExtractRequest, list[Window], list[list[dict[str, Any]]]]:
    """Tagging may run past the job deadline (``docmodels_budget`` lets it
    reach deadline + T): re-check the job budget after it, so a document
    that has run out still reads its first window (lite, outside the
    deadline) instead of getting a negative window budget and no graph."""
    if request.deadline is None or not request.job_spent():
        return request, run, hints
    lite = _within_job_budget(request, notes)
    first = _selected(windows, lite.mode, lite.settings)
    return lite, first, hints[: len(first)]


def _run_windows(
    request: ExtractRequest,
    windows: list[Window],
    run: list[Window],
    hints: list[list[dict[str, Any]]],
    vocab: Any,
) -> tuple[list[WindowResult], int]:
    # The per-document window budget starts after the hints: tagging has its
    # own budget (ingest_models.docmodels_budget), outside the LLM one.
    started = time.monotonic()
    caller = _Caller(request.chat)
    system = system_prompt(vocab)
    jobs = [
        functools.partial(
            _extract_window, caller, system, w, len(windows), h, request.model
        )
        for w, h in zip(run, hints, strict=True)
    ]
    remaining = request.remaining(started)
    results, late = run_bounded(jobs, request.settings.concurrency, remaining)
    return [r for r in results if r is not None], late


def extract_document(
    text: str, request: ExtractRequest, notes: list[str]
) -> dict[str, Any] | None:
    """One document's merged ``{nodes, edges, extractor, parser}`` payload, or
    None with the reason in ``notes``. Blocking: run it in a worker thread."""
    vocab = request.vocab or default_vocabulary()
    windows = pack_windows(text, request.settings.window_chars)
    if not windows:
        return None
    request = _within_job_budget(request, notes)
    run = _selected(windows, request.mode, request.settings)
    hints = _hints(request.tagger, run)
    request, run, hints = _after_hints(request, windows, run, hints, notes)
    done, late = _run_windows(request, windows, run, hints, vocab)
    good = [r for r in done if r.data is not None]
    if not good:
        reason = done[0].error if done else "extraction budget exhausted"
        notes.append(f"semantic extraction unavailable: {reason}")
        notes.extend(_unparsed_notes(done))
        return None
    skipped = 0 if request.mode == "lite" else len(windows) - len(run)
    notes.extend(_window_notes(skipped, late, len(done) - len(good)))
    notes.extend(_unparsed_notes(done))
    return merge_windows(good, vocab, request.parser_id)


__all__ = [
    "EXTRACTOR_ID",
    "LITE_FLOOR_S",
    "BuiltinVocabulary",
    "ExtractRequest",
    "ExtractSettings",
    "Window",
    "default_vocabulary",
    "extract_document",
    "is_graph",
    "merge_windows",
    "norm_name",
    "normalize_mode",
    "pack_windows",
    "parse_reply",
    "resolve_vocabulary",
    "run_bounded",
    "session_history_text",
    "system_prompt",
]
