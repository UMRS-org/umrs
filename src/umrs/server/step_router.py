"""Step-level router.

Resolves a :class:`umrs.pipeline_sdk.workflow.StepSpec` to a concrete upstream
``(api_base, api_key, model, endpoint_id, headers)`` tuple before the step
runs. Designed so the workflow runner asks the router *once per step* and
templates (VisionStep / LLMStep / ...) only ever read fields from
``ctx.config`` — they do not need to know about endpoint discovery.

Resolution order
----------------
1. Mock mode (``UMRS_STEP_ROUTER_MOCK=1``) → static decision; useful for
   unit / integration tests that should not hit any real network.
2. Explicit ``config["api_base"]`` declared on the step → trust the
   workflow author; use as-is. This is today's behaviour and stays for
   back-compat.
3. Capability match: query the model registry for models satisfying
   ``step.requires`` / ``step.model_type``, then pick the corresponding
   active endpoint via the new capability filter. Prefers the system
   tier first. The caller's own project wins; an endpoint of another
   project is used when its ADR-078 access list admits the caller. When no
   live endpoint exists but one is starting (PENDING / PROVISIONING /
   STARTING), :class:`StepEndpointPending` tells the runner to wait
   (:func:`resolve_waiting`, ADR-109 D-4).
4. Env-var fallback: ``OPENAI_BASE_URL`` / ``R4_GATEWAY_URL``. This is
   what kept the templates working before this router existed.

A step that explicitly declared capability requirements (``requires`` or
``model_type``) and whose registry lookup fails will raise
``StepRoutingError``. Otherwise the router silently falls through to env
so legacy unit tests stay green.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from umrs.pipeline_sdk.workflow import StepSpec

logger = logging.getLogger(__name__)


class StepRoutingError(RuntimeError):
    """Raised when a step declares routing requirements that cannot be met."""


class StepEndpointPending(StepRoutingError):
    """No live endpoint yet, but one the caller may use is starting (ADR-109 D-4).

    The runner waits on this (bounded by ``UMRS_STEP_ENDPOINT_WAIT_SEC``)
    instead of failing the step at once; any other ``StepRoutingError`` still
    fails it.
    """

    def __init__(self, step_name: str, model: str, status: str) -> None:
        super().__init__(
            f"Step '{step_name}': endpoint for {model} is {status or 'starting'}"
        )
        self.step_name = step_name
        self.model = model
        self.status = status


@dataclass
class RoutingDecision:
    """Concrete upstream parameters resolved by :class:`StepRouter`."""

    api_base: str | None = None
    api_key: str | None = None
    model: str | None = None
    endpoint_id: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    resolved_via: str = "env_fallback"
    routed_capabilities: list[str] = field(default_factory=list)
    lora: str | None = None
    model_type: str | None = None
    # Affinity tiebreaker applied inside the registry-capability path.
    # ``affinity_used`` is True when the chosen endpoint matched at least
    # one parent's endpoint id; ``affinity_matched_parents`` lists the
    # parent step names whose endpoint id equals the chosen one. Both
    # default to falsy so existing tests / consumers see no change.
    affinity_used: bool = False
    affinity_matched_parents: list[str] = field(default_factory=list)
    # Load-aware scoring (Slice 4). ``load_score`` is the normalised
    # utilisation of the chosen endpoint at decision time (0.0 = idle,
    # 1.0 = saturated; values >1 are possible if a backend reports
    # over-subscription).
    load_score: float = 0.0
    # Latency-aware scoring (Slice 5). ``latency_score`` and
    # ``queue_score`` are normalised additional penalties
    # (avg_latency_ms / latency_ref, queue_time_ms / queue_ref). Both
    # default to 0.0 when the endpoint reports no telemetry, so legacy
    # paths and existing tests are unaffected.
    latency_score: float = 0.0
    queue_score: float = 0.0
    # Feedback-aware scoring (Slice 6). ``feedback_score`` is the
    # endpoint's EWMA of realised exec_time_ms (across all prior
    # successful steps in this process) divided by the feedback
    # reference. 0.0 for cold endpoints; consumed alongside
    # ``latency_score`` so the router can refine decisions even when
    # the registry doesn't report latency directly.
    feedback_score: float = 0.0
    # Failure-rate EWMA in [0, 1] for the chosen endpoint (Slice 7).
    # 0.0 for cold endpoints — the router treats unobserved endpoints
    # as healthy until proven otherwise.
    failure_score: float = 0.0
    # Capability-preference score (Slice 17). The fraction of
    # ``routing_hints.prefer`` capabilities matched by the chosen
    # endpoint's metadata.  Range [0, 1]; 0.0 when no preferences
    # declared or endpoint has no metadata.
    capability_score: float = 0.0
    # True when this decision was driven by ε-greedy exploration
    # (random pick or cold-endpoint forced exploration), False when
    # the deterministic score path picked the winner.
    exploration_used: bool = False
    # ``total_score`` is the combined affinity + load number used to
    # pick this endpoint within its scope tier.
    total_score: float = 0.0
    # Weighted score breakdown (Slice 20-prep).  Each entry is the
    # *weighted* contribution to ``total_score`` (sign-included), so
    # ``sum(decision_breakdown.values()) ≈ total_score``.
    decision_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # ``asdict`` produces a JSON-safe view (only primitives + dicts).
        return asdict(self)


@dataclass
class AffinityHints:
    """Per-execution affinity signal passed by the runner into ``resolve``.

    The runner builds this immediately before each ``resolve()`` call from
    the ``RoutingDecision.endpoint_id`` of completed parent steps. The
    router treats ``parent_endpoint_ids`` as a tiebreaker among
    capability-satisfying candidates: an endpoint that matches more
    parents ranks higher.

    ``disabled=True`` short-circuits affinity entirely (workflow author
    set ``routing_hints.affinity = "off"``).

    ``lora`` is reserved for the next slice (LoRA-aware routing) and is
    accepted now so the shape is stable; it is not consumed yet.
    """

    parent_endpoint_ids: dict[str, str] = field(default_factory=dict)
    lora: str | None = None
    disabled: bool = False

    @classmethod
    def from_step(
        cls,
        step: StepSpec,
        *,
        parent_endpoint_ids: dict[str, str] | None = None,
    ) -> AffinityHints:
        """Build hints from a step plus a parent-name -> endpoint_id map.

        Reads ``step.routing_hints.affinity`` to decide whether the
        per-step opt-out is in effect. Any value other than the literal
        strings ``"off"`` / ``"disabled"`` / ``False`` keeps affinity
        enabled (forward-compatible with future values like ``"strict"``).
        """
        hints = step.routing_hints if isinstance(step.routing_hints, dict) else {}
        raw = hints.get("affinity")
        disabled = raw in ("off", "disabled", False)
        return cls(
            parent_endpoint_ids=dict(parent_endpoint_ids or {}),
            lora=step.lora,
            disabled=disabled,
        )


# ----------------------------------------------------------------------
# Discovery callbacks
# ----------------------------------------------------------------------
# The router is decoupled from the model-registry / endpoint-registry by
# accepting two callbacks at construction time. Tests inject lambdas; the
# real server wires them to ``model_registry.store`` + the active endpoint
# registry. Keeping the router free of FastAPI/HTTP code makes it cheap to
# unit-test and lets it work both in-process and in worker subprocesses.
# ----------------------------------------------------------------------

#: ``find_models(capabilities, model_type) -> list[ModelEntry-as-dict]``.
ModelLookup = Callable[[list[str], str | None], list[dict[str, Any]]]

#: ``find_endpoint(model_name, project_id, owner_hint) -> Optional[EndpointDict]``.
EndpointLookup = Callable[[str, str | None, str | None], dict[str, Any] | None]

#: ``find_pending(model_name, project_id) -> Optional[EndpointDict]``: a
#: PENDING/PROVISIONING/STARTING row of the model the caller may use.
PendingLookup = Callable[[str, str | None], dict[str, Any] | None]

#: ``alias_models(name) -> models``: the models a role alias (``svc-*``) names.
AliasLookup = Callable[[str], tuple[str, ...]]


def _endpoint_id_of(ep: dict[str, Any]) -> str:
    """Return the stable identity used for affinity matching.

    Matches the same precedence used when populating
    ``RoutingDecision.endpoint_id`` so a parent's recorded id will line
    up with a candidate's id during the affinity tiebreaker.
    """
    return str(ep.get("job_id") or ep.get("endpoint_id") or "")


def _score_endpoint_for_affinity(
    ep: dict[str, Any], hints: AffinityHints
) -> tuple[int, list[str]]:
    """Return ``(score, matched_parent_names)`` for a candidate endpoint.

    Score is the count of parent steps whose recorded ``endpoint_id``
    equals this candidate's id. Empty endpoint id never matches.
    """
    endpoint_id = _endpoint_id_of(ep)
    if not endpoint_id or not hints.parent_endpoint_ids:
        return 0, []
    matches = [
        parent
        for parent, parent_id in hints.parent_endpoint_ids.items()
        if parent_id and parent_id == endpoint_id
    ]
    return len(matches), matches


def _endpoint_load_norm(ep: dict[str, Any]) -> float:
    """Return the normalised load of an endpoint in approximately ``[0, 1]``.

    The router reads load signals out of the endpoint dict so the
    registry side can populate them incrementally without changing the
    router contract. Lookup precedence:

    1. ``ep["load_norm"]`` (top-level override) — pre-normalised float.
    2. ``ep["metadata"]["load_norm"]`` — same, on the metadata dict.
    3. ``metadata.active_requests / metadata.max_concurrency`` —
       computed from raw counters. ``max_concurrency`` defaults to 1.
    4. ``metadata.queue_depth`` divided by ``max_concurrency``.

    Anything missing or non-numeric returns ``0.0``. Values may exceed
    1.0 if a backend reports over-subscription; the scoring function
    deals with that by penalising proportionally rather than clamping.
    """
    md = ep.get("metadata")
    if not isinstance(md, dict):
        md = {}

    def _as_float(v: Any) -> float | None:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    direct = _as_float(ep.get("load_norm"))
    if direct is not None and direct >= 0.0:
        return direct
    md_norm = _as_float(md.get("load_norm"))
    if md_norm is not None and md_norm >= 0.0:
        return md_norm

    cap = _as_float(md.get("max_concurrency")) or 1.0
    if cap <= 0:
        cap = 1.0
    active = _as_float(md.get("active_requests"))
    if active is not None and active >= 0.0:
        return active / cap
    qd = _as_float(md.get("queue_depth"))
    if qd is not None and qd >= 0.0:
        return qd / cap
    return 0.0


def _env_float(name: str, default: float) -> float:
    """Return an env-driven float, clamped to ``>= 0``.

    Empty / unset / non-numeric values fall back to ``default``. Used by
    every router weight / reference knob so behaviour is consistent.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, value)


def _router_weights() -> tuple[float, float, float, float]:
    """Return ``(w_affinity, w_load, w_latency, w_queue)`` from env.

    Defaults: affinity=1.0, load=0.5, latency=0.3, queue=0.3. Affinity
    dominates equal-load candidates; load + latency + queue together
    swing the choice when one endpoint is materially busier or slower.

    Knobs:
        UMRS_ROUTER_AFFINITY_WEIGHT (default 1.0)
        UMRS_ROUTER_LOAD_WEIGHT     (default 0.5)
        UMRS_ROUTER_LATENCY_WEIGHT  (default 0.3)
        UMRS_ROUTER_QUEUE_WEIGHT    (default 0.3)
    """
    return (
        _env_float("UMRS_ROUTER_AFFINITY_WEIGHT", 1.0),
        _env_float("UMRS_ROUTER_LOAD_WEIGHT", 0.5),
        _env_float("UMRS_ROUTER_LATENCY_WEIGHT", 0.3),
        _env_float("UMRS_ROUTER_QUEUE_WEIGHT", 0.3),
    )


# ----------------------------------------------------------------------
# Feedback store (Slice 6)
# ----------------------------------------------------------------------
# A process-local, thread-safe EWMA-of-exec_time_ms keyed by endpoint
# id. The runner records successful step outcomes here; the router
# consults it as a soft feedback signal that complements the registry-
# reported latency. This closes the loop:
#
#     decide -> measure -> bias next decision
#
# Kept deliberately simple (single dict + lock + EWMA) so it works
# in-process without external state. A future slice can swap this for
# a shared store (Redis / Postgres) without touching call sites.
# ----------------------------------------------------------------------


class EndpointFeedbackStore:
    """Thread-safe per-endpoint outcome store.

    Tracks two independent signals:

    * ``_ewma`` — EWMA of realised ``exec_time_ms`` for SUCCESSFUL
      steps. Failures are excluded so a crashing step does not
      poison the latency signal.
    * ``_failure_ewma`` — EWMA of a 0/1 failure indicator (0 on
      success, 1 on failure) recorded for every terminal outcome.
      Acts as a smoothed failure rate in [0, 1].
    * ``_visits`` — total number of recorded outcomes per endpoint.
      Used by the router to decide whether an endpoint is "cold"
      enough to warrant forced exploration.

    The runner calls :meth:`record` on success and
    :meth:`record_failure` on failure; the router consults
    :meth:`get_norm`, :meth:`get_failure_norm`, and :meth:`get_visits`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ewma: dict[str, float] = {}
        self._failure_ewma: dict[str, float] = {}
        self._visits: dict[str, int] = {}

    @staticmethod
    def _alpha() -> float:
        alpha = _env_float("UMRS_ROUTER_FEEDBACK_EWMA_ALPHA", 0.3)
        # Clamp alpha to (0, 1]; 0 would freeze the EWMA and 1 makes
        # it equal to the latest sample (no smoothing).
        if alpha <= 0.0 or alpha > 1.0:
            alpha = 0.3
        return alpha

    def record(self, endpoint_id: str, exec_time_ms: float) -> None:
        """Record a successful outcome (latency + zero-failure)."""
        if not endpoint_id or exec_time_ms <= 0:
            return
        alpha = self._alpha()
        with self._lock:
            prev = self._ewma.get(endpoint_id)
            if prev is None:
                self._ewma[endpoint_id] = float(exec_time_ms)
            else:
                self._ewma[endpoint_id] = (
                    alpha * float(exec_time_ms) + (1.0 - alpha) * prev
                )
            # Mix a 0 (no-failure) sample into the failure EWMA so the
            # rate decays naturally as successes accumulate.
            prev_f = self._failure_ewma.get(endpoint_id, 0.0)
            self._failure_ewma[endpoint_id] = (1.0 - alpha) * prev_f
            self._visits[endpoint_id] = self._visits.get(endpoint_id, 0) + 1

    def record_failure(self, endpoint_id: str) -> None:
        """Record a terminal failure (1.0 sample into failure EWMA).

        Latency EWMA is intentionally untouched — see class docstring.
        """
        if not endpoint_id:
            return
        alpha = self._alpha()
        with self._lock:
            prev_f = self._failure_ewma.get(endpoint_id, 0.0)
            self._failure_ewma[endpoint_id] = alpha * 1.0 + (1.0 - alpha) * prev_f
            self._visits[endpoint_id] = self._visits.get(endpoint_id, 0) + 1

    def get_norm(self, endpoint_id: str) -> float:
        """Return the latency EWMA divided by ``UMRS_ROUTER_FEEDBACK_REF_MS``.

        Returns ``0.0`` for endpoints with no successful runs yet
        (cold endpoints get no latency penalty).
        """
        if not endpoint_id:
            return 0.0
        with self._lock:
            ewma = self._ewma.get(endpoint_id)
        if ewma is None or ewma <= 0:
            return 0.0
        ref = _env_float("UMRS_ROUTER_FEEDBACK_REF_MS", 1000.0) or 1000.0
        return ewma / ref

    def get_failure_norm(self, endpoint_id: str) -> float:
        """Return the failure-rate EWMA in ``[0.0, 1.0]``.

        Returns ``0.0`` for endpoints with no recorded outcomes yet —
        cold endpoints are assumed healthy until proven otherwise.
        """
        if not endpoint_id:
            return 0.0
        with self._lock:
            f = self._failure_ewma.get(endpoint_id, 0.0)
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f

    def get_visits(self, endpoint_id: str) -> int:
        """Return total recorded outcomes (success + failure) for the endpoint."""
        if not endpoint_id:
            return 0
        with self._lock:
            return int(self._visits.get(endpoint_id, 0))

    def get_total_visits(self) -> int:
        """Return sum of all recorded outcomes across every endpoint."""
        with self._lock:
            return int(sum(self._visits.values()))

    def reset(self) -> None:
        """Clear all recorded state. Used by tests."""
        with self._lock:
            self._ewma.clear()
            self._failure_ewma.clear()
            self._visits.clear()


# Module-level singleton. Tests call ``reset_feedback_store`` to wipe
# accumulated state between cases.
_feedback_store = EndpointFeedbackStore()


def get_feedback_store() -> EndpointFeedbackStore:
    """Return the process-wide feedback store."""
    return _feedback_store


def reset_feedback_store() -> None:
    """Test helper: clear all recorded outcomes."""
    _feedback_store.reset()


def _endpoint_feedback_norm(ep: dict[str, Any]) -> float:
    """Return the feedback EWMA norm for an endpoint, or 0.0 if cold.

    Always consulted (independent of ``metadata.avg_latency_ms``) so
    the learned signal refines routing even when the registry doesn't
    surface latency. Operators who don't want feedback in the score
    set ``UMRS_ROUTER_FEEDBACK_WEIGHT=0``.
    """
    endpoint_id = _endpoint_id_of(ep)
    return _feedback_store.get_norm(endpoint_id)


def _endpoint_failure_norm(ep: dict[str, Any]) -> float:
    """Return the failure-rate EWMA for an endpoint, or 0.0 if cold."""
    endpoint_id = _endpoint_id_of(ep)
    return _feedback_store.get_failure_norm(endpoint_id)


def _endpoint_latency_norm(ep: dict[str, Any]) -> float:
    """Return ``avg_latency_ms / latency_ref_ms`` (0.0 if missing).

    Reference configurable via ``UMRS_ROUTER_LATENCY_REF_MS`` (default
    1000ms = 1s). An endpoint reporting 500ms average latency against a
    1000ms reference scores 0.5; an endpoint reporting 2s scores 2.0
    ("twice as slow as the reference"). Missing or non-numeric -> 0.0
    so endpoints without telemetry pay no penalty.
    """
    md = ep.get("metadata")
    if not isinstance(md, dict):
        return 0.0
    raw = md.get("avg_latency_ms")
    try:
        ms = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if ms <= 0:
        return 0.0
    ref = _env_float("UMRS_ROUTER_LATENCY_REF_MS", 1000.0) or 1000.0
    return ms / ref


def _endpoint_queue_norm(ep: dict[str, Any]) -> float:
    """Return ``queue_time_ms / queue_ref_ms`` (0.0 if missing).

    Reference configurable via ``UMRS_ROUTER_QUEUE_REF_MS`` (default
    500ms). Same semantics as :func:`_endpoint_latency_norm`.
    """
    md = ep.get("metadata")
    if not isinstance(md, dict):
        return 0.0
    raw = md.get("queue_time_ms")
    try:
        ms = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if ms <= 0:
        return 0.0
    ref = _env_float("UMRS_ROUTER_QUEUE_REF_MS", 500.0) or 500.0
    return ms / ref


def _capability_match(
    ep_metadata: dict[str, Any],
    preferred: list[Any],
) -> float:
    """Score how well an endpoint's metadata matches preferred capabilities.

    ``preferred`` items can be:
    - ``str`` — treated as weight 1.0
    - ``dict`` with ``{"name": str, "weight": float}`` — explicit weight

    Returns a value in ``[0.0, 1.0]`` (weighted matches / total weight).
    Returns ``0.0`` when *preferred* is empty or total weight is zero.
    """
    if not preferred:
        return 0.0
    total_weight = 0.0
    matched_weight = 0.0
    for item in preferred:
        if isinstance(item, str):
            name, weight = item, 1.0
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            name = item["name"]
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            if weight <= 0:
                continue
        else:
            continue
        total_weight += weight
        if ep_metadata.get(name):
            matched_weight += weight
    if total_weight <= 0:
        return 0.0
    raw = matched_weight / total_weight
    # Clamp to [0, 1] to prevent bad metadata from exploding scores.
    return max(0.0, min(1.0, raw))


def _softcap_penalty(load_norm: float) -> float:
    """Return an extra penalty when utilisation crosses the soft cap.

    Below ``UMRS_ROUTER_LOAD_SOFTCAP`` (default 0.9) returns 0. Above,
    returns ``(load_norm - softcap) * UMRS_ROUTER_SOFTCAP_PENALTY``
    (default factor 5.0). The default makes a 1.0-utilisation endpoint
    incur an extra 0.5 penalty on top of the linear load term, which
    matters for tiebreakers around the saturation boundary while still
    letting affinity win for moderately busy endpoints.
    """
    softcap = _env_float("UMRS_ROUTER_LOAD_SOFTCAP", 0.9)
    if load_norm <= softcap:
        return 0.0
    factor = _env_float("UMRS_ROUTER_SOFTCAP_PENALTY", 5.0)
    return (load_norm - softcap) * factor


#: Model scope tiers, cheapest (shared infra) first.
_SCOPE_ORDER = {"system": 0, "project": 1, "user": 2}


def _scope_tier(entry: dict[str, Any]) -> int:
    return _SCOPE_ORDER.get(str(entry.get("scope", "system")), 99)


@dataclass(frozen=True)
class _ScoreWeights:
    """Every scoring knob, read from the environment once per resolve."""

    aff: float
    load: float
    lat: float
    q: float
    fb: float
    fail: float
    lat_decorr: float
    ucb: float
    ucb_c: float
    cap: float

    @classmethod
    def from_env(cls) -> _ScoreWeights:
        w_aff, w_load, w_lat, w_q = _router_weights()
        # Slice 7: when an endpoint has accumulated feedback, treat the
        # registry-reported latency as a prior and shrink its contribution.
        # Default 0.5 halves the latency term for warm endpoints; 1.0
        # disables decorrelation.
        lat_decorr = _env_float("UMRS_ROUTER_LATENCY_DECORR_FACTOR", 0.5)
        return cls(
            aff=w_aff,
            load=w_load,
            lat=w_lat,
            q=w_q,
            fb=_env_float("UMRS_ROUTER_FEEDBACK_WEIGHT", 0.2),
            fail=_env_float("UMRS_ROUTER_FAILURE_WEIGHT", 0.5),
            lat_decorr=min(max(lat_decorr, 0.0), 1.0),
            # UCB exploration (Slice 15): a bonus that decays with visits.
            ucb=_env_float("UMRS_ROUTER_UCB_WEIGHT", 0.3),
            ucb_c=_env_float("UMRS_ROUTER_UCB_C", 1.0),
            # Capability preference (Slice 17): a soft nudge, not a filter.
            cap=_env_float("UMRS_ROUTER_CAPABILITY_WEIGHT", 0.2),
        )

    def effective_latency(self, lat_norm: float, fb_norm: float) -> float:
        """Latency/feedback decorrelation: with observed exec times the
        registry's reported latency becomes a less-trusted prior."""
        return float(lat_norm) * self.lat_decorr if fb_norm > 0.0 else float(lat_norm)


def _owner_hint(step: StepSpec) -> str | None:
    hints = step.routing_hints if isinstance(step.routing_hints, dict) else {}
    raw = hints.get("owner")
    return raw if isinstance(raw, str) else None


def _preferred_caps(step: StepSpec) -> list[Any]:
    """``routing_hints.prefer``: names or ``{"name", "weight"}`` dicts."""
    hints = step.routing_hints if isinstance(step.routing_hints, dict) else {}
    raw_prefer = hints.get("prefer")
    if not isinstance(raw_prefer, list):
        return []
    return [
        c
        for c in raw_prefer
        if isinstance(c, str)
        and c
        or isinstance(c, dict)
        and isinstance(c.get("name"), str)
    ]


def _ucb_bonus(ep: dict[str, Any], ucb_c: float, total_visits: int) -> float:
    """Cold endpoints (0 visits) get the maximum bonus; it decays as
    sqrt(log(N+1) / (n+1))."""
    visits = _feedback_store.get_visits(_endpoint_id_of(ep))
    if visits:
        return ucb_c * math.sqrt(math.log(total_visits + 1) / (visits + 1))
    return ucb_c * math.sqrt(math.log(total_visits + 1)) if total_visits > 0 else ucb_c


def _score_item(
    idx: int,
    item: dict[str, Any],
    w: _ScoreWeights,
    hints: AffinityHints | None,
    preferred_caps: list[Any],
    total_visits: int,
) -> dict[str, Any]:
    """One viable endpoint's signals and weighted total (higher wins).

    The soft cap is scaled by the load weight, so ``UMRS_ROUTER_LOAD_WEIGHT=0``
    disables the whole load family (linear + soft cap).
    """
    ep = item["ep"]
    aff_score, matches = (
        _score_endpoint_for_affinity(ep, hints) if hints is not None else (0, [])
    )
    load_norm = _endpoint_load_norm(ep)
    lat_norm = _endpoint_latency_norm(ep)
    q_norm = _endpoint_queue_norm(ep)
    fb_norm = _endpoint_feedback_norm(ep)
    fail_norm = _endpoint_failure_norm(ep)
    ucb_bonus = _ucb_bonus(ep, w.ucb_c, total_visits)
    ep_md = ep.get("metadata")
    cap_score = _capability_match(
        ep_md if isinstance(ep_md, dict) else {}, preferred_caps
    )
    total = (
        (w.aff * float(aff_score))
        - (w.load * float(load_norm))
        - (w.lat * w.effective_latency(lat_norm, fb_norm))
        - (w.q * float(q_norm))
        - (w.fb * float(fb_norm))
        - (w.fail * float(fail_norm))
        - (w.load * float(_softcap_penalty(load_norm)))
        + (w.ucb * ucb_bonus)
        + (w.cap * cap_score)
    )
    return {
        **item,
        "aff_score": aff_score,
        "matches": matches,
        "load_norm": load_norm,
        "latency_norm": lat_norm,
        "queue_norm": q_norm,
        "feedback_norm": fb_norm,
        "failure_norm": fail_norm,
        "ucb_bonus": ucb_bonus,
        "cap_score": cap_score,
        "total": total,
        "decl_index": idx,
    }


def _decision_breakdown(chosen: dict[str, Any], w: _ScoreWeights) -> dict[str, float]:
    """Each weighted contribution to ``total_score`` (sign included)."""
    return {
        "affinity": w.aff * float(chosen["aff_score"]),
        "load": -(w.load * float(chosen["load_norm"])),
        "latency": -(
            w.lat * w.effective_latency(chosen["latency_norm"], chosen["feedback_norm"])
        ),
        "queue": -(w.q * float(chosen["queue_norm"])),
        "feedback": -(w.fb * float(chosen["feedback_norm"])),
        "failure": -(w.fail * float(chosen["failure_norm"])),
        "softcap": -(w.load * _softcap_penalty(float(chosen["load_norm"]))),
        "ucb": w.ucb * float(chosen["ucb_bonus"]),
        "capability": w.cap * float(chosen.get("cap_score", 0.0)),
    }


def _active_affinity(hints: AffinityHints | None) -> AffinityHints | None:
    """The hints when affinity is in effect for this step, else None."""
    if hints and not hints.disabled and hints.parent_endpoint_ids:
        return hints
    return None


@dataclass
class StepRouter:
    """Resolves :class:`StepSpec` -> :class:`RoutingDecision`."""

    find_models: ModelLookup | None = None
    find_endpoint: EndpointLookup | None = None
    default_api_base: str | None = None
    default_api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    find_pending: PendingLookup | None = None
    alias_models: AliasLookup | None = None

    def resolve(
        self,
        step: StepSpec,
        *,
        project_id: str | None = None,
        user_id: str | None = None,
        affinity_hints: AffinityHints | None = None,
    ) -> RoutingDecision:
        # 1. Mock mode short-circuit (affinity has no meaning here).
        if os.environ.get("UMRS_STEP_ROUTER_MOCK") == "1":
            return self._mock_decision(step)

        config = step.config or {}

        # 2. Explicit api_base on the step config -> trust it. Workflow
        # authors who pin api_base have already opted out of routing;
        # affinity is silently bypassed.
        if config.get("api_base"):
            api_base = str(config["api_base"])
            return RoutingDecision(
                api_base=api_base,
                api_key=self._explicit_key(config, api_base),
                model=step.model or config.get("model"),
                headers=dict(self.extra_headers),
                resolved_via="explicit_config",
                routed_capabilities=list(step.requires),
                lora=step.lora,
                model_type=step.model_type,
            )

        # 3. Capability match via registry. Affinity (if any) is applied
        # as a tiebreaker among capability-satisfying candidates.
        wants_routing = bool(step.requires) or bool(step.model_type)
        if wants_routing and self.find_models is not None:
            decision = self._resolve_from_registry(
                step,
                project_id=project_id,
                affinity_hints=affinity_hints,
            )
            if decision is not None:
                return decision
            # ADR-109 D-4: an endpoint still starting is "wait", not "fail".
            self._raise_if_pending(step, project_id)
            # Strict mode when explicit requirements were declared: fail
            # rather than silently picking up an env endpoint that may not
            # satisfy them.
            raise StepRoutingError(
                f"Step '{step.name}' requires "
                f"capabilities={list(step.requires)} model_type={step.model_type!r} "
                "but no matching active endpoint was found."
            )

        # 4. Env fallback (today's behaviour)
        return self._env_fallback_decision(step)

    # ------------------------------------------------------------------
    # Internal resolution helpers
    # ------------------------------------------------------------------

    def _resolve_from_registry(
        self,
        step: StepSpec,
        *,
        project_id: str | None,
        affinity_hints: AffinityHints | None = None,
    ) -> RoutingDecision | None:
        # Prefer system-tier models (cheapest path / shared infra).
        candidates = sorted(self.registry_candidates(step), key=_scope_tier)
        # Resolve every viable candidate so we can score them. Without
        # affinity hints and with no load info, every score is 0 and the
        # first-by-scope candidate wins (legacy behaviour preserved).
        viable = self._viable(candidates, project_id, _owner_hint(step))
        if not viable:
            return None

        # Scope tier is a HARD policy gate, not a tiebreaker: only score
        # within the lowest tier that has a viable endpoint. Mixing
        # tiers via score would let a project endpoint outrank a system
        # endpoint on load, which is a policy surprise we don't want.
        lowest_tier = min(item["scope_tier"] for item in viable)
        tier_pool = [item for item in viable if item["scope_tier"] == lowest_tier]

        hints = _active_affinity(affinity_hints)
        weights = _ScoreWeights.from_env()
        preferred_caps = _preferred_caps(step)
        total_visits = _feedback_store.get_total_visits()
        scored = [
            _score_item(idx, item, weights, hints, preferred_caps, total_visits)
            for idx, item in enumerate(tier_pool)
        ]
        # Highest total wins; ties broken by declaration order so legacy
        # tests with all-zero scores stay deterministic. The UCB bonus is
        # already folded into ``total`` (Slice 15): no separate random pick.
        scored.sort(key=lambda s: (-s["total"], s["decl_index"]))
        return self._registry_decision(step, scored[0], hints, weights)

    def _viable(
        self,
        candidates: list[dict[str, Any]],
        project_id: str | None,
        owner_hint: str | None,
    ) -> list[dict[str, Any]]:
        """Candidates with a live endpoint that has an upstream URL."""
        viable: list[dict[str, Any]] = []
        for entry in candidates:
            model_name = str(entry.get("name") or "")
            ep = (
                self.find_endpoint(model_name, project_id, owner_hint)
                if model_name and self.find_endpoint is not None
                else None
            )
            api_base = (ep.get("litellm_url") or ep.get("api_base")) if ep else None
            if api_base:
                viable.append(
                    {
                        "entry": entry,
                        "ep": ep,
                        "api_base": api_base,
                        "scope_tier": _scope_tier(entry),
                    }
                )
        return viable

    def _registry_decision(
        self,
        step: StepSpec,
        chosen: dict[str, Any],
        hints: AffinityHints | None,
        w: _ScoreWeights,
    ) -> RoutingDecision:
        entry = chosen["entry"]
        ep = chosen["ep"]
        endpoint_id = str(ep.get("job_id") or ep.get("endpoint_id") or "")
        return RoutingDecision(
            api_base=str(chosen["api_base"]),
            api_key=ep.get("api_key") or self._env_api_key(),
            model=ep.get("served_model_name") or str(entry.get("name") or ""),
            endpoint_id=endpoint_id,
            headers=dict(self.extra_headers),
            resolved_via="registry_capability",
            routed_capabilities=list(step.requires),
            lora=step.lora,
            model_type=step.model_type
            or (str(entry.get("model_type")) if entry.get("model_type") else None),
            affinity_used=bool(hints is not None and chosen["aff_score"] > 0),
            affinity_matched_parents=list(chosen["matches"]),
            load_score=float(chosen["load_norm"]),
            latency_score=float(chosen["latency_norm"]),
            queue_score=float(chosen["queue_norm"]),
            feedback_score=float(chosen["feedback_norm"]),
            failure_score=float(chosen["failure_norm"]),
            capability_score=float(chosen.get("cap_score", 0.0)),
            exploration_used=False,
            total_score=float(chosen["total"]),
            decision_breakdown=_decision_breakdown(chosen, w),
        )

    # ------------------------------------------------------------------
    # Candidate + endpoint-state helpers (shared with the trigger-time ensure)
    # ------------------------------------------------------------------

    def pinned_names(self, step: StepSpec) -> set[str]:
        """The model names the step pinned: ``step.model`` and, for a role
        alias (ADR-106 D-1), the models whose roster entries carry it."""
        if not step.model:
            return set()
        names = {step.model}
        if self.alias_models is not None:
            names.update(self.alias_models(step.model))
        return names

    def registry_candidates(self, step: StepSpec) -> list[dict[str, Any]]:
        """Registry models satisfying the step; narrowed to the pinned model
        (or the models of a pinned role alias) when any of them is present."""
        if self.find_models is None:
            return []
        candidates = self.find_models(list(step.requires), step.model_type) or []
        pinned = self.pinned_names(step)
        preferred = [m for m in candidates if m.get("name") in pinned]
        return preferred or candidates

    def pending_endpoint(
        self, model_name: str, project_id: str | None
    ) -> dict[str, Any] | None:
        """A starting endpoint of *model_name* the caller may use, or None."""
        if self.find_pending is None:
            return None
        return self.find_pending(model_name, project_id)

    def has_usable_endpoint(self, names: list[str], project_id: str | None) -> bool:
        """True when any of *names* has a live or starting endpoint for the caller."""
        for name in names:
            if self.find_endpoint is not None and self.find_endpoint(
                name, project_id, None
            ):
                return True
            if self.pending_endpoint(name, project_id) is not None:
                return True
        return False

    def _raise_if_pending(self, step: StepSpec, project_id: str | None) -> None:
        for entry in self.registry_candidates(step):
            name = str(entry.get("name") or "")
            pending = self.pending_endpoint(name, project_id) if name else None
            if pending is not None:
                raise StepEndpointPending(
                    step.name, name, str(pending.get("status") or "")
                )

    def _env_fallback_decision(self, step: StepSpec) -> RoutingDecision:
        api_base = (
            self.default_api_base
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("R4_GATEWAY_URL")
        )
        return RoutingDecision(
            api_base=api_base,
            api_key=self._env_api_key(),
            model=step.model or (step.config or {}).get("model"),
            headers=dict(self.extra_headers),
            resolved_via="env_fallback",
            routed_capabilities=list(step.requires),
            lora=step.lora,
            model_type=step.model_type,
        )

    def _env_api_key(self) -> str | None:
        return self.default_api_key or os.environ.get("OPENAI_API_KEY")

    def _explicit_key(self, config: dict[str, Any], api_base: str) -> str | None:
        """The step's own key, else the platform key — but the platform key
        rides only to the platform's own base: a workflow- or param-chosen
        host got it as a Bearer token (2026-09-25)."""
        if config.get("api_key"):
            return str(config["api_key"])
        return self._env_api_key() if self._is_platform_base(api_base) else None

    def _is_platform_base(self, api_base: str) -> bool:
        """True when *api_base* is one this deployment configured itself."""
        bases = (
            self.default_api_base,
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("R4_GATEWAY_URL"),
        )
        return api_base.rstrip("/") in {b.rstrip("/") for b in bases if b}

    def _mock_decision(self, step: StepSpec) -> RoutingDecision:
        return RoutingDecision(
            api_base="http://router-mock.invalid",
            api_key="sk-mock",
            model=step.model or (step.config or {}).get("model") or "mock-model",
            endpoint_id="mock-endpoint",
            headers={"X-Step-Router": "mock"},
            resolved_via="mock",
            routed_capabilities=list(step.requires),
            lora=step.lora,
            model_type=step.model_type,
        )


# ----------------------------------------------------------------------
# Convenience: default router for the in-process runner
# ----------------------------------------------------------------------


def build_default_router() -> StepRouter:
    """Construct a router with no live registry hooks.

    Safe to use anywhere — the runner falls back to env on miss as long as
    no step declares capability requirements. Real deployments should pass
    a router constructed with concrete ``find_models`` / ``find_endpoint``
    callbacks (the registry handler does this when running under UMRS).
    """
    return StepRouter()


# ----------------------------------------------------------------------
# ADR-109 D-4: a step waits (bounded) for an endpoint that is starting
# ----------------------------------------------------------------------

#: Queue plus model load of a cold endpoint (site override:
#: ``step_executor.endpoint_wait_sec`` -> ``UMRS_STEP_ENDPOINT_WAIT_SEC``).
DEFAULT_STEP_ENDPOINT_WAIT_SEC = 45 * 60
DEFAULT_STEP_ENDPOINT_POLL_SEC = 15.0


def step_endpoint_wait_sec() -> float:
    """How long a step waits for a starting endpoint (0 = fail at once)."""
    raw = _env_float("UMRS_STEP_ENDPOINT_WAIT_SEC", DEFAULT_STEP_ENDPOINT_WAIT_SEC)
    return max(raw, 0.0)


def _step_endpoint_poll_sec() -> float:
    raw = _env_float("UMRS_STEP_ENDPOINT_POLL_SEC", DEFAULT_STEP_ENDPOINT_POLL_SEC)
    return raw if raw > 0 else DEFAULT_STEP_ENDPOINT_POLL_SEC


def resolve_waiting(
    router: StepRouter,
    step: StepSpec,
    *,
    cancelled: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    **resolve_kwargs: Any,
) -> RoutingDecision:
    """``router.resolve`` that waits while the step's endpoint is starting.

    Re-resolves every poll interval until a live endpoint appears, then
    returns its decision. Past the bounded clock (or on cancel) it raises
    :class:`StepRoutingError` naming the model; any other routing error
    propagates at once, as before.
    """
    budget = step_endpoint_wait_sec()
    deadline = clock() + budget
    while True:
        try:
            return router.resolve(step, **resolve_kwargs)
        except StepEndpointPending as pending:
            _wait_or_give_up(pending, deadline, budget, clock, sleep, cancelled)


def _wait_or_give_up(
    pending: StepEndpointPending,
    deadline: float,
    budget: float,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    cancelled: Callable[[], bool] | None,
) -> None:
    left = deadline - clock()
    if left <= 0:
        raise StepRoutingError(
            f"Step '{pending.step_name}': endpoint for {pending.model} not READY "
            f"within {int(budget)} s (last status: {pending.status or 'unknown'})"
        ) from pending
    if cancelled is not None and cancelled():
        raise StepRoutingError(
            f"Step '{pending.step_name}': run cancelled while waiting for "
            f"endpoint {pending.model}"
        ) from pending
    logger.info("%s; waiting (%.0f s left)", pending, left)
    sleep(min(_step_endpoint_poll_sec(), left))


def _registry_models(
    capabilities: list[str], model_type: str | None
) -> list[dict[str, Any]]:
    """Model-registry entries of *model_type* carrying every capability."""
    try:
        from umrs.api.endpoints import _capability_satisfied
        from umrs.api.models import get_model_store
        from umrs.model_registry.store import AccessContext

        store = get_model_store()
        ctx = AccessContext(user_id="", is_admin=True)
        entries = [e.to_dict() for e in store.list_models(ctx, limit=500)]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Model registry lookup failed: %s", exc)
        return []
    return [
        entry
        for entry in entries
        if (not model_type or entry.get("model_type") == model_type)
        and all(_capability_satisfied(entry, tag) for tag in capabilities)
    ]


def _endpoint_caller(user_id: str | None, project_id: str) -> Any:
    """The ADR-078 routing caller of a run: its user in its project. Never
    admin (admin does not widen routing); circle scopes are not resolved
    here, so a circle-only share is not routable from a step."""
    from umrs.orchestrator.endpoint_access import EndpointCaller

    return EndpointCaller(user_id=user_id or "", project_id=project_id)


def _serves_model(ep: Any, model_name: str) -> bool:
    served = (ep.metadata or {}).get("served_model_name")
    return served == model_name or ep.model == model_name


def _own_live_endpoint(registry: Any, project: str, model_name: str) -> Any:
    from umrs.orchestrator.endpoint_registry import EndpointStatus

    ep = registry.get(project, model_name, prefer_warm=True)
    if ep is not None:
        return ep
    # Fall back to scanning all ready endpoints for served model match.
    for candidate in registry.list_all(
        project_id=project or None,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    ):
        if _serves_model(candidate, model_name):
            return candidate
    return None


def _registry_endpoint(
    model_name: str, project: str, user_id: str | None
) -> dict[str, Any] | None:
    """Own project first, then an endpoint shared with the caller (ADR-078 D-2)."""
    try:
        from umrs.orchestrator.endpoint_registry import get_registry

        registry = get_registry()
        ep = _own_live_endpoint(registry, project, model_name)
        if ep is None:
            ep = registry.select_shared_endpoint(
                _endpoint_caller(user_id, project), model_name
            )
        return ep.to_dict() if ep is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("Endpoint registry lookup failed: %s", exc)
    return None


def _registry_pending(
    model_name: str, project: str, user_id: str | None
) -> dict[str, Any] | None:
    """A PENDING/PROVISIONING/STARTING row of the model the caller may use
    (own project, or shared with the caller: ADR-078)."""
    try:
        from umrs.orchestrator.endpoint_access import can_use
        from umrs.orchestrator.endpoint_registry import EndpointStatus, get_registry

        caller = _endpoint_caller(user_id, project)
        starting = [
            EndpointStatus.PENDING,
            EndpointStatus.PROVISIONING,
            EndpointStatus.STARTING,
        ]
        for ep in get_registry().list_all(status_filter=starting):
            if _serves_model(ep, model_name) and can_use(ep, caller):
                return ep.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Pending endpoint lookup failed: %s", exc)
    return None


def _role_alias_models(name: str) -> tuple[str, ...]:
    try:
        from umrs.orchestrator.provision_roster import role_alias_models

        return role_alias_models(name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Role alias lookup failed: %s", exc)
        return ()


def build_registry_backed_router(
    *,
    project_id: str | None = None,
    user_id: str | None = None,
) -> StepRouter:
    """Build a router that consults the live UMRS model + endpoint registries.

    Imports are lazy so this module stays cheap to import in worker
    subprocesses that do not need the registry backends. ``user_id`` is the
    run's triggering user: an endpoint of another project is routable when
    its ADR-078 access list admits that user in the run's project.
    """

    def _find_endpoint(
        model_name: str, proj: str | None, owner_hint: str | None
    ) -> dict[str, Any] | None:
        return _registry_endpoint(model_name, proj or project_id or "", user_id)

    def _find_pending(model_name: str, proj: str | None) -> dict[str, Any] | None:
        return _registry_pending(model_name, proj or project_id or "", user_id)

    return StepRouter(
        find_models=_registry_models,
        find_endpoint=_find_endpoint,
        find_pending=_find_pending,
        alias_models=_role_alias_models,
    )
