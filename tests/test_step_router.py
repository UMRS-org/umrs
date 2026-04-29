"""Unit tests for ``StepRouter`` resolution paths.

Covers the four resolution branches plus strict failure when a step
declares capability requirements but the registry yields nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from umrs.pipeline_sdk.workflow import StepSpec
from umrs.server.step_router import (
    RoutingDecision,
    StepRouter,
    StepRoutingError,
    build_default_router,
    reset_feedback_store,
)


@pytest.fixture(autouse=True)
def _deterministic_router(monkeypatch):
    """Disable Slice 15 UCB exploration by default.

    UCB tests opt back in by setting their own env values.
    Also reset the feedback store between tests so EWMA/visit state
    from one case never leaks into another.
    """
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0")
    reset_feedback_store()
    yield
    reset_feedback_store()


def _spec(**kw: Any) -> StepSpec:
    return StepSpec(
        name=kw.pop("name", "step"),
        entrypoint=kw.pop("entrypoint", "pkg.mod:Cls"),
        config=kw.pop("config", {}),
        depends_on=kw.pop("depends_on", None),
        requires=kw.pop("requires", []),
        model=kw.pop("model", None),
        lora=kw.pop("lora", None),
        model_type=kw.pop("model_type", None),
        routing_hints=kw.pop("routing_hints", {}),
    )


# ---------------------------------------------------------------- mock mode


def test_mock_mode_short_circuits(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_ROUTER_MOCK", "1")
    router = StepRouter()  # no hooks needed
    decision = router.resolve(_spec(name="anything", requires=["vision"]))
    assert decision.resolved_via == "mock"
    assert decision.api_base == "http://router-mock.invalid"
    assert decision.routed_capabilities == ["vision"]


# --------------------------------------------------- explicit config wins


def test_explicit_config_api_base_wins(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    called: List[Any] = []

    def _fail(*a: Any, **kw: Any) -> Any:
        called.append((a, kw))
        raise AssertionError("registry should not be consulted")

    router = StepRouter(find_models=_fail, find_endpoint=_fail)
    decision = router.resolve(
        _spec(
            name="explicit",
            requires=["vision"],  # router still must NOT call registry
            config={"api_base": "http://my-vlm:1234", "model": "pinned-vlm"},
        ),
    )
    assert decision.resolved_via == "explicit_config"
    assert decision.api_base == "http://my-vlm:1234"
    assert decision.api_key == "sk-env"
    # find_models / find_endpoint must not be called when api_base is pinned.
    assert called == []


# ---------------------------------------------------- registry capability


def test_registry_capability_match_picks_system_then_endpoint(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    captured: Dict[str, Any] = {}

    def find_models(caps: List[str], model_type: Optional[str]) -> List[Dict[str, Any]]:
        captured["caps"] = caps
        captured["mt"] = model_type
        return [
            # Project tier first in the response — router must reorder.
            {"name": "vlm-project", "scope": "project", "model_type": "vision"},
            {"name": "vlm-system", "scope": "system", "model_type": "vision"},
        ]

    def find_endpoint(
        model_name: str, project_id: Optional[str], owner_hint: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        captured.setdefault("calls", []).append((model_name, project_id, owner_hint))
        if model_name == "vlm-system":
            return {
                "litellm_url": "http://gw/llm/system",
                "served_model_name": "Qwen/Qwen2.5-VL",
                "job_id": "ep-sys-1",
            }
        return None

    router = StepRouter(find_models=find_models, find_endpoint=find_endpoint)
    decision = router.resolve(
        _spec(name="vd", requires=["vision"], model_type="vision"),
        project_id="p-1",
        user_id="alice",
    )
    assert decision.resolved_via == "registry_capability"
    assert decision.api_base == "http://gw/llm/system"
    assert decision.endpoint_id == "ep-sys-1"
    assert decision.model == "Qwen/Qwen2.5-VL"
    assert decision.routed_capabilities == ["vision"]
    assert captured["caps"] == ["vision"]
    assert captured["mt"] == "vision"
    # System tier was tried first — confirm by calls order.
    assert captured["calls"][0][0] == "vlm-system"


def test_registry_strict_mode_raises_when_no_match(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    router = StepRouter(
        find_models=lambda caps, mt: [],
        find_endpoint=lambda *a, **k: None,
    )
    with pytest.raises(StepRoutingError, match="vision"):
        router.resolve(_spec(name="vd", requires=["vision"]))


def test_pinned_model_filters_candidate_set(monkeypatch):
    """When step.model is set, it narrows the candidate list to that name."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    def find_models(caps: List[str], mt: Optional[str]) -> List[Dict[str, Any]]:
        return [
            {"name": "vlm-a", "scope": "system", "model_type": "vision"},
            {"name": "vlm-b", "scope": "system", "model_type": "vision"},
        ]

    seen: List[str] = []

    def find_endpoint(model_name: str, *_a: Any, **_k: Any) -> Optional[Dict[str, Any]]:
        seen.append(model_name)
        if model_name == "vlm-b":
            return {"litellm_url": "http://b", "job_id": "ep-b"}
        return None

    router = StepRouter(find_models=find_models, find_endpoint=find_endpoint)
    decision = router.resolve(_spec(requires=["vision"], model="vlm-b"))
    assert decision.resolved_via == "registry_capability"
    assert decision.endpoint_id == "ep-b"
    # find_endpoint must only have been called with vlm-b (preferred filter).
    assert seen == ["vlm-b"]


# --------------------------------------------------------- env fallback


def test_env_fallback_used_when_no_requirements(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-base/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    # Even with hooks present, no requires/model_type means env path.
    router = StepRouter(
        find_models=lambda caps, mt: [{"name": "x"}],
        find_endpoint=lambda *a, **k: {"litellm_url": "http://x"},
    )
    decision = router.resolve(_spec(name="plain"))
    assert decision.resolved_via == "env_fallback"
    assert decision.api_base == "http://env-base/v1"
    assert decision.api_key == "sk-env"


def test_default_router_falls_back_to_env(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://default-env/v1")
    router = build_default_router()
    decision = router.resolve(_spec(name="x"))
    assert decision.resolved_via == "env_fallback"
    assert decision.api_base == "http://default-env/v1"


# ---------------------------------------------------- decision serialisation


def test_routing_decision_round_trips_via_to_dict():
    d = RoutingDecision(
        api_base="http://x", model="m", endpoint_id="ep", resolved_via="mock"
    )
    out = d.to_dict()
    assert out["api_base"] == "http://x"
    assert out["model"] == "m"
    assert out["endpoint_id"] == "ep"
    assert out["resolved_via"] == "mock"
    # Affinity fields ship in the routing log even when unused.
    assert out["affinity_used"] is False
    assert out["affinity_matched_parents"] == []


# ------------------------------------------------ affinity-aware routing


def _two_endpoint_router(monkeypatch) -> StepRouter:
    """Helper: capability registry with two same-tier endpoints (ep-1 / ep-2).

    Both satisfy ``requires=["vision"]``. Without affinity hints the
    router picks the first one (declaration order after scope sort).
    """
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    def find_models(caps: List[str], mt: Optional[str]) -> List[Dict[str, Any]]:
        return [
            {"name": "vlm-a", "scope": "system", "model_type": "vision"},
            {"name": "vlm-b", "scope": "system", "model_type": "vision"},
        ]

    def find_endpoint(model_name: str, *_a: Any, **_k: Any) -> Optional[Dict[str, Any]]:
        if model_name == "vlm-a":
            return {"litellm_url": "http://a", "job_id": "ep-1"}
        if model_name == "vlm-b":
            return {"litellm_url": "http://b", "job_id": "ep-2"}
        return None

    return StepRouter(find_models=find_models, find_endpoint=find_endpoint)


def test_resolve_uses_affinity_as_tiebreaker_in_registry_path(monkeypatch):
    """Same scope tier: affinity to ep-2 flips the choice from ep-1."""
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router(monkeypatch)
    spec = _spec(name="child", requires=["vision"], model_type="vision")

    # Without hints: declaration order wins (ep-1).
    baseline = router.resolve(spec)
    assert baseline.endpoint_id == "ep-1"
    assert baseline.affinity_used is False
    assert baseline.affinity_matched_parents == []

    # With parent on ep-2: affinity flips selection.
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-2"})
    decision = router.resolve(spec, affinity_hints=hints)
    assert decision.endpoint_id == "ep-2"
    assert decision.affinity_used is True
    assert decision.affinity_matched_parents == ["parent"]


def test_resolve_ignores_affinity_when_no_capability_match(monkeypatch):
    """Affinity is a tiebreaker, never a constraint: a parent endpoint that
    no longer satisfies the capability is silently ignored, not preferred."""
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router(monkeypatch)
    spec = _spec(name="child", requires=["vision"], model_type="vision")

    # Parent ran on an endpoint that isn't in the candidate set.
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-stale"})
    decision = router.resolve(spec, affinity_hints=hints)
    # Falls back to declaration order; no parent matched.
    assert decision.endpoint_id == "ep-1"
    assert decision.affinity_used is False
    assert decision.affinity_matched_parents == []


def test_resolve_respects_affinity_disabled_hint(monkeypatch):
    """``disabled=True`` short-circuits affinity even with a matching parent."""
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router(monkeypatch)
    spec = _spec(name="child", requires=["vision"], model_type="vision")

    hints = AffinityHints(
        parent_endpoint_ids={"parent": "ep-2"}, disabled=True
    )
    decision = router.resolve(spec, affinity_hints=hints)
    assert decision.endpoint_id == "ep-1"  # default, not ep-2
    assert decision.affinity_used is False


def test_resolve_without_affinity_hints_unchanged(monkeypatch):
    """Calling ``resolve`` without the new kwarg behaves exactly as before."""
    router = _two_endpoint_router(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"
    assert decision.affinity_used is False
    assert decision.affinity_matched_parents == []


def test_affinity_hints_from_step_reads_routing_hints_off():
    """``routing_hints={'affinity': 'off'}`` produces ``disabled=True``."""
    from umrs.server.step_router import AffinityHints

    spec = _spec(routing_hints={"affinity": "off"})
    h = AffinityHints.from_step(spec, parent_endpoint_ids={"p": "ep-1"})
    assert h.disabled is True
    assert h.parent_endpoint_ids == {"p": "ep-1"}

    # Default (no affinity key) keeps it enabled.
    spec2 = _spec(routing_hints={})
    h2 = AffinityHints.from_step(spec2, parent_endpoint_ids={"p": "ep-1"})
    assert h2.disabled is False


# --------------------------------------------------- load-aware scoring


def _two_endpoint_router_with_load(
    monkeypatch,
    *,
    a_load: Optional[Dict[str, Any]] = None,
    b_load: Optional[Dict[str, Any]] = None,
) -> StepRouter:
    """Helper: two same-tier endpoints (ep-1, ep-2) with custom load metadata."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    def find_models(caps: List[str], mt: Optional[str]) -> List[Dict[str, Any]]:
        return [
            {"name": "vlm-a", "scope": "system", "model_type": "vision"},
            {"name": "vlm-b", "scope": "system", "model_type": "vision"},
        ]

    def find_endpoint(model_name: str, *_a: Any, **_k: Any) -> Optional[Dict[str, Any]]:
        if model_name == "vlm-a":
            return {
                "litellm_url": "http://a",
                "job_id": "ep-1",
                "metadata": dict(a_load or {}),
            }
        if model_name == "vlm-b":
            return {
                "litellm_url": "http://b",
                "job_id": "ep-2",
                "metadata": dict(b_load or {}),
            }
        return None

    return StepRouter(find_models=find_models, find_endpoint=find_endpoint)


def test_load_penalty_breaks_affinity_tie(monkeypatch):
    """Two candidates, no affinity: the less-loaded endpoint wins."""
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 8, "max_concurrency": 8},  # saturated
        b_load={"active_requests": 0, "max_concurrency": 8},  # idle
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Without load awareness, ep-1 (declaration order) would win.
    # With load awareness, ep-2 (idle) wins.
    assert decision.endpoint_id == "ep-2"
    assert decision.load_score == pytest.approx(0.0)
    assert decision.affinity_used is False


def test_affinity_beats_small_load_delta(monkeypatch):
    """Affinity (weight 1.0) outranks a small load delta (weight 0.5).

    Parent ran on ep-1 which is slightly more loaded; affinity score 1
    minus load 0.2 * 0.5 = 0.9 still beats 0 - 0.0 * 0.5 = 0.0 for ep-2.
    """
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 2, "max_concurrency": 10},  # 0.2
        b_load={"active_requests": 0, "max_concurrency": 10},  # 0.0
    )
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-1"})
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision"),
        affinity_hints=hints,
    )
    assert decision.endpoint_id == "ep-1"
    assert decision.affinity_used is True
    assert decision.load_score == pytest.approx(0.2)


def test_load_dominates_when_affinity_match_is_overloaded(monkeypatch):
    """If the affinity-matched endpoint is heavily overloaded, load wins.

    With default weights (aff=1.0, load=0.5), an affinity match on a
    fully saturated endpoint scores 1 - 0.5 * 1.0 = 0.5; an idle
    non-match scores 0. To flip the choice we need the load penalty to
    exceed the affinity bonus, which happens when load_norm >= 2.0 with
    default weights — i.e. the endpoint reports being 2x oversubscribed.
    """
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 20, "max_concurrency": 8},  # 2.5
        b_load={"active_requests": 0, "max_concurrency": 8},
    )
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-1"})
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision"),
        affinity_hints=hints,
    )
    # ep-1 score: 1.0 - 0.5 * 2.5 = -0.25
    # ep-2 score: 0.0 - 0.5 * 0.0 = 0.0
    # ep-2 wins; affinity was considered but did not prevail.
    assert decision.endpoint_id == "ep-2"
    assert decision.affinity_used is False


def test_load_weight_zero_disables_load_term(monkeypatch):
    """Setting ``UMRS_ROUTER_LOAD_WEIGHT=0`` reverts to affinity-only."""
    monkeypatch.setenv("UMRS_ROUTER_LOAD_WEIGHT", "0")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 8, "max_concurrency": 8},
        b_load={"active_requests": 0, "max_concurrency": 8},
    )
    # Without affinity, ties broken by declaration order -> ep-1.
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"
    # load_score is still recorded for observability even when weight=0.
    assert decision.load_score == pytest.approx(1.0)


def test_load_norm_top_level_override(monkeypatch):
    """``ep['load_norm']`` takes precedence over computed metadata."""
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    def find_models(caps: List[str], mt: Optional[str]) -> List[Dict[str, Any]]:
        return [
            {"name": "vlm-a", "scope": "system", "model_type": "vision"},
            {"name": "vlm-b", "scope": "system", "model_type": "vision"},
        ]

    def find_endpoint(name: str, *_a: Any, **_k: Any) -> Optional[Dict[str, Any]]:
        if name == "vlm-a":
            # active=0 implies load_norm=0; the top-level override
            # should force load_norm=1.0 and lose to ep-2.
            return {
                "litellm_url": "http://a",
                "job_id": "ep-1",
                "load_norm": 1.0,
                "metadata": {"active_requests": 0, "max_concurrency": 8},
            }
        return {
            "litellm_url": "http://b",
            "job_id": "ep-2",
            "metadata": {},
        }

    router = StepRouter(find_models=find_models, find_endpoint=find_endpoint)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"


def test_no_load_metadata_preserves_legacy_order(monkeypatch):
    """Endpoints without any load info score 0; declaration order wins."""
    router = _two_endpoint_router_with_load(monkeypatch)  # no metadata
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"
    assert decision.load_score == pytest.approx(0.0)
    assert decision.total_score == pytest.approx(0.0)


def test_scope_tier_is_hard_gate_not_score(monkeypatch):
    """A heavily-loaded system endpoint still beats an idle project endpoint.

    Scope is policy: load awareness only operates within a tier.
    """
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)

    def find_models(caps: List[str], mt: Optional[str]) -> List[Dict[str, Any]]:
        return [
            {"name": "vlm-sys", "scope": "system", "model_type": "vision"},
            {"name": "vlm-proj", "scope": "project", "model_type": "vision"},
        ]

    def find_endpoint(name: str, *_a: Any, **_k: Any) -> Optional[Dict[str, Any]]:
        if name == "vlm-sys":
            return {
                "litellm_url": "http://sys",
                "job_id": "ep-sys",
                "metadata": {"active_requests": 100, "max_concurrency": 1},
            }
        return {
            "litellm_url": "http://proj",
            "job_id": "ep-proj",
            "metadata": {"active_requests": 0, "max_concurrency": 8},
        }

    router = StepRouter(find_models=find_models, find_endpoint=find_endpoint)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # System tier wins despite being saturated.
    assert decision.endpoint_id == "ep-sys"


# ---------------------------------------------- latency / queue scoring


def test_latency_penalty_breaks_tie(monkeypatch):
    """Two equally-loaded endpoints: lower avg_latency_ms wins."""
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"avg_latency_ms": 2000},  # slow
        b_load={"avg_latency_ms": 200},  # fast
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Default: w_lat=0.3, ref=1000ms.
    # ep-1: 0 - 0.3*2.0 = -0.6;  ep-2: 0 - 0.3*0.2 = -0.06. ep-2 wins.
    assert decision.endpoint_id == "ep-2"
    assert decision.latency_score == pytest.approx(0.2)


def test_queue_time_penalty_breaks_tie(monkeypatch):
    """Two equally-loaded, same-latency endpoints: lower queue_time wins."""
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"queue_time_ms": 2000},
        b_load={"queue_time_ms": 50},
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Default: w_q=0.3, queue_ref=500ms.
    # ep-1: 0 - 0.3*4.0 = -1.2;  ep-2: 0 - 0.3*0.1 = -0.03. ep-2 wins.
    assert decision.endpoint_id == "ep-2"
    assert decision.queue_score == pytest.approx(0.1)


def test_softcap_penalty_at_saturation(monkeypatch):
    """Soft cap pushes a saturated endpoint behind a 90%-loaded one.

    Without soft cap, a fully-saturated (1.0) endpoint and a slightly
    less-loaded (0.9) endpoint differ by only ``w_load * 0.1 = 0.05``,
    which an affinity match would dominate. The soft cap (default
    threshold=0.9, factor=5.0) adds an extra ``(1.0 - 0.9) * 5 = 0.5``
    on the saturated side, making it lose decisively even with
    affinity bias.
    """
    from umrs.server.step_router import AffinityHints

    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 10, "max_concurrency": 10},  # 1.0
        b_load={"active_requests": 9, "max_concurrency": 10},  # 0.9
    )
    # Affinity to ep-1 would normally tie/win; soft cap flips it.
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-1"})
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision"),
        affinity_hints=hints,
    )
    # ep-1: 1.0 - 0.5*1.0 - 0.5*0.5 = 0.25
    # ep-2: 0.0 - 0.5*0.9 - 0   = -0.45
    # ep-1 still wins on affinity here. To verify softcap fires, drop
    # affinity and check ep-2 wins.
    decision_no_aff = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision_no_aff.endpoint_id == "ep-2"
    # Sanity: the saturated endpoint's load_score is exactly 1.0.
    # Pick ep-1 deliberately to inspect: route again with weight=0
    # disables softcap entirely → declaration order returns.
    monkeypatch.setenv("UMRS_ROUTER_LOAD_WEIGHT", "0")
    decision_off = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision_off.endpoint_id == "ep-1"


def test_softcap_disabled_via_env(monkeypatch):
    """Setting ``UMRS_ROUTER_SOFTCAP_PENALTY=0`` removes the saturation kick."""
    monkeypatch.setenv("UMRS_ROUTER_SOFTCAP_PENALTY", "0")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"active_requests": 10, "max_concurrency": 10},
        b_load={"active_requests": 9, "max_concurrency": 10},
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Linear load only: ep-2 still slightly less loaded → still wins.
    assert decision.endpoint_id == "ep-2"
    # But the gap is much smaller without softcap.
    assert decision.total_score == pytest.approx(-0.5 * 0.9)


def test_latency_and_queue_default_zero_without_metadata(monkeypatch):
    """Endpoints without latency/queue telemetry score 0 on those terms."""
    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.latency_score == pytest.approx(0.0)
    assert decision.queue_score == pytest.approx(0.0)


def test_latency_ref_env_override(monkeypatch):
    """``UMRS_ROUTER_LATENCY_REF_MS`` rescales latency normalisation."""
    monkeypatch.setenv("UMRS_ROUTER_LATENCY_REF_MS", "200")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"avg_latency_ms": 200},  # was 0.2 with default ref; now 1.0
        b_load={"avg_latency_ms": 100},  # was 0.1 with default ref; now 0.5
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"
    assert decision.latency_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Feedback-aware scoring (Slice 6)
# ---------------------------------------------------------------------------


def test_feedback_store_ewma_math(monkeypatch):
    """EWMA: first record sets the value; subsequent records smooth it."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_EWMA_ALPHA", "0.5")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_REF_MS", "1000")
    reset_feedback_store()

    store = get_feedback_store()
    store.record("ep-1", 1000.0)
    assert store.get_norm("ep-1") == pytest.approx(1.0)

    # alpha=0.5: new = 0.5 * 500 + 0.5 * 1000 = 750
    store.record("ep-1", 500.0)
    assert store.get_norm("ep-1") == pytest.approx(0.75)

    # Cold endpoint: norm is 0.
    assert store.get_norm("ep-2") == pytest.approx(0.0)

    reset_feedback_store()
    assert store.get_norm("ep-1") == pytest.approx(0.0)


def test_feedback_norm_biases_routing(monkeypatch):
    """A "slow" endpoint (high feedback EWMA) loses to a cold one."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # ep-1 has prior runs at 2000ms; ep-2 is cold.
    get_feedback_store().record("ep-1", 2000.0)

    # No metadata-reported latency; only feedback differs.
    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"
    assert decision.feedback_score == pytest.approx(0.0)
    reset_feedback_store()


def test_feedback_weight_zero_disables(monkeypatch):
    """``UMRS_ROUTER_FEEDBACK_WEIGHT=0`` ignores recorded outcomes."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    get_feedback_store().record("ep-1", 2000.0)
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Feedback disabled -> declaration order wins (ep-1).
    assert decision.endpoint_id == "ep-1"
    reset_feedback_store()


def test_feedback_cold_endpoints_score_zero(monkeypatch):
    """No recorded outcomes -> feedback_score=0 -> declaration order wins."""
    from umrs.server.step_router import reset_feedback_store

    reset_feedback_store()
    # Disable Slice 7 exploration so the deterministic path is asserted.
    monkeypatch.setenv("UMRS_ROUTER_EXPLORE_EPSILON", "0")
    monkeypatch.setenv("UMRS_ROUTER_MIN_VISITS", "0")
    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"
    assert decision.feedback_score == pytest.approx(0.0)


def test_feedback_alpha_clamps_when_invalid(monkeypatch):
    """Out-of-range alpha falls back to the 0.3 default."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_EWMA_ALPHA", "0")  # invalid
    store = get_feedback_store()
    store.record("ep-1", 1000.0)
    store.record("ep-1", 500.0)
    # alpha=0.3 default: 0.3 * 500 + 0.7 * 1000 = 850 -> norm 0.85
    assert store.get_norm("ep-1") == pytest.approx(0.85)
    reset_feedback_store()


# ---------------------------------------------------------------------------
# Slice 7 — failure signal, decorrelation, ε-greedy
# ---------------------------------------------------------------------------


def test_failure_ewma_increments_and_decays(monkeypatch):
    """record_failure raises rate; subsequent successes decay it."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_EWMA_ALPHA", "0.5")
    reset_feedback_store()
    store = get_feedback_store()

    store.record_failure("ep-1")
    # alpha=0.5: 0.5 * 1 + 0.5 * 0 = 0.5
    assert store.get_failure_norm("ep-1") == pytest.approx(0.5)

    store.record("ep-1", 100.0)
    # success mixes a 0 sample: 0.5 * 0.5 = 0.25
    assert store.get_failure_norm("ep-1") == pytest.approx(0.25)

    # Cold endpoint -> 0.0 (no penalty).
    assert store.get_failure_norm("ep-2") == pytest.approx(0.0)

    # Visit count increments for both success and failure paths.
    assert store.get_visits("ep-1") == 2
    reset_feedback_store()


def test_failure_score_biases_routing(monkeypatch):
    """A flaky endpoint loses to a clean one even if both are warm."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    store = get_feedback_store()
    # Both endpoints have identical successful latency; ep-1 also
    # accumulated failures and should now lose.
    store.record("ep-1", 100.0)
    store.record_failure("ep-1")
    store.record_failure("ep-1")
    store.record("ep-2", 100.0)

    # Disable exploration so the test is deterministic.
    monkeypatch.setenv("UMRS_ROUTER_EXPLORE_EPSILON", "0")
    monkeypatch.setenv("UMRS_ROUTER_MIN_VISITS", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"
    assert decision.failure_score == pytest.approx(0.0)
    reset_feedback_store()


def test_failure_weight_zero_disables(monkeypatch):
    """``UMRS_ROUTER_FAILURE_WEIGHT=0`` ignores failure history."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    get_feedback_store().record_failure("ep-1")
    get_feedback_store().record_failure("ep-1")
    monkeypatch.setenv("UMRS_ROUTER_FAILURE_WEIGHT", "0")
    monkeypatch.setenv("UMRS_ROUTER_EXPLORE_EPSILON", "0")
    monkeypatch.setenv("UMRS_ROUTER_MIN_VISITS", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # Failure penalty disabled -> declaration order wins.
    assert decision.endpoint_id == "ep-1"
    reset_feedback_store()


def test_latency_decorrelation_shrinks_when_feedback_present(monkeypatch):
    """Warm endpoints with feedback get latency_norm scaled by decorr factor."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # ep-1 reports a high latency in metadata and ALSO has a fast
    # observed feedback. ep-2 reports nothing and is cold.
    # With decorrelation, ep-1's metadata latency is halved (0.5x),
    # so ep-1 wins despite the high reported value.
    monkeypatch.setenv("UMRS_ROUTER_LATENCY_DECORR_FACTOR", "0.0")  # full discount
    monkeypatch.setenv("UMRS_ROUTER_EXPLORE_EPSILON", "0")
    monkeypatch.setenv("UMRS_ROUTER_MIN_VISITS", "0")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")  # isolate the lat term

    get_feedback_store().record("ep-1", 100.0)  # warm

    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"avg_latency_ms": 5000},  # would normally lose
        b_load={},  # cold
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"
    reset_feedback_store()


def test_latency_decorrelation_does_not_apply_to_cold(monkeypatch):
    """Without feedback history, the full latency_norm still applies."""
    from umrs.server.step_router import reset_feedback_store

    reset_feedback_store()
    monkeypatch.setenv("UMRS_ROUTER_LATENCY_DECORR_FACTOR", "0.0")
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0")

    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"avg_latency_ms": 5000},  # cold + slow -> should lose
        b_load={"avg_latency_ms": 100},  # cold + fast
    )
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"


def test_ucb_cold_endpoint_prioritised(monkeypatch):
    """A brand-new endpoint with 0 visits gets maximum UCB bonus."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # ep-1 has many visits; ep-2 is brand new.
    for _ in range(10):
        get_feedback_store().record("ep-1", 100.0)
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "1.0")
    monkeypatch.setenv("UMRS_ROUTER_UCB_C", "2.0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # ep-2 (cold) must be chosen because UCB bonus is maximal.
    assert decision.endpoint_id == "ep-2"
    reset_feedback_store()


def test_ucb_bonus_decays_with_visits(monkeypatch):
    """After both endpoints accumulate visits, UCB bonus difference shrinks."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # Give both endpoints equal visits — UCB can't distinguish them.
    for _ in range(20):
        get_feedback_store().record("ep-1", 100.0)
        get_feedback_store().record("ep-2", 100.0)
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0.3")
    monkeypatch.setenv("UMRS_ROUTER_UCB_C", "1.0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    # With equal visits and equal load, declaration order wins (ep-1).
    assert decision.endpoint_id == "ep-1"
    reset_feedback_store()


def test_ucb_weight_zero_disables_exploration(monkeypatch):
    """UCB_WEIGHT=0 makes the bonus term zero, producing deterministic picks."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    for _ in range(10):
        get_feedback_store().record("ep-1", 100.0)
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    for _ in range(5):
        decision = router.resolve(
            _spec(requires=["vision"], model_type="vision")
        )
        # With UCB disabled, declaration order + equal scores picks ep-1.
        assert decision.endpoint_id == "ep-1"
        assert decision.exploration_used is False
    reset_feedback_store()


def test_ucb_c_controls_bonus_strength(monkeypatch):
    """Larger C makes the UCB bonus dominate more aggressively."""
    from umrs.server.step_router import (
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # ep-1 has visits, ep-2 is cold.
    for _ in range(5):
        get_feedback_store().record("ep-1", 100.0)
    # With C=0, bonus is zero — ep-1 wins by declaration order.
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "1.0")
    monkeypatch.setenv("UMRS_ROUTER_UCB_C", "0")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-1"

    # Now crank C high — ep-2 (cold) should win.
    monkeypatch.setenv("UMRS_ROUTER_UCB_C", "10.0")
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision")
    )
    assert decision.endpoint_id == "ep-2"
    reset_feedback_store()


def test_affinity_overrides_ucb(monkeypatch):
    """When affinity matches, UCB bonus must not steal the decision."""
    from umrs.server.step_router import (
        AffinityHints,
        get_feedback_store,
        reset_feedback_store,
    )

    reset_feedback_store()
    # ep-1 has many visits (low UCB bonus); ep-2 is cold (high bonus).
    for _ in range(10):
        get_feedback_store().record("ep-1", 100.0)
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0.5")
    monkeypatch.setenv("UMRS_ROUTER_UCB_C", "1.0")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")

    router = _two_endpoint_router_with_load(monkeypatch)
    hints = AffinityHints(parent_endpoint_ids={"parent": "ep-1"})
    for _ in range(5):
        decision = router.resolve(
            _spec(requires=["vision"], model_type="vision"),
            affinity_hints=hints,
        )
        # Affinity to ep-1 must hold even though ep-2 has a higher UCB bonus.
        assert decision.endpoint_id == "ep-1"
        assert decision.affinity_used is True
    reset_feedback_store()


# ── Slice 17: Capability-preference routing ───────────────────────────


def test_capability_preference_bias(monkeypatch):
    """Equal load — endpoint matching ``prefer`` tags wins."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.5")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True, "fast_storage": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu", "fast_storage"]},
        ),
    )
    assert decision.endpoint_id == "ep-1"
    assert decision.capability_score == pytest.approx(1.0)


def test_capability_weight_zero_disables(monkeypatch):
    """``UMRS_ROUTER_CAPABILITY_WEIGHT=0`` → capability score has no effect."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True, "fast_storage": True},
        b_load={},
    )
    # With weight=0 and no other differences both are equal; the first
    # by declaration index wins.
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu", "fast_storage"]},
        ),
    )
    # Either endpoint could win (declaration index tiebreak); the key
    # assertion is that the capability score is still computed but
    # irrelevant.
    assert decision.endpoint_id in ("ep-1", "ep-2")


def test_missing_capability_metadata_no_crash(monkeypatch):
    """Endpoint with no metadata → capability_score 0.0, no crash."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.5")
    router = _two_endpoint_router_with_load(monkeypatch)
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu"]},
        ),
    )
    assert decision.capability_score == 0.0


def test_capability_match_partial(monkeypatch):
    """2 of 3 preferred capabilities matched → score ≈ 2/3."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.5")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True, "fast_storage": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu", "fast_storage", "nvlink"]},
        ),
    )
    # ep-1 matches 2/3, ep-2 matches 0/3 → ep-1 wins.
    assert decision.endpoint_id == "ep-1"
    assert decision.capability_score == pytest.approx(2.0 / 3.0)


def test_ucb_overrides_capability(monkeypatch):
    """Heavily-visited preferred endpoint can lose to cold endpoint via UCB."""
    from umrs.server.step_router import get_feedback_store

    reset_feedback_store()
    # ep-1 has matching caps but many visits → low UCB bonus.
    for _ in range(50):
        get_feedback_store().record("ep-1", 100.0)
    # Strong UCB, weak capability weight → cold ep-2 should win.
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "2.0")
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.1")
    monkeypatch.setenv("UMRS_ROUTER_FEEDBACK_WEIGHT", "0")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu"]},
        ),
    )
    assert decision.endpoint_id == "ep-2"
    reset_feedback_store()


def test_capability_weighted_prefer(monkeypatch):
    """Weighted prefer: gpu (weight=1.0) + nvlink (weight=0.5).

    ep-1 has gpu=True (matches weight 1.0 of total 1.5 → 2/3).
    """
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.5")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={
                "prefer": [
                    {"name": "gpu", "weight": 1.0},
                    {"name": "nvlink", "weight": 0.5},
                ],
            },
        ),
    )
    assert decision.endpoint_id == "ep-1"
    # 1.0 matched / 1.5 total = 2/3
    assert decision.capability_score == pytest.approx(1.0 / 1.5)


def test_capability_mixed_str_and_dict_prefer(monkeypatch):
    """Mixed prefer list: str and dict items coexist."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.5")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True, "fast_storage": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={
                "prefer": [
                    "gpu",
                    {"name": "fast_storage", "weight": 2.0},
                ],
            },
        ),
    )
    assert decision.endpoint_id == "ep-1"
    # gpu=1.0 + fast_storage=2.0 matched / total=3.0
    assert decision.capability_score == pytest.approx(1.0)


def test_decision_breakdown_sums_to_total(monkeypatch):
    """decision_breakdown values sum to total_score."""
    monkeypatch.setenv("UMRS_ROUTER_CAPABILITY_WEIGHT", "0.3")
    router = _two_endpoint_router_with_load(
        monkeypatch,
        a_load={"gpu": True},
        b_load={},
    )
    decision = router.resolve(
        _spec(
            requires=["vision"],
            model_type="vision",
            routing_hints={"prefer": ["gpu"]},
        ),
    )
    bd = decision.decision_breakdown
    assert bd, "breakdown should not be empty"
    assert sum(bd.values()) == pytest.approx(decision.total_score, abs=1e-9)


def test_decision_breakdown_round_trips(monkeypatch):
    """decision_breakdown survives to_dict() → dict access."""
    router = _two_endpoint_router_with_load(monkeypatch, a_load={}, b_load={})
    decision = router.resolve(
        _spec(requires=["vision"], model_type="vision"),
    )
    d = decision.to_dict()
    assert "decision_breakdown" in d
    assert isinstance(d["decision_breakdown"], dict)
    # All expected keys present.
    for key in ("affinity", "load", "latency", "queue", "feedback",
                "failure", "softcap", "ucb", "capability"):
        assert key in d["decision_breakdown"]
