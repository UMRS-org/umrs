"""Model-size rule and retrieval profiles (ADR-093 D-8, D-14).

Stdlib only (ADR-005). One size rule serves both repositories: R4 sizes its
pre-retrieval pack from the session model, and UMRS sizes its cut, pool,
merge and answer limit from the registry-resolved answer model. Both read
the model's *name* through the same functions here, so they agree without a
registry round trip.

The rule (lead decision for the maintainer, 2026-10-10: ONE size rule
everywhere, the EFFECTIVE size):

- the class is read from the effective size: sqrt(total x active) for a
  mixture-of-experts model when both sizes are known (an ``-a<n>b`` tag, or
  the published-size table :data:`KNOWN_MODEL_FAMILIES`), else the total. A
  30B-A3B reads like ~9.5B and is medium, as is ``gpt-oss-20b``; kimi-k2
  (1T total, 32B active) is large;
- a parameter tag in the name gives the total: ``llama3.2:3b`` is small,
  ``qwen3.8-27b`` is large, ``mixtral-8x7b`` counts 56B when no published
  size is known;
- hosted families that publish no size are large: kimi, glm-5, deepseek-v/r1,
  gpt-4/5, claude, gemini;
- anything else is ``unknown``, and reads with the MEDIUM profile (C22): a
  balanced pack, neither starved nor flooded. R4 adds one notice per session
  saying how to set the scale (``r4 model scale set``).

Thresholds: small <= 4B, large > 14B. :func:`model_size_class`,
:func:`scale_from_name` and :func:`profile_for` all apply this one rule, and
``r4.model_context`` re-exports it unchanged.

The profile table is the D-14 table. The stand-alone stack's reader
(``llama3.2:3b``) reads with the small profile; the remote deployment's
readers (rikyu ``kimi-k3`` and ``glm-5.3``, frontier MoE models with 1M-token
windows) read with the large one. A profile carries numbers only: relevance,
the cut and sufficiency stay UMRS's (CONSTITUTION §1.3).

C22 (2026-10-10) adds the shared half of model-scale *detection*
(``r4.model_scale``): :func:`effective_params_b`, the published-size table
:data:`KNOWN_MODEL_FAMILIES`, :func:`scale_from_name` and
:func:`profile_for_class`, which R4 calls with a class its probes detected.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# --- The size rule (ADR-093 D-8) ---------------------------------------------

SIZE_SMALL_MAX_B = 4.0  # small: <= 4B parameters
SIZE_LARGE_MIN_B = 14.0  # large: > 14B parameters
_PARAM_COUNT_RE = re.compile(r"(?<![a-z0-9.])(?:(\d+)x)?(\d+(?:\.\d+)?)([bm])(?![a-z0-9])")
_LARGE_UNSIZED_FAMILIES = (
    "claude",
    "gpt-4",
    "gpt-5",
    "gemini",
    "kimi",
    "deepseek-v",
    "deepseek-r1",
    "glm-5",
    "glm-4.5",
    "glm-4.6",
    "mistral-large",
)


def model_param_billions(*names: str | None) -> float | None:
    """The parameter count (billions) a model name states, or ``None``.

    The largest ``<n>b`` tag across the names wins (a ``-1m`` context suffix
    never shrinks a 7B model); ``<n>m`` counts only when no ``b`` tag exists.
    """
    tags = _param_tags(" ".join(name for name in names if name))
    for unit, scale in (("b", 1.0), ("m", 0.001)):
        values = [value for value, tag_unit in tags if tag_unit == unit]
        if values:
            return max(values) * scale
    return None


def _param_tags(name: str) -> list[tuple[float, str]]:
    """``(count, unit)`` for every size tag in ``name`` (``8x7b`` → 56, ``b``)."""
    return [
        (float(count) * (int(experts) if experts else 1), unit)
        for experts, count, unit in _PARAM_COUNT_RE.findall(name.lower())
    ]


def _large_unsized(names: tuple[str | None, ...]) -> bool:
    haystacks = [n.lower() for n in names if n]
    return any(fam in h for h in haystacks for fam in _LARGE_UNSIZED_FAMILIES)


def model_size_class(*names: str | None) -> str:
    """``small`` (<= 4B), ``medium``, ``large`` (> 14B) or ``unknown``.

    The one size rule: the class of the EFFECTIVE size the names state
    (:func:`scale_from_name`), so ``qwen3-30b-a3b`` and ``gpt-oss-20b`` are
    medium (lead decision, 2026-10-10).
    """
    return scale_from_name(*names).size_class


# --- The retrieval profiles (ADR-093 D-14) -----------------------------------


@dataclass(frozen=True)
class RetrievalProfile:
    """How much retrieved evidence one reader gets, sized by the reading model.

    UMRS reads ``k_max``, ``partial_k``, ``pool``, ``merge``,
    ``answer_max_tokens`` and ``lexical``; R4 reads ``pack_passages``,
    ``pack_tokens`` and ``expand``. Explicit request parameters and env overrides still win over
    these defaults; the profile only replaces hard-wired small-model limits.
    """

    k_max: int  # most passages the UMRS cut keeps
    partial_k: int  # passages kept when sufficiency is only partial
    pool: int  # candidate pool before the cut
    merge: str  # neighbour merge (D-5): "gate" = the site's UMRS_RAG_MERGE, or "on"
    answer_max_tokens: int | None  # compose output limit; None = no explicit limit
    pack_passages: int  # R4 pre-retrieval pack: passages
    pack_tokens: int  # R4 pack token cap (the caller still holds it to 12% of the window)
    expand: str  # LLM query expansion: "off", or "multi" (two variants, fused in UMRS)
    # Lexical lanes in the reader's cut: "gate" = the site's UMRS_RAG_LEXICAL,
    # or "off". Every profile reads "gate" (ADR-093 Phase 9, 2026-10-11): the
    # small reader's "off" of 2026-10-10 cost the answer path 15 points of
    # citing the evidence and let a nonsense question through.
    lexical: str


#: The profile a model of unknown scale reads with (C22; lead decision
#: 2026-10-10, "unknown -> medium everywhere"): balanced, neither the 3B pack
#: that starves a large reader nor the frontier pack that sinks a 3B.
UNKNOWN_SCALE_PROFILE = "medium"
#: What :func:`profile_for` falls back to: the unknown-scale profile.
DEFAULT_PROFILE = UNKNOWN_SCALE_PROFILE

#: The values a profile override (``UMRS_RAG_PROFILE``, ``R4_RAG_PROFILE``, the
#: retrieve verb's ``profile``) may take; ``auto`` resolves from the model name.
PROFILE_OVERRIDES: tuple[str, ...] = ("auto", "small", "medium", "large")

RETRIEVAL_PROFILES: dict[str, RetrievalProfile] = {
    "small": RetrievalProfile(
        k_max=4,
        partial_k=2,
        pool=50,
        merge="gate",
        answer_max_tokens=400,
        pack_passages=3,
        pack_tokens=1_200,
        expand="off",
        lexical="gate",
    ),
    "medium": RetrievalProfile(
        k_max=6,
        partial_k=3,
        pool=50,
        merge="on",
        answer_max_tokens=600,
        pack_passages=4,
        pack_tokens=2_000,
        expand="off",
        lexical="gate",
    ),
    # ADR-093 MD-14 (eval Phase 19, 2026-10-13): read by kimi-k3 and glm-5.3,
    # the D-14 sizes (10 passages, 8,000 tokens, pool 100) were not shown
    # non-inferior to the small pack for kimi-k3 (-1.0 [-4.7, +2.7]) and raised
    # verdicts on NEI claims, so the pre-registered fallback applies: the large
    # profile retrieves and packs at the medium sizes. Its output limit, query
    # expansion and graph lane stay large-only.
    "large": RetrievalProfile(
        k_max=6,
        partial_k=3,
        pool=50,
        merge="on",
        answer_max_tokens=None,
        pack_passages=4,
        pack_tokens=2_000,
        expand="multi",
        lexical="gate",
    ),
}


def profile_for(
    model_name: str | None, override: str | None = None
) -> tuple[str, RetrievalProfile]:
    """The profile ``(name, profile)`` for the model that will read the passages.

    ``override`` names a profile outright (case and surrounding space ignored);
    ``None``, ``""``, ``auto`` or an unrecognised value fall through to the
    size rule (:func:`model_size_class`, the effective size). A model whose
    scale is ``unknown`` (or no model at all) reads with
    :data:`UNKNOWN_SCALE_PROFILE` (medium), exactly as a detected ``unknown``
    does in :func:`profile_for_class`. A caller with a site default of its
    own for "no reader named" (UMRS: the site's env defaults) applies it
    before calling this.
    """
    return profile_for_class(model_size_class(model_name), override)


# --- Detected model scale (ADR-093 D-14, C22) --------------------------------
#
# R4 detects the scale of a model registered through any source (ollama,
# a platform registry, rikyu or any other OpenAI-compatible endpoint). The
# helpers below are the shared part of that rule and the name rule itself
# (model_size_class above delegates here), so UMRS sizes a model the same
# way: the size class comes from the EFFECTIVE size, which for a
# mixture-of-experts model is sqrt(total x active) (a 30B-A3B reads like a
# ~9.5B model: medium; kimi-k2 at 1T/32B is large).

_ACTIVE_PARAM_RE = re.compile(r"(?<![a-z0-9.])a(\d+(?:\.\d+)?)b(?![a-z0-9])")
_NAME_SEPARATORS_RE = re.compile(r"[\s_:]+")
#: A size a name states within this fraction of a published size is the same model.
SIZE_AGREEMENT = 0.25


@dataclass(frozen=True)
class ModelFamily:
    """Published sizes (billions of parameters) of one model family.

    ``total_b`` ``None`` means the family publishes no size and is large
    (hosted frontier, or a release whose size is not public). ``active_b`` is
    the per-token active size of a mixture-of-experts model.
    """

    patterns: tuple[str, ...]
    total_b: float | None
    active_b: float | None = None


#: First match wins, so specific patterns precede general ones. Names are
#: matched lower-cased, with ``_``, ``:`` and spaces read as ``-``
#: (``llama4:scout`` matches ``llama4-scout``, ollama's ``qwen3:30b`` matches
#: ``qwen3-30b``). A size tag in the name that agrees with neither published
#: size (a distill such as ``deepseek-r1-distill-qwen-7b``) wins over the
#: family entry, and a name that states both sizes (``qwen3-30b-a3b``) is
#: read as stated.
#:
#: Ollama names mixture-of-experts models by their TOTAL size with no
#: ``-a<n>b`` tag (``qwen3:30b`` is Qwen3-30B-A3B, ``deepseek-v2:16b`` is
#: DeepSeek-V2-Lite, ``mixtral`` is 8x7B), and UMRS sizes a model by its
#: registry name alone, so those tags carry their active size here (review
#: of ADR-093, 2026-10-10: ``qwen3:30b`` was classed large and turned the
#: graph lane on).
KNOWN_MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(("kimi-k3",), None),  # rikyu reader; size not public
    ModelFamily(("kimi-k2",), 1_000.0, 32.0),  # Moonshot: 1T total, 32B active
    ModelFamily(("glm-5",), None),  # glm-5, glm-5.x (rikyu glm-5.3); size not public
    ModelFamily(("glm-4.5-air",), 106.0, 12.0),
    ModelFamily(("glm-4.5", "glm-4.6"), 355.0, 32.0),
    ModelFamily(("deepseek-v3", "deepseek-r1"), 671.0, 37.0),
    # DeepSeek-V2-Lite and Coder-V2-Lite (ollama deepseek-v2:16b, deepseek-coder-v2:16b).
    ModelFamily(("deepseek-v2-16b", "deepseek-v2-lite"), 15.7, 2.4),
    ModelFamily(("deepseek-coder-v2-16b", "deepseek-coder-v2-lite"), 15.7, 2.4),
    ModelFamily(("qwen3-235b",), 235.0, 22.0),  # Qwen3-235B-A22B (ollama qwen3:235b)
    # Qwen3-30B-A3B and Qwen3-Coder-30B-A3B (ollama qwen3:30b, qwen3-coder:30b).
    ModelFamily(("qwen3-30b", "qwen3-coder-30b"), 30.5, 3.3),
    ModelFamily(("qwen3-vl-30b",), 30.0, 3.0),  # Qwen3-VL-30B-A3B (ollama qwen3-vl:30b)
    ModelFamily(("llama-4-maverick", "llama4-maverick"), 400.0, 17.0),
    ModelFamily(("llama-4-scout", "llama4-scout"), 109.0, 17.0),
    ModelFamily(("gpt-oss-120b",), 117.0, 5.1),
    ModelFamily(("gpt-oss-20b",), 21.0, 3.6),
    ModelFamily(("mixtral-8x22b",), 141.0, 39.0),
    ModelFamily(("mixtral-8x7b",), 46.7, 12.9),
    ModelFamily(("mixtral",), 46.7, 12.9),  # ollama's untagged mixtral is 8x7B
)


@dataclass(frozen=True)
class NameScale:
    """What the names alone say about a model's scale.

    ``basis`` is ``family`` (the known-family table, or a large family that
    publishes no size), ``name`` (a parameter tag) or ``unknown``.
    """

    size_class: str
    total_b: float | None = None
    active_b: float | None = None
    effective_b: float | None = None
    basis: str = "unknown"


def _normalized_names(names: tuple[str | None, ...]) -> str:
    return " ".join(_NAME_SEPARATORS_RE.sub("-", n.lower()) for n in names if n)


def active_param_billions(*names: str | None) -> float | None:
    """The active-expert size an ``-a<n>b`` tag states (``qwen3-30b-a3b`` → 3.0)."""
    values = [float(v) for v in _ACTIVE_PARAM_RE.findall(_normalized_names(names))]
    return max(values) if values else None


def effective_params_b(total: float | None, active: float | None = None) -> float | None:
    """The size a model reads like: sqrt(total x active) for MoE, else the total.

    ``active`` counts only when it is positive and below ``total``; ``None``
    when the total is unknown.
    """
    if total is None or total <= 0:
        return None
    if active is not None and 0 < active < total:
        return math.sqrt(total * active)
    return float(total)


def size_class_for(params_b: float | None) -> str:
    """``small`` (<= 4B), ``medium``, ``large`` (> 14B) or ``unknown`` for a size."""
    if params_b is None:
        return "unknown"
    if params_b <= SIZE_SMALL_MAX_B:
        return "small"
    return "large" if params_b > SIZE_LARGE_MIN_B else "medium"


def known_model_family(*names: str | None) -> ModelFamily | None:
    """The first :data:`KNOWN_MODEL_FAMILIES` entry the names match, or ``None``."""
    haystack = _normalized_names(names)
    return next(
        (fam for fam in KNOWN_MODEL_FAMILIES if any(p in haystack for p in fam.patterns)),
        None,
    )


def _agrees_with(stated: float | None, family: ModelFamily) -> bool:
    if stated is None:
        return True
    published = [size for size in (family.total_b, family.active_b) if size]
    return any(abs(stated - size) <= SIZE_AGREEMENT * size for size in published)


def _sized(total: float, active: float | None, basis: str) -> NameScale:
    effective = effective_params_b(total, active)
    moe_active = active if active is not None and 0 < active < total else None
    return NameScale(size_class_for(effective), total, moe_active, effective, basis)


def scale_from_name(*names: str | None) -> NameScale:
    """Total, active and effective size, and the class, the names state (C22).

    A name that states both its total and its active size (``qwen3-30b-a3b``)
    is read as stated; else the known-family table (unless a stated size
    disagrees with it), then a parameter tag, then the large families that
    publish no size; anything else is ``unknown``.
    """
    stated = model_param_billions(*names)
    active = active_param_billions(*names)
    if stated is not None and active is not None:
        return _sized(stated, active, "name")
    family = known_model_family(*names)
    if family is not None and _agrees_with(stated, family):
        if family.total_b is None:
            return NameScale("large", basis="family")
        return _sized(family.total_b, family.active_b, "family")
    if stated is not None:
        return _sized(stated, None, "name")
    if _large_unsized(names):
        return NameScale("large", basis="family")
    return NameScale("unknown")


def profile_for_class(
    size_class: str | None, override: str | None = None
) -> tuple[str, RetrievalProfile]:
    """The profile ``(name, profile)`` for a size class (C22).

    ``override`` names a profile outright, as in :func:`profile_for`. A
    class that is not a profile (``unknown``, ``None``) reads with
    :data:`UNKNOWN_SCALE_PROFILE` (medium). R4 calls this with the class its
    probes detected; :func:`profile_for` calls it with the name rule's.
    """
    for candidate in (override, size_class):
        name = (candidate or "").strip().lower()
        if name in RETRIEVAL_PROFILES:
            return name, RETRIEVAL_PROFILES[name]
    return UNKNOWN_SCALE_PROFILE, RETRIEVAL_PROFILES[UNKNOWN_SCALE_PROFILE]
