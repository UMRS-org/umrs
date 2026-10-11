"""The retrieval profile UMRS reads with (ADR-093 D-14, component C18).

One design, sized by the model that reads the passages. The stand-alone
reference reader is ``llama3.2:3b`` (the ``small`` profile); the rikyu
deployment reads with ``kimi-k3`` and ``glm-5.3``, frontier MoE models with
1M-token windows (the ``large`` profile). The size rule and the profile table
live in ``r4_contracts.models`` so R4 and UMRS cannot drift apart.

Which profile:

1. a request's ``profile`` (``small|medium|large``) names it outright;
2. else ``UMRS_RAG_PROFILE`` (``auto|small|medium|large``, default ``auto``);
3. else (``auto``) no reading model named at all (an older R4, a client that
   sends none) reads with the site's own defaults, the ``small`` profile,
   exactly as before D-14;
4. else the ONE size rule over the reading model's name
   (``r4_contracts.models.profile_for``, which R4 applies too): the class of
   the EFFECTIVE size, sqrt(total x active) for a mixture-of-experts model,
   so ``qwen3-30b-a3b`` and ``gpt-oss-20b`` read with ``medium``; a named
   model of unknown scale reads with ``medium`` (lead decision for the
   maintainer, 2026-10-10: one rule, and unknown -> medium everywhere).

What a profile changes, and what wins over it:

- explicit request parameters always win;
- ``small`` reads the site's env defaults (``UMRS_RAG_FINAL_K_MAX``,
  ``UMRS_RAG_POOL``, ``UMRS_RAG_ANSWER_MAX_TOKENS``) exactly as before D-14,
  so the stand-alone stack behaves as it did;
- a larger profile takes its own value, unless the site's env default is
  larger still (a site never gets less than it configured);
- ``merge``: ``gate`` defers to the site's ``UMRS_RAG_MERGE`` (the S2 gate),
  ``on`` turns the neighbour merge on unless the request says ``off``;
- the answer limit: ``None`` sends no explicit limit (the provider's family
  default, ADR-087).

Relevance, the cut and sufficiency stay where they are (D-3): a profile only
sizes them. Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import os

from r4_contracts.models import RETRIEVAL_PROFILES, RetrievalProfile, profile_for

PROFILE_ENV = "UMRS_RAG_PROFILE"
#: The profile whose numbers are the site's env defaults (``UMRS_RAG_FINAL_K_MAX``
#: etc.), and what a request that names no reading model reads with.
SITE_PROFILE = "small"
#: ``auto`` resolves from the reading model; the others name a profile.
PROFILE_CHOICES: tuple[str, ...] = ("auto", *RETRIEVAL_PROFILES)


def _choice(value: str | None) -> str:
    """A profile name, or ``""`` for ``auto``/unset/unrecognised."""
    name = (value or "").strip().lower()
    return name if name in RETRIEVAL_PROFILES else ""


def site_profile() -> str:
    """``UMRS_RAG_PROFILE``: a profile name, or ``auto`` (also for a bad value)."""
    return _choice(os.environ.get(PROFILE_ENV, "auto")) or "auto"


def resolve_profile(
    reader_model: str | None, override: str | None = None
) -> tuple[str, RetrievalProfile]:
    """``(name, profile)`` for the model that will read the passages.

    ``override`` is the request's ``profile``; it wins over
    ``UMRS_RAG_PROFILE``, which wins over the size rule (``auto``). No
    reading model named reads with :data:`SITE_PROFILE`; a named one goes
    through the shared rule, so R4 and UMRS size one reader alike.
    """
    for choice in (override, site_profile()):
        name = _choice(choice)
        if name:
            return name, RETRIEVAL_PROFILES[name]
    if not (reader_model or "").strip():
        return SITE_PROFILE, RETRIEVAL_PROFILES[SITE_PROFILE]
    return profile_for(reader_model)


def sized(name: str, site: int, wanted: int) -> int:
    """A count default: the site's own for ``small``; else the profile's,
    never below what the site configured."""
    return site if name == SITE_PROFILE else max(site, wanted)


def merge_mode(requested: str | None, profile: RetrievalProfile) -> str:
    """The spec's ``merge``: an explicit ``on``/``off`` wins; ``auto`` takes
    the profile's default (``gate`` = the site's ``UMRS_RAG_MERGE``)."""
    asked = (requested or "auto").strip().lower()
    if asked in ("on", "off"):
        return asked
    return "on" if profile.merge == "on" else "auto"


def lexical_mode(requested: str | None, profile: RetrievalProfile, name: str) -> str:
    """The spec's ``lexical``: an explicit ``on``/``off`` wins; ``auto`` takes
    the profile's default (``gate`` = the site's ``UMRS_RAG_LEXICAL``).

    The site profile (``small``) reads the site's ``UMRS_RAG_LEXICAL`` like
    every other small-profile value. ADR-093 Phase 9 (2026-10-11) measured
    the answer path with lexical inclusion off for the small reader: it cited
    the evidence document 15 points less often, abstained wrongly more, and
    let a nonsense question through, so the 2026-10-10 ``off`` is reverted
    whatever the shared table still says."""
    asked = (requested or "auto").strip().lower()
    if asked in ("on", "off"):
        return asked
    if name == SITE_PROFILE:
        return "auto"
    return "off" if profile.lexical == "off" else "auto"


def answer_limit(name: str, profile: RetrievalProfile, site: int) -> int | None:
    """``max_tokens`` for compose: the site's for ``small``; ``None`` (no
    explicit limit) when the profile defers to the family default."""
    if name == SITE_PROFILE:
        return site
    if profile.answer_max_tokens is None:
        return None
    return max(site, profile.answer_max_tokens)


def report(name: str, reader_model: str | None) -> dict[str, str]:
    """``lanes.profile``: which profile sized this call, read by which model."""
    return {"status": "ok", "name": name, "reader_model": reader_model or ""}


__all__ = [
    "PROFILE_CHOICES",
    "PROFILE_ENV",
    "SITE_PROFILE",
    "answer_limit",
    "lexical_mode",
    "merge_mode",
    "report",
    "resolve_profile",
    "site_profile",
    "sized",
]
