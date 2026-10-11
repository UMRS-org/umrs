"""Ask-the-scientist policy vocabulary (ADR-066 + amendment 3, gap row D-33).

- which trigger classes an engine action carries (a set: a physical workflow
  trigger is both ``cost`` and ``physical_action``). The class names and
  the kind → class table are ``r4_contracts.ask_policy``'s, shared with r4's
  local loop; a team-plan run, ``plan_run``, is ``cost`` there (ADR-092
  slice 6, registered before any loop runs a plan), so an ask-mode loop
  parks at it and a profile naming ``cost`` does too;
- the maturity-stage presets of the persisted per-project policy, and the
  reverse lookup that names the stage a stored policy corresponds to.

The stored mode + trigger list stay the single truth; a stage is only a name
for a preset (never derived from catalog metrics).
"""

from __future__ import annotations

from typing import Any

from r4_contracts.ask_policy import (
    ACTION_KIND_CLASSES,
    COMPETING,
    COST,
    DIRECTION,
    PHYSICAL,
    PLAN_RUN,
    TRIGGER_CLASSES,
)

# The kinds the shared table classes ``cost`` (a plan's own gates are not
# actions: they always ask, ADR-092 slice 2, whatever the loop's mode).
COST_ACTION_KINDS = tuple(k for k, c in ACTION_KIND_CLASSES.items() if COST in c)

# Maturity stage → (ask_mode, pause_triggers). physical_action is kept on
# every profile preset: hardware actions keep asking even when autonomous.
STAGE_PRESETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "bootstrapped": ("ask", ()),
    "learning": ("profile", (COST, COMPETING, DIRECTION, PHYSICAL)),
    "competent": ("profile", (COMPETING, DIRECTION, PHYSICAL)),
    "autonomous": ("profile", (PHYSICAL,)),
}
CUSTOM_STAGE = "custom"


def action_trigger_classes(action: dict[str, Any]) -> frozenset[str]:
    """The trigger classes one engine action carries (unknown kinds: none —
    a profile-mode loop never silently parks on them)."""
    kind = str(action.get("kind") or "")
    classes = ACTION_KIND_CLASSES.get(kind)
    if classes:
        return classes | {PHYSICAL} if action.get("physical") else classes
    if kind in (COMPETING, DIRECTION):
        return frozenset({kind})
    return frozenset()


def stage_preset(stage: str) -> tuple[str, list[str]]:
    """(ask_mode, pause_triggers) of one named stage; KeyError names the stage."""
    if stage not in STAGE_PRESETS:
        raise KeyError(
            f"unknown maturity stage {stage!r} (known: {sorted(STAGE_PRESETS)})"
        )
    mode, triggers = STAGE_PRESETS[stage]
    return mode, list(triggers)


def stage_for(ask_mode: str, pause_triggers: list[str]) -> str:
    """The stage whose preset the stored policy equals; ``custom`` otherwise
    ('' when no policy is stored)."""
    if not ask_mode:
        return ""
    wanted = (ask_mode, frozenset(pause_triggers if ask_mode == "profile" else ()))
    for stage, (mode, triggers) in STAGE_PRESETS.items():
        if (mode, frozenset(triggers)) == wanted:
            return stage
    return CUSTOM_STAGE


__all__ = [
    "COMPETING",
    "COST",
    "COST_ACTION_KINDS",
    "CUSTOM_STAGE",
    "DIRECTION",
    "PHYSICAL",
    "PLAN_RUN",
    "STAGE_PRESETS",
    "TRIGGER_CLASSES",
    "action_trigger_classes",
    "stage_for",
    "stage_preset",
]
