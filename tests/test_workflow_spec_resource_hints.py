"""Unit tests for the optional routing/resource hint fields on StepSpec.

These guard the back-compat invariants:
  * existing workflow.json without hints round-trips byte-stable
  * hints round-trip when present
  * unknown capability tags warn but don't fail
  * unknown model_type fails validation
"""

from __future__ import annotations

import logging

import pytest

from umrs.pipeline_sdk.workflow import (
    KNOWN_CAPABILITY_TAGS,
    KNOWN_MODEL_TYPES,
    StepSpec,
)


def test_legacy_step_round_trips_without_new_fields():
    """A spec that doesn't declare hints must round-trip without leaking
    default-empty routing fields into the JSON shape."""

    raw = {
        "name": "embed",
        "entrypoint": "umrs.pipeline_sdk.templates.llm_step:LLMStep",
        "config": {"model": "rivault-llm-static"},
    }
    spec = StepSpec.from_dict(raw)
    spec.validate()

    out = spec.to_dict()
    # No hint keys must appear when defaults are unchanged.
    assert "requires" not in out
    assert "model" not in out
    assert "lora" not in out
    assert "model_type" not in out
    assert "routing_hints" not in out

    # And the original keys are preserved exactly.
    assert out == {
        "name": "embed",
        "entrypoint": "umrs.pipeline_sdk.templates.llm_step:LLMStep",
        "config": {"model": "rivault-llm-static"},
    }


def test_step_with_hints_round_trips():
    raw = {
        "name": "vision_dense",
        "entrypoint": "umrs.pipeline_sdk.templates.vision_step:VisionStep",
        "config": {"mode": "dense"},
        "requires": ["vision", "gpu"],
        "model": "vlm-large",
        "lora": "planner_v1",
        "model_type": "vision",
        "routing_hints": {"latency": "high"},
    }
    spec = StepSpec.from_dict(raw)
    spec.validate()
    assert spec.requires == ["vision", "gpu"]
    assert spec.model == "vlm-large"
    assert spec.lora == "planner_v1"
    assert spec.model_type == "vision"
    assert spec.routing_hints == {"latency": "high"}

    out = spec.to_dict()
    assert out["requires"] == ["vision", "gpu"]
    assert out["model"] == "vlm-large"
    assert out["lora"] == "planner_v1"
    assert out["model_type"] == "vision"
    assert out["routing_hints"] == {"latency": "high"}


def test_unknown_capability_tag_warns_but_validates(caplog):
    spec = StepSpec(
        name="exotic",
        entrypoint="pkg.mod:Cls",
        requires=["vision", "fancy_new_thing"],
    )
    with caplog.at_level(logging.WARNING, logger="umrs.pipeline_sdk.workflow"):
        spec.validate()  # must NOT raise
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "fancy_new_thing" in msgs


def test_invalid_requires_shape_raises():
    spec = StepSpec(
        name="bad",
        entrypoint="pkg.mod:Cls",
        requires=["", "  vision  "],  # empty string is rejected
    )
    with pytest.raises(ValueError, match="requires must be a list"):
        spec.validate()


def test_invalid_model_type_raises():
    spec = StepSpec(
        name="bad",
        entrypoint="pkg.mod:Cls",
        model_type="speech",
    )
    with pytest.raises(ValueError, match="model_type must be one of"):
        spec.validate()


def test_known_model_type_validates():
    for mt in sorted(KNOWN_MODEL_TYPES):
        spec = StepSpec(
            name=f"step_{mt}",
            entrypoint="pkg.mod:Cls",
            model_type=mt,
        )
        spec.validate()


def test_known_capability_tags_present():
    # Sanity: at least the tags advertised by the plan are present so that
    # the reference workflow doesn't trigger the unknown-tag warning.
    for tag in ("vision", "tool_calling", "structured_output", "gpu"):
        assert tag in KNOWN_CAPABILITY_TAGS
