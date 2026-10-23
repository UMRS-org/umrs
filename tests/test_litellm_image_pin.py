"""Which LiteLLM image a serving job runs, and how long it may live (2026-10-23).

serve-model.yaml ran ghcr.io/berriai/litellm-non_root:main-stable, a floating
tag. The first fix pinned every venue to the digest the bdrt nodes ran; the
review (F4) found it verified on bdrt only, while all four fronts deploy the
same pin and nothing stages the image, so a GPU node without that digest or
registry access would not start LiteLLM. Now the floating tag stays the
default and a venue pins a digest in its site.yaml partition profile
(``litellm_image``), which the wrapper receives as LITELLM_IMAGE; bdrt does.

Review F1: with one uvicorn worker, MAX_REQUESTS_BEFORE_RESTART exits the
whole proxy after 10,000 requests. The wrapper now leaves it empty (no limit)
for one worker and keeps 10000 for several.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml
from umrs.orchestrator.site_config import load_site_config

UMRS = Path(__file__).resolve().parents[1]
MINI_RIVAULT = UMRS / "src" / "umrs" / "mini-rivault-llm"
SERVE_MODEL = MINI_RIVAULT / "serve-model.yaml"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
REPO = "ghcr.io/berriai/litellm-non_root"
FLOATING = f"{REPO}:main-stable"
DIGEST_REF = re.compile(r"^ghcr\.io/berriai/litellm-non_root@sha256:[0-9a-f]{64}$")
PINNED_VENUES = {"bdrt", "bdrt-front"}


def _resolution_snippet() -> str:
    text = SERVE_MODEL_SLURM.read_text()
    start = text.index('if [[ -z "${LITELLM_IMAGE:-}" ]]; then')
    end = text.index("\n# Owner-only from the start", start)
    return text[start:end] + "\n"


def _resolve(**env: str) -> tuple[str, str]:
    merged = {k: v for k, v in os.environ.items() if not k.startswith("LITELLM_")}
    merged.update(env)
    out = subprocess.run(
        [
            "bash",
            "-c",
            _resolution_snippet()
            + 'printf "%s|%s" "$LITELLM_IMAGE" "$LITELLM_MAX_REQUESTS_BEFORE_RESTART"',
        ],
        env=merged,
        capture_output=True,
        text=True,
        check=True,
    )
    image, max_requests = out.stdout.split("|")
    return image, max_requests


def _litellm_service() -> dict:
    return yaml.safe_load(SERVE_MODEL.read_text())["services"]["litellm"]


def test_the_compose_default_is_the_floating_tag() -> None:
    assert _litellm_service()["image"] == f"${{LITELLM_IMAGE:-{FLOATING}}}"


def test_the_wrapper_defaults_to_the_floating_tag() -> None:
    assert _resolve()[0] == FLOATING


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"LITELLM_DOCKER_TAG": "v1.80.0"}, f"{REPO}:v1.80.0"),
        ({"LITELLM_IMAGE": "localhost/litellm:test"}, "localhost/litellm:test"),
        (
            {"LITELLM_IMAGE": "localhost/litellm:test", "LITELLM_DOCKER_TAG": "x"},
            "localhost/litellm:test",
        ),
    ],
)
def test_the_overrides_win(env: dict[str, str], expected: str) -> None:
    assert _resolve(**env)[0] == expected


@pytest.mark.parametrize("partition", ["spark", "edge-double"])
def test_bdrt_pins_the_digest_through_its_partition_profiles(partition: str) -> None:
    cfg = load_site_config(str(UMRS / "deploy" / "bdrt" / "site.yaml"))
    assert DIGEST_REF.match(cfg.partition_env(partition)["LITELLM_IMAGE"])


def test_no_other_venue_pins_litellm() -> None:
    """A digest is an opt-in per venue, verified on that venue's GPU nodes."""
    for site in sorted((UMRS / "deploy").glob("*/site.yaml")):
        if site.parent.name in PINNED_VENUES:
            continue
        profiles = (yaml.safe_load(site.read_text()) or {}).get("partition_profiles") or {}
        pinned = [name for name, p in profiles.items() if (p or {}).get("litellm_image")]
        assert pinned == [], f"{site}: {pinned}"


def test_an_unpinned_partition_exports_no_litellm_image() -> None:
    cfg = load_site_config(str(UMRS / "deploy" / "bdrt" / "site.yaml"))
    assert "LITELLM_IMAGE" not in cfg.partition_env("a100")


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, ""),
        ({"LITELLM_CPU_LIMIT": "1"}, ""),
        ({"LITELLM_CPU_LIMIT": "4"}, "10000"),
        ({"LITELLM_CPU_LIMIT": "1", "LITELLM_MAX_REQUESTS_BEFORE_RESTART": "500"}, "500"),
    ],
)
def test_one_worker_never_exits_on_its_request_count(env: dict[str, str], expected: str) -> None:
    assert _resolve(**env)[1] == expected


def test_compose_reads_the_request_limit_and_leaves_it_empty_by_default() -> None:
    env = _litellm_service()["environment"]
    assert env["MAX_REQUESTS_BEFORE_RESTART"] == "${LITELLM_MAX_REQUESTS_BEFORE_RESTART:-}"


def test_compose_and_the_migration_step_read_the_resolved_values() -> None:
    text = SERVE_MODEL_SLURM.read_text()
    assert "\nLITELLM_IMAGE=${LITELLM_IMAGE}\n" in text  # into env.<uuid>
    assert "\nLITELLM_MAX_REQUESTS_BEFORE_RESTART=${LITELLM_MAX_REQUESTS_BEFORE_RESTART}\n" in text
    assert 'local litellm_image="${LITELLM_IMAGE}"' in text
