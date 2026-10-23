"""LiteLLM's default image is pinned by digest (2026-10-23).

serve-model.yaml ran ghcr.io/berriai/litellm-non_root:main-stable, a floating
tag: two nodes could run two different proxies under one name and a re-pull
could change the proxy under a running fleet. The default is now the digest
of the multi-arch index the bdrt nodes ran on 2026-10-22 (read with podman
image inspect inside job 8218). LITELLM_IMAGE and LITELLM_DOCKER_TAG still
override it.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

MINI_RIVAULT = Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm"
SERVE_MODEL = MINI_RIVAULT / "serve-model.yaml"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
REPO = "ghcr.io/berriai/litellm-non_root"
DIGEST_REF = re.compile(r"^ghcr\.io/berriai/litellm-non_root@sha256:[0-9a-f]{64}$")


def _resolution_snippet() -> str:
    text = SERVE_MODEL_SLURM.read_text()
    start = text.index("LITELLM_PINNED_IMAGE=")
    end_marker = 'LITELLM_IMAGE="${LITELLM_IMAGE:-${LITELLM_PINNED_IMAGE}}"\n'
    return text[start : text.index(end_marker) + len(end_marker)]


def _resolve(**env: str) -> str:
    merged = {k: v for k, v in os.environ.items() if not k.startswith("LITELLM_")}
    merged.update(env)
    out = subprocess.run(
        ["bash", "-c", _resolution_snippet() + 'printf "%s" "$LITELLM_IMAGE"'],
        env=merged,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def _compose_default() -> str:
    image = yaml.safe_load(SERVE_MODEL.read_text())["services"]["litellm"]["image"]
    match = re.fullmatch(r"\$\{LITELLM_IMAGE:-(.+)\}", image)
    assert match, image
    return match.group(1)


def test_the_compose_default_is_a_digest_not_a_tag() -> None:
    assert DIGEST_REF.match(_compose_default())


def test_the_wrapper_defaults_to_the_same_digest_as_compose() -> None:
    assert _resolve() == _compose_default()


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"LITELLM_DOCKER_TAG": "main-stable"}, f"{REPO}:main-stable"),
        ({"LITELLM_IMAGE": "localhost/litellm:test"}, "localhost/litellm:test"),
        (
            {"LITELLM_IMAGE": "localhost/litellm:test", "LITELLM_DOCKER_TAG": "x"},
            "localhost/litellm:test",
        ),
    ],
)
def test_the_overrides_still_win(env: dict[str, str], expected: str) -> None:
    assert _resolve(**env) == expected


def test_compose_and_the_migration_step_read_the_resolved_image() -> None:
    text = SERVE_MODEL_SLURM.read_text()
    assert "\nLITELLM_IMAGE=${LITELLM_IMAGE}\n" in text  # into env.<uuid>
    assert 'local litellm_image="${LITELLM_IMAGE}"' in text
    assert f"{REPO}:${{LITELLM_DOCKER_TAG:-main-stable}}" not in text
