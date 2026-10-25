"""The server image's spaCy stack is pinned (2026-10-25).

docker/Dockerfile.server installed ``spacy`` unpinned. The rebuild of
2026-10-25 picked up preshed 3.0.14, with which ``en_core_web_sm`` fails to
load (OverflowError in Tokenizer._flush_specials), and the text-embedding
experience on rccs lost its text and text-file modalities at boot. The image
now pins spaCy and its compiled dependencies to the set that ran in
production before (image 968706f0).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.server"
SPACY_STACK = ("spacy", "thinc", "preshed", "cymem", "srsly", "murmurhash", "blis")
BROKEN = {"preshed": "3.0.14"}


def _pins() -> dict[str, str]:
    text = DOCKERFILE.read_text()
    return dict(re.findall(r'"([a-z][a-z0-9_-]*)==([0-9][0-9.]*)"', text))


def _unpinned_spacy_install() -> bool:
    return re.search(r"(?m)^\s+spacy\s", DOCKERFILE.read_text()) is not None


@pytest.mark.parametrize("package", SPACY_STACK)
def test_the_server_image_pins_each_spacy_stack_package(package):
    assert package in _pins(), f"{package} is not pinned in {DOCKERFILE.name}"


def test_spacy_is_not_also_installed_unpinned():
    assert not _unpinned_spacy_install()


def test_the_pins_avoid_the_release_that_breaks_the_model_load():
    pins = _pins()
    assert all(pins.get(name) != version for name, version in BROKEN.items())
