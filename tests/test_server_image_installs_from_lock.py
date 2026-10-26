"""The server image installs every third-party package from its lock (ADR-100).

docker/Dockerfile.server pip-installed an unpinned package list and fetched
the spaCy model with ``spacy download``. The rebuild of 2026-10-25 picked up
preshed 3.0.14, ``spacy.load("en_core_web_sm")`` raised OverflowError, and the
rccs Text Embedding experience lost its text modalities (wave ledger (81)).
The hotfix d41165f5 pinned only the spaCy stack, in the Dockerfile; the rest
stayed unpinned. Since 2026-10-26 the image installs the lock in
docker/server-image/ (exported from the ai4s-platform workspace uv.lock by
scripts/deps/lockctl.py) with ``--no-deps``, and runs smoke.py at build.

Replaces test_server_image_spacy_pin.py. On d41165f5's Dockerfile the
install-rule tests fail (bare package names, no --no-deps, spacy download).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

UMRS = Path(__file__).resolve().parents[1]
DOCKERFILE = UMRS / "docker" / "Dockerfile.server"
LOCK_DIR = UMRS / "docker" / "server-image"
FULL = LOCK_DIR / "requirements.lock.txt"
CPU = LOCK_DIR / "requirements-cpu.lock.txt"
TORCH_CPU = LOCK_DIR / "torch-cpu.lock.txt"
SMOKE = LOCK_DIR / "smoke.py"

SPACY_STACK = ("spacy", "thinc", "preshed", "cymem", "srsly", "murmurhash", "blis")
KNOWN_BAD = {"preshed": "3.0.14"}
VALUE_FLAGS = {"-r", "--requirement", "-c", "--constraint", "-e", "--editable",
               "-i", "--index-url"}
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==([^\s;]+)")
URL_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]* @ https://\S+#sha256=[0-9a-f]{64}$")


def _run_commands(text: str) -> list[str]:
    """Shell commands of every RUN instruction, continuations joined."""
    joined = re.sub(r"\\\n", " ", text)
    runs = [line[4:] for line in joined.splitlines() if line.startswith("RUN ")]
    separators = r"&&|\|\||;|\bthen\b|\belse\b|\bfi\b"
    return [part.strip() for run in runs for part in re.split(separators, run)]


def pip_installs(text: str) -> list[list[str]]:
    """The argument list of every `pip install` in a Dockerfile."""
    found = []
    for command in _run_commands(text):
        match = re.search(r"\bpip install\b(.*)", command)
        if match:
            found.append(shlex.split(match.group(1)))
    return found


def _value_problem(flag: str, value: str) -> str:
    if flag in ("-r", "--requirement") and not value.endswith(".lock.txt"):
        return f"-r {value} is not a lock"
    if flag in ("-e", "--editable") and not value.startswith(("/", ".")):
        return f"-e {value} is not a local path"
    return ""


def install_problems(args: list[str]) -> list[str]:
    """Why one `pip install` is not locked ([] when it is)."""
    problems = [] if "--no-deps" in args else ["no --no-deps"]
    items = iter(args)
    for arg in items:
        if arg in VALUE_FLAGS:
            problems.append(_value_problem(arg, next(items, "")))
        elif not arg.startswith(("-", "/", ".")):
            problems.append(f"unlocked package {arg}")
    return [problem for problem in problems if problem]


def _requirements(path: Path) -> list[str]:
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line for line in lines if line and not line.startswith(("#", "-"))]


def _pins(path: Path) -> dict[str, str]:
    matches = (PIN.match(line) for line in _requirements(path))
    return {m.group(1).lower(): m.group(3) for m in matches if m}


def test_every_pip_install_in_the_server_image_is_locked():
    installs = pip_installs(DOCKERFILE.read_text())
    assert installs, "Dockerfile.server installs nothing?"
    problems = {" ".join(args): install_problems(args) for args in installs}
    assert {cmd: why for cmd, why in problems.items() if why} == {}


def test_the_image_fetches_no_model_outside_the_lock():
    text = DOCKERFILE.read_text()
    assert "spacy download" not in text
    assert "en-core-web-sm @ https://" in FULL.read_text()


@pytest.mark.parametrize("lock", [FULL, CPU, TORCH_CPU], ids=lambda p: p.name)
def test_every_lock_line_is_an_exact_pin(lock):
    loose = [line for line in _requirements(lock) if not (PIN.match(line) or URL_PIN.match(line))]
    assert loose == []


@pytest.mark.parametrize("lock", [FULL, CPU], ids=lambda p: p.name)
def test_the_spacy_stack_is_locked_and_avoids_the_known_bad_release(lock):
    pins = _pins(lock)
    assert [name for name in SPACY_STACK if name not in pins] == []
    assert [name for name, bad in KNOWN_BAD.items() if pins[name] == bad] == []


def test_the_cpu_variant_is_the_lock_without_torch_and_cuda():
    full, cpu, torch = _pins(FULL), _pins(CPU), _pins(TORCH_CPU)
    assert torch == {"torch": full["torch"], "torchvision": full["torchvision"]}
    dropped = set(full) - set(cpu)
    assert {"torch", "torchvision"} <= dropped
    assert all(re.match(r"torch$|torchvision$|nvidia-|cuda-|triton$", name) for name in dropped)
    assert {name: cpu[name] for name in cpu} == {name: full[name] for name in cpu}
    # Not CUDA runtime: bentoml needs it (the first CPU build's pip check).
    assert "nvidia-ml-py" in cpu


def test_the_server_extra_excludes_the_known_bad_release():
    pyproject = (UMRS / "src" / "pyproject.toml").read_text()
    assert all(f'"{name}!={bad}"' in pyproject for name, bad in KNOWN_BAD.items())


def test_the_build_runs_the_smoke_test_that_loads_the_model():
    text = DOCKERFILE.read_text()
    assert "RUN python /opt/umrs/server-image-smoke.py" in text
    smoke = SMOKE.read_text()
    assert 'SPACY_MODEL = "en_core_web_sm"' in smoke and "spacy.load(SPACY_MODEL" in smoke


def test_the_rule_rejects_the_unpinned_install_of_d41165f5():
    """The checker itself: the hotfix's install line is not locked."""
    old = (
        'RUN pip install --no-cache-dir \\\n    pyzmq boto3 "spacy==3.8.16" \\\n'
        "    && (python -c \"import spacy\" && python -m spacy download en_core_web_sm || true)\n"
    )
    (first, *_) = pip_installs(old)
    assert install_problems(first) == ["no --no-deps", "unlocked package pyzmq",
                                       "unlocked package boto3", "unlocked package spacy==3.8.16"]
