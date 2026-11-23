"""Retired error codes (DEC-H10, decision record 2026-11-24, slice S-H10).

``pipelines.*`` and ``verification.*`` lost their routes in hardening H-10.1
(2026-10-18). They leave the live catalog ``known_error_codes()``, which is
what clients and docs enumerate, but stay decodable: a client talking to an
older self-hosted UMRS can still receive one. umrs-protocol 1.0 deletes them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from umrs_protocol import errors

HERE = Path(__file__).resolve().parent
PROTOCOL_ROOT = HERE.parent
UMRS_ROOT = PROTOCOL_ROOT.parent
GOLDEN = json.loads((HERE / "golden" / "wire_vectors.json").read_text(encoding="utf-8"))
RETIRED_FAMILIES = ("pipelines", "verification")


def _retired_family(code: str) -> bool:
    return code.split(".", 1)[0] in RETIRED_FAMILIES


def test_is_retired_names_the_retired_families() -> None:
    assert errors.is_retired("pipelines.not_found")
    assert errors.is_retired("verification.unavailable")
    assert not errors.is_retired("jobs.not_found")
    assert not errors.is_retired("pipelines.made_up")


def test_registry_holds_both_whole_families() -> None:
    expected = {**errors.PIPELINES_ERRORS, **errors.VERIFICATION_ERRORS}
    assert expected == errors.RETIRED_CODES
    assert all(_retired_family(code) for code in errors.RETIRED_CODES)


def test_live_catalog_excludes_retired_codes() -> None:
    live = errors.known_error_codes()
    assert not set(live) & set(errors.RETIRED_CODES)
    assert not [code for code in live if _retired_family(code)]


def test_golden_codes_stay_decodable_with_their_status() -> None:
    """Retiring is not removing: every golden code decodes with its golden status."""
    decodable = errors.decodable_error_codes()
    golden = GOLDEN["errors"]["known_error_codes"]
    assert {code: decodable.get(code) for code in golden} == golden


@pytest.mark.parametrize("code", sorted(errors.RETIRED_CODES))
def test_retired_code_round_trips(code: str) -> None:
    wire = {"code": code, "message": "m", "http_status": errors.RETIRED_CODES[code], "detail": {}}
    decoded = errors.PlatformError.from_dict(wire)
    assert decoded.to_dict() == wire
    assert errors.is_retired(decoded.code)


def _protocol_version() -> tuple[int, ...]:
    text = (PROTOCOL_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)', text, re.MULTILINE)
    assert match, "umrs-protocol pyproject has no version"
    return tuple(int(part) for part in match.groups())


def test_removal_is_scheduled_for_protocol_1_0() -> None:
    """At 1.0 the registry must be empty: this test fails the release that forgets."""
    assert errors.RETIRED_CODES_REMOVED_IN == "1.0"
    if _protocol_version() >= (1, 0):
        assert errors.RETIRED_CODES == {}, "umrs-protocol 1.0 deletes RETIRED_CODES"


RETIRED_PREFIXES = tuple(f"{family}." for family in RETIRED_FAMILIES)


def _raises_retired(node: ast.AST) -> bool:
    """Any ``.pipelines(...)``/``.verification(...)`` call or a retired-family string.

    The owner is not checked, so ``errors.PlatformError.pipelines(...)`` and an
    aliased ``PE.verification(...)`` count. Any string constant that starts with
    a retired family prefix counts, which covers f-string pieces
    (``f"pipelines.{name}"``) and concatenation (``"verification." + name``).
    Not caught: a code assembled at runtime from separate pieces such as
    ``"verification" + ".forbidden"``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.startswith(RETIRED_PREFIXES)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in RETIRED_FAMILIES
    )


def _server_sources() -> list[Path]:
    roots = [UMRS_ROOT / "src", *sorted((UMRS_ROOT / "packages").glob("*/src"))]
    return [path for root in roots if root.is_dir() for path in sorted(root.rglob("*.py"))]


def test_no_server_code_raises_a_retired_code() -> None:
    sources = _server_sources()
    if not sources:
        pytest.skip("umrs-protocol checked out without the UMRS server tree")
    offenders = [
        f"{path.relative_to(UMRS_ROOT)}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if _raises_retired(node)
    ]
    assert offenders == []


def test_fence_detects_a_retired_raise() -> None:
    tree = ast.parse('raise_typed(PlatformError.pipelines("not_found", "m"))\nx = "verification.forbidden"')
    assert sum(_raises_retired(node) for node in ast.walk(tree)) == 2


def _flagged(source: str) -> int:
    return sum(_raises_retired(node) for node in ast.walk(ast.parse(source)))


@pytest.mark.parametrize(
    "source",
    [
        'errors.PlatformError.pipelines("not_found", "m")',
        'PE.verification("forbidden", "m")',
        'PlatformError(code=f"pipelines.{name}", message="m")',
        'PlatformError(code="verification." + name, message="m")',
        'HTTPException(409, headers={"X-Code": "pipelines." + suffix})',
        'PlatformError(code="pipelines.not_found", message="m")',
    ],
)
def test_fence_catches_bypass_forms(source: str) -> None:
    assert _flagged(source) >= 1


def test_fence_ignores_live_codes() -> None:
    source = 'PlatformError.jobs("not_found", "m")\nx = "jobs.not_found"\ny = f"workflows.{name}"'
    assert _flagged(source) == 0
