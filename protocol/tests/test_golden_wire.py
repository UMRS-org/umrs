"""Golden wire vectors: umrs_protocol reproduces what r4_contracts produced.

``golden/wire_vectors.json`` was generated once from ``r4_contracts`` at the
U2.0 P0 commit, before the modules moved (ADR-117 D-2, U2.0 spec §7). These
tests replay it against ``umrs_protocol``. A failing vector is a wire change
between UMRS and its clients (gateway signing, ZMQ handshake, error codes):
review it as one, never regenerate the file to make it pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from umrs_protocol import errors, handshake_hmac, hpc_gate, protocol

VECTORS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "golden" / "wire_vectors.json").read_text(encoding="utf-8")
)
HMAC = VECTORS["handshake_hmac"]
PROTO = VECTORS["protocol"]
ERRS = VECTORS["errors"]


def _flip(hex_digest: str) -> str:
    """The same digest with its first hex digit changed."""
    head = "1" if hex_digest[0] == "0" else "0"
    return head + hex_digest[1:]


# ── handshake_hmac ──────────────────────────────────────────────────────────


def test_hmac_constants() -> None:
    assert HMAC["ENV_VAR"] == handshake_hmac.ENV_VAR
    assert handshake_hmac.MEMBERSHIPS_DOMAIN == HMAC["MEMBERSHIPS_DOMAIN"] == "memberships-v1|"
    live = {n: getattr(handshake_hmac, n) for n in vars(handshake_hmac) if n.startswith("HEADER_")}
    assert live == HMAC["headers"]
    assert len(live) == 8


@pytest.mark.parametrize(
    "row", HMAC["sign_claims"], ids=lambda r: f"{r['user_id']}-{r['is_admin']}"
)
def test_sign_claims(row: dict[str, Any]) -> None:
    args = (row["user_id"], row["project_id"], row["is_admin"])
    assert handshake_hmac.sign_claims(row["secret"], *args) == row["hmac"]
    assert handshake_hmac.verify_claims(row["secret"], row["hmac"], *args)
    assert not handshake_hmac.verify_claims(row["secret"], _flip(row["hmac"]), *args)
    assert not handshake_hmac.verify_claims(
        row["secret"], row["hmac"], args[0], args[1], not args[2]
    )


@pytest.mark.parametrize(
    "row", HMAC["sign_workflow_author_claim"], ids=lambda r: f"{r['user_id']}-{r['author']}"
)
def test_sign_workflow_author_claim(row: dict[str, Any]) -> None:
    args = (row["user_id"], row["project_id"], row["author"])
    assert handshake_hmac.sign_workflow_author_claim(row["secret"], *args) == row["hmac"]
    assert handshake_hmac.verify_workflow_author_claim(row["secret"], row["hmac"], *args)
    flipped = _flip(row["hmac"])
    assert not handshake_hmac.verify_workflow_author_claim(row["secret"], flipped, *args)


@pytest.mark.parametrize("row", HMAC["sign_memberships"], ids=lambda r: repr(r["memberships"]))
def test_sign_memberships(row: dict[str, Any]) -> None:
    args = (row["user_id"], row["project_id"], row["is_admin"], row["memberships"])
    assert handshake_hmac.sign_memberships(row["secret"], *args) == row["hmac"]
    listed = handshake_hmac.canonical_memberships(row["memberships"])
    # An empty list never verifies, whatever its signature (verify_memberships).
    assert handshake_hmac.verify_memberships(row["secret"], row["hmac"], *args) is bool(listed)
    assert not handshake_hmac.verify_memberships(row["secret"], _flip(row["hmac"]), *args)


@pytest.mark.parametrize("row", HMAC["canonical_memberships"], ids=lambda r: repr(r["input"]))
def test_canonical_memberships(row: dict[str, Any]) -> None:
    assert handshake_hmac.canonical_memberships(row["input"]) == row["output"]


def test_hmac_vectors_cover_the_spec_cases() -> None:
    """Admin and non-admin, author true and false."""
    assert {r["is_admin"] for r in HMAC["sign_claims"]} == {True, False}
    assert {r["author"] for r in HMAC["sign_workflow_author_claim"]} == {True, False}


def test_membership_vectors_cover_the_spec_cases() -> None:
    """Empty, single, unordered-with-duplicates and string membership inputs."""
    listed = [r["memberships"] for r in HMAC["sign_memberships"]]
    assert set(map(type, listed)) == {list, str}
    lists = [value for value in listed if isinstance(value, list)]
    assert {0, 1} <= set(map(len, lists))
    assert any(map(_unordered_with_duplicates, lists))


def _unordered_with_duplicates(value: list[str]) -> bool:
    return len(value) > len(set(value)) and value != sorted(value)


# ── protocol ────────────────────────────────────────────────────────────────


def test_protocol_version() -> None:
    assert protocol.PROTOCOL_VERSION == PROTO["PROTOCOL_VERSION"] == "1.0"


@pytest.mark.parametrize("row", PROTO["handshake_reply"], ids=lambda r: repr(r["args"]))
def test_handshake_reply(row: dict[str, Any]) -> None:
    assert protocol.handshake_reply(*row["args"]) == row["output"]


@pytest.mark.parametrize("row", PROTO["parse_handshake_reply"], ids=lambda r: repr(r["input"]))
def test_parse_handshake_reply(row: dict[str, Any]) -> None:
    assert list(protocol.parse_handshake_reply(row["input"])) == row["output"]


@pytest.mark.parametrize(
    "row", PROTO["is_protocol_compatible"], ids=lambda r: f"{r['server']!r}-{r['client']!r}"
)
def test_is_protocol_compatible(row: dict[str, Any]) -> None:
    extra = [row["client"]] if row["client"] else []
    assert protocol.is_protocol_compatible(row["server"], *extra) is row["output"]


# ── errors ──────────────────────────────────────────────────────────────────


def test_error_code_header() -> None:
    assert errors.ERROR_CODE_HEADER == ERRS["ERROR_CODE_HEADER"] == "x-platform-error-code"


def test_known_error_codes_superset() -> None:
    """Codes are append-only: additions pass, a removal or status change fails."""
    live = errors.known_error_codes()
    missing = sorted(set(ERRS["known_error_codes"]) - set(live))
    changed = {
        code: (status, live[code])
        for code, status in ERRS["known_error_codes"].items()
        if code in live and live[code] != status
    }
    assert not missing, f"error codes removed: {missing}"
    assert not changed, f"error code status changed (golden, live): {changed}"


@pytest.mark.parametrize("row", ERRS["factories"], ids=lambda r: f"{r['factory']}.{r['name']}")
def test_error_factories(row: dict[str, Any]) -> None:
    factory = getattr(errors.PlatformError, row["factory"])
    err = factory(row["name"], row["message"], **row["detail"])
    assert err.to_dict() == row["to_dict"]
    assert err.to_http_parts() == row["to_http_parts"]
    assert errors.known_error_codes()[err.code] == row["to_dict"]["http_status"]
    assert errors.PlatformError.from_dict(row["to_dict"]) == err


# ── hpc_gate ────────────────────────────────────────────────────────────────


def test_gate_script_name() -> None:
    assert hpc_gate.GATE_SCRIPT == VECTORS["hpc_gate"]["GATE_SCRIPT"] == "umrs_gate.py"
