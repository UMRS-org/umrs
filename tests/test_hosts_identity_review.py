"""Adversarial review of SH-H1b (ADR-098 D-5/D-6): defects found and pinned.

1. A host token's claims are parsed before its signature is checked, so any
   caller reaches ``json.loads`` + ``int()``. ``1e999`` (``inf``) raised
   ``OverflowError`` and a deeply nested payload ``RecursionError``: both
   escaped ``HostAuthError`` and became a 500 instead of a named 401.
2. ``--rekey`` was checked only when the token was issued. A second plain
   token issued before the host enrolled could later replace the enrolled
   key, so a leaked unused token took over an enrolled host's identity.
   The token now remembers whether it may rekey; redemption refuses
   ``enroll.already_enrolled`` otherwise and leaves the token unused.

Keys are generated per run; nothing here is a real credential.
"""

from __future__ import annotations

import pytest
from tests.test_hosts_identity import pg_pool  # noqa: F401  (fixture)
from umrs.hosts.enroll import (
    EnrollmentError,
    enrollment_request,
    generate_host_key,
    issue_enrollment,
    redeem_enrollment,
)
from umrs.hosts.identity import (
    HostAuthError,
    HostKeyStore,
    b64url,
    sign_host_token,
    verify_host_token,
)

pytest.importorskip("cryptography")

VENUE = "site-a"


def _forged(payload: bytes) -> str:
    return f"umrs-host.v1.{b64url(payload)}.{b64url(b'x' * 64)}"


def _redeem(store: HostKeyStore, token: str) -> object:
    body = enrollment_request(token, generate_host_key())
    return redeem_enrollment(store, body["token"], body["public_key"])


@pytest.fixture()
def store() -> HostKeyStore:
    return HostKeyStore(venue=VENUE)


_HEAD = b'{"aud":"umrs-host","venue":"site-a","host":"hostd-1",'


@pytest.mark.parametrize(
    "payload",
    [
        _HEAD + b'"iat":1e999,"exp":1e999}',
        _HEAD + b'"iat":-1e999,"exp":1}',
        _HEAD + b'"iat":NaN,"exp":1}',
        b"[" * 50_000 + b"]" * 50_000,
    ],
    ids=["inf", "neg-inf", "nan", "deep-nesting"],
)
def test_hostile_claims_are_a_named_401(store, payload):
    with pytest.raises(HostAuthError) as exc:
        verify_host_token(_forged(payload), store)
    assert exc.value.reason == "host_token_invalid" and exc.value.status == 401


def test_rail_answers_401_not_500_for_hostile_claims(store):
    from umrs_r4.api._rail_identity import RailAuthError, resolve_rail_principal

    headers = {"x-umrs-host-token": _forged(_HEAD + b'"iat":1e999,"exp":1}')}
    with pytest.raises(RailAuthError) as exc:
        resolve_rail_principal(
            token="t",
            session_id="s",
            client_ip="",
            broker=None,
            headers=headers,
            host_keys=store,
        )
    assert exc.value.reason == "host_token_invalid"


def _plain_token_after_enrollment(store: HostKeyStore) -> str:
    """Two plain tokens issued before enrollment; the first one enrolls."""
    first = issue_enrollment(store, "hostd-1")
    second = issue_enrollment(store, "hostd-1")
    _redeem(store, first.token)
    return second.token


def test_plain_token_cannot_replace_an_enrolled_key(store):
    spare = _plain_token_after_enrollment(store)
    before = store.public_key("hostd-1")
    with pytest.raises(EnrollmentError) as exc:
        _redeem(store, spare)
    assert exc.value.code == "enroll.already_enrolled"
    assert store.public_key("hostd-1").public_key == before.public_key


def test_refused_plain_token_stays_unused_until_unenroll(store):
    spare = _plain_token_after_enrollment(store)
    with pytest.raises(EnrollmentError):
        _redeem(store, spare)
    store.unenroll("hostd-1")
    issued = issue_enrollment(store, "hostd-1")
    _redeem(store, issued.token)


def test_rekey_token_still_replaces_the_key(store):
    _redeem(store, issue_enrollment(store, "hostd-1").token)
    new = generate_host_key()
    issued = issue_enrollment(store, "hostd-1", rekey=True)
    body = enrollment_request(issued.token, new)
    redeem_enrollment(store, body["token"], body["public_key"])
    verify_host_token(sign_host_token(new, venue=VENUE, host_id="hostd-1"), store)


def test_plain_token_refused_on_postgres(pg_pool):  # noqa: F811
    store = HostKeyStore(pg_pool, venue=VENUE)
    spare = _plain_token_after_enrollment(store)
    before = store.public_key("hostd-1")
    with pytest.raises(EnrollmentError) as exc:
        _redeem(store, spare)
    assert exc.value.code == "enroll.already_enrolled"
    assert store.public_key("hostd-1").public_key == before.public_key
    rekey = issue_enrollment(store, "hostd-1", rekey=True)
    assert _redeem(store, rekey.token).host_id == "hostd-1"
