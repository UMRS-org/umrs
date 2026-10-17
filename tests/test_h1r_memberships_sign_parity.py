"""H-1 memberships lane (2026-10-18): the sidecar's memberships signer is a
stdlib twin of ``r4_contracts.handshake_hmac.sign_memberships`` (the sidecar
image does not ship r4-contracts). Drift would make UMRS ignore every
gateway-issued list, silently narrowing multi-project users to their
active project; this test makes it loud.
"""

from __future__ import annotations

import pytest

# Reuse the allowlisted sibling-tree loader (tools/test_import_hygiene_allowlist.txt
# lists test_sidecar_sign_parity.py) instead of a second spec_from_file_location
# harness, so the ADR-053 import-hygiene validator stays green.
from tests.test_sidecar_sign_parity import _load_sidecar

from r4_contracts.handshake_hmac import (
    HEADER_MEMBERSHIPS_HMAC,
    HEADER_PROJECT_MEMBERSHIPS,
    sign_claims,
    sign_memberships,
    verify_memberships,
)


@pytest.mark.parametrize(
    ("secret", "user", "project", "admin", "members"),
    [
        ("s3crt", "alice", "proj-a", False, ["proj-a", "proj-b"]),
        ("s3crt", "alice", "proj-a", True, ["proj-b", "proj-a", "proj-b"]),
        ("other", "b_ob", "proj-b", False, [" proj-c ", "proj-b"]),
        ("x" * 64, "u-y", "proj-c", True, ["proj-c"]),
    ],
)
def test_sidecar_memberships_signer_matches_contract(secret, user, project, admin, members):
    sidecar = _load_sidecar()
    got = sidecar._sign_memberships_hmac(secret, user, project, admin, members)
    assert got == sign_memberships(secret, user, project, admin, members)


def test_sidecar_headers_verify_against_the_contract(monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "parity-secret")
    sidecar = _load_sidecar()
    claims = {
        "user_id": "alice",
        "project_id": "proj-a",
        "roles": [],
        "project_memberships": ["proj-b", "proj-a"],
    }
    headers = sidecar._verified_project_headers(claims)
    assert headers["X-Verified-HMAC"] == sign_claims("parity-secret", "alice", "proj-a", False)
    assert verify_memberships(
        "parity-secret",
        headers[HEADER_MEMBERSHIPS_HMAC],
        "alice",
        "proj-a",
        False,
        headers[HEADER_PROJECT_MEMBERSHIPS],
    )
