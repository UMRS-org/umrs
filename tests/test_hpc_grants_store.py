"""Grants at the server (ADR-112 wave C; plan §6 C/D.0 C4, K8, K11, K18).

- storing: one refusal per C4 reason, in order, and an idempotent store;
- the view never carries the signature; ``covers_now`` / ``stale`` are
  computed now from the server's own digests, images and links (K11);
- a revoke (and so the account cascade) or a suspension ends the person's
  active grants (``account_revoked``);
- the same contract over Postgres (0137; skipped without a database).
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

import pytest
from grant_fixtures import (
    IMAGE,
    SIG,
    SITE,
    account_store,
    context,
    encoded,
    grant_body,
    signed,
)
from hpc_ledger_pg import execute, ledger_pool_or_skip
from umrs.hpc_accounts import grants as g
from umrs.hpc_accounts.store import SiteAccountStore
from umrs.hpc_accounts.usage_ledger import (
    BudgetExceeded,
    InMemoryUsageLedger,
    LedgerUnit,
    Resources,
)


def _ledger(budget: float = 0.0) -> InMemoryUsageLedger:
    return InMemoryUsageLedger(tables={"rows": [], "budgets": {"p1": budget}})


def _check(body: dict[str, Any], store: Any = None, ledger: Any = None, **ctx_over: Any) -> None:
    store = store or account_store()
    g.check_grant(signed(body), context(store, **ctx_over), g.grant_store_for(store), ledger or _ledger())


# ── decode (C4 step 1) ───────────────────────────────────────────────────────


def test_decode_parses_with_the_gates_own_parser_and_schema():
    body = grant_body()
    grant = g.decode_grant(encoded(body), SIG)
    assert grant.body == body and grant.grant_id == body["grant_id"]


@pytest.mark.parametrize(
    ("payload", "signature", "reason"),
    [
        ("not base64!", SIG, "bad_envelope"),
        (base64.b64encode(b'{"v": 1}').decode(), SIG, "bad_schema"),
        ("QUFB" * (g.GRANT_MAX_BYTES // 3 + 2), SIG, "too_large"),
        (None, "not armored", "bad_envelope"),
    ],
    ids=["not_base64", "bad_schema", "too_large", "bad_signature"],
)
def test_decode_refuses_by_name(payload, signature, reason):
    payload = encoded(grant_body()) if payload is None else payload
    with pytest.raises(g.GrantInvalid) as info:
        g.decode_grant(payload, signature)
    assert info.value.reason == reason


def test_decode_refuses_an_issued_outside_five_minutes():
    """K8: the store refuses ``issued`` outside ``now ± 300 s``."""
    now = int(time.time())
    late = grant_body(issued=now - 400, not_before=now - 400, expires=now + 3600)
    with pytest.raises(g.GrantInvalid, match="300 s") as info:
        g.decode_grant(encoded(late), SIG)
    assert info.value.reason == "bad_schema"


def test_a_float_anywhere_is_bad_schema():
    body = grant_body()
    raw = json.dumps({**body, "caps": {**body["caps"], "max_jobs": 1.5}}).encode()
    with pytest.raises(g.GrantInvalid) as info:
        g.decode_grant(base64.b64encode(raw).decode(), SIG)
    assert info.value.reason == "bad_schema"


# ── the checks (C4 steps 2-7) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"site": "other-site"}, "wrong_site"),
        ({"principal": "bob"}, "wrong_principal"),
        ({"login": "mallory"}, "wrong_login"),
        ({"projects": ["p9"]}, "not_linked"),
        ({"accounts": ["grp-z"]}, "not_linked"),
        ({"workflows": [{"name": "wf", "version": "9.9.9", "src_manifest_sha256": "1" * 64, "steps_sha256": "2" * 64, "params": []}]}, "workflow_unknown"),
        ({"workflows": [{"name": "wf", "version": "1.0.0", "src_manifest_sha256": "3" * 64, "steps_sha256": "2" * 64, "params": []}]}, "workflow_mismatch"),
        ({"workflows": [{"name": "wf", "version": "1.0.0", "src_manifest_sha256": "1" * 64, "steps_sha256": "4" * 64, "params": []}]}, "workflow_mismatch"),
        ({"images": [{**IMAGE, "image_id": "c" * 64}]}, "image_unknown"),
    ],
)
def test_each_check_refuses_by_its_reason(over, reason):
    with pytest.raises(g.GrantInvalid) as info:
        _check(grant_body(**over))
    assert info.value.reason == reason


def test_a_project_the_person_left_is_not_linked():
    with pytest.raises(g.GrantInvalid) as info:
        _check(grant_body(), memberships=frozenset({"p2"}))
    assert info.value.reason == "not_linked"


def test_a_gate_older_than_three_refuses_gate_version():
    """K18: ``hpcaccess.gate_version`` (``grants_need_gate_3``), not grant_invalid."""
    store = account_store()
    store.record_verified("alice", "tk-000000000001", {**store.get_account("alice"), "gate_version": "2"})
    with pytest.raises(g.GrantGateStale) as info:
        _check(grant_body(), store=store)
    assert (info.value.code, info.value.reason) == ("hpcaccess.gate_version", "grants_need_gate_3")


def test_too_many_active_grants():
    store = account_store()
    grants = g.grant_store_for(store)
    for _ in range(g.MAX_ACTIVE):
        grants.insert(g.row_from_grant(SITE, "alice", signed(grant_body()), {}))
    with pytest.raises(g.GrantInvalid) as info:
        _check(grant_body(), store=store)
    assert info.value.reason == "too_many"


def test_a_grant_that_could_never_fit_the_budget_is_refused():
    """D3, K14: caps.max_gpu_hours must fit what is left; a grant reserves nothing."""
    ledger = _ledger(budget=3.0)
    _check(grant_body(), ledger=ledger)  # 2 h <= 3 h
    unit = LedgerUnit(venue=SITE, attempt_id="r/s/1", person="bob", lane="staged_step", tag="t", project_id="p1")
    ledger.pending(unit, Resources(1, 1, 7200))
    with pytest.raises(BudgetExceeded):
        _check(grant_body(), ledger=ledger)  # 2 h > 3 h - 2 h reserved


def test_storing_is_idempotent_and_another_body_under_one_id_is_conflict():
    store = g.InMemoryGrantStore(SITE)
    body = grant_body()
    row = g.row_from_grant(SITE, "alice", signed(body), {})
    assert store.insert(row)["grant_sha256"] == store.insert(dict(row))["grant_sha256"]
    other = g.row_from_grant(SITE, "alice", signed(grant_body(grant_id=body["grant_id"])), {})
    with pytest.raises(g.GrantInvalid) as info:
        store.insert(other)
    assert info.value.reason == "conflict"


# ── the view (K11) and the lifecycle ─────────────────────────────────────────


def test_the_view_carries_no_signature_and_covers_now():
    store = account_store()
    row = g.grant_store_for(store).insert(g.row_from_grant(SITE, "alice", signed(grant_body()), {"jobs": 1}))
    view = g.grant_view(row, context(store))
    assert set(view) == {*g.VIEW, "covers_now", "stale"}
    assert SIG not in json.dumps(view) and "payload_b64" not in view
    assert (view["covers_now"], view["stale"], view["usage"]) == (True, [], {"jobs": 1})


def test_a_changed_image_or_digest_or_link_makes_the_grant_stale():
    """K11: after the site's image digests change, ``covers_now`` is false
    with the reason; the admission's candidates skip it."""
    store = account_store()
    row = g.grant_store_for(store).insert(g.row_from_grant(SITE, "alice", signed(grant_body()), {}))
    view = g.grant_view(row, context(store, images=frozenset()))
    assert view["covers_now"] is False and view["stale"] == [f"image_unknown: {'a' * 12}"]
    view = g.grant_view(row, context(store, digests_for=lambda _n, _v: ("9" * 64, "2" * 64)))
    assert view["stale"] == ["workflow_mismatch: wf@1.0.0"]
    store.suspend_links("alice", "test", "p1")
    assert g.grant_view(row, context(store))["stale"] == ["link_inactive: p1"]
    assert g.candidates([row], context(store, images=frozenset()), project="p1", workflow=("wf", "1.0.0"), account="grp-a") == []


def test_expired_rows_are_marked_lazily():
    store = g.InMemoryGrantStore(SITE)
    row = g.row_from_grant(SITE, "alice", signed(grant_body()), {})
    store.insert({**row, "expires_at": row["created_at"].replace(year=2020)})
    (listed,) = store.list("alice")
    assert (listed["status"], listed["ended_reason"]) == ("expired", "expired")


def test_a_revoke_ends_every_active_grant_of_the_person(tmp_path, monkeypatch):
    from umrs.hpc_accounts.revoke import revoke_account

    monkeypatch.setenv("UMRS_TRANSPORT_KEY_DIR", str(tmp_path))
    store = account_store()
    grants = g.grant_store_for(store)
    row = grants.insert(g.row_from_grant(SITE, "alice", signed(grant_body()), {}))
    revoke_account(store, "alice", root=tmp_path)
    ended = grants.get("alice", row["grant_id"])
    assert (ended["status"], ended["ended_reason"]) == ("cancelled", "account_revoked")


def test_another_venue_never_sees_the_grant():
    tables: dict[str, Any] = {}
    here = g.InMemoryGrantStore(SITE, tables)
    row = here.insert(g.row_from_grant(SITE, "alice", signed(grant_body()), {}))
    assert g.InMemoryGrantStore("other-front", tables).get("alice", row["grant_id"]) is None
    with pytest.raises(g.GrantNotFound):
        g.check_grant_id("../etc")


# ── Postgres (0137) ──────────────────────────────────────────────────────────


@pytest.mark.integration
def test_the_postgres_store_keeps_the_same_contract():
    pool = ledger_pool_or_skip()
    venue = f"g-{uuid.uuid4().hex[:8]}"
    accounts = SiteAccountStore(pool, venue)
    accounts.issue_key("alice", "tk-000000000001", "SHA256:fp")
    try:
        grants = g.grant_store_for(accounts)
        assert isinstance(grants, g.PgGrantStore)
        body = grant_body(site=venue)
        row = grants.insert(g.row_from_grant(venue, "alice", signed(body), {"jobs": 0}))
        assert grants.insert(g.row_from_grant(venue, "alice", signed(body), {}))["grant_sha256"] == row["grant_sha256"]
        with pytest.raises(g.GrantInvalid):
            grants.insert(g.row_from_grant(venue, "alice", signed(grant_body(site=venue, grant_id=body["grant_id"])), {}))
        grants.set_usage("alice", row["grant_id"], {"jobs": 3})
        assert grants.get("alice", row["grant_id"])["usage"] == {"jobs": 3}
        assert grants.end_for_account("alice") == 1
        assert grants.get("alice", row["grant_id"])["ended_reason"] == "account_revoked"
        assert grants.set_status("alice", row["grant_id"], "cancelled", "cancelled", "pending")["gate_cancel"] == "pending"
    finally:
        execute(pool, "DELETE FROM site_accounts WHERE venue = %s", (venue,))
