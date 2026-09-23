"""D-03 pins: the UMRS account record is the CURVE key authority.

Long-lived per-user public keys live on the account record; the ZAP
provider registers them via the keystore sync; revocation on the record
unregisters at the next sync; session keys ride the same callback path
unchanged.
"""

from __future__ import annotations

import time

from umrs.admin.user_store import UserStore
from umrs.core.session_curve_provider import SessionCredentialsProvider

_PUB_A = "FQSq#7vW*e<!J<;.6&!*}b0OfiYOIZ!MtFZjqT8}"
_PUB_B = "Yne#EW0Gz.&-HQxAToP}QTkFI!BzYyM6IF4S<b98"
_BAD = "short"


def _store_with_key(tmp_path, username="alice", key=_PUB_A, revoked=""):
    store = UserStore(tmp_path)
    store.create(username)
    store.update(username, curve_public_key=key, curve_key_revoked_at=revoked)
    return store


def test_record_round_trips_the_key(tmp_path):
    store = _store_with_key(tmp_path)
    rec = store.get("alice")
    assert rec.curve_public_key == _PUB_A
    assert rec.curve_key_revoked_at == ""
    # persists through a fresh store over the same files
    rec2 = UserStore(tmp_path).get("alice")
    assert rec2.curve_public_key == _PUB_A


def test_sync_registers_account_keys(tmp_path):
    store = _store_with_key(tmp_path)
    provider = SessionCredentialsProvider()
    count = provider.sync_account_keys(store.list())
    assert count == 1
    assert provider.callback("*", _PUB_A.encode()) is True


def test_sync_honors_revocation_and_disable(tmp_path):
    provider = SessionCredentialsProvider()
    store = _store_with_key(tmp_path, revoked="2026-09-23T00:00:00+00:00")
    assert provider.sync_account_keys(store.list()) == 0  # revoked → never registered
    store.update("alice", curve_key_revoked_at="", disabled=True)
    assert provider.sync_account_keys(store.list()) == 0  # disabled → refused


def test_sync_drops_stale_keys_on_revoke(tmp_path):
    store = _store_with_key(tmp_path)
    provider = SessionCredentialsProvider()
    provider.sync_account_keys(store.list())
    assert provider.callback("*", _PUB_A.encode()) is True
    store.update("alice", curve_key_revoked_at="2026-09-23T01:00:00+00:00")
    assert provider.sync_account_keys(store.list()) == 0
    assert provider.callback("*", _PUB_A.encode()) is False  # ZAP refuses post-sync


def test_sync_replaces_changed_keys(tmp_path):
    store = _store_with_key(tmp_path)
    provider = SessionCredentialsProvider()
    provider.sync_account_keys(store.list())
    store.update("alice", curve_public_key=_PUB_B)
    assert provider.sync_account_keys(store.list()) == 1
    assert provider.callback("*", _PUB_A.encode()) is False
    assert provider.callback("*", _PUB_B.encode()) is True


def test_account_keys_do_not_expire_by_clock():
    """D-03: the far-future sentinel TTL must bypass the session cap."""
    provider = SessionCredentialsProvider(max_ttl_seconds=1)
    entry = provider.register_account_key(_PUB_A, "alice")
    assert entry.ttl_remaining > 10 * 365 * 86400 - 60
    time.sleep(1.1)
    assert provider.callback("*", _PUB_A.encode()) is True  # clock is not the switch


def test_session_keys_still_clock_capped():
    provider = SessionCredentialsProvider(max_ttl_seconds=1)
    provider.register(_PUB_B, "bob", ttl_seconds=3600)
    time.sleep(1.1)
    assert provider.callback("*", _PUB_B.encode()) is False  # session cap intact


def test_session_keys_untouched_by_sync(tmp_path):
    store = _store_with_key(tmp_path)
    provider = SessionCredentialsProvider()
    provider.register(_PUB_B, "bob", ttl_seconds=3600)
    provider.sync_account_keys(store.list())
    assert provider.callback("*", _PUB_B.encode()) is True  # bob survives syncs


def test_zap_refuses_unknown_keys():
    provider = SessionCredentialsProvider()
    assert provider.callback("*", _PUB_A.encode()) is False


def test_cli_revoke_verb_stamps_the_record(tmp_path):

    store = _store_with_key(tmp_path)
    from types import SimpleNamespace

    from umrs.cli.admin.main import _cmd_zmq_revoke_curve_key

    args = SimpleNamespace(name="alice")
    rc = _cmd_zmq_revoke_curve_key(args, store)
    assert rc == 0
    assert store.get("alice").curve_key_revoked_at.startswith("2026-")
