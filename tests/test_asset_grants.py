"""Asset grant lifecycle + fetch-path authorization seam (ADR-041, P10-C2)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from umrs.grants import AssetGrant, GrantStore, grant_authorizes, grant_is_live
from umrs.model_registry.models import ModelEntry, ModelScope, ModelType
from umrs.model_registry.store.acl import AccessContext, ACLEnforcer

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)


def _grant(**overrides) -> AssetGrant:
    base = {
        "grant_id": "grt-1",
        "asset_kind": "model",
        "asset_id": "mdl-1",
        "grantee_kind": "user",
        "grantee_id": "alice",
    }
    base.update(overrides)
    return AssetGrant(**base)


class TestGrantLifecycle:
    def test_live_grant(self):
        assert grant_is_live(_grant(), now=NOW)

    def test_tombstoned_grant_is_dead(self):
        dead = _grant(tombstoned_at=NOW - timedelta(hours=1))
        assert not grant_is_live(dead, now=NOW)

    def test_expired_grant_is_dead(self):
        dead = _grant(expires_at=NOW - timedelta(seconds=1))
        assert not grant_is_live(dead, now=NOW)

    def test_future_expiry_is_live(self):
        live = _grant(expires_at=NOW + timedelta(days=30))
        assert grant_is_live(live, now=NOW)


class TestGrantAuthorizes:
    def test_user_grantee_match(self):
        assert grant_authorizes(
            _grant(),
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            now=NOW,
        )

    def test_wrong_user_denied(self):
        assert not grant_authorizes(
            _grant(),
            asset_kind="model",
            asset_id="mdl-1",
            user_id="bob",
            project_id="",
            now=NOW,
        )

    def test_project_grantee_match(self):
        grant = _grant(grantee_kind="project", grantee_id="proj-x")
        assert grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="bob",
            project_id="proj-x",
            now=NOW,
        )

    def test_project_grantee_needs_a_verified_project(self):
        grant = _grant(grantee_kind="project", grantee_id="proj-x")
        assert not grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="bob",
            project_id="",
            now=NOW,
        )

    def test_asset_mismatch_denied(self):
        assert not grant_authorizes(
            _grant(),
            asset_kind="model",
            asset_id="mdl-2",
            user_id="alice",
            project_id="",
            now=NOW,
        )

    def test_tombstone_denies(self):
        grant = _grant(tombstoned_at=NOW - timedelta(minutes=1))
        assert not grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            now=NOW,
        )

    def test_expiry_denies(self):
        grant = _grant(expires_at=NOW - timedelta(seconds=1))
        assert not grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            now=NOW,
        )

    def test_version_pin_match(self):
        grant = _grant(version_pin="v3")
        assert grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            asset_version="v3",
            now=NOW,
        )

    def test_version_pin_mismatch_denies(self):
        grant = _grant(version_pin="v3")
        assert not grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            asset_version="v4",
            now=NOW,
        )

    def test_version_pin_fail_closed_on_unversioned_asset(self):
        grant = _grant(version_pin="v3")
        assert not grant_authorizes(
            grant,
            asset_kind="model",
            asset_id="mdl-1",
            user_id="alice",
            project_id="",
            asset_version=None,
            now=NOW,
        )


class _FakeConn:
    """Minimal psycopg2 stand-in for GrantStore SQL-shape tests."""

    def __init__(self):
        self.rows = {}
        self.committed = 0

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *_a):
                return False

            def execute(self, sql, params):
                self._last = sql.strip().lower()
                if self._last.startswith("insert"):
                    gid = params[0]
                    conn.rows[gid] = list(params) + [None, None]
                elif self._last.startswith("update"):
                    marker, gid = params
                    for row in conn.rows.values():
                        if row[0] == gid and row[-2] is None:
                            row[-2] = marker
                            self.rowcount = 1
                            break
                    else:
                        self.rowcount = 0
                elif self._last.startswith("select"):
                    gid = params[0]
                    row = conn.rows.get(gid)
                    self._one = tuple(row) if row else None

            def fetchone(self):
                return getattr(self, "_one", None)

            def fetchall(self):
                return []

        return _Cur()

    def commit(self):
        self.committed += 1


class TestGrantStore:
    def test_create_grant_inserts_live_row(self):
        conn = _FakeConn()
        grant = GrantStore(conn).create_grant(
            asset_kind="model",
            asset_id="mdl-1",
            grantee_kind="user",
            grantee_id="alice",
            created_by="carol",
        )
        assert grant.grant_id.startswith("grt-")
        assert conn.rows[grant.grant_id][6] is None  # no tombstone on create
        assert conn.committed == 1

    def test_create_rejects_unknown_kinds(self):
        store = GrantStore(_FakeConn())
        for kwargs in (
            {"asset_kind": "endpoint"},
            {"grantee_kind": "org"},
        ):
            merged = {
                "asset_kind": "model",
                "asset_id": "m",
                "grantee_kind": "user",
                "grantee_id": "u",
                "created_by": "c",
            } | kwargs
            try:
                store.create_grant(**merged)
                raise AssertionError("should have raised ValueError")
            except ValueError:
                pass

    def test_revoke_tombstones_once(self):
        conn = _FakeConn()
        store = GrantStore(conn)
        grant = store.create_grant(
            asset_kind="model",
            asset_id="mdl-1",
            grantee_kind="user",
            grantee_id="alice",
            created_by="carol",
        )
        assert store.revoke_grant(grant.grant_id, now=NOW)
        assert not store.revoke_grant(grant.grant_id, now=NOW)  # idempotent

    def test_get_grant_maps_row(self):
        conn = _FakeConn()
        store = GrantStore(conn)
        grant = store.create_grant(
            asset_kind="knowledge_graph",
            asset_id="g-1",
            grantee_kind="project",
            grantee_id="proj-x",
            created_by="carol",
        )
        fetched = store.get_grant(grant.grant_id)
        assert fetched is not None
        assert fetched.asset_kind == "knowledge_graph"
        assert fetched.grantee_id == "proj-x"
        assert store.get_grant("grt-absent") is None


def _model(scope: ModelScope, **kw) -> ModelEntry:
    base = {
        "model_id": "mdl-1",
        "name": "m",
        "storage_ref": "/x",
        "model_type": ModelType.LLM,
    }
    base.update(kw)
    return ModelEntry(scope=scope, **base)


def _ctx(user: str, projects=(), admin=False) -> AccessContext:
    return AccessContext(
        user_id=user,
        is_admin=admin,
        project_memberships=set(projects),
    )


class TestACLLAdditiveGrantSeam:
    """Fetch-path authorization: grants extend the deny path, never writes."""

    def test_static_deny_without_reader(self):
        model = _model(ModelScope.USER, owner_id="carol")
        assert not ACLEnforcer().can_read(model, _ctx("alice"))

    def test_live_grant_authorizes_user_scope_read(self):
        model = _model(ModelScope.USER, owner_id="carol")
        enforcer = ACLEnforcer(grant_reader=lambda *_: [_grant()])
        assert enforcer.can_read(model, _ctx("alice"))

    def test_live_grant_authorizes_project_scope_read(self):
        model = _model(ModelScope.PROJECT, project_id="proj-y")
        enforcer = ACLEnforcer(grant_reader=lambda *_: [_grant()])
        assert enforcer.can_read(model, _ctx("alice"))

    def test_project_granted_to_member_project(self):
        model = _model(ModelScope.USER, owner_id="carol")
        grant = _grant(grantee_kind="project", grantee_id="proj-x")
        enforcer = ACLEnforcer(grant_reader=lambda *_: [grant])
        assert enforcer.can_read(model, _ctx("bob", projects={"proj-x"}))
        assert not enforcer.can_read(model, _ctx("bob", projects={"proj-z"}))

    def test_tombstoned_grant_denies(self):
        model = _model(ModelScope.USER, owner_id="carol")
        dead = _grant(tombstoned_at=NOW - timedelta(hours=1))
        enforcer = ACLEnforcer(grant_reader=lambda *_: [dead])
        assert not enforcer.can_read(model, _ctx("alice"))

    def test_wrong_asset_grant_denies(self):
        model = _model(ModelScope.USER, owner_id="carol")
        other = _grant(asset_id="mdl-other")
        enforcer = ACLEnforcer(grant_reader=lambda *_: [other])
        assert not enforcer.can_read(model, _ctx("alice"))

    def test_write_never_grant_authorized(self):
        model = _model(ModelScope.USER, owner_id="carol")
        enforcer = ACLEnforcer(grant_reader=lambda *_: [_grant()])
        assert not enforcer.can_write(model, _ctx("alice"))

    def test_static_allow_untouched_by_reader(self):
        model = _model(ModelScope.SYSTEM)
        enforcer = ACLEnforcer(grant_reader=lambda *_: [])
        assert enforcer.can_read(model, _ctx("alice"))

    def test_reader_failure_means_deny(self):
        model = _model(ModelScope.USER, owner_id="carol")

        def boom(*_):
            raise RuntimeError("grant backend down")

        enforcer = ACLEnforcer(grant_reader=boom)
        # fail-closed: an unreadable grant store must not authorize
        try:
            allowed = enforcer.can_read(model, _ctx("alice"))
        except RuntimeError:
            allowed = "raised"
        assert allowed in (False, "raised")
