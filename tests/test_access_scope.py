"""Three-tier access boundary unit tests (system / project / user).

Predicate coverage for umrs/knowledge/access_scope + identity resolution
semantics of umrs/api/_scope_identity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from umrs.knowledge.access_scope import (
    CallerScope,
    graph_readable,
    graph_writable,
    personal_project,
    project_readable,
    project_writable,
)

ALICE = CallerScope(user_id="alice", project_id="proj-a")
BOB = CallerScope(user_id="bob", project_id="proj-b")
ADMIN = CallerScope(user_id="root", project_id="ops", is_admin=True)
ANON = CallerScope(anonymous=True)
LEGACY = CallerScope(user_id="", project_id="proj-a")  # mvp X-Project-ID only


class TestProjectNamespace:
    def test_member_reads_own_project(self):
        assert project_readable("proj-a", ALICE)
        assert ALICE and not project_readable("proj-b", ALICE)

    def test_user_namespace_owner_only(self):
        assert project_readable(personal_project("alice"), ALICE)
        assert not project_readable(personal_project("alice"), BOB)

    def test_system_space_read_for_all_but_admin_only_writes(self):
        for caller in (ALICE, BOB, LEGACY, ANON):
            assert project_readable("system", caller)
        assert not project_writable("system", ALICE)
        assert project_writable("system", ADMIN)

    def test_global_and_blank_read_but_not_write(self):
        assert project_readable("", BOB)
        assert not project_writable("", BOB)

    def test_admin_bypass(self):
        assert project_readable("proj-b", ADMIN)
        assert project_writable("proj-b", ADMIN)

    def test_anonymous_mvp_stays_legacy_permissive(self):
        assert project_readable("proj-b", ANON)
        assert project_writable("proj-b", ANON)

    def test_legacy_x_project_id_gets_group_access_not_owner(self):
        assert project_readable("proj-a", LEGACY)
        assert not project_readable(personal_project("alice"), LEGACY)


class TestGraphTiers:
    def test_system_graph_read_all_admin_write_only(self):
        readable = graph_readable
        assert readable(kind="system", project_id=None, owner_id="", caller=ANON)
        assert readable(kind="system", project_id=None, owner_id="", caller=BOB)
        assert not graph_writable(
            kind="system", project_id=None, owner_id="", caller=BOB
        )
        assert graph_writable(kind="system", project_id=None, owner_id="", caller=ADMIN)

    def test_user_graph_owner_only(self):
        for fn in (graph_readable, graph_writable):
            assert fn(kind="user", project_id=None, owner_id="alice", caller=ALICE)
            assert not fn(kind="user", project_id=None, owner_id="alice", caller=BOB)
            assert fn(kind="user", project_id=None, owner_id="alice", caller=ADMIN)

    def test_personal_alias(self):
        assert graph_readable(
            kind="personal", project_id=None, owner_id="alice", caller=ALICE
        )

    def test_project_kind_member_rule(self):
        assert graph_readable(
            kind="project", project_id="proj-a", owner_id="", caller=ALICE
        )
        assert not graph_writable(
            kind="project", project_id="proj-a", owner_id="", caller=BOB
        )

    def test_legacy_research_kind_behaves_as_project(self):
        assert graph_readable(
            kind="research", project_id="proj-a", owner_id="", caller=ALICE
        )
        assert not graph_readable(
            kind="research", project_id="proj-a", owner_id="", caller=BOB
        )

    def test_graph_project_user_namespace(self):
        pid = personal_project("alice")
        assert graph_writable(kind="project", project_id=pid, owner_id="alice", caller=ALICE)
        assert not graph_readable(kind="project", project_id=pid, owner_id="alice", caller=BOB)


class TestCallerScopeAdaptation:
    def test_from_identity_none_is_anonymous(self):
        assert CallerScope.from_identity(None) == ANON

    def test_from_identity_maps_fields(self):
        scope = CallerScope.from_identity(
            SimpleNamespace(user_id="alice", project_id="proj-a", is_admin=False)
        )
        assert scope == ALICE


class TestScopeIdentityResolution:
    """HTTP-level resolution (mvp tolerant, strict closed)."""

    def _request(self, headers: dict[str, str]):
        return SimpleNamespace(headers={k.lower(): v for k, v in headers.items()})

    def test_verified_tuple_raises_without_hmac_in_strict(self, monkeypatch):
        from umrs.api import _scope_identity as si

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        req = self._request(
            {"x-user-id": "alice", "x-verified-project-id": "proj-a"}
        )
        with pytest.raises(Exception):
            si.request_caller_scope(req)

    def test_strict_anonymous_rejected(self, monkeypatch):
        from umrs.api import _scope_identity as si

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        with pytest.raises(Exception):
            si.request_caller_scope(self._request({}))

    def test_mvp_legacy_project_header(self, monkeypatch):
        from umrs.api import _scope_identity as si

        # mvp is an explicit dev opt-in since B12 (strict default)
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        scope = si.request_caller_scope(self._request({"x-project-id": "proj-a"}))
        assert scope == LEGACY

    def test_mvp_no_headers_is_anonymous(self, monkeypatch):
        from umrs.api import _scope_identity as si

        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        assert si.request_caller_scope(self._request({})) == ANON

    def test_mvp_verified_tuple_skips_hmac_but_keeps_scope(self, monkeypatch):
        from umrs.api import _scope_identity as si

        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        req = self._request(
            {
                "x-user-id": "alice",
                "x-verified-project-id": "proj-a",
                "x-verified-is-admin": "true",
            }
        )
        scope = si.request_caller_scope(req)
        assert scope.user_id == "alice"
        assert scope.project_id == "proj-a"
        assert scope.is_admin
