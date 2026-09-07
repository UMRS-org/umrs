"""Canonical scope grammar (shared by trust store + asset index)."""

from __future__ import annotations

import pytest
from umrs.shared.scope_grammar import (
    Scope,
    ScopeError,
    fallback_chain,
    is_descendant_of,
    parse,
    parse_lenient,
)


class TestParse:
    def test_bare_system(self):
        assert parse("system") == Scope(kind="system")

    def test_each_prefixed_kind(self):
        for kind in ("project", "group", "user", "session"):
            scope = parse(f"{kind}:abc")
            assert scope.kind == kind and scope.id == "abc"

    def test_id_with_inner_lexemes(self):
        assert parse("project:p1/sub") == Scope(kind="project", id="p1/sub")

    def test_empty_rejected(self):
        with pytest.raises(ScopeError):
            parse("")

    def test_garbage_rejected(self):
        with pytest.raises(ScopeError):
            parse("cosmic")

    def test_prefix_without_id_rejected(self):
        with pytest.raises(ScopeError):
            parse("project:")

    def test_legacy_alias_rejected_on_writes(self):
        with pytest.raises(ScopeError):
            parse("global")

    def test_legacy_alias_normalized_on_reads(self):
        assert parse_lenient("global") == Scope(kind="system")
        assert parse_lenient(None) == Scope(kind="system")


class TestPrecedence:
    def test_fallback_chain_bare(self):
        assert fallback_chain("system") == ["system"]

    def test_fallback_chain_deep(self):
        chain = fallback_chain("session:s1")
        assert chain[0] == "session:s1"
        assert chain[-1] == "system"

    def test_fallback_chain_project(self):
        assert fallback_chain("project:p1") == ["project:p1", "system"]

    def test_descendant(self):
        assert is_descendant_of("session:s1", "system") is True
        assert is_descendant_of("system", "project:p1") is False
        assert is_descendant_of("project:p1", "project:p1") is True

    def test_str_round_trip(self):
        scope = parse("group:lab-ml")
        assert str(scope) == "group:lab-ml"
        assert str(Scope(kind="system", id="")) == "system"


# ---------------------------------------------------------------------------
# Authorization ladder (ADR-041) — one ladder for every asset record type
# ---------------------------------------------------------------------------

from umrs.knowledge.access_scope import (  # noqa: E402
    CallerScope,
    graph_readable,
    graph_writable,
)
from umrs.shared.scope_grammar import (  # noqa: E402
    normalize_asset_scope,
    scope_readable,
    scope_writable,
)


def _read(rep, owner, project, **caller):
    ref = normalize_asset_scope(rep, owner_id=owner, project_id=project)
    return scope_readable(ref, **caller)


def _write(rep, owner, project, **caller):
    ref = normalize_asset_scope(rep, owner_id=owner, project_id=project)
    return scope_writable(ref, **caller)


ADMIN = {"user_id": "root", "project_id": "p", "is_admin": True}
ALICE_PX = {"user_id": "alice", "project_id": "proj-x"}
BOB_PY = {"user_id": "bob", "project_id": "proj-y"}
ANON = {"user_id": "", "project_id": "", "anonymous": True}


class TestNormalizeAssetScope:
    @pytest.mark.parametrize(
        "representation,kind",
        [
            ("system", "system"),
            ("SYSTEM", "system"),
            ("user", "user"),
            ("personal", "user"),
            ("project", "project"),
            ("research", "project"),
            ("group", "project"),
            ("team", "project"),
            ("", "project"),
        ],
    )
    def test_kind_mapping(self, representation, kind):
        ref = normalize_asset_scope(representation, owner_id="alice", project_id="proj-x")
        assert ref.kind == kind
        assert ref.owner_id == "alice"
        assert ref.project_id == "proj-x"


class TestSingleLadder:
    def test_admin_does_everything(self):
        assert _read("user", "alice", "", **ADMIN)
        assert _write("system", "", "", **ADMIN)

    def test_anonymous_stays_permissive(self):
        assert _read("user", "alice", "", **ANON)
        assert _write("system", "", "", **ANON)

    def test_system_reads_for_all_writes_for_admins(self):
        assert _read("system", "", "", **ALICE_PX)
        assert not _write("system", "", "", **ALICE_PX)

    def test_user_scope_owner_only(self):
        assert _read("user", "alice", "", **ALICE_PX)
        assert not _read("user", "alice", "", **BOB_PY)
        assert _write("user", "alice", "", **ALICE_PX)
        assert not _write("user", "alice", "", **BOB_PY)

    def test_project_scope_members(self):
        assert _read("project", "", "proj-x", **ALICE_PX)
        assert not _read("project", "", "proj-x", **BOB_PY)
        assert _write("project", "", "proj-x", **ALICE_PX)
        assert not _write("project", "", "proj-x", **BOB_PY)

    def test_global_project_row_any_authenticated(self):
        assert _read("research", "", "", **ALICE_PX)
        assert _read("project", "", "", **BOB_PY)
        assert not _write("project", "", "", **ALICE_PX)

    def test_system_namespace_shared_read_admin_write(self):
        assert _read("project", "", "system", **BOB_PY)
        assert not _write("project", "", "system", **BOB_PY)

    def test_personal_namespace_bound_to_owner(self):
        assert _read("project", "", "user:alice", **ALICE_PX)
        assert not _read("project", "", "user:alice", **BOB_PY)
        assert _write("project", "", "user:alice", **ALICE_PX)
        assert not _write("project", "", "user:alice", **BOB_PY)

    def test_scope_string_round_trip_into_ladder(self):
        # The label half and the ladder half share one vocabulary.
        ref = normalize_asset_scope("user", owner_id="alice")
        assert str(parse(f"user:{ref.owner_id}")) == "user:alice"


def _caller(**kw) -> CallerScope:
    base = {"user_id": "alice", "project_id": "proj-x"}
    base.update(kw)
    return CallerScope(**base)


class TestGraphDelegationEquivalence:
    """graph_readable/writable must equal the ladder for the same inputs."""

    def test_system_graph(self):
        assert graph_readable(kind="system", project_id=None, owner_id="", caller=_caller())
        assert not graph_writable(kind="system", project_id=None, owner_id="", caller=_caller())
        assert graph_writable(
            kind="system",
            project_id=None,
            owner_id="",
            caller=_caller(is_admin=True),
        )

    def test_personal_graph_owner_only(self):
        caller = _caller()
        assert graph_readable(kind="user", project_id=None, owner_id="alice", caller=caller)
        assert not graph_readable(kind="personal", project_id=None, owner_id="carol", caller=caller)
        assert graph_writable(kind="user", project_id=None, owner_id="alice", caller=caller)
        assert not graph_writable(kind="user", project_id=None, owner_id="carol", caller=caller)

    def test_project_graph_member_rule(self):
        assert graph_readable(kind="research", project_id="proj-x", owner_id="", caller=_caller())
        assert not graph_readable(
            kind="project", project_id="proj-y", owner_id="", caller=_caller()
        )
        assert graph_writable(kind="project", project_id="proj-x", owner_id="", caller=_caller())
        assert not graph_writable(
            kind="project", project_id="proj-y", owner_id="", caller=_caller()
        )

    def test_anonymous_caller_legacy_permissive(self):
        anon = CallerScope(anonymous=True)
        assert graph_readable(kind="user", project_id=None, owner_id="alice", caller=anon)
        assert graph_writable(kind="system", project_id=None, owner_id="", caller=anon)

    def test_system_namespace_graph_admin_managed(self):
        assert graph_readable(kind="project", project_id="system", owner_id="", caller=_caller())
        assert not graph_writable(
            kind="project", project_id="system", owner_id="", caller=_caller()
        )
