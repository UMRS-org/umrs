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
