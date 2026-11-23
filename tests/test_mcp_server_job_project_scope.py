"""SH-M1 (ADR-098 D-11): MCP_SERVER resolves (project_id, id|name) only.

Migration 0140 made MCP server names unique per project. Before this slice
the job spec dropped ``project_id`` and the backend looked servers up by
name alone, so project A's job could bind project B's server ``x``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs.mcp_server import McpServerBackend, McpServerStatusChecker
from umrs.mcp.project_scope import resolve_server_for_project, server_visible_to


class _Row:
    def __init__(self, server_id, name, project_id, url="http://127.0.0.1:9"):
        self.server_id = server_id
        self.name = name
        self.project_id = project_id
        self.url = url


_A = _Row("mcp-a", "x", "proj-a")
_B = _Row("mcp-b", "x", "proj-b")
_G = _Row("mcp-g", "shared", None)


class _NaiveRegistry:
    """Ignores the project hint, as the pre-0140 lookup did (last row wins)."""

    def __init__(self, rows):
        self._rows = list(rows)

    def get_server(self, server_id):
        for row in self._rows:
            if row.server_id == server_id:
                return row
        raise KeyError(server_id)

    def get_server_by_name(self, name, *, project_id=None):
        self.last_hint = project_id
        matches = [row for row in self._rows if row.name == name]
        if not matches:
            raise KeyError(name)
        return matches[-1]


class _ScopedRegistry(_NaiveRegistry):
    """Filters like ``MCPServerStore._name_lookup`` (own row, then global)."""

    def get_server_by_name(self, name, *, project_id=None):
        self.last_hint = project_id
        ranked = sorted(
            (r for r in self._rows if r.name == name and _owner(r) in ("", project_id)),
            key=lambda r: _owner(r) == "",
        )
        if not ranked:
            raise KeyError(name)
        return ranked[0]


def _owner(row):
    return (row.project_id or "").strip()


def _job(job_project, **spec):
    return SimpleNamespace(job_id="j", user_id="u", project_id=job_project, spec=spec)


def _attempt(ref=""):
    return SimpleNamespace(attempt_id="a", metrics={}, backend_ref=ref)


def _submit(registry, job):
    return asyncio.run(McpServerBackend(registry=registry).submit_job(job, _attempt()))


@pytest.mark.parametrize("registry_cls", [_NaiveRegistry, _ScopedRegistry])
def test_project_a_job_resolves_only_a_row(registry_cls):
    registry = registry_cls([_A, _B])
    ref = _submit(registry, _job("proj-a", server_name="x", project_id="proj-a"))
    if registry_cls is _NaiveRegistry:
        # A registry that ignores the hint returned B's row: refused, never bound.
        assert ref == "mcp_server|error=server_not_in_registry"
    else:
        assert ref == "mcp_server|id=mcp-a"
    assert registry.last_hint == "proj-a"  # the project reaches the store lookup
    assert _submit(registry_cls([_B]), _job("proj-a", server_name="x", project_id="proj-a")) == (
        "mcp_server|error=server_not_in_registry"
    )


def test_other_projects_server_id_refused():
    ref = _submit(_ScopedRegistry([_A, _B]), _job("proj-a", server_id="mcp-b", project_id="proj-a"))
    assert ref == "mcp_server|error=server_not_in_registry"


def test_global_server_visible_to_every_project():
    ref = _submit(_ScopedRegistry([_G]), _job("proj-a", server_name="shared", project_id="proj-a"))
    assert ref == "mcp_server|id=mcp-g"


def test_missing_project_id_refused_by_name():
    with pytest.raises(ValueError, match="mcp_server.project_required"):
        _submit(_ScopedRegistry([_A]), _job("proj-a", server_name="x"))


def test_spec_project_must_match_job_project():
    with pytest.raises(ValueError, match="mcp_server.project_mismatch"):
        _submit(_ScopedRegistry([_B]), _job("proj-a", server_name="x", project_id="proj-b"))


def test_checker_refuses_row_of_another_project(monkeypatch):
    from umrs.jobs import mcp_server as mod

    async def healthy(url):
        return True, "http 200"

    monkeypatch.setattr(mod, "_probe_server", healthy)
    checker = McpServerStatusChecker(registry_factory=_ScopedRegistry([_A, _B]))
    result = asyncio.run(checker.check_status(_attempt("mcp_server|id=mcp-b"), _job("proj-a")))
    assert result.status == "failed"
    result = asyncio.run(checker.check_status(_attempt("mcp_server|id=mcp-a"), _job("proj-a")))
    assert result.status == "running"


def test_helper_fails_closed():
    registry = _ScopedRegistry([_A, _G])
    assert resolve_server_for_project(registry, project_id="", server_id="mcp-g") is None
    assert resolve_server_for_project(registry, project_id="proj-a", server_name="nope") is None
    assert server_visible_to(_Row("s", "n", "  "), "proj-a")
    assert not server_visible_to(_B, "proj-a")
