"""S-D39a task s-d39a-03: catalog member refs mapped onto local ids (import).

``umrs.knowledge.catalog_member_map.map_members`` resolves the bundle's
content identities through a resolver already scoped to the target project;
the bundle's own ids and trust classes are never used.
"""

from __future__ import annotations

from tests._lane_pending import pending
from umrs.knowledge import catalog_member_map
from umrs.knowledge.catalog_member_map import MemberMapping, map_members


class _Resolver:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def default_graph_id(self) -> str:
        self.calls.append(("default",))
        return "g-local-default"

    def graph_id_by_name(self, name: str) -> str | None:
        self.calls.append(("graph", name))
        return {"spectra": "g-local-spectra"}.get(name)

    def skill_id(self, scope: str, name: str, version: str) -> str | None:
        self.calls.append(("skill", scope, name, version))
        return {("project", "fit", "1.0.0"): "sk-local-fit"}.get((scope, name, version))

    def mcp_server(self, name: str, version: str) -> tuple[str, str] | None:
        self.calls.append(("mcp", name, version))
        if (name, version) == ("microscope", "0.2.0"):
            return ("mcp-local-1", "front-end")
        return None


pytestmark = pending(
    "s-d39a-03", lambda: catalog_member_map.map_members(None, _Resolver())
)


def _members() -> dict:
    return {
        "graphs": [{"default": True, "name": ""}, {"default": False, "name": "spectra"}],
        "skills": [{"scope": "project", "name": "fit", "version": "1.0.0"}],
        "mcp_servers": [
            {"name": "microscope", "version": "0.2.0", "trust_class": "local-safe"}
        ],
        "embedding_models": ["intfloat/e5-base-v2"],
        "unresolved": 0,
    }


def test_all_members_map_to_local_ids() -> None:
    out = map_members(_members(), _Resolver())
    assert out == MemberMapping(
        member_graph_ids=["g-local-default", "g-local-spectra"],
        skill_ids=["sk-local-fit"],
        mcp_server_ids=["mcp-local-1"],
        embedding_model_ids=["intfloat/e5-base-v2"],
        mcp_trust_classes={"mcp-local-1": "front-end"},
        unmapped=[],
    )


def test_trust_class_comes_from_the_resolver_not_the_bundle() -> None:
    out = map_members(_members(), _Resolver())
    assert out.mcp_trust_classes == {"mcp-local-1": "front-end"}


def test_misses_become_labels() -> None:
    members = {
        "graphs": [{"default": False, "name": "absent"}],
        "skills": [{"scope": "user", "name": "mine", "version": "2"}],
        "mcp_servers": [{"name": "microscope", "version": "9.9"}],
    }
    out = map_members(members, _Resolver())
    assert out.member_graph_ids == [] and out.skill_ids == []
    assert out.mcp_server_ids == [] and out.mcp_trust_classes == {}
    assert out.unmapped == ["graph:absent", "skill:user:mine@2", "mcp:microscope@9.9"]


def test_missing_scope_and_version_read_empty() -> None:
    resolver = _Resolver()
    out = map_members({"skills": [{"name": "fit"}], "mcp_servers": [{"name": "m"}]}, resolver)
    assert ("skill", "", "fit", "") in resolver.calls
    assert ("mcp", "m", "") in resolver.calls
    assert out.unmapped == ["skill::fit@", "mcp:m@"]


def test_malformed_entries_are_named() -> None:
    members = {
        "graphs": ["g-1", {"default": False}, {"default": False, "name": 7}],
        "skills": [None, {"scope": "project"}],
        "mcp_servers": [{"name": ""}],
        "embedding_models": [3, "", "m"],
    }
    out = map_members(members, _Resolver())
    assert out.unmapped == [
        "malformed:graph",
        "malformed:graph",
        "malformed:graph",
        "malformed:skill",
        "malformed:skill",
        "malformed:mcp",
        "malformed:embedding",
    ]
    assert out.embedding_model_ids == ["m"]


def test_non_dict_block_and_non_list_keys_are_empty() -> None:
    assert map_members(None, _Resolver()) == MemberMapping()
    assert map_members(["x"], _Resolver()) == MemberMapping()
    assert map_members({"graphs": "spectra", "skills": None}, _Resolver()) == MemberMapping()


def test_ids_deduplicated_labels_kept() -> None:
    members = {
        "graphs": [
            {"default": True},
            {"default": True, "name": "ignored"},
            {"default": False, "name": "absent"},
            {"default": False, "name": "absent"},
        ],
        "embedding_models": ["a", "a"],
    }
    out = map_members(members, _Resolver())
    assert out.member_graph_ids == ["g-local-default"]
    assert out.embedding_model_ids == ["a"]
    assert out.unmapped == ["graph:absent", "graph:absent"]


def test_bundle_ids_are_never_passed_through() -> None:
    members = {"graphs": [{"default": False, "name": "spectra", "graph_id": "g-src"}]}
    out = map_members(members, _Resolver())
    assert out.member_graph_ids == ["g-local-spectra"]
