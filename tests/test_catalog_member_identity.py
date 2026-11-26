"""S-D39a task s-d39a-02: catalog member refs by content identity (export).

``umrs.knowledge.catalog_member_identity.member_identities`` turns a catalog
row's installation-local member ids into the bundle's ``members`` block.
"""

from __future__ import annotations

from umrs.knowledge.catalog_member_identity import member_identities

_DEFAULT = "g-default"
_GRAPHS = [{"graph_id": "g-2", "name": "spectra"}, {"graph_id": "g-3", "name": None}]
_SKILLS = [
    {"skill_id": "sk-1", "scope": "project", "name": "fit", "version": "1.0.0"},
    {"skill_id": "sk-2", "scope": "system", "name": "plot", "version": None},
]
_MCPS = [
    {"server_id": "mcp-1", "name": "microscope", "version": "0.2.0", "trust_class": "x"}
]


def _row(**over) -> dict:
    base = {
        "member_graph_ids": [_DEFAULT, "g-2"],
        "skill_ids": ["sk-1", "sk-2"],
        "mcp_server_ids": ["mcp-1"],
        "embedding_model_ids": ["intfloat/e5-base-v2"],
    }
    base.update(over)
    return base


def test_full_block_shape_and_values() -> None:
    out = member_identities(_row(), _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert out == {
        "graphs": [
            {"default": True, "name": ""},
            {"default": False, "name": "spectra"},
        ],
        "skills": [
            {"scope": "project", "name": "fit", "version": "1.0.0"},
            {"scope": "system", "name": "plot", "version": ""},
        ],
        "mcp_servers": [{"name": "microscope", "version": "0.2.0"}],
        "embedding_models": ["intfloat/e5-base-v2"],
        "unresolved": 0,
    }


def test_trust_class_is_never_exported() -> None:
    out = member_identities(_row(), _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert all(set(m) == {"name", "version"} for m in out["mcp_servers"])


def test_default_graph_needs_no_registry_row() -> None:
    row = _row(member_graph_ids=[_DEFAULT], skill_ids=[], mcp_server_ids=[])
    out = member_identities(row, [], [], [], _DEFAULT)
    assert out["graphs"] == [{"default": True, "name": ""}]
    assert out["unresolved"] == 0


def test_unknown_ids_are_counted_once_and_omitted() -> None:
    row = _row(
        member_graph_ids=["g-x", "g-x", "g-2"],
        skill_ids=["sk-none"],
        mcp_server_ids=["mcp-none", "mcp-none"],
    )
    out = member_identities(row, _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert out["graphs"] == [{"default": False, "name": "spectra"}]
    assert out["skills"] == []
    assert out["mcp_servers"] == []
    assert out["unresolved"] == 3


def test_none_name_reads_as_empty_string() -> None:
    out = member_identities(_row(member_graph_ids=["g-3"]), _GRAPHS, [], [], _DEFAULT)
    assert out["graphs"] == [{"default": False, "name": ""}]


def test_order_kept_and_repeats_collapsed() -> None:
    row = _row(
        member_graph_ids=["g-2", _DEFAULT, "g-2"],
        skill_ids=["sk-2", "sk-1", "sk-2"],
        embedding_model_ids=["b", "a", "b", ""],
    )
    out = member_identities(row, _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert [g["default"] for g in out["graphs"]] == [False, True]
    assert [s["name"] for s in out["skills"]] == ["plot", "fit"]
    assert out["embedding_models"] == ["b", "a"]


def test_missing_none_and_non_str_ids_read_as_empty() -> None:
    row = {"member_graph_ids": None, "skill_ids": [1, None, "sk-1"]}
    out = member_identities(row, _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert out == {
        "graphs": [],
        "skills": [{"scope": "project", "name": "fit", "version": "1.0.0"}],
        "mcp_servers": [],
        "embedding_models": [],
        "unresolved": 0,
    }


def test_inputs_are_not_mutated() -> None:
    row, graphs = _row(), [dict(g) for g in _GRAPHS]
    member_identities(row, graphs, _SKILLS, _MCPS, _DEFAULT)
    assert row == _row()
    assert graphs == _GRAPHS


def test_a_str_where_a_list_belongs_reads_as_empty() -> None:
    row = {"member_graph_ids": "g-2", "embedding_model_ids": "e5"}
    out = member_identities(row, _GRAPHS, _SKILLS, _MCPS, _DEFAULT)
    assert (out["graphs"], out["embedding_models"], out["unresolved"]) == ([], [], 0)
