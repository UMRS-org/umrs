"""S-D39a task s-d39a-01: the manifest catalog identity block (DEC-D39).

Pure functions in ``umrs.knowledge.catalog_identity``: the change-note chain
up to the exported version, and the block that adds ``catalog_id``,
``change_notes``, ``members`` and ``origin_fingerprint`` to the record.
"""

from __future__ import annotations

import copy

from umrs.knowledge.catalog_identity import catalog_block, change_note_chain

_VERSIONS = [
    {"version": "0.1.0", "change_note": "first"},
    {"version": "0.2.0", "change_note": "added spectra"},
    {"version": "0.3.0", "change_note": None},
]


def test_chain_stops_after_the_exported_version() -> None:
    assert change_note_chain(_VERSIONS, "0.2.0") == [
        {"version": "0.1.0", "change_note": "first"},
        {"version": "0.2.0", "change_note": "added spectra"},
    ]


def test_chain_reads_none_and_missing_as_empty() -> None:
    rows = [{"version": "0.1.0"}, {"change_note": "x"}, {"version": None}]
    assert change_note_chain(rows, "nope") == [
        {"version": "0.1.0", "change_note": ""},
        {"version": "", "change_note": "x"},
        {"version": "", "change_note": ""},
    ]


def test_chain_without_the_version_returns_every_row_in_order() -> None:
    chain = change_note_chain(_VERSIONS, "9.9.9")
    assert [e["version"] for e in chain] == ["0.1.0", "0.2.0", "0.3.0"]
    assert chain[2]["change_note"] == ""


def test_chain_skips_non_dict_rows_and_handles_empty() -> None:
    assert change_note_chain([], "0.1.0") == []
    rows = ["junk", None, {"version": "0.1.0", "change_note": "a"}]
    assert change_note_chain(rows, "0.1.0") == [  # type: ignore[arg-type]
        {"version": "0.1.0", "change_note": "a"}
    ]


def test_chain_stops_at_the_first_match() -> None:
    rows = [
        {"version": "1", "change_note": "a"},
        {"version": "1", "change_note": "b"},
    ]
    assert change_note_chain(rows, "1") == [{"version": "1", "change_note": "a"}]


def _record() -> dict:
    return {
        "catalog_id": "scat-0123456789abcdef",
        "name": "mof-defects",
        "version": "0.2.0",
        "licence": "CC-BY-4.0",
        "contributors": ["alice"],
        "provenance": {"source": "lab"},
    }


def test_block_adds_identity_keys_and_keeps_record_keys() -> None:
    members = {"graphs": [{"default": True, "name": ""}], "unresolved": 0}
    chain = [{"version": "0.2.0", "change_note": "n"}]
    block = catalog_block(_record(), chain, members, "inst-" + "a" * 32)
    for key, value in _record().items():
        assert block[key] == value
    assert block["catalog_id"] == "scat-0123456789abcdef"
    assert block["change_notes"] == chain
    assert block["members"] == members
    assert block["origin_fingerprint"] == "inst-" + "a" * 32


def test_block_never_mutates_or_aliases_its_inputs() -> None:
    record, members = _record(), {"graphs": []}
    chain = [{"version": "0.2.0", "change_note": "n"}]
    before = copy.deepcopy((record, chain, members))
    block = catalog_block(record, chain, members, "fp")
    assert (record, chain, members) == before
    assert block is not record
    block["change_notes"][0]["change_note"] = "changed"
    block["members"]["graphs"] = ["x"]
    assert chain[0]["change_note"] == "n"
    assert members == {"graphs": []}


def test_block_missing_id_and_none_fingerprint_read_empty() -> None:
    record = _record()
    del record["catalog_id"]
    block = catalog_block(record, [], {}, None)  # type: ignore[arg-type]
    assert block["catalog_id"] == ""
    assert block["origin_fingerprint"] == ""
    assert block["change_notes"] == []
    assert block["members"] == {}


def test_empty_record_is_never_invented() -> None:
    chain = [{"version": "1", "change_note": "x"}]
    assert catalog_block({}, chain, {"graphs": []}, "inst-x") == {}
