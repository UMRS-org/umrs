"""S-D39a: catalog origin labels are hashed, shape-checked and bounded.

``umrs.knowledge.catalog_origin`` is the import's trust boundary for the
values a bundle claims about where its catalog came from.
"""

from __future__ import annotations

import hashlib

import pytest
from umrs.knowledge import catalog_origin as co

_FP = "inst-" + "0" * 32


def test_fingerprint_is_a_domain_separated_hash_of_the_id() -> None:
    fp = co.installation_fingerprint("6f1c-uuid")
    digest = hashlib.sha256(b"umrs-installation:6f1c-uuid").hexdigest()
    assert fp == "inst-" + digest[:32]
    assert co.installation_fingerprint("6f1c-uuid") == fp  # stable
    assert co.installation_fingerprint("other-id") != fp


def test_fingerprint_of_nothing_is_empty() -> None:
    assert co.installation_fingerprint("") == ""
    assert co.installation_fingerprint("   ") == ""
    assert co.installation_fingerprint(None) == ""  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["scat-0123456789abcdef", "a", "proj:cat@0.1.0", "x" * 128],
)
def test_well_formed_catalog_ids_kept(value: str) -> None:
    assert co.clean_catalog_id(value) == value


@pytest.mark.parametrize(
    "value",
    ["", " scat", "../etc", "a b", "<script>", "x" * 129, "-lead", 7, None, ["a"]],
)
def test_malformed_catalog_ids_dropped(value: object) -> None:
    assert co.clean_catalog_id(value) == ""


@pytest.mark.parametrize(
    "value", ["inst-", "inst-" + "G" * 32, "inst-" + "a" * 31, _FP + "0", "x" + _FP, 5]
)
def test_malformed_fingerprints_dropped(value: object) -> None:
    assert co.clean_fingerprint(value) == ""


def test_origin_from_block() -> None:
    block = {"catalog_id": "scat-1", "origin_fingerprint": _FP}
    assert co.origin_from_block(block) == ("scat-1", _FP)
    assert co.origin_from_block({"catalog_id": "a b"}) == ("", "")
    assert co.origin_from_block("not a block") == ("", "")
    assert co.origin_from_block(None) == ("", "")


def test_change_notes_shape_checked_truncated_and_bounded() -> None:
    long_note = "n" * (co.MAX_NOTE_CHARS + 50)
    raw = [
        {"version": "0.1.0", "change_note": long_note},
        {"version": "", "change_note": "dropped"},
        {"version": 3},
        "junk",
        {"version": "0.2.0", "change_note": 9},
    ]
    chain = co.clean_change_notes(raw)
    assert chain == [
        {"version": "0.1.0", "change_note": "n" * co.MAX_NOTE_CHARS},
        {"version": "0.2.0", "change_note": ""},
    ]
    many = [{"version": str(i), "change_note": ""} for i in range(co.MAX_CHANGE_NOTES + 5)]
    kept = co.clean_change_notes(many)
    assert len(kept) == co.MAX_CHANGE_NOTES
    assert kept[-1]["version"] == str(co.MAX_CHANGE_NOTES + 4)
    assert co.clean_change_notes({"version": "1"}) == []


def test_version_note_picks_the_last_entry_for_the_version() -> None:
    chain = [
        {"version": "0.1.0", "change_note": "a"},
        {"version": "0.1.0", "change_note": "b"},
    ]
    assert co.version_note(chain, "0.1.0") == "b"
    assert co.version_note(chain, "0.2.0") == ""
