"""``fields=`` projection (ADR-105 D-6 S6a, task a105-s6a-03)."""

from __future__ import annotations

import pytest
from _a105_s6a_pending import pending
from umrs.api.read_fields import MAX_FIELDS, FieldsError, parse_fields, project

pytestmark = pending("a105-s6a-03")

ALLOWED = frozenset({"job_id", "status", "job_type", "created_at"})


def test_fields_error_is_a_value_error() -> None:
    assert issubclass(FieldsError, ValueError)


@pytest.mark.parametrize("raw", [None, ""])
def test_absent_means_all_fields(raw: str | None) -> None:
    assert parse_fields(raw, ALLOWED, ("job_id",)) is None


def test_always_fields_come_first() -> None:
    assert parse_fields("status,job_type", ALLOWED, ("job_id",)) == (
        "job_id",
        "status",
        "job_type",
    )


def test_names_are_stripped_and_deduplicated_in_first_seen_order() -> None:
    got = parse_fields(" job_type , status,job_type,, job_id ", ALLOWED, ("job_id",))
    assert got == ("job_id", "job_type", "status")


def test_without_always() -> None:
    assert parse_fields("created_at", ALLOWED) == ("created_at",)


def test_always_names_need_not_be_allowed() -> None:
    assert parse_fields("status", frozenset({"status"}), ("execution_id",)) == (
        "execution_id",
        "status",
    )


@pytest.mark.parametrize("raw", [",", " , ", " ", ",,,"])
def test_no_names_is_an_error(raw: str) -> None:
    with pytest.raises(FieldsError) as exc:
        parse_fields(raw, ALLOWED, ("job_id",))
    assert str(exc.value) == "fields: no field names"


def test_unknown_names_are_listed_with_the_allowed_ones() -> None:
    with pytest.raises(FieldsError) as exc:
        parse_fields("zz,status,bogus,zz", ALLOWED, ("job_id",))
    assert str(exc.value) == (
        "fields: unknown bogus, zz; allowed: created_at, job_id, job_type, status"
    )


def test_too_many_names() -> None:
    raw = ",".join(["status"] * (MAX_FIELDS + 1))
    with pytest.raises(FieldsError) as exc:
        parse_fields(raw, ALLOWED)
    assert str(exc.value) == f"fields: at most {MAX_FIELDS} names"
    assert MAX_FIELDS == 32


def test_exactly_max_names_is_fine() -> None:
    raw = ",".join(["status"] * MAX_FIELDS)
    assert parse_fields(raw, ALLOWED) == ("status",)


def test_project_keeps_order_and_skips_missing() -> None:
    spec = {"a": 1}
    row = {"job_id": "j1", "status": "running", "spec": spec, "x": None}
    out = project(row, ("status", "missing", "job_id", "spec", "x"))
    assert out == {"status": "running", "job_id": "j1", "spec": spec, "x": None}
    assert list(out) == ["status", "job_id", "spec", "x"]
    assert out["spec"] is spec
    assert out is not row


def test_project_empty_fields() -> None:
    assert project({"a": 1}, ()) == {}
