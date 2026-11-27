"""``fields=`` projection for list reads (ADR-105 D-6, slice S6a).

A caller (an agent most of all) asks for only the columns it needs:
``GET /umrs/jobs?fields=status,job_type`` answers rows with ``job_id``,
``status`` and ``job_type`` only. Each route declares its allowed names
and the key fields it always keeps; an unknown name is a typed 400 that
lists the allowed names, never a silently dropped column.

Every failure raises ``FieldsError``; ``str(exc)`` is safe to return.
Pure: no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Most names one ``fields=`` value may carry (before de-duplication).
MAX_FIELDS = 32


class FieldsError(ValueError):
    """An unusable ``fields=`` value; ``str(exc)`` is safe to return."""


def parse_fields(
    raw: str | None,
    allowed: frozenset[str],
    always: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    """The projection a ``fields=`` value asks for, or None for "all fields".

    - ``raw`` None or ``""`` (no parameter): return None.
    - Split ``raw`` on ``","``, strip each name, drop empty names.
    - No name left (e.g. ``" , "``): ``FieldsError("fields: no field
      names")``.
    - More than ``MAX_FIELDS`` names (counted after dropping empties,
      before de-duplication): ``FieldsError("fields: at most 32 names")``
      (the number is ``MAX_FIELDS``).
    - Names not in ``allowed``: ``FieldsError("fields: unknown <a, b>;
      allowed: <x, y, z>")`` with the unknown names de-duplicated and
      sorted, joined by ``", "``, and every allowed name sorted, joined by
      ``", "``. Example: ``"fields: unknown bogus, zz; allowed: job_id,
      status"``.
    - Otherwise return ``always`` first (in its order), then the asked
      names in first-seen order, each name once. Names in ``always`` need
      not be in ``allowed``.
    """
    if not raw:
        return None
    names = _split_names(raw)
    _refuse_unknown(names, allowed)
    return tuple(dict.fromkeys((*always, *names)))


def _split_names(raw: str) -> list[str]:
    """The non-empty, stripped names of ``raw``, at most ``MAX_FIELDS``."""
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise FieldsError("fields: no field names")
    if len(names) > MAX_FIELDS:
        raise FieldsError(f"fields: at most {MAX_FIELDS} names")
    return names


def _refuse_unknown(names: list[str], allowed: frozenset[str]) -> None:
    """Raise the typed error naming every unknown and every allowed name."""
    unknown = sorted({name for name in names if name not in allowed})
    if unknown:
        raise FieldsError(
            f"fields: unknown {', '.join(unknown)}; allowed: {', '.join(sorted(allowed))}"
        )


def project(row: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    """A new dict with only ``fields`` of ``row``, in the order of ``fields``.

    A name missing from ``row`` is left out (no ``None`` placeholder).
    Values are the same objects (no copy).
    """
    return {name: row[name] for name in fields if name in row}
