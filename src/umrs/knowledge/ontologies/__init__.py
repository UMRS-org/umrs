"""Bundled ontologies as package data (ADR-081 D-3, KN-2; slice KN-S7).

``std@1`` and ``r4-core@2`` ship as YAML files in this package, so a
stand-alone or air-gapped install gets them from the seed migration with no
fetch and no admin step. :func:`parse_doc` turns one YAML mapping into an
:class:`OntologyDoc`; :func:`load_bundled` parses every ``*.yaml`` file of
this package.

Document format (top-level keys; any other key is an error):

- ``ontology`` (str, required, ``[a-z0-9-]+``) and ``version`` (str or int,
  required; an int becomes its str) — the ref is ``<ontology>@<version>``.
- ``prefixes`` (mapping str -> str, required).
- ``iri_base`` (str, default ``""``), ``description`` (str, default ``""``),
  ``origin`` (default term origin, one of ``ORIGINS``, default
  ``"minted"``), ``imports`` (list of refs matching
  ``^[a-z0-9-]+@[0-9A-Za-z.]+$``, default ``[]``), ``profile`` (mapping,
  default ``{}``).
- ``classes``, ``relations``, ``attributes`` (lists of term mappings,
  default ``[]``); the section gives the term's ``kind``.

Term keys (any other key is an error): ``id`` (required CURIE),
``label`` (default: the CURIE's local part), ``parents``, ``bucket``,
``domain`` (-> ``domain_classes``), ``range`` (-> ``range_classes``),
``datatype``, ``unit``, ``transitive``, ``functional``, ``inverse_of``,
``aliases``, ``synonyms``, ``definition``, ``required_on``, ``origin``
(overrides the document default), ``verified`` (default True), ``source``
(default the document ref), ``status`` (``active``|``deprecated``, default
``active``), ``replaced_by``, ``mappings`` (mapping). Lists become tuples.

Validation (each failure raises :class:`OntologyDocError` naming the
document ref and, where there is one, the term id):

- a CURIE is ``<prefix>:<local>`` with a non-empty local part and a prefix
  declared in ``prefixes``; this holds for ``id``, ``parents``, ``domain``,
  ``range``, ``required_on``, ``inverse_of`` and ``replaced_by``
  (``unit`` is free text). ``iri`` = ``prefixes[prefix] + local``.
- a class needs a ``bucket`` in ``BUCKETS`` and has no ``datatype``;
  relations and attributes have no ``bucket``; a relation has no
  ``datatype``; an attribute's ``datatype``, when given, is in
  ``DATATYPES``.
- ``origin`` in ``ORIGINS``; ``status`` in ``active``/``deprecated``.
- no duplicate term id in one document; no alias used twice within one
  kind in one document (compared case-insensitively).

Terms keep the order of the document (classes, then relations, then
attributes).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..term_types import OntologyDoc

__all__ = [
    "OntologyDocError",
    "bundled_by_ref",
    "expand_curie",
    "load_bundled",
    "parse_doc",
]


class OntologyDocError(ValueError):
    """An ontology document is malformed."""


def expand_curie(curie: str, prefixes: Mapping[str, str]) -> str:
    """The IRI of ``curie``; :class:`OntologyDocError` when it is not a CURIE
    or its prefix is not declared."""
    raise NotImplementedError("kn-s7-01")


def parse_doc(mapping: Mapping[str, Any], *, source: str = "") -> OntologyDoc:
    """Parse one document (``source`` names it in error messages)."""
    raise NotImplementedError("kn-s7-01")


def load_bundled() -> tuple[OntologyDoc, ...]:
    """Every ``*.yaml`` document of this package, parsed with
    ``yaml.safe_load`` (files read through ``importlib.resources``), sorted
    by ref. Two files with one ref raise :class:`OntologyDocError`."""
    raise NotImplementedError("kn-s7-01")


def bundled_by_ref() -> dict[str, OntologyDoc]:
    """:func:`load_bundled` keyed by ref."""
    raise NotImplementedError("kn-s7-01")
