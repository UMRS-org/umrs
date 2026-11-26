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

Validation (each failure raises :class:`OntologyDocError`; its message
names the document ref once it is known, the term id where there is one,
and the offending key or value, e.g. ``"demo@1: d:Thing: class needs a
bucket"`` or ``"demo@1: unknown key 'colour'"``):

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

import importlib.resources
import re
from collections.abc import Mapping
from typing import Any, NoReturn

import yaml

from ..term_types import BUCKETS, DATATYPES, ORIGINS, OntologyDoc, TermSpec

__all__ = [
    "OntologyDocError",
    "bundled_by_ref",
    "expand_curie",
    "load_bundled",
    "parse_doc",
]


class OntologyDocError(ValueError):
    """An ontology document is malformed."""


_DOC_KEYS = frozenset(
    {
        "ontology",
        "version",
        "prefixes",
        "iri_base",
        "description",
        "origin",
        "imports",
        "profile",
        "classes",
        "relations",
        "attributes",
    }
)
_TERM_KEYS = frozenset(
    {
        "id",
        "label",
        "parents",
        "bucket",
        "domain",
        "range",
        "datatype",
        "unit",
        "transitive",
        "functional",
        "inverse_of",
        "aliases",
        "synonyms",
        "definition",
        "required_on",
        "origin",
        "verified",
        "source",
        "status",
        "replaced_by",
        "mappings",
    }
)
_SECTIONS = (
    ("classes", "class"),
    ("relations", "relation"),
    ("attributes", "attribute"),
)


def _fail(ref: str, term_id: str, detail: str) -> NoReturn:
    location = ref or "<document>"
    if term_id:
        location = f"{location}: {term_id}"
    raise OntologyDocError(f"{location}: {detail}")


def _required(mapping: Mapping[str, Any], key: str, ref: str, term_id: str = "") -> Any:
    if key not in mapping:
        _fail(ref, term_id, f"missing required key {key!r}")
    return mapping[key]


def _check_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], ref: str, term_id: str = ""
) -> None:
    for key in mapping:
        if key not in allowed:
            _fail(ref, term_id, f"unknown key {key!r}")


def _text(value: Any, key: str, ref: str, term_id: str = "") -> str:
    if not isinstance(value, str):
        _fail(ref, term_id, f"{key} must be a string, got {value!r}")
    return value


def _optional_text(value: Any, key: str, ref: str, term_id: str = "") -> str | None:
    if value is None:
        return None
    return _text(value, key, ref, term_id)


def _bool(value: Any, key: str, ref: str, term_id: str = "") -> bool:
    if not isinstance(value, bool):
        _fail(ref, term_id, f"{key} must be a boolean, got {value!r}")
    return value


def _mapping(value: Any, key: str, ref: str, term_id: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(ref, term_id, f"{key} must be a mapping, got {value!r}")
    return dict(value)


def _list(value: Any, key: str, ref: str, term_id: str = "") -> list[Any]:
    if not isinstance(value, list):
        _fail(ref, term_id, f"{key} must be a list, got {value!r}")
    return value


def _choice(value: Any, allowed: Any, key: str, ref: str, term_id: str = "") -> str:
    text = _text(value, key, ref, term_id)
    if text not in allowed:
        _fail(ref, term_id, f"{key} {text!r} is not one of {tuple(allowed)!r}")
    return text


def _strings(value: Any, key: str, ref: str, term_id: str = "") -> tuple[str, ...]:
    return tuple(
        _text(item, key, ref, term_id) for item in _list(value, key, ref, term_id)
    )


def _curie_iri(
    value: str, prefixes: Mapping[str, str], key: str, ref: str, term_id: str = ""
) -> str:
    try:
        return expand_curie(value, prefixes)
    except OntologyDocError as exc:
        _fail(ref, term_id, f"{key} {value!r}: {exc}")


def _curies(
    value: Any, prefixes: Mapping[str, str], key: str, ref: str, term_id: str = ""
) -> tuple[str, ...]:
    values = _strings(value, key, ref, term_id)
    for curie in values:
        _curie_iri(curie, prefixes, key, ref, term_id)
    return values


def _optional_curie(
    value: Any, prefixes: Mapping[str, str], key: str, ref: str, term_id: str
) -> str | None:
    if value is None:
        return None
    text = _text(value, key, ref, term_id)
    _curie_iri(text, prefixes, key, ref, term_id)
    return text


def _version(value: Any, ref: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        _fail(ref, "", f"version must be a string or integer, got {value!r}")
    version = str(value)
    if not version:
        _fail(ref, "", "version must not be empty")
    return version


def _prefixes(value: Any, ref: str) -> dict[str, str]:
    raw = _mapping(value, "prefixes", ref)
    result: dict[str, str] = {}
    for prefix, iri in raw.items():
        if not isinstance(prefix, str) or not isinstance(iri, str):
            _fail(ref, "", f"prefixes entry {prefix!r} must map a string to a string")
        result[prefix] = iri
    return result


def _imports(value: Any, ref: str) -> tuple[str, ...]:
    imports = _strings(value, "imports", ref)
    for imported in imports:
        if re.fullmatch(r"[a-z0-9-]+@[0-9A-Za-z.]+", imported) is None:
            _fail(ref, "", f"invalid import ref {imported!r}")
    return imports


def _validate_class(
    bucket: str | None, datatype: str | None, ref: str, term_id: str
) -> None:
    if bucket is None:
        _fail(ref, term_id, "class needs a bucket")
    if bucket not in BUCKETS:
        _fail(ref, term_id, f"invalid bucket {bucket!r}")
    if datatype is not None:
        _fail(ref, term_id, f"class must not have datatype {datatype!r}")


def _validate_non_class(
    kind: str, bucket: str | None, datatype: str | None, ref: str, term_id: str
) -> None:
    if bucket is not None:
        _fail(ref, term_id, f"{kind} must not have bucket {bucket!r}")
    if kind == "relation" and datatype is not None:
        _fail(ref, term_id, f"relation must not have datatype {datatype!r}")
    if kind == "attribute" and datatype is not None and datatype not in DATATYPES:
        _fail(ref, term_id, f"invalid attribute datatype {datatype!r}")


def _parse_term(
    raw: Any, kind: str, prefixes: Mapping[str, str], ref: str, document_origin: str
) -> TermSpec:
    if not isinstance(raw, Mapping):
        _fail(ref, "", f"{kind} entry must be a mapping, got {raw!r}")
    hint_value = raw.get("id", "")
    hint = hint_value if isinstance(hint_value, str) else ""
    _check_keys(raw, _TERM_KEYS, ref, hint)
    term_id = _text(_required(raw, "id", ref, hint), "id", ref, hint)
    iri = _curie_iri(term_id, prefixes, "id", ref, hint)
    bucket = _optional_text(raw.get("bucket"), "bucket", ref, term_id)
    datatype = _optional_text(raw.get("datatype"), "datatype", ref, term_id)
    if kind == "class":
        _validate_class(bucket, datatype, ref, term_id)
    else:
        _validate_non_class(kind, bucket, datatype, ref, term_id)
    return TermSpec(
        term_id=term_id,
        iri=iri,
        kind=kind,
        label=_text(raw.get("label", term_id.partition(":")[2]), "label", ref, term_id),
        parents=_curies(raw.get("parents", []), prefixes, "parents", ref, term_id),
        bucket=bucket,
        domain_classes=_curies(raw.get("domain", []), prefixes, "domain", ref, term_id),
        range_classes=_curies(raw.get("range", []), prefixes, "range", ref, term_id),
        datatype=datatype,
        unit=_optional_text(raw.get("unit"), "unit", ref, term_id),
        transitive=_bool(raw.get("transitive", False), "transitive", ref, term_id),
        functional=_bool(raw.get("functional", False), "functional", ref, term_id),
        inverse_of=_optional_curie(
            raw.get("inverse_of"), prefixes, "inverse_of", ref, term_id
        ),
        aliases=_strings(raw.get("aliases", []), "aliases", ref, term_id),
        synonyms=_strings(raw.get("synonyms", []), "synonyms", ref, term_id),
        definition=_text(raw.get("definition", ""), "definition", ref, term_id),
        required_on=_curies(
            raw.get("required_on", []), prefixes, "required_on", ref, term_id
        ),
        origin=_choice(
            raw.get("origin", document_origin), ORIGINS, "origin", ref, term_id
        ),
        verified=_bool(raw.get("verified", True), "verified", ref, term_id),
        source=_text(raw.get("source", ref), "source", ref, term_id),
        status=_choice(
            raw.get("status", "active"),
            ("active", "deprecated"),
            "status",
            ref,
            term_id,
        ),
        replaced_by=_optional_curie(
            raw.get("replaced_by"), prefixes, "replaced_by", ref, term_id
        ),
        mappings=_mapping(raw.get("mappings", {}), "mappings", ref, term_id),
    )


def _check_duplicate(
    term: TermSpec, seen_ids: set[str], aliases: dict[tuple[str, str], str], ref: str
) -> None:
    if term.term_id in seen_ids:
        _fail(ref, term.term_id, f"duplicate term id {term.term_id!r}")
    seen_ids.add(term.term_id)
    for alias in term.aliases:
        key = (term.kind, alias.casefold())
        if key in aliases:
            _fail(
                ref,
                term.term_id,
                f"duplicate alias {alias!r} also used by {aliases[key]!r}",
            )
        aliases[key] = term.term_id


def _parse_terms(
    mapping: Mapping[str, Any], prefixes: Mapping[str, str], ref: str, origin: str
) -> tuple[TermSpec, ...]:
    terms: list[TermSpec] = []
    seen_ids: set[str] = set()
    aliases: dict[tuple[str, str], str] = {}
    for section, kind in _SECTIONS:
        entries = _list(mapping.get(section, []), section, ref)
        for raw in entries:
            term = _parse_term(raw, kind, prefixes, ref, origin)
            _check_duplicate(term, seen_ids, aliases, ref)
            terms.append(term)
    return tuple(terms)


def expand_curie(curie: str, prefixes: Mapping[str, str]) -> str:
    """The IRI of ``curie``; :class:`OntologyDocError` when it is not a CURIE
    or its prefix is not declared."""
    if not isinstance(curie, str):
        raise OntologyDocError(f"invalid CURIE {curie!r}: expected text")
    prefix, separator, local = curie.partition(":")
    if not separator or not prefix or not local:
        raise OntologyDocError(f"invalid CURIE {curie!r}: expected <prefix>:<local>")
    if prefix not in prefixes:
        raise OntologyDocError(f"invalid CURIE {curie!r}: undeclared prefix {prefix!r}")
    return prefixes[prefix] + local


def parse_doc(mapping: Mapping[str, Any], *, source: str = "") -> OntologyDoc:
    """Parse one document (``source`` names it in error messages)."""
    where = source or "<document>"
    if not isinstance(mapping, Mapping):
        _fail(where, "", f"document must be a mapping, got {mapping!r}")
    ontology = _text(_required(mapping, "ontology", where), "ontology", where)
    if re.fullmatch(r"[a-z0-9-]+", ontology) is None:
        _fail(where, "", f"invalid ontology id {ontology!r}")
    version = _version(_required(mapping, "version", where), where)
    ref = f"{ontology}@{version}"
    _check_keys(mapping, _DOC_KEYS, ref)
    prefixes = _prefixes(_required(mapping, "prefixes", ref), ref)
    origin = _choice(mapping.get("origin", "minted"), ORIGINS, "origin", ref)
    return OntologyDoc(
        ontology_id=ontology,
        version=version,
        iri_base=_text(mapping.get("iri_base", ""), "iri_base", ref),
        prefixes=prefixes,
        imports=_imports(mapping.get("imports", []), ref),
        description=_text(mapping.get("description", ""), "description", ref),
        terms=_parse_terms(mapping, prefixes, ref, origin),
        profile=_mapping(mapping.get("profile", {}), "profile", ref),
    )


def _read_bundled(path: Any) -> OntologyDoc:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise OntologyDocError(f"{path.name}: cannot read YAML: {exc}") from exc
    return parse_doc(raw, source=path.name)


def load_bundled() -> tuple[OntologyDoc, ...]:
    """Every ``*.yaml`` document of this package, parsed with
    ``yaml.safe_load`` (files read through ``importlib.resources``), sorted
    by ref. Two files with one ref raise :class:`OntologyDocError`."""
    docs: dict[str, OntologyDoc] = {}
    paths = importlib.resources.files("umrs.knowledge.ontologies").iterdir()
    for path in sorted(paths, key=lambda item: item.name):
        if not path.name.endswith(".yaml"):
            continue
        doc = _read_bundled(path)
        if doc.ref in docs:
            _fail(doc.ref, "", f"duplicate document ref in {path.name!r}")
        docs[doc.ref] = doc
    return tuple(docs[ref] for ref in sorted(docs))


def bundled_by_ref() -> dict[str, OntologyDoc]:
    """:func:`load_bundled` keyed by ref."""
    return {doc.ref: doc for doc in load_bundled()}
