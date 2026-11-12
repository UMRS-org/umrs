"""Read a LinkML schema subset into an :class:`Ontology` (ADR-116 D-4).

The server parses the YAML with PyYAML ``safe_load``; the LinkML runtime is
not a dependency. Supported:

* schema ``name`` (the ontology id), ``id`` and ``version`` (all required),
  ``description``, ``prefixes`` (optional);
* YAML anchors and aliases (``&a`` / ``*a``) are refused: an alias tree of a
  few hundred bytes expands to gigabytes once a value is turned into text;
  a document of more than ``MAX_YAML_NODES`` nodes or deeper than
  ``MAX_YAML_DEPTH`` levels is refused too;
* ``imports``: only ``linkml:types``; any other import is refused, so the
  server never fetches a remote or local file;
* ``classes``: ``is_a`` (one parent), ``aliases``, ``description``,
  ``abstract``, ``annotations.r4_bucket`` (one of the 7 node_type buckets,
  inherited along ``is_a``, default ``entity``); ``mixin`` / ``mixins`` are
  refused;
* ``slots`` and class ``attributes`` whose ``range`` is a class become
  relation types with (domain, range) signatures. The domain is the slot's
  ``domain``, else each class that lists the slot under ``slots``, else any
  class (``""``); an attribute's domain is its class. ``transitive``,
  ``aliases`` and ``description`` are kept. Slots whose range is a type or
  an enum are skipped.

Class and relation names are normalized to lowercase snake_case.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server`` (foundation module).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, NoReturn

from .ontology import (
    NODE_TYPE_BUCKETS,
    Ontology,
    normalize_term,
    ontology_from_dict,
)

MAX_SOURCE_BYTES = 256 * 1024
MAX_TERMS = 2000
MAX_ALIASES_PER_TERM = 50
# A schema of MAX_TERMS terms with aliases and descriptions stays far below
# both; they bound composing a hostile document (no aliases to expand).
MAX_YAML_NODES = 100_000
MAX_YAML_DEPTH = 32
ALLOWED_IMPORTS = frozenset({"linkml:types"})
DEFAULT_BUCKET = "entity"


class OntologySourceError(ValueError):
    """The registered source is not an ontology R4 accepts."""


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,31}$")


@dataclass
class _Class:
    name: str
    parent: str | None
    aliases: tuple[str, ...]
    description: str
    abstract: bool
    bucket: str | None


@dataclass
class _Relation:
    name: str
    signatures: set[tuple[str, str]] = field(default_factory=set)
    transitive: bool = False
    aliases: list[str] = field(default_factory=list)
    description: str = ""


def linkml_to_ontology(yaml_text: str) -> Ontology:
    """Parse LinkML YAML text into an Ontology; ValueError names what is wrong."""
    doc = _load_yaml(yaml_text)
    ontology_id, version = _schema_coordinate(doc)
    _check_imports(doc.get("imports"))
    raw_classes = _mapping_or_empty(doc.get("classes"), "classes")
    if not raw_classes:
        raise OntologySourceError("LinkML schema declares no classes")
    _check_term_count(doc, raw_classes)
    classes = _parse_classes(raw_classes)
    relations = _parse_relations(doc, raw_classes, classes)
    return check_limits(_build_ontology(doc, ontology_id, version, classes, relations))


def check_limits(ontology: Ontology) -> Ontology:
    """The ontology, unless it has more than ``MAX_TERMS`` classes and
    relations or a term with more than ``MAX_ALIASES_PER_TERM`` aliases
    (both registration formats meet this)."""
    total = len(ontology.entity_types) + len(ontology.relation_types)
    if total > MAX_TERMS:
        raise OntologySourceError(
            f"ontology declares {total} terms (limit {MAX_TERMS})"
        )
    crowded = sorted(
        t for t, names in ontology.aliases if len(names) > MAX_ALIASES_PER_TERM
    )
    if crowded:
        raise OntologySourceError(
            f"terms with more than {MAX_ALIASES_PER_TERM} aliases: {crowded[:5]}"
        )
    return ontology


@lru_cache(maxsize=1)
def _no_alias_loader() -> type:
    """A SafeLoader that refuses YAML anchors and aliases (billion-laughs
    expansion) and caps the node count and the nesting depth, so composing
    stays bounded work and never recurses past ``MAX_YAML_DEPTH``."""
    import yaml

    def refuse(problem: str, event: Any) -> NoReturn:
        raise yaml.composer.ComposerError(None, None, problem, event.start_mark)

    class NoAliasLoader(yaml.SafeLoader):
        _depth = 0
        _nodes = 0

        def compose_node(self, parent: Any, index: Any) -> Any:
            event = self.peek_event()
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                refuse("YAML anchors and aliases (&name, *name) are refused", event)
            self._nodes += 1
            if self._nodes > MAX_YAML_NODES:
                refuse(
                    f"LinkML source has more than {MAX_YAML_NODES} YAML nodes", event
                )
            if self._depth >= MAX_YAML_DEPTH:
                refuse(
                    f"LinkML source nests deeper than {MAX_YAML_DEPTH} levels", event
                )
            self._depth += 1
            try:
                return super().compose_node(parent, index)
            finally:
                self._depth -= 1

    return NoAliasLoader


def _parse_yaml(yaml_text: str) -> Any:
    import yaml

    try:
        return yaml.load(yaml_text, Loader=_no_alias_loader())
    except yaml.YAMLError as exc:
        raise OntologySourceError(f"LinkML source is not valid YAML: {exc}") from exc
    except RecursionError as exc:
        raise OntologySourceError("LinkML source is nested too deeply") from exc


def _load_yaml(yaml_text: Any) -> Mapping[str, Any]:
    if not isinstance(yaml_text, str):
        raise OntologySourceError("LinkML source must be text")
    if len(yaml_text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise OntologySourceError(
            f"LinkML source is larger than {MAX_SOURCE_BYTES} bytes"
        )
    doc = _parse_yaml(yaml_text)
    if not isinstance(doc, Mapping):
        raise OntologySourceError(
            "LinkML source must be a YAML mapping (a schema document)"
        )
    return doc


def _text(value: Any, label: str) -> str:
    """A scalar as stripped text ("" for None); anything else is refused
    before ``str()`` could turn a large structure into text."""
    if value is None:
        return ""
    if not isinstance(value, (str, int, float)):
        raise OntologySourceError(f"{label} must be text, not {type(value).__name__}")
    return str(value).strip()


def _schema_coordinate(doc: Mapping[str, Any]) -> tuple[str, str]:
    fields = {
        key: _text(doc.get(key), f"schema {key!r}") for key in ("name", "id", "version")
    }
    missing = [key for key, value in fields.items() if not value]
    if missing:
        raise OntologySourceError(
            f"LinkML schema is missing required field(s) {missing}: R4 needs "
            "'name' (the ontology id), 'id' and 'version'"
        )
    validate_coordinate(fields["name"], fields["version"])
    return fields["name"], fields["version"]


def validate_coordinate(ontology_id: str, version: str) -> None:
    """ValueError unless the id and version are safe registry coordinates."""
    if not _ID_RE.match(ontology_id):
        raise OntologySourceError(
            f"ontology id {ontology_id!r} must be 1-64 letters, digits, '.', '_' or '-'"
        )
    if not _VERSION_RE.match(version):
        raise OntologySourceError(
            f"ontology version {version!r} must be 1-32 letters, digits, '.', '_', '+' or '-'"
        )


def _check_imports(value: Any) -> None:
    if value is None:
        return
    imports = value if isinstance(value, list) else [value]
    refused = sorted({str(item) for item in imports} - ALLOWED_IMPORTS)
    if refused:
        raise OntologySourceError(
            f"imports {refused} are refused: only 'linkml:types' is allowed "
            "(the server never fetches remote or local schema files)"
        )


def _mapping_or_empty(value: Any, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise OntologySourceError(f"'{label}' must be a mapping of name to definition")
    return value


def _body(value: Any, label: str) -> Mapping[str, Any]:
    """A class/slot definition body; a bare name (None) means an empty one."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise OntologySourceError(f"{label} must be a mapping")
    return value


def _check_term_count(doc: Mapping[str, Any], raw_classes: Mapping[str, Any]) -> None:
    slots = _mapping_or_empty(doc.get("slots"), "slots")
    attributes = sum(
        len(_body(body, f"class {name!r}").get("attributes") or {})
        for name, body in raw_classes.items()
    )
    total = len(raw_classes) + len(slots) + attributes
    if total > MAX_TERMS:
        raise OntologySourceError(
            f"LinkML schema declares {total} terms (limit {MAX_TERMS})"
        )


def _term_name(raw: Any, kind: str) -> str:
    name = normalize_term(raw)
    if not name:
        raise OntologySourceError(f"{kind} name {raw!r} has no letters or digits")
    return name


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, list) else [value]
    if not all(isinstance(item, (str, int, float)) for item in items):
        raise OntologySourceError(f"{label} must be a list of strings")
    return tuple(str(item).strip() for item in items if str(item).strip())


# ── classes ──────────────────────────────────────────────────────────────


def _parse_classes(raw_classes: Mapping[str, Any]) -> dict[str, _Class]:
    classes: dict[str, _Class] = {}
    for raw_name, raw_body in raw_classes.items():
        name = _term_name(raw_name, "class")
        if name in classes:
            raise OntologySourceError(
                f"class {raw_name!r} collides with another class as {name!r}"
            )
        classes[name] = _class_from_body(name, _body(raw_body, f"class {raw_name!r}"))
    _check_lattice(classes)
    for cls in classes.values():
        cls.bucket = _inherited_bucket(cls.name, classes)
    return classes


def _class_from_body(name: str, body: Mapping[str, Any]) -> _Class:
    if body.get("mixin") or body.get("mixins"):
        raise OntologySourceError(
            f"class {name!r} uses mixins; R4 ontologies support single "
            "inheritance (is_a) only"
        )
    parent = body.get("is_a")
    return _Class(
        name=name,
        parent=_term_name(parent, "is_a") if parent else None,
        aliases=_string_list(body.get("aliases"), f"class {name!r} aliases"),
        description=_text(body.get("description"), f"class {name!r} description"),
        abstract=bool(body.get("abstract")),
        bucket=_bucket_annotation(body.get("annotations"), name),
    )


def _bucket_annotation(annotations: Any, name: str) -> str | None:
    if not isinstance(annotations, Mapping):
        return None
    value = annotations.get("r4_bucket")
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return None
    bucket = _text(value, f"class {name!r} r4_bucket").lower()
    if bucket not in NODE_TYPE_BUCKETS:
        raise OntologySourceError(
            f"class {name!r}: annotations.r4_bucket {value!r} is not one of "
            f"{sorted(NODE_TYPE_BUCKETS)}"
        )
    return bucket


def _check_lattice(classes: Mapping[str, _Class]) -> None:
    for cls in classes.values():
        if cls.parent is not None and cls.parent not in classes:
            raise OntologySourceError(
                f"class {cls.name!r}: is_a {cls.parent!r} is not a declared class"
            )
    for name in classes:
        _ancestors(name, classes)


def _ancestors(name: str, classes: Mapping[str, _Class]) -> list[str]:
    """The is_a chain above ``name``; ValueError on a cycle."""
    chain: list[str] = []
    current = classes[name].parent
    while current is not None:
        if current == name or current in chain:
            raise OntologySourceError(
                f"class {name!r}: is_a chain forms a cycle at {current!r}"
            )
        chain.append(current)
        current = classes[current].parent
    return chain


def _inherited_bucket(name: str, classes: Mapping[str, _Class]) -> str:
    for candidate in [name, *_ancestors(name, classes)]:
        bucket = classes[candidate].bucket
        if bucket:
            return bucket
    return DEFAULT_BUCKET


# ── relations ────────────────────────────────────────────────────────────


def _class_range(body: Mapping[str, Any], classes: Mapping[str, _Class]) -> str | None:
    """The normalized range when it names a class; None for types and enums."""
    raw = body.get("range")
    if not raw:
        return None
    name = normalize_term(raw)
    return name if name in classes else None


def _slot_users(raw_classes: Mapping[str, Any]) -> dict[str, list[str]]:
    """slot name -> classes that list it under ``slots``."""
    users: dict[str, list[str]] = {}
    for raw_name, raw_body in raw_classes.items():
        body = _body(raw_body, f"class {raw_name!r}")
        cls = normalize_term(raw_name)
        for slot in _string_list(body.get("slots"), f"class {raw_name!r} slots"):
            users.setdefault(normalize_term(slot), []).append(cls)
    return users


def _parse_relations(
    doc: Mapping[str, Any],
    raw_classes: Mapping[str, Any],
    classes: Mapping[str, _Class],
) -> dict[str, _Relation]:
    relations: dict[str, _Relation] = {}
    users = _slot_users(raw_classes)
    for raw_name, raw_body in _mapping_or_empty(doc.get("slots"), "slots").items():
        name = _term_name(raw_name, "slot")
        body = _body(raw_body, f"slot {raw_name!r}")
        domains = _slot_domains(body, users.get(name), classes, name)
        _add_relation(relations, name, body, domains, classes)
    for raw_class, raw_body in raw_classes.items():
        owner = normalize_term(raw_class)
        attributes = _mapping_or_empty(
            _body(raw_body, f"class {raw_class!r}").get("attributes"), "attributes"
        )
        for raw_name, attr_body in attributes.items():
            name = _term_name(raw_name, "attribute")
            body = _body(attr_body, f"attribute {raw_name!r}")
            _add_relation(relations, name, body, [owner], classes)
    return relations


def _slot_domains(
    body: Mapping[str, Any],
    users: list[str] | None,
    classes: Mapping[str, _Class],
    name: str,
) -> list[str]:
    domain = body.get("domain")
    if not domain:
        return list(users or [""])
    normalized = normalize_term(domain)
    if normalized not in classes:
        raise OntologySourceError(
            f"slot {name!r}: domain {domain!r} is not a declared class"
        )
    return [normalized]


def _add_relation(
    relations: dict[str, _Relation],
    name: str,
    body: Mapping[str, Any],
    domains: list[str],
    classes: Mapping[str, _Class],
) -> None:
    range_class = _class_range(body, classes)
    if range_class is None:
        return
    relation = relations.setdefault(name, _Relation(name=name))
    relation.signatures.update((domain, range_class) for domain in domains)
    relation.transitive = relation.transitive or bool(body.get("transitive"))
    for alias in _string_list(body.get("aliases"), f"slot {name!r} aliases"):
        if alias not in relation.aliases:
            relation.aliases.append(alias)
    relation.description = relation.description or _text(
        body.get("description"), f"slot {name!r} description"
    )


# ── assembly ─────────────────────────────────────────────────────────────


def _prefixes(value: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for prefix, target in _mapping_or_empty(value, "prefixes").items():
        if isinstance(target, Mapping):
            target = target.get("prefix_reference")
        if target:
            out[_text(prefix, "prefix")] = _text(target, f"prefix {prefix!r}")
    return out


def _term_aliases(
    classes: Mapping[str, _Class], relations: Mapping[str, _Relation]
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for cls in classes.values():
        if cls.aliases:
            aliases.setdefault(cls.name, []).extend(cls.aliases)
    for rel in relations.values():
        if rel.aliases:
            aliases.setdefault(rel.name, []).extend(rel.aliases)
    return aliases


def _term_descriptions(
    classes: Mapping[str, _Class], relations: Mapping[str, _Relation]
) -> dict[str, str]:
    descriptions = {c.name: c.description for c in classes.values() if c.description}
    for rel in relations.values():
        if rel.description:
            descriptions.setdefault(rel.name, rel.description)
    return descriptions


def _class_fields(classes: Mapping[str, _Class]) -> dict[str, Any]:
    return {
        "entity_types": sorted(classes),
        "subtype_of": {c.name: c.parent for c in classes.values() if c.parent},
        "class_buckets": {c.name: c.bucket or DEFAULT_BUCKET for c in classes.values()},
        "abstract_types": sorted(c.name for c in classes.values() if c.abstract),
    }


def _relation_fields(relations: Mapping[str, _Relation]) -> dict[str, Any]:
    return {
        "relation_types": sorted(relations),
        "transitive_relations": sorted(
            r.name for r in relations.values() if r.transitive
        ),
        "relation_signatures": [
            {"relation": rel.name, "domain": dom, "range": rng}
            for rel in relations.values()
            for dom, rng in sorted(rel.signatures)
        ],
    }


def _build_ontology(
    doc: Mapping[str, Any],
    ontology_id: str,
    version: str,
    classes: Mapping[str, _Class],
    relations: Mapping[str, _Relation],
) -> Ontology:
    spec = {
        "ontology_id": ontology_id,
        "version": version,
        "description": _text(doc.get("description"), "schema 'description'"),
        **_class_fields(classes),
        **_relation_fields(relations),
        "aliases": _term_aliases(classes, relations),
        "descriptions": _term_descriptions(classes, relations),
        "prefixes": _prefixes(doc.get("prefixes")),
        "source_format": "linkml",
    }
    # The strict reader checks every cross-reference by the same rules a
    # JSON registration meets.
    return ontology_from_dict(spec)


__all__ = [
    "ALLOWED_IMPORTS",
    "MAX_ALIASES_PER_TERM",
    "MAX_SOURCE_BYTES",
    "MAX_TERMS",
    "MAX_YAML_DEPTH",
    "MAX_YAML_NODES",
    "OntologySourceError",
    "check_limits",
    "linkml_to_ontology",
    "validate_coordinate",
]
