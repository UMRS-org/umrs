"""Embedding spaces declared in the site file (ADR-083 D-1, slice KN-S5).

``site.yaml``::

    knowledge:
      embedding_spaces:
        - space_id: me5b                       # required, SPACE_ID_PATTERN
          model_name: /models/me5b             # required, MODEL_NAME_PATTERN
          dimension: 768                       # required, int 1..16000
          model_ref: intfloat/multilingual-e5-base   # optional, default ""
          revision: "<hf commit>"              # optional, default ""
          modalities: [text]                   # optional, default [text]
          metric: cosine                       # optional, default cosine
          query_prefix: "query: "              # optional, default ""
          document_prefix: "passage: "         # optional, default ""
          locality: site                       # optional, default site
          legacy_names: [me5b]                 # optional, default []

A declared space is always ``status="active"`` and ``source="site"``:
``status`` and ``source`` are not accepted keys (retiring is an admin act,
``umrs-server admin embedding-space retire``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from umrs.knowledge.embedding_spaces import (
    CODE_INVALID,
    SOURCE_SITE,
    STATUS_ACTIVE,
    EmbeddingSpace,
    EmbeddingSpaceError,
    validate_space,
)

SITE_KEY = "embedding_spaces"
REQUIRED_KEYS = frozenset({"space_id", "model_name", "dimension"})
OPTIONAL_KEYS = frozenset(
    {
        "model_ref",
        "revision",
        "modalities",
        "metric",
        "query_prefix",
        "document_prefix",
        "locality",
        "legacy_names",
    }
)
_LIST_PREFIX = "site.yaml knowledge.embedding_spaces: "
_STRING_KEYS = (
    "space_id",
    "model_name",
    "model_ref",
    "revision",
    "metric",
    "query_prefix",
    "document_prefix",
    "locality",
)
_LIST_KEYS = ("modalities", "legacy_names")


def _invalid(prefix: str, message: str) -> EmbeddingSpaceError:
    return EmbeddingSpaceError(CODE_INVALID, prefix + message)


def _entry_prefix(index: int) -> str:
    return f"site.yaml knowledge.embedding_spaces[{index}]: "


def _check_keys(entry: Mapping[str, Any], prefix: str) -> None:
    for key in sorted(REQUIRED_KEYS):
        if key not in entry:
            raise _invalid(prefix, f"missing required key {key!r}")
    for key in entry:
        if key not in REQUIRED_KEYS | OPTIONAL_KEYS:
            raise _invalid(prefix, f"unknown key {key!r}")


def _string_tuple(value: Any, key: str, prefix: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _invalid(prefix, f"{key} must be a list of strings")
    return tuple(value)


def _entry_values(entry: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    dimension = entry["dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise _invalid(prefix, "dimension must be an int (not bool)")
    for key in _STRING_KEYS:
        if key in entry and not isinstance(entry[key], str):
            raise _invalid(prefix, f"{key} must be a string")
    values = {
        key: value
        for key, value in entry.items()
        if key in REQUIRED_KEYS | OPTIONAL_KEYS
    }
    for key in _LIST_KEYS:
        if key in values:
            values[key] = _string_tuple(values[key], key, prefix)
    return values


def _space_from_entry(entry: Any, index: int) -> EmbeddingSpace:
    prefix = _entry_prefix(index)
    if not isinstance(entry, Mapping):
        raise _invalid(prefix, "entry must be a mapping")
    _check_keys(entry, prefix)
    values = _entry_values(entry, prefix)
    try:
        space = EmbeddingSpace(**values, status=STATUS_ACTIVE, source=SOURCE_SITE)
        return validate_space(space)
    except EmbeddingSpaceError as exc:
        text = str(exc).split(": ", 1)[1]
        raise EmbeddingSpaceError(CODE_INVALID, prefix + text) from exc


def _check_duplicate(
    space: EmbeddingSpace,
    index: int,
    space_ids: set[str],
    model_names: set[str],
) -> None:
    prefix = _entry_prefix(index)
    if space.space_id in space_ids:
        raise _invalid(prefix, f"duplicate space_id {space.space_id!r}")
    if space.model_name in model_names:
        raise _invalid(prefix, f"duplicate model_name {space.model_name!r}")
    space_ids.add(space.space_id)
    model_names.add(space.model_name)


def spaces_from_site(knowledge: Mapping[str, Any] | None) -> list[EmbeddingSpace]:
    """Parse the ``knowledge`` mapping of a site file into validated spaces.

    Rules (each violation raises
    ``umrs.knowledge.embedding_spaces.EmbeddingSpaceError`` with code
    ``CODE_INVALID`` and a message that starts with
    ``site.yaml knowledge.embedding_spaces[<i>]: `` for an entry, or
    ``site.yaml knowledge.embedding_spaces: `` for the list itself):

    - ``knowledge`` None, empty, or without ``embedding_spaces`` (or the key
      set to None) → ``[]``.
    - ``knowledge`` itself not a mapping → error (list-level message).
    - ``embedding_spaces`` not a list → error (list-level message).
    - an entry that is not a mapping → error naming its index.
    - a missing required key → error naming the key; an unknown key
      (anything outside REQUIRED_KEYS | OPTIONAL_KEYS, e.g. ``status``) →
      error naming the key.
    - ``dimension`` must be an ``int`` (not bool, not a string).
    - ``modalities`` and ``legacy_names``, when given, must be lists of
      strings; they become tuples in list order.
    - string fields must be ``str`` when given.
    - the built ``EmbeddingSpace`` (status ``active``, source ``site``) is
      passed through ``validate_space``; its error is re-raised with the
      entry prefix (keep the original code ``CODE_INVALID``).
    - two entries with the same ``space_id``, or the same ``model_name``,
      → error naming the later index.

    Returns the spaces in file order.
    """
    if knowledge is None:
        return []
    if not isinstance(knowledge, Mapping):
        raise _invalid(_LIST_PREFIX, "knowledge must be a mapping")
    declared = knowledge.get(SITE_KEY)
    if declared is None:
        return []
    if not isinstance(declared, list):
        raise _invalid(_LIST_PREFIX, "embedding_spaces must be a list")

    spaces: list[EmbeddingSpace] = []
    space_ids: set[str] = set()
    model_names: set[str] = set()
    for index, entry in enumerate(declared):
        space = _space_from_entry(entry, index)
        _check_duplicate(space, index, space_ids, model_names)
        spaces.append(space)
    return spaces


def site_embedding_spaces() -> list[EmbeddingSpace]:
    """The current site's declared spaces:
    ``spaces_from_site(umrs.orchestrator.site_config.get_site_config().knowledge)``.

    Import ``get_site_config`` inside the function (the site config module
    is heavy). Errors propagate unchanged.
    """
    from umrs.orchestrator.site_config import get_site_config

    return spaces_from_site(get_site_config().knowledge)
