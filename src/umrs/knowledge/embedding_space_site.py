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

from umrs.knowledge.embedding_spaces import EmbeddingSpace

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
    raise NotImplementedError("kn-s5-01")


def site_embedding_spaces() -> list[EmbeddingSpace]:
    """The current site's declared spaces:
    ``spaces_from_site(umrs.orchestrator.site_config.get_site_config().knowledge)``.

    Import ``get_site_config`` inside the function (the site config module
    is heavy). Errors propagate unchanged.
    """
    raise NotImplementedError("kn-s5-01")
