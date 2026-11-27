"""Shared helpers for fakes that answer the knowledge vector reads (slice KN-S6).

Since KN-S6 the vector lanes (the retrieval core's dense lane and the K3
similarity lane) first read the embedding-space registry
(``SELECT ... FROM knowledge_embedding_spaces``), then run one ANN statement
whose space filter is an inline ``e.model_name IN ('a', 'b')`` list (no
``LIKE`` parameter). Fakes recognise both with the markers below and find
the space's names with :func:`filter_names`.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

REGISTRY_MARKER = "FROM knowledge_embedding_spaces"
ANN_MARKER = "AS ann_vec"
#: The K3 similarity lane's ANN read selects the node label; the dense one does not.
K3_ANN_MARKER = "n.label AS label"
_IN_LIST = re.compile(r"\bmodel_name IN \(([^)]*)\)")


def registry_row(
    space_id: str,
    model_name: str,
    dimension: int,
    *,
    legacy_names: tuple[str, ...] = (),
    status: str = "active",
) -> tuple[Any, ...]:
    """One registry row in ``embedding_spaces.COLUMNS`` order."""
    return (
        space_id,
        model_name,
        "",
        "",
        dimension,
        ["text"],
        "cosine",
        "",
        "",
        "site",
        list(legacy_names),
        status,
        "admin",
    )


def filter_names(sql: str) -> list[str]:
    """The names of the first ``model_name IN (...)`` list in ``sql``."""
    found = _IN_LIST.search(sql)
    if found is None:
        return []
    return [part.strip().strip("'") for part in found.group(1).split(",")]


def filter_space(sql: str) -> str | None:
    """The basename of the first filtered name (the fakes' space key)."""
    names = filter_names(sql)
    return PurePosixPath(names[0]).name if names else None


def like_space(params: list[Any]) -> str | None:
    """The space of a legacy ``LIKE '%/<space>'`` parameter, if any."""
    for param in params:
        if isinstance(param, str) and param.startswith("%/"):
            return param[2:].replace("\\_", "_").replace("\\%", "%")
    return None


def space_of(sql: str, params: list[Any]) -> str:
    """The space a vector read filters on: its inline list, else its LIKE."""
    space = filter_space(sql) or like_space(params)
    if space is None:
        raise AssertionError(f"vector read without a space filter: {sql[:160]}")
    return space
