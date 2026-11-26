"""The writer-side refusal (ADR-083 D-1, slice KN-S5).

:func:`require_writer_space` runs on the embed writer's asyncpg connection,
inside the artifact's transaction, before any vector is written:

1. read every registry row (active and retired);
2. resolve the model name among them; an empty registry (no row at all, a
   state no migration leaves) stands for the bundled space alone, so a
   database the migration has not seeded yet keeps the shipped default;
3. on a miss, consult the site file's ``knowledge.embedding_spaces``: a
   declared space that the registry does not already hold (same
   ``space_id`` or ``model_name``, in any status — a retired row is never
   resurrected) is inserted and admitted, unless one of its names belongs
   to another row (any status), which raises ``EmbeddingSpaceError``
   (``CODE_NAME_TAKEN``) and inserts nothing;
4. otherwise refuse with :class:`UnregisteredEmbeddingModel`.

The site-file reader is passed in (``site_spaces``) and called only on a
miss, so a write under a registered model never reads the site file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    COLUMNS,
    SOURCE_SITE,
    STATUS_ACTIVE,
    TABLE,
    EmbeddingRegistryUnavailable,
    EmbeddingSpace,
    UnregisteredEmbeddingModel,
    require_names_free,
    resolve_space,
    space_from_row,
    space_to_row,
)

SELECT_SPACES_SQL = f"SELECT {', '.join(COLUMNS)} FROM {TABLE} ORDER BY space_id"
INSERT_IF_ABSENT_SQL = (
    f"INSERT INTO {TABLE} ({', '.join(COLUMNS)}) VALUES ("
    + ", ".join(f"${i}" for i in range(1, len(COLUMNS) + 1))
    + ") ON CONFLICT DO NOTHING"
)

SiteSpaces = Callable[[], Sequence[EmbeddingSpace]]


async def read_spaces(conn: Any) -> list[EmbeddingSpace]:
    """Every registry row (asyncpg ``conn``). A missing table raises
    :class:`EmbeddingRegistryUnavailable` naming the migrate verb."""
    try:
        rows = await conn.fetch(SELECT_SPACES_SQL)
    except Exception as exc:  # named by class: no asyncpg import here
        if type(exc).__name__ == "UndefinedTableError":
            raise EmbeddingRegistryUnavailable(
                f"{TABLE} is missing: run `umrs-server admin migrate`"
            ) from exc
        raise
    return [space_from_row(row) for row in rows]


def _seedable(
    declared: Sequence[EmbeddingSpace], registry: Sequence[EmbeddingSpace]
) -> list[EmbeddingSpace]:
    """Declared spaces the registry holds no row for (by id or model name)."""
    ids = {s.space_id for s in registry}
    names = {s.model_name for s in registry}
    return [s for s in declared if s.space_id not in ids and s.model_name not in names]


def _as_site_row(space: EmbeddingSpace) -> EmbeddingSpace:
    from dataclasses import replace

    return replace(space, status=STATUS_ACTIVE, source=SOURCE_SITE)


async def require_writer_space(
    conn: Any, model_name: str, *, site_spaces: SiteSpaces
) -> EmbeddingSpace:
    """The active space ``model_name`` belongs to, seeding a site-declared
    one on first use; else :class:`UnregisteredEmbeddingModel`."""
    registry = await read_spaces(conn)
    space = resolve_space(model_name, registry or [BUNDLED_SPACE])
    if space is not None:
        return space
    candidates = [_as_site_row(s) for s in _seedable(site_spaces(), registry)]
    space = resolve_space(model_name, candidates)
    if space is None:
        raise UnregisteredEmbeddingModel(model_name)
    # A declared space reusing a name another row (any status) owns is
    # refused, not seeded: the seed would make that name ambiguous for every
    # writer, or revive a retired space's model under a new id.
    require_names_free(space, registry or [BUNDLED_SPACE])
    await conn.execute(INSERT_IF_ABSENT_SQL, *space_to_row(space))
    return space
