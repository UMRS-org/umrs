"""The embedding-space registry (ADR-083 D-1, decision KN-3, slice KN-S5).

A **space** is one encoder's vector space: every vector in
``knowledge_node_embeddings`` carries the ``model_name`` it was encoded
under, and the registry (``knowledge_embedding_spaces``) says which names
belong to which space. Writers refuse a vector whose model name resolves to
no active space, so two models never share a space by accident.

Where spaces come from, in both deploy modes (hosted fronts and a
self-hosted R4 + UMRS):

- the migration seeds ONE row, the bundled e5-base-v2 text space
  (:data:`BUNDLED_SPACE`), with the names sites stage it under, so a
  stand-alone install has an active space with no admin step;
- the site file's ``knowledge.embedding_spaces`` declares more
  (``embedding_space_site``); the writer seeds a declared space on its
  first write, and ``umrs-server admin embedding-space seed-site`` seeds
  them all;
- an admin registers, retires and re-activates spaces with
  ``umrs-server admin embedding-space`` (``umrs.cli.admin.embedding_spaces``).

Name resolution (:func:`resolve_space`): a name matches a space when it
equals the space's ``model_name`` or one of its ``legacy_names`` (after
trimming whitespace and trailing slashes); a **bare** registered name (one
without ``/``) also matches any path whose last segment is that name, the
convention sites already stage models by (``/models/e5-base-v2``,
``/opt/umrs/models/e5-base-v2`` and ``intfloat/e5-base-v2`` are one space).
Exact matches win over basename matches; two different spaces matching at
the same step is an error, never a guess.

Identifier safety: ``space_id`` becomes part of an index name
(``hnsw_<space_id>``, slice KN-S6) and ``model_name`` part of a partial
index predicate, so both are restricted to the character sets
:data:`SPACE_ID_PATTERN` and :data:`MODEL_NAME_PATTERN` here and by CHECK
constraints in the migration.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from umrs.knowledge.text_embedder import (
    BUNDLED_TEXT_EMBEDDER_DIM,
    BUNDLED_TEXT_EMBEDDER_DIR,
    BUNDLED_TEXT_EMBEDDER_REPO,
    BUNDLED_TEXT_EMBEDDER_REVISION,
    embedding_space,
)

TABLE = "knowledge_embedding_spaces"

# Same patterns as the migration's CHECK constraints (a test compares them).
SPACE_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{0,47}$"
MODEL_NAME_PATTERN = r"^[A-Za-z0-9._/:@+-]{1,255}$"
_SPACE_ID_RE = re.compile(SPACE_ID_PATTERN)
_MODEL_NAME_RE = re.compile(MODEL_NAME_PATTERN)

STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"
VALID_STATUSES = frozenset({STATUS_ACTIVE, STATUS_RETIRED})
VALID_MODALITIES = frozenset({"text", "image", "video", "audio", "volume"})
VALID_METRICS = frozenset({"cosine"})
VALID_LOCALITIES = frozenset({"site"})
SOURCE_BUNDLED = "bundled"
SOURCE_SITE = "site"
SOURCE_ADMIN = "admin"
VALID_SOURCES = frozenset({SOURCE_BUNDLED, SOURCE_SITE, SOURCE_ADMIN})
MAX_DIMENSION = 16000
MAX_PREFIX_CHARS = 64

# Error codes (named refusals; the writer's error message starts with one).
CODE_INVALID = "knowledge.embedding_space_invalid"
CODE_AMBIGUOUS = "knowledge.embedding_space_ambiguous"
CODE_UNREGISTERED = "knowledge.embedding_space_unregistered"
CODE_EXISTS = "knowledge.embedding_space_exists"
CODE_NAME_TAKEN = "knowledge.embedding_space_name_taken"
CODE_REGISTRY_UNAVAILABLE = "knowledge.embedding_space_registry_unavailable"

# Column order of every SELECT/INSERT on the registry (``space_to_row``
# returns values in this order; ``space_from_row`` reads these keys).
COLUMNS: tuple[str, ...] = (
    "space_id",
    "model_name",
    "model_ref",
    "revision",
    "dimension",
    "modalities",
    "metric",
    "query_prefix",
    "document_prefix",
    "locality",
    "legacy_names",
    "status",
    "source",
)


class EmbeddingSpaceError(ValueError):
    """An invalid space, an ambiguous name or a registry conflict.

    ``code`` is one of the ``CODE_*`` constants; ``str(exc)`` starts with it.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class UnregisteredEmbeddingModel(RuntimeError):
    """A writer was asked to store vectors of a model no active space owns.

    Not ``permanent`` (the embed job's failure record): registering the
    space and re-running the embed repairs it, no re-ingest needed.
    """

    code = CODE_UNREGISTERED
    permanent = False

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"{CODE_UNREGISTERED}: no active embedding space owns model "
            f"{model_name!r}; declare it in site.yaml knowledge.embedding_spaces "
            "or run `umrs-server admin embedding-space register` "
            "(ADR-083 D-1)"
        )
        self.model_name = model_name


class EmbeddingRegistryUnavailable(RuntimeError):
    """The registry table cannot be read (not migrated, or the read failed)."""

    code = CODE_REGISTRY_UNAVAILABLE
    permanent = False


@dataclass(frozen=True)
class EmbeddingSpace:
    """One registry row. ``index_name`` is derived, never stored apart."""

    space_id: str
    model_name: str
    dimension: int
    model_ref: str = ""
    revision: str = ""
    modalities: tuple[str, ...] = ("text",)
    metric: str = "cosine"
    query_prefix: str = ""
    document_prefix: str = ""
    locality: str = "site"
    legacy_names: tuple[str, ...] = field(default_factory=tuple)
    status: str = STATUS_ACTIVE
    source: str = SOURCE_ADMIN

    @property
    def index_name(self) -> str:
        """The per-space partial HNSW index name (ADR-083 D-2, slice KN-S6)."""
        return f"hnsw_{self.space_id}"

    @property
    def active(self) -> bool:
        return self.status == STATUS_ACTIVE


# The bundled text space: the migration seeds exactly this row.
BUNDLED_SPACE_ID = "e5_base_v2"
BUNDLED_SPACE = EmbeddingSpace(
    space_id=BUNDLED_SPACE_ID,
    model_name=BUNDLED_TEXT_EMBEDDER_DIR,
    dimension=BUNDLED_TEXT_EMBEDDER_DIM,
    model_ref=BUNDLED_TEXT_EMBEDDER_REPO,
    revision=BUNDLED_TEXT_EMBEDDER_REVISION,
    modalities=("text",),
    metric="cosine",
    query_prefix="query: ",
    document_prefix="passage: ",
    locality="site",
    legacy_names=(
        BUNDLED_TEXT_EMBEDDER_REPO,
        "/models/e5-base-v2",
        embedding_space(BUNDLED_TEXT_EMBEDDER_DIR),
    ),
    status=STATUS_ACTIVE,
    source=SOURCE_BUNDLED,
)


# ── validation (identifier safety: implemented here, not delegated) ─────────


def _invalid(message: str) -> EmbeddingSpaceError:
    return EmbeddingSpaceError(CODE_INVALID, message)


def _check_name(name: Any, what: str) -> None:
    if not isinstance(name, str) or not _MODEL_NAME_RE.fullmatch(name):
        raise _invalid(f"{what} {name!r} must match {MODEL_NAME_PATTERN}")


def _check_identity(space: EmbeddingSpace) -> None:
    if not isinstance(space.space_id, str) or not _SPACE_ID_RE.fullmatch(
        space.space_id
    ):
        raise _invalid(f"space_id {space.space_id!r} must match {SPACE_ID_PATTERN}")
    _check_name(space.model_name, "model_name")
    for name in space.legacy_names:
        _check_name(name, "legacy name")
    dim = space.dimension
    if (
        isinstance(dim, bool)
        or not isinstance(dim, int)
        or not 0 < dim <= MAX_DIMENSION
    ):
        raise _invalid(f"dimension {dim!r} must be an integer in 1..{MAX_DIMENSION}")


def _check_one_of(value: Any, allowed: frozenset[str], what: str) -> None:
    if value not in allowed:
        raise _invalid(f"{what} {value!r} must be one of {sorted(allowed)}")


def _check_shape(space: EmbeddingSpace) -> None:
    if not space.modalities:
        raise _invalid("modalities must not be empty")
    for modality in space.modalities:
        _check_one_of(modality, VALID_MODALITIES, "modality")
    _check_one_of(space.metric, VALID_METRICS, "metric")
    _check_one_of(space.locality, VALID_LOCALITIES, "locality")
    _check_one_of(space.status, VALID_STATUSES, "status")
    _check_one_of(space.source, VALID_SOURCES, "source")


def _check_texts(space: EmbeddingSpace) -> None:
    for what in ("model_ref", "revision", "query_prefix", "document_prefix"):
        value = getattr(space, what)
        if not isinstance(value, str):
            raise _invalid(f"{what} must be a string")
        if any(ord(ch) < 32 and ch != " " for ch in value):
            raise _invalid(f"{what} must not contain control characters")
    for what in ("query_prefix", "document_prefix"):
        if len(getattr(space, what)) > MAX_PREFIX_CHARS:
            raise _invalid(f"{what} is longer than {MAX_PREFIX_CHARS} characters")


def validate_space(space: EmbeddingSpace) -> EmbeddingSpace:
    """Return ``space`` unchanged, or raise ``EmbeddingSpaceError`` (code
    :data:`CODE_INVALID`) naming the first bad field."""
    _check_identity(space)
    _check_shape(space)
    _check_texts(space)
    return space


# ── names and resolution ────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Trim whitespace and trailing slashes (``"/m/e5/ "`` → ``"/m/e5"``)."""
    return str(name or "").strip().rstrip("/")


def space_names(space: EmbeddingSpace) -> tuple[str, ...]:
    """The space's ``model_name`` then its ``legacy_names``, normalized,
    without empties or duplicates, first occurrence kept."""
    seen: dict[str, None] = {}
    for name in (space.model_name, *space.legacy_names):
        norm = normalize_name(name)
        if norm:
            seen.setdefault(norm, None)
    return tuple(seen)


def matches_exactly(space: EmbeddingSpace, name: str) -> bool:
    """``name`` (normalized) is one of :func:`space_names`. Status ignored."""
    norm = normalize_name(name)
    return bool(norm) and norm in space_names(space)


def matches_by_basename(space: EmbeddingSpace, name: str) -> bool:
    """A bare registered name (no ``/``) equals ``name``'s last path
    segment (:func:`umrs.knowledge.text_embedder.embedding_space`).
    Status ignored."""
    norm = normalize_name(name)
    if not norm:
        return False
    base = embedding_space(norm)
    return any("/" not in bare and bare == base for bare in space_names(space))


def matches(space: EmbeddingSpace, name: str) -> bool:
    """Exact or basename match (status ignored)."""
    return matches_exactly(space, name) or matches_by_basename(space, name)


def shared_name(space: EmbeddingSpace, other: EmbeddingSpace) -> str | None:
    """The first name two spaces both answer to (exact or basename, status
    ignored), or None. Two rows sharing a name make it ambiguous forever."""
    for name in space_names(space):
        if matches(other, name):
            return name
    for name in space_names(other):
        if matches(space, name):
            return name
    return None


def require_names_free(space: EmbeddingSpace, others: Iterable[EmbeddingSpace]) -> None:
    """Refuse ``space`` when a name of it is owned by another space (any
    status, a different ``space_id``): ``EmbeddingSpaceError``
    (:data:`CODE_NAME_TAKEN`) naming both ids and the name."""
    for other in others:
        if other.space_id == space.space_id:
            continue
        name = shared_name(space, other)
        if name is not None:
            raise EmbeddingSpaceError(
                CODE_NAME_TAKEN,
                f"name {name!r} is shared by spaces {space.space_id!r} "
                f"and {other.space_id!r}",
            )


def _single(found: Sequence[EmbeddingSpace], name: str) -> EmbeddingSpace | None:
    ids = sorted({s.space_id for s in found})
    if len(ids) > 1:
        raise EmbeddingSpaceError(
            CODE_AMBIGUOUS, f"model {name!r} matches spaces {ids}"
        )
    return found[0] if found else None


def resolve_space(name: str, spaces: Iterable[EmbeddingSpace]) -> EmbeddingSpace | None:
    """The ACTIVE space ``name`` denotes, or None.

    Exact matches first; only when none, basename matches. More than one
    distinct space at the deciding step raises ``EmbeddingSpaceError``
    (:data:`CODE_AMBIGUOUS`). Retired spaces never resolve."""
    active = [s for s in spaces if s.active]
    exact = _single([s for s in active if matches_exactly(s, name)], name)
    if exact is not None:
        return exact
    return _single([s for s in active if matches_by_basename(s, name)], name)


def require_space(name: str, spaces: Iterable[EmbeddingSpace]) -> EmbeddingSpace:
    """:func:`resolve_space`, or ``UnregisteredEmbeddingModel``."""
    space = resolve_space(name, spaces)
    if space is None:
        raise UnregisteredEmbeddingModel(normalize_name(name) or str(name))
    return space


# ── rows ────────────────────────────────────────────────────────────────────


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise _invalid(f"expected a list of strings, got {value!r}")
    return tuple(str(v) for v in value)


def space_from_row(row: Mapping[str, Any]) -> EmbeddingSpace:
    """A registry row (psycopg2 RealDict, asyncpg Record or plain dict keyed
    by :data:`COLUMNS`) as a validated :class:`EmbeddingSpace`. Array
    columns become tuples; NULL text columns become ``""``."""

    def text(key: str, default: str = "") -> str:
        value = row[key]
        return default if value is None else str(value)

    return validate_space(
        EmbeddingSpace(
            space_id=text("space_id"),
            model_name=text("model_name"),
            dimension=int(row["dimension"]),
            model_ref=text("model_ref"),
            revision=text("revision"),
            modalities=_text_tuple(row["modalities"]),
            metric=text("metric", "cosine"),
            query_prefix=text("query_prefix"),
            document_prefix=text("document_prefix"),
            locality=text("locality", "site"),
            legacy_names=_text_tuple(row["legacy_names"]),
            status=text("status", STATUS_ACTIVE),
            source=text("source", SOURCE_ADMIN),
        )
    )


def space_to_row(space: EmbeddingSpace) -> tuple[Any, ...]:
    """Values in :data:`COLUMNS` order (arrays as lists), after validation."""
    validate_space(space)
    return (
        space.space_id,
        space.model_name,
        space.model_ref,
        space.revision,
        space.dimension,
        list(space.modalities),
        space.metric,
        space.query_prefix,
        space.document_prefix,
        space.locality,
        list(space.legacy_names),
        space.status,
        space.source,
    )


# ── index and filter SQL (ADR-083 D-2, slice KN-S6) ─────────────────────────
#
# One partial expression HNSW per space on the untyped 0075 column:
#
#   CREATE INDEX CONCURRENTLY IF NOT EXISTS hnsw_<space_id>
#     ON knowledge_node_embeddings
#     USING hnsw (((embedding)::vector(<D>)) vector_cosine_ops)
#     WHERE model_name IN ('<name>', ...)
#
# The predicate lists the space's names (``space_names``: model_name, then
# the legacy names). A read carries the same predicate, built by the same
# function (:func:`space_filter_sql`), so the planner can prove it and use
# the index. A row stored under another path that only basename-matches the
# space is outside both; ``uncovered_names`` lists such paths so an admin can
# register them as legacy names (and rebuild the index).
#
# Names are inlined as SQL literals: every name passed the MODEL_NAME_PATTERN
# check (no quote, no backslash, no whitespace), and :func:`sql_literal`
# re-checks it, so no value reaches DDL or a read unvalidated.

HNSW_OPCLASS = "vector_cosine_ops"
INDEX_COMMENT_PREFIX = "umrs-space:v1"
_ALIAS_RE = re.compile(r"^[a-z_][a-z0-9_]{0,31}$")


def sql_literal(name: str) -> str:
    """``'name'`` for a name that matches :data:`MODEL_NAME_PATTERN`; anything
    else raises ``EmbeddingSpaceError`` (:data:`CODE_INVALID`)."""
    _check_name(name, "name")
    return "'" + name + "'"


def _names_sql(space: EmbeddingSpace) -> str:
    names = space_names(space)
    if not names:
        raise _invalid(f"space {space.space_id!r} has no name")
    return ", ".join(sql_literal(name) for name in names)


def index_predicate_sql(space: EmbeddingSpace) -> str:
    """The partial-index predicate: ``model_name IN ('a', 'b')``."""
    validate_space(space)
    return f"model_name IN ({_names_sql(space)})"


def space_filter_sql(space: EmbeddingSpace, alias: str = "e") -> str:
    """The read-side twin of :func:`index_predicate_sql` over ``alias``:
    ``e.model_name IN ('a', 'b')``. Every vector read of the knowledge read
    lanes carries a model filter, and the ANN reads carry exactly this one
    (the KN-S6 fence test)."""
    if not _ALIAS_RE.fullmatch(alias):
        raise _invalid(f"alias {alias!r} is not a plain SQL identifier")
    return f"{alias}.{index_predicate_sql(space)}"


def vector_cast_sql(space: EmbeddingSpace, expr: str) -> str:
    """``(<expr>)::vector(<D>)``, the index expression's cast. ``expr`` is
    caller-written SQL (a column or a placeholder), never a value."""
    validate_space(space)
    return f"({expr})::vector({int(space.dimension)})"


def hnsw_index_ddl(space: EmbeddingSpace, *, concurrently: bool = True) -> str:
    """``CREATE INDEX [CONCURRENTLY] IF NOT EXISTS hnsw_<id> ...`` for one
    space. CONCURRENTLY cannot run inside a transaction block."""
    how = "CONCURRENTLY " if concurrently else ""
    return (
        f"CREATE INDEX {how}IF NOT EXISTS {space.index_name} "
        "ON knowledge_node_embeddings USING hnsw "
        f"(({vector_cast_sql(space, 'embedding')}) {HNSW_OPCLASS}) "
        f"WHERE {index_predicate_sql(space)}"
    )


def drop_index_ddl(space: EmbeddingSpace, *, concurrently: bool = True) -> str:
    """``DROP INDEX [CONCURRENTLY] IF EXISTS hnsw_<id>``."""
    validate_space(space)
    how = "CONCURRENTLY " if concurrently else ""
    return f"DROP INDEX {how}IF EXISTS {space.index_name}"


@dataclass(frozen=True)
class IndexState:
    """What the database holds under a space's index name.

    ``exists``: a relation named ``hnsw_<space_id>`` exists; ``valid``: its
    ``pg_index.indisvalid`` (False after a failed CONCURRENTLY build);
    ``comment``: its ``obj_description`` (None when it has none)."""

    exists: bool = False
    valid: bool = False
    comment: str | None = None


def index_signature(space: EmbeddingSpace) -> str:
    """The comment a current index of ``space`` carries (task kn-s6-01).

    Exactly ``f"{INDEX_COMMENT_PREFIX} space={space.space_id} "
    f"dim={space.dimension} names={','.join(space_names(space))}"``,
    after ``validate_space(space)`` (an invalid space raises
    ``EmbeddingSpaceError``). Example for the bundled space:
    ``umrs-space:v1 space=e5_base_v2 dim=768 names=/opt/umrs/models/e5-base-v2,``
    ``intfloat/e5-base-v2,/models/e5-base-v2,e5-base-v2`` (one line)."""
    validate_space(space)
    return (
        f"{INDEX_COMMENT_PREFIX} space={space.space_id} "
        f"dim={space.dimension} names={','.join(space_names(space))}"
    )


def comment_index_sql(space: EmbeddingSpace) -> str:
    """``f"COMMENT ON INDEX {space.index_name} IS '{index_signature(space)}'"``
    (task kn-s6-01). The signature holds no quote by construction."""
    return f"COMMENT ON INDEX {space.index_name} IS '{index_signature(space)}'"


def index_actions(space: EmbeddingSpace, state: IndexState) -> tuple[str, ...]:
    """The statements that bring ``state`` to the space's index, in order
    (task kn-s6-01). All use the CONCURRENTLY forms (the defaults).

    - ``space.active`` and ``state.exists and state.valid and
      state.comment == index_signature(space)``: ``()`` (current);
    - ``space.active`` and ``state.exists`` otherwise (invalid build, or a
      stale/missing comment, e.g. after legacy names changed):
      ``(drop_index_ddl(space), hnsw_index_ddl(space), comment_index_sql(space))``;
    - ``space.active`` and not ``state.exists``:
      ``(hnsw_index_ddl(space), comment_index_sql(space))``;
    - retired space: ``(drop_index_ddl(space),)`` when ``state.exists``,
      else ``()``."""
    if not space.active:
        return (drop_index_ddl(space),) if state.exists else ()
    if state.exists and state.valid and state.comment == index_signature(space):
        return ()
    if state.exists:
        return (
            drop_index_ddl(space),
            hnsw_index_ddl(space),
            comment_index_sql(space),
        )
    return hnsw_index_ddl(space), comment_index_sql(space)


def uncovered_names(
    space: EmbeddingSpace, stored_names: Iterable[str]
) -> tuple[str, ...]:
    """Stored ``model_name`` values that belong to ``space`` but sit outside
    its index predicate (task kn-s6-01).

    For each name of ``stored_names`` in order: keep ``normalize_name(name)``
    when ``matches(space, name)`` is True and the normalized name is NOT in
    ``space_names(space)``. No duplicates (first occurrence kept); empty
    names are skipped. Status is ignored."""
    covered = space_names(space)
    uncovered: dict[str, None] = {}
    for name in stored_names:
        normalized = normalize_name(name)
        if not normalized:
            continue
        if matches(space, name) and normalized not in covered:
            uncovered.setdefault(normalized, None)
    return tuple(uncovered)
