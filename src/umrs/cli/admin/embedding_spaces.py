"""``umrs-server admin embedding-space`` (ADR-083 D-1, slice KN-S5).

Spaces are registered by an admin (system tier). The verbs talk to the
database the server reads (``cascade.cli_db_pool``), one transaction per
command::

    umrs-server admin embedding-space list [--all] [--json]
    umrs-server admin embedding-space register --space-id me5b \\
        --model-name /models/me5b --dimension 768 \\
        [--model-ref intfloat/multilingual-e5-base] [--revision SHA] \\
        [--modality text ...] [--query-prefix 'query: '] \\
        [--document-prefix 'passage: '] [--legacy-name NAME ...] [--replace]
    umrs-server admin embedding-space retire SPACE_ID
    umrs-server admin embedding-space activate SPACE_ID
    umrs-server admin embedding-space seed-site [--dry-run]
    umrs-server admin embedding-space index [--space SPACE_ID ...] [--dry-run]

Handlers call the store and site modules through their module objects
(``embedding_space_store.list_spaces(...)``,
``embedding_space_site.site_embedding_spaces()``) so tests can patch them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Callable, Iterator
from typing import Any

from umrs.knowledge import embedding_space_site, embedding_space_store
from umrs.knowledge.embedding_spaces import (
    COLUMNS,
    SOURCE_ADMIN,
    STATUS_ACTIVE,
    STATUS_RETIRED,
    EmbeddingSpace,
    EmbeddingSpaceError,
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_NO_DB = 2


class RegistryUnavailable(RuntimeError):
    """No site database is configured for the admin CLI."""


@contextlib.contextmanager
def registry_conn() -> Iterator[Any]:
    """One psycopg2 connection to the server's database: commit when the
    block ends normally, roll back when it raises. Raises
    :class:`RegistryUnavailable` when no database is configured."""
    from umrs.cli.admin.cascade import cli_db_pool

    pool = cli_db_pool()
    if pool is None:
        raise RegistryUnavailable(
            "no site database configured (DATABASE_URL / DB_HOST, or "
            "UMRS_ADMIN_CLI_DB=off): the embedding-space registry lives there"
        )
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextlib.contextmanager
def index_conn() -> Iterator[Any]:
    """One AUTOCOMMIT psycopg2 connection (``CREATE INDEX CONCURRENTLY``
    cannot run in a transaction block, slice KN-S6). The connection's
    previous autocommit setting is restored before it goes back to the
    pool. Raises :class:`RegistryUnavailable` when no database is
    configured."""
    from umrs.cli.admin.cascade import cli_db_pool

    pool = cli_db_pool()
    if pool is None:
        raise RegistryUnavailable(
            "no site database configured (DATABASE_URL / DB_HOST, or "
            "UMRS_ADMIN_CLI_DB=off): the embedding-space registry lives there"
        )
    conn = pool.getconn()
    previous = conn.autocommit
    try:
        conn.rollback()  # autocommit cannot change inside a transaction
        conn.autocommit = True
        yield conn
    finally:
        conn.autocommit = previous
        pool.putconn(conn)


def _run_with_index_conn(operation: Callable[[Any], int]) -> int:
    """:func:`_run_with_registry`'s error rules on an :func:`index_conn`."""
    try:
        with index_conn() as conn:
            return operation(conn)
    except RegistryUnavailable as exc:
        _error(str(exc))
        return EXIT_NO_DB
    except EmbeddingSpaceError as exc:
        _error(str(exc))
        return EXIT_REFUSED


# ── handlers (slice task kn-s5-03) ──────────────────────────────────────────
#
# Shared rules for every handler below (signature ``(args, _store) -> int``;
# ``_store`` is the admin UserStore, unused):
# - open the connection with ``with registry_conn() as conn:``; on
#   ``RegistryUnavailable`` print ``error: <message>`` to stderr and return
#   EXIT_NO_DB.
# - on ``EmbeddingSpaceError`` print ``error: <str(exc)>`` to stderr and
#   return EXIT_REFUSED.
# - normal output goes to stdout with ``print``.


def space_as_dict(space: EmbeddingSpace) -> dict[str, Any]:
    """``{column: value}`` for every name in COLUMNS (tuples as lists),
    plus ``"index_name": space.index_name``."""
    data: dict[str, Any] = {}
    for column in COLUMNS:
        value = getattr(space, column)
        data[column] = list(value) if isinstance(value, tuple) else value
    data["index_name"] = space.index_name
    return data


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def _run_with_registry(operation: Callable[[Any], int]) -> int:
    """Run one registry operation with the shared connection/error rules."""
    try:
        with registry_conn() as conn:
            return operation(conn)
    except RegistryUnavailable as exc:
        _error(str(exc))
        return EXIT_NO_DB
    except EmbeddingSpaceError as exc:
        _error(str(exc))
        return EXIT_REFUSED


def cmd_list(args: argparse.Namespace, _store: Any = None) -> int:
    """``list``: ``embedding_space_store.list_spaces(conn,
    include_retired=args.all)``. With ``args.json`` print
    ``json.dumps([space_as_dict(s) ...], indent=2)``; else print one line
    per space: ``f"{s.space_id}\\t{s.status}\\t{s.dimension}\\t{s.model_name}\\t"
    f"{','.join(s.legacy_names)}"``, or ``(no embedding spaces)`` when the
    list is empty. Return EXIT_OK."""

    def list_spaces(conn: Any) -> int:
        spaces = embedding_space_store.list_spaces(conn, include_retired=args.all)
        if args.json:
            print(json.dumps([space_as_dict(space) for space in spaces], indent=2))
        elif not spaces:
            print("(no embedding spaces)")
        else:
            for space in spaces:
                print(
                    f"{space.space_id}\t{space.status}\t{space.dimension}\t"
                    f"{space.model_name}\t{','.join(space.legacy_names)}"
                )
        return EXIT_OK

    return _run_with_registry(list_spaces)


def space_from_args(args: argparse.Namespace) -> EmbeddingSpace:
    """The ``EmbeddingSpace`` a ``register`` command describes:
    space_id, model_name, dimension, model_ref, revision, query_prefix,
    document_prefix from the same-named args; ``modalities`` =
    ``tuple(args.modality)`` or ``("text",)`` when empty/None;
    ``legacy_names`` = ``tuple(args.legacy_name or ())``; status
    ``"active"``; source ``SOURCE_ADMIN``. No validation here."""
    return EmbeddingSpace(
        space_id=args.space_id,
        model_name=args.model_name,
        dimension=args.dimension,
        model_ref=args.model_ref,
        revision=args.revision,
        modalities=tuple(args.modality or ()) or ("text",),
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
        legacy_names=tuple(args.legacy_name or ()),
        status=STATUS_ACTIVE,
        source=SOURCE_ADMIN,
    )


def cmd_register(args: argparse.Namespace, _store: Any = None) -> int:
    """``register``: ``embedding_space_store.register_space(conn,
    space_from_args(args), replace=args.replace)``; print
    ``registered <space_id>`` when it returns True, ``replaced <space_id>``
    when False. Return EXIT_OK."""

    def register(conn: Any) -> int:
        space = space_from_args(args)
        inserted = embedding_space_store.register_space(
            conn, space, replace=args.replace
        )
        print(f"{'registered' if inserted else 'replaced'} {space.space_id}")
        return EXIT_OK

    return _run_with_registry(register)


def _set_status(args: argparse.Namespace, status: str, word: str) -> int:
    def set_status(conn: Any) -> int:
        found = embedding_space_store.set_status(conn, args.space_id, status)
        if not found:
            _error(f"no embedding space {args.space_id}")
            return EXIT_REFUSED
        print(f"{word} {args.space_id}")
        return EXIT_OK

    return _run_with_registry(set_status)


def cmd_retire(args: argparse.Namespace, _store: Any = None) -> int:
    """``retire``: ``embedding_space_store.set_status(conn, args.space_id,
    "retired")``; True → print ``retired <space_id>``, EXIT_OK; False →
    stderr ``error: no embedding space <space_id>``, EXIT_REFUSED."""
    return _set_status(args, STATUS_RETIRED, "retired")


def cmd_activate(args: argparse.Namespace, _store: Any = None) -> int:
    """``activate``: as ``cmd_retire`` with status ``"active"`` and the
    word ``activated``."""
    return _set_status(args, STATUS_ACTIVE, "activated")


def _seed_dry_run(conn: Any, declared: list[EmbeddingSpace]) -> int:
    for space in declared:
        existing = embedding_space_store.get_space(conn, space.space_id)
        print(
            f"{'already registered' if existing is not None else 'would seed'} "
            f"{space.space_id}"
        )
    return EXIT_OK


def _seed(conn: Any, declared: list[EmbeddingSpace]) -> int:
    inserted = embedding_space_store.seed_spaces(conn, declared)
    for space in declared:
        word = "seeded" if space.space_id in inserted else "already registered"
        print(f"{word} {space.space_id}")
    return EXIT_OK


def cmd_seed_site(args: argparse.Namespace, _store: Any = None) -> int:
    """``seed-site``: ``declared = embedding_space_site.site_embedding_spaces()``
    (read BEFORE opening the connection; an ``EmbeddingSpaceError`` there
    → EXIT_REFUSED). No declared spaces → print
    ``no knowledge.embedding_spaces in the site file`` and return EXIT_OK
    without opening a connection.

    With ``args.dry_run``: for each declared space print
    ``would seed <id>`` when ``embedding_space_store.get_space(conn, id)``
    is None, else ``already registered <id>``; write nothing.
    Without: ``inserted = embedding_space_store.seed_spaces(conn, declared)``;
    for each declared space in order print ``seeded <id>`` when its id is
    in ``inserted``, else ``already registered <id>``. Return EXIT_OK."""
    try:
        declared = embedding_space_site.site_embedding_spaces()
    except EmbeddingSpaceError as exc:
        _error(str(exc))
        return EXIT_REFUSED
    if not declared:
        print("no knowledge.embedding_spaces in the site file")
        return EXIT_OK
    operation = _seed_dry_run if args.dry_run else _seed
    return _run_with_registry(lambda conn: operation(conn, declared))


def cmd_index(args: argparse.Namespace, _store: Any = None) -> int:
    """``index``: build, rebuild or drop the per-space HNSW indexes (slice
    KN-S6, task kn-s6-03).

    Return ``_run_with_index_conn(operation)`` where ``operation(conn)``:

    1. ``spaces = embedding_space_store.list_spaces(conn, include_retired=True)``;
    2. ``states = embedding_space_index.read_index_states(conn, spaces)``;
    3. ``work = embedding_space_index.plan_index_work(spaces, states,
       only=args.space or None)``;
    4. for each line of ``embedding_space_index.coverage_warnings(spaces,
       embedding_space_index.stored_model_names(conn))`` print
       ``warning: <line>`` to stderr;
    5. no work: print ``all embedding-space indexes are current``; return
       EXIT_OK;
    6. with ``args.dry_run``: for each item of ``work``, for each of its
       statements, print ``would run: <statement>``; run nothing; return
       EXIT_OK;
    7. else ``ran = embedding_space_index.apply_index_work(conn, work)``; print
       ``ran: <statement>`` for each statement of ``ran``; return EXIT_OK.

    Call the modules through their module objects (as above) so tests can
    patch them."""
    raise NotImplementedError("kn-s6-03")


# ── parsers (wired by umrs.cli.admin.main) ──────────────────────────────────


def _add_register_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    reg = sub.add_parser("register", help="Register (or --replace) a space")
    reg.add_argument("--space-id", required=True)
    reg.add_argument(
        "--model-name", required=True, help="the name vectors are stored under"
    )
    reg.add_argument("--dimension", required=True, type=int)
    reg.add_argument("--model-ref", default="", help="hub id the weights came from")
    reg.add_argument("--revision", default="", help="pinned snapshot hash")
    reg.add_argument(
        "--modality", action="append", default=[], help="text|image|video|audio|volume"
    )
    reg.add_argument("--query-prefix", default="")
    reg.add_argument("--document-prefix", default="")
    reg.add_argument(
        "--legacy-name", action="append", default=[], help="another name of this space"
    )
    reg.add_argument("--replace", action="store_true", help="overwrite an existing id")
    reg.set_defaults(func=cmd_register)


def add_embedding_space_parsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """The ``embedding-space`` verbs on ``sub`` (``dest="action"``)."""
    list_p = sub.add_parser("list", help="List registered spaces")
    list_p.add_argument("--all", action="store_true", help="include retired spaces")
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=cmd_list)
    _add_register_parser(sub)
    for verb, func, text in (
        ("retire", cmd_retire, "Retire a space (writers refuse it)"),
        ("activate", cmd_activate, "Re-activate a retired space"),
    ):
        verb_p = sub.add_parser(verb, help=text)
        verb_p.add_argument("space_id")
        verb_p.set_defaults(func=func)
    seed_p = sub.add_parser(
        "seed-site", help="Insert the site file's knowledge.embedding_spaces"
    )
    seed_p.add_argument("--dry-run", action="store_true")
    seed_p.set_defaults(func=cmd_seed_site)
    index_p = sub.add_parser(
        "index", help="Build, rebuild or drop the per-space HNSW indexes"
    )
    index_p.add_argument(
        "--space", action="append", default=[], help="only this space id (repeatable)"
    )
    index_p.add_argument("--dry-run", action="store_true")
    index_p.set_defaults(func=cmd_index)
