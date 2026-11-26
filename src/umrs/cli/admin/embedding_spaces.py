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

Handlers call the store and site modules through their module objects
(``embedding_space_store.list_spaces(...)``,
``embedding_space_site.site_embedding_spaces()``) so tests can patch them.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from typing import Any

from umrs.knowledge import embedding_space_site, embedding_space_store  # noqa: F401
from umrs.knowledge.embedding_spaces import (  # noqa: F401
    COLUMNS,
    SOURCE_ADMIN,
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
    raise NotImplementedError("kn-s5-03")


def cmd_list(args: argparse.Namespace, _store: Any = None) -> int:
    """``list``: ``embedding_space_store.list_spaces(conn,
    include_retired=args.all)``. With ``args.json`` print
    ``json.dumps([space_as_dict(s) ...], indent=2)``; else print one line
    per space: ``f"{s.space_id}\\t{s.status}\\t{s.dimension}\\t{s.model_name}\\t"
    f"{','.join(s.legacy_names)}"``, or ``(no embedding spaces)`` when the
    list is empty. Return EXIT_OK."""
    raise NotImplementedError("kn-s5-03")


def space_from_args(args: argparse.Namespace) -> EmbeddingSpace:
    """The ``EmbeddingSpace`` a ``register`` command describes:
    space_id, model_name, dimension, model_ref, revision, query_prefix,
    document_prefix from the same-named args; ``modalities`` =
    ``tuple(args.modality)`` or ``("text",)`` when empty/None;
    ``legacy_names`` = ``tuple(args.legacy_name or ())``; status
    ``"active"``; source ``SOURCE_ADMIN``. No validation here."""
    raise NotImplementedError("kn-s5-03")


def cmd_register(args: argparse.Namespace, _store: Any = None) -> int:
    """``register``: ``embedding_space_store.register_space(conn,
    space_from_args(args), replace=args.replace)``; print
    ``registered <space_id>`` when it returns True, ``replaced <space_id>``
    when False. Return EXIT_OK."""
    raise NotImplementedError("kn-s5-03")


def cmd_retire(args: argparse.Namespace, _store: Any = None) -> int:
    """``retire``: ``embedding_space_store.set_status(conn, args.space_id,
    "retired")``; True → print ``retired <space_id>``, EXIT_OK; False →
    stderr ``error: no embedding space <space_id>``, EXIT_REFUSED."""
    raise NotImplementedError("kn-s5-03")


def cmd_activate(args: argparse.Namespace, _store: Any = None) -> int:
    """``activate``: as ``cmd_retire`` with status ``"active"`` and the
    word ``activated``."""
    raise NotImplementedError("kn-s5-03")


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
    raise NotImplementedError("kn-s5-03")


# ── parsers (wired by umrs.cli.admin.main) ──────────────────────────────────


def _add_register_parser(sub: argparse._SubParsersAction) -> None:
    reg = sub.add_parser("register", help="Register (or --replace) a space")
    reg.add_argument("--space-id", required=True)
    reg.add_argument("--model-name", required=True, help="the name vectors are stored under")
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


def add_embedding_space_parsers(sub: argparse._SubParsersAction) -> None:
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
