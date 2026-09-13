"""Canonical server CLI entrypoint for UMRS.

The ``admin`` subcommand and ``-h``/``--help`` are dispatched here
*before* importing ``server_umrs`` so that heavy ML-library imports
(torch, transformers, spacy, sentence-transformers, …) are avoided
when they are not needed.
"""

import sys


def _handle_completion_command(argv: list[str]) -> int:
    import argparse

    from umrs.cli.completion import (
        install_completion,
        print_completion_script,
        uninstall_completion,
    )

    parser = argparse.ArgumentParser(prog="umrs-server completion")
    sub = parser.add_subparsers(dest="completion_action")
    sub.add_parser("bash", help="Print bash completion script")
    sub.add_parser("zsh", help="Print zsh completion script")
    inst_p = sub.add_parser(
        "install", help="Install completion into your shell RC file"
    )
    inst_p.add_argument("--shell", choices=("bash", "zsh"), default=None)
    uninst_p = sub.add_parser(
        "uninstall", help="Remove completion from your shell RC file"
    )
    uninst_p.add_argument("--shell", choices=("bash", "zsh"), default=None)

    args = parser.parse_args(argv)
    action = getattr(args, "completion_action", None)
    if action == "install":
        print(install_completion("umrs-server", shell=args.shell))
        return 0
    if action == "uninstall":
        print(uninstall_completion("umrs-server", shell=args.shell))
        return 0
    if action in ("bash", "zsh"):
        print_completion_script("umrs-server", "_UMRS_SERVER_COMPLETE", action)
        return 0
    print_completion_script("umrs-server", "_UMRS_SERVER_COMPLETE", "bash")
    return 0


def _build_admin_migrate_parser():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="umrs-server admin migrate",
        description=(
            "Apply pending UMRS schema migrations (python -m umrs.migrations "
            "wrapper). With --check, exit 0 when the chain is current and 1 "
            "when migrations are pending (the pending list is printed)."
        ),
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="PostgreSQL connection URL (default: $DATABASE_URL)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report pending migrations without applying; exit 1 if any",
    )
    parser.add_argument("--target", help="Apply up to this version only")
    return parser


def _open_migrate_connection(db_url: str):
    """psycopg2 connection for the migrate verb (R-14 hygiene kwargs applied)."""
    import psycopg2

    from umrs.shared.pg_settings import pg_connect_kwargs

    return psycopg2.connect(db_url, **pg_connect_kwargs("admin-migrate"))


def _handle_admin_migrate(argv: list[str]) -> int:
    """``umrs-server admin migrate [--check] [--db-url URL] [--target V]``.

    Thin dispatch to :mod:`umrs.migrations`: ``--check`` maps to
    ``get_pending_migrations`` (exit 0 = current, 1 = pending, list printed);
    without it ``apply_migrations`` runs and the applied count is printed.
    Connection failures and missing DSNs exit 2 with a one-line reason.
    """
    args = _build_admin_migrate_parser().parse_args(argv)
    if not args.db_url:
        print(
            "umrs-server admin migrate: --db-url or DATABASE_URL required",
            file=sys.stderr,
        )
        return 2

    from umrs.migrations import (
        apply_migrations,
        get_current_version,
        get_pending_migrations,
    )

    try:
        conn = _open_migrate_connection(args.db_url)
    except Exception as exc:  # noqa: BLE001 — operator-facing one-liner
        print(f"umrs-server admin migrate: cannot connect: {exc}", file=sys.stderr)
        return 2
    try:
        if args.check:
            pending = get_pending_migrations(conn)
            print(f"Current version: {get_current_version(conn) or '(none)'}")
            print(f"Pending migrations: {len(pending)}")
            for version, path in pending:
                print(f"  - {version}: {path.name}")
            return 1 if pending else 0
        count = apply_migrations(conn, target_version=args.target)
        print(f"Applied {count} migration(s)")
        return 0
    finally:
        conn.close()


def _build_completion_parser():
    import argparse

    parser = argparse.ArgumentParser(prog="umrs-server")
    parser.add_argument("-c", "--conf")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--check-objects", action="store_true")
    parser.add_argument("--pretty-print", action="store_true")
    parser.add_argument("--pool-max-workers")
    parser.add_argument("--pool-max-pending")
    subcommands = parser.add_subparsers(dest="command")
    subcommands.add_parser("admin")
    subcommands.add_parser("completion")
    return parser


def _emit_completion_if_requested() -> bool:
    import os

    if not os.environ.get("_UMRS_SERVER_COMPLETE"):
        return False

    from umrs.cli.admin.main import build_parser as build_admin_parser
    from umrs.cli.completion import (
        normalize_completion_words,
        print_completion_candidates,
    )

    try:
        cword = int(os.environ.get("COMP_CWORD", "0"))
    except ValueError:
        cword = 0
    words, adjusted_cword = normalize_completion_words(
        sys.argv[1:], cword, "umrs-server"
    )
    if words and words[0] == "admin":
        admin_words = words[1:]
        admin_cword = max(adjusted_cword - 1, 0)
        print_completion_candidates(build_admin_parser(), admin_words, admin_cword)
    else:
        print_completion_candidates(_build_completion_parser(), words, adjusted_cword)
    return True


def _handle_admin(argv: list[str]) -> int:
    """``umrs-server admin …`` without loading the full server module.

    ``admin migrate [--check]`` is a thin dispatch to umrs.migrations (R-10
    support item, 2026-09-13); every other verb goes to the admin CLI.
    """
    if argv and argv[0] == "migrate":
        return _handle_admin_migrate(argv[1:])
    from umrs.cli.admin import main as admin_main

    return admin_main(argv)


def _fast_path(argv: list[str]) -> int | None:
    """Subcommands, help and rejections that must not load torch/transformers.

    Returns an exit code when handled, None when the full server should start.
    """
    head = argv[0] if argv else ""
    if head == "completion":
        return _handle_completion_command(argv[1:])
    if head == "admin":
        return _handle_admin(argv[1:])
    # Lightweight help when no daemon-class args are given.
    if not argv or (len(argv) == 1 and head in ("-h", "--help")):
        _print_toplevel_help()
        return 0
    # Reject unknown subcommands before loading heavy ML imports.
    if not head.startswith("-"):
        print(
            f"umrs-server: unknown command '{head}'\nRun 'umrs-server -h' for usage.",
            file=sys.stderr,
        )
        return 2
    return None


def main():
    if _emit_completion_if_requested():
        return 0

    handled = _fast_path(sys.argv[1:])
    if handled is not None:
        return handled

    # Full server start – heavy imports happen only here.
    from server_umrs import main as legacy_main

    return legacy_main()


def _print_toplevel_help():
    """Print a lightweight help message without importing heavy deps."""
    print(
        "usage: umrs-server [-h] {admin,...} ...\n"
        "\n"
        "UMRS server and admin CLI.\n"
        "\n"
        "subcommands:\n"
        "  admin        Offline admin CLI (user/token/slurm management)\n"
        "  admin migrate [--check]\n"
        "               Apply pending schema migrations (--check: exit 1 if pending)\n"
        "  completion   Print shell completion setup\n"
        "\n"
        "server mode options:\n"
        "  -c, --conf   Config file path\n"
        "  --force-recompute\n"
        "               Recompute all embeddings\n"
        "  --force-update\n"
        "               Recompute only missing/mismatched embeddings\n"
        "  --check-objects\n"
        "               Check S3 objects against database\n"
        "  --pretty-print\n"
        "               Enable pretty-printed tables\n"
        "  --pool-max-workers N\n"
        "               Max worker threads (default: 16)\n"
        "  --pool-max-pending N\n"
        "               Max pending tasks (default: 64)\n"
        "\n"
        "Operator onboarding:\n"
        '  umrs-server admin user add <username> --fullname "Your Name" --role user\n'
        "  umrs-server admin user list\n"
        "  umrs-server admin user show <username>\n"
        '  eval "$(umrs-server completion bash)"\n'
        "\n"
        "Run 'umrs-server admin -h' for admin subcommands.\n",
        file=sys.stderr,
    )
