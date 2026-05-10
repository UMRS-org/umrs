"""Canonical server CLI entrypoint for UMRS.

The ``admin`` subcommand and ``-h``/``--help`` are dispatched here
*before* importing ``server_umrs`` so that heavy ML-library imports
(torch, transformers, spacy, sentence-transformers, …) are avoided
when they are not needed.
"""

import sys


def _handle_completion_command(argv: list[str]) -> int:
    import argparse

    from umrs.cli.completion import install_completion, print_completion_script, uninstall_completion

    parser = argparse.ArgumentParser(prog="umrs-server completion")
    sub = parser.add_subparsers(dest="completion_action")
    sub.add_parser("bash", help="Print bash completion script")
    sub.add_parser("zsh", help="Print zsh completion script")
    inst_p = sub.add_parser("install", help="Install completion into your shell RC file")
    inst_p.add_argument("--shell", choices=("bash", "zsh"), default=None)
    uninst_p = sub.add_parser("uninstall", help="Remove completion from your shell RC file")
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
    words, adjusted_cword = normalize_completion_words(sys.argv[1:], cword, "umrs-server")
    if words and words[0] == "admin":
        admin_words = words[1:]
        admin_cword = max(adjusted_cword - 1, 0)
        print_completion_candidates(build_admin_parser(), admin_words, admin_cword)
    else:
        print_completion_candidates(_build_completion_parser(), words, adjusted_cword)
    return True


def main():
    if _emit_completion_if_requested():
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "completion":
        return _handle_completion_command(sys.argv[2:])

    # Fast-path: route `umrs-server admin …` without loading the full
    # server module (which pulls in torch/transformers/spacy/…).
    if len(sys.argv) > 1 and sys.argv[1] == "admin":
        from umrs.cli.admin import main as admin_main

        return admin_main(sys.argv[2:])

    # Fast-path: show lightweight help when no daemon-class args are given.
    if len(sys.argv) == 1 or (
        len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")
    ):
        _print_toplevel_help()
        return 0

    # Reject unknown subcommands before loading heavy ML imports.
    first = sys.argv[1]
    if not first.startswith("-"):
        print(
            f"umrs-server: unknown command '{first}'\n"
            f"Run 'umrs-server -h' for usage.",
            file=sys.stderr,
        )
        return 2

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
        "  umrs-server admin user add <username> --fullname \"Your Name\" --role user\n"
        "  umrs-server admin user list\n"
        "  umrs-server admin user show <username>\n"
        "  eval \"$(umrs-server completion bash)\"\n"
        "\n"
        "Run 'umrs-server admin -h' for admin subcommands.\n"
    )
