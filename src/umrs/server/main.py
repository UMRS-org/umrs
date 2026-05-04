"""Canonical server CLI entrypoint for UMRS.

The ``admin`` subcommand and ``-h``/``--help`` are dispatched here
*before* importing ``server_umrs`` so that heavy ML-library imports
(torch, transformers, spacy, sentence-transformers, …) are avoided
when they are not needed.
"""

import sys


def main():
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
        "Run 'umrs-server admin -h' for admin subcommands.\n"
    )
