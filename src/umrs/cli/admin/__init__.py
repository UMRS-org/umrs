"""``umrs-server admin`` command line entry point.

Dispatched from ``server_umrs.main()`` when invoked as ``umrs-server admin ...``.
Works offline (does not require a running UMRS daemon) and manipulates the
file-backed user records and admin token store directly.
"""

from umrs.cli.admin.main import main

__all__ = ["main"]
