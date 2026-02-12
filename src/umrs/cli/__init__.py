"""
UMRS Command Line Interface.

Provides CLI commands for managing UMRS resources.

Usage:
    python -m umrs.cli context list
    python -m umrs.cli context show <conversation_id>
    python -m umrs.cli context prune --dry-run
    python -m umrs.cli memory list
    python -m umrs.cli memory search "query"
"""

from umrs.cli.main import main


__all__ = ["main"]
