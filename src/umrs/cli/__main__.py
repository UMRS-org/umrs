"""
UMRS CLI entry point.

Usage:
    python -m umrs.cli context list
    python -m umrs.cli memory search "query"
"""

import sys
from umrs.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
