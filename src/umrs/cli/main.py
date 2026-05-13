"""
UMRS CLI main module.

Note: Context and memory commands excluded from selective merge.
This CLI provides a minimal interface for UMRS functionality.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="umrs",
        description="UMRS Command Line Interface (minimal build)",
        epilog="""
Note: Context and memory commands are not available in this build.
Use the r4 CLI or REST API for full functionality.

UMRS provides:
- Data transfer to S3/MinIO
- Embedding calculation
- RAG pipeline
- Model registry
- Job scheduling (Slurm, K8s, Compose)
""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="umrs (minimal build)",
    )

    parser.add_argument(
        "--url",
        dest="base_url",
        help="UMRS base URL",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show UMRS server status")
    status_parser.set_defaults(func=cmd_status)

    return parser


def cmd_status(args: argparse.Namespace, config: dict) -> None:
    """Show server status."""
    print("UMRS CLI (minimal build)")
    print(f"Server URL: {config.get('base_url', 'not configured')}")
    print("\nAvailable functionality:")
    print("  - Data transfer (S3/MinIO)")
    print("  - Embedding calculation")
    print("  - RAG pipeline")
    print("  - Model registry")
    print("  - Job scheduling")
    print("\nNote: Use server_umrs.py or REST API for full access.")


def get_env_config() -> dict:
    """Get configuration from environment variables."""
    import os
    return {
        "base_url": os.environ.get("UMRS_BASE_URL", "http://localhost:8000"),
    }


def main(argv: list = None) -> int:
    """Main entry point for CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    # Build config from env + args
    config = get_env_config()
    if args.base_url:
        config["base_url"] = args.base_url

    # Call the appropriate function
    if hasattr(args, "func"):
        args.func(args, config)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
