"""Canonical server CLI entrypoint for UMRS."""


def main():
    from server_umrs import main as legacy_main

    return legacy_main()