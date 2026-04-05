"""Canonical client CLI entrypoint for UMRS."""


def main():
    from client_umrs import main as legacy_main

    return legacy_main()