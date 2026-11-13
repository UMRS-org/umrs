"""umrs-protocol: the stdlib-only contract between UMRS and its clients.

This package holds what both sides of the UMRS boundary must agree on byte for
byte: the ZMQ handshake (``protocol``), the gateway identity signing
(``handshake_hmac``), the platform error codes (``errors``), the HPC gate run
under sshd (``hpc_gate``), and the shared schemas for models, retrieval,
research provenance, recipes, skills, security scanners and workflow packages.

It depends on the Python standard library only (ADR-117 D-3, enforced by
``tests/test_stdlib_only.py``). Under ADR-117 D-2 it is the source of truth for
the modules that ``r4_contracts`` used to own; ``r4_contracts`` becomes an
alias layer over it.

Nothing is imported here on purpose: ``import umrs_protocol`` loads no
submodule (``tests/test_no_eager_imports.py``). Import the module you need,
e.g. ``from umrs_protocol import hpc_gate``. The version lives in the project
metadata only: ``importlib.metadata.version("umrs-protocol")``.
"""
