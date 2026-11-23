# umrs-protocol

The stdlib-only contract between **UMRS** and its clients (ADR-117 D-2, D-3).

It holds what both sides of the UMRS boundary must agree on byte for byte:

| Module | What it is |
|---|---|
| `protocol` | ZMQ handshake reply and version compatibility |
| `handshake_hmac` | gateway identity, workflow-author and memberships HMAC signing |
| `errors` | platform error codes, their HTTP status and wire shape; `RETIRED_CODES` for codes no route raises any more |
| `hpc_gate` | the gate script sshd runs on an HPC login node (ADR-112) |
| `models`, `retrieval`, `context_pack` | model profiles, retrieval sizing, context packing |
| `provenance`, `research`, `a2l` | execution provenance, research primitives, A2L conservation records |
| `ask_policy` | ask-the-scientist action vocabulary shared by both loop venues |
| `recipes`, `skills`, `security`, `workflow_package`, `workflows` | shared schemas and pre-flight validators |

It has **no runtime dependencies** beyond the Python standard library.
`import umrs_protocol` loads no submodule; import the one you need.

## Status (U2.0)

The modules were copied byte for byte from `r4-contracts` (`r4_contracts`),
which becomes an alias layer over this package in U2.0b. Until then nothing in
`umrs/src` imports `umrs_protocol`; docstrings that still say "r4-contracts"
are reworded per module family in U2.1 onward (never in `hpc_gate.py`, whose
bytes are the gate release).

## The gate

`hpc_gate.py` is installed on HPC login nodes and its sha256 is recorded at
enrolment. Its bytes are pinned in `tests/golden/gate.json`: any edit is a gate
release under ADR-112 (a reviewed row in that file and normally a
`GATE_VERSION` bump, which forces re-enrolment). The formatter excludes it.

## Tests

```bash
cd protocol && python -m pytest tests -q
```

- `test_golden_wire.py`: wire vectors (`tests/golden/wire_vectors.json`)
  generated once from `r4_contracts` before the move;
- `test_gate_release.py`: gate bytes and version against `gate.json`;
- `test_stdlib_only.py`: every import is stdlib or `umrs_protocol`;
- `test_no_eager_imports.py`: `import umrs_protocol` loads no submodule;
- `test_retired_codes.py`: retired codes are out of `known_error_codes()`,
  still decode, no server module raises one, and 1.0 ships them deleted.

## Retired error codes

A code whose routes are gone is retired, not removed (DEC-H10, decision
record 2026-11-24). It moves from `known_error_codes()` (the live catalog)
to `RETIRED_CODES`; `is_retired(code)` names it and
`decodable_error_codes()` still maps it to its status, so a client talking
to an older self-hosted UMRS decodes it. umrs-protocol 1.0 deletes the
registry. See [CHANGELOG.md](CHANGELOG.md).
