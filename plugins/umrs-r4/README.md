# umrs-r4

The R4 runtime hosted by UMRS, packaged as a UMRS extension (ADR-117 §6, plan
`docs/development/umrs-independent-main-plan-2026-11-13.md` in mnagaso/r4).

- Core `umrs` never imports `umrs_r4` (fence: `tests/test_core_r4_extension_fence.py`).
- Entry point: group `umrs.extensions`, name `r4`, `umrs_r4:register`.
- Install from the checkout, after umrs: `pip install -e src -e plugins/umrs-r4[mcp]`.
- Tests: `python -m pytest plugins/umrs-r4/tests`.

Contents so far (U3 S2): `umrs_r4.verifiers.mcp_adapter`, the MCP knowledge
verifier adapter (core's only former `r4_mcp` user).
