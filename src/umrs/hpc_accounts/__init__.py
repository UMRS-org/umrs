"""User-registered HPC site accounts (ADR-112 D-3/D-7, wave B1).

- ``store``: the ``site_accounts`` / ``project_site_allocations`` tables
  (0134/0135), venue-scoped, plus an in-memory twin for tests.
- ``transport_keys``: the per-person, per-site transport key the server
  generates. The private half is a file under ``UMRS_TRANSPORT_KEY_DIR``;
  the database holds only its id and public fingerprint.
- ``gate_client``: the server's SSH call to the user's gate (plan §4 B1.5a
  wire contract): exact argv, JSON in and out, refusals by name.

Nothing here runs work under a user's login. Wave B1 registers and verifies
accounts only; execution arrives in B2.

Waves C and D (plan §6 C/D.0):

- ``grants`` / ``grant_service``: the standing grants a person signed
  (migration 0137), their checks, views and the grant proposal;
- ``usage_ledger`` / ``usage_reports`` / ``ledger_sweep``: the append-only
  HPC usage ledger of user work (migration 0138), the binding budget check,
  the reports and the recovery sweep.
"""
