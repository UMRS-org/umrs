# umrs-protocol changelog

Error codes are append-only: a code is retired before it is removed, and
removals happen only at a major version (see `errors.py`).

## Unreleased (0.2.x)

### Retired

- `pipelines.*` (`not_found`, `forbidden`, `conflict`, `bad_spec`,
  `rag_profile_not_found`) and `verification.*` (`unavailable`, `forbidden`,
  `bad_request`) move to the new `RETIRED_CODES` registry (DEC-H10, decision
  record 2026-11-24). Their routes were deleted in hardening H-10.1
  (2026-10-18), so no current UMRS raises them.
  - `known_error_codes()`, the live catalog, no longer lists them.
  - They stay decodable: `PlatformError.from_dict` accepts them,
    `decodable_error_codes()` maps them to their old status, and
    `is_retired(code)` names them. A self-hosted UMRS that upgrades later can
    still send them to a newer client.
  - The `PlatformError.pipelines` and `PlatformError.verification` factories
    stay until removal; no server module may call them
    (`tests/test_retired_codes.py`).

### Scheduled removal

- **1.0:** `RETIRED_CODES`, `PIPELINES_ERRORS`, `VERIFICATION_ERRORS` and the
  two factories are deleted (`RETIRED_CODES_REMOVED_IN = "1.0"`;
  `test_removal_is_scheduled_for_protocol_1_0` fails a 1.0 version that still
  carries them).
