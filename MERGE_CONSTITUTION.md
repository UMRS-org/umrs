# UMRS Merge-Safety Constitution

> **Audience:** Human developers and AI coding agents working in the `umrs/` submodule
> **Status:** CANONICAL — governs every merge from `main` into `ai4s-platform`
> **Scope:** The `umrs/` git submodule only. The repo-root [CONSTITUTION.md](../CONSTITUTION.md) still governs the wider platform.
> **Last updated:** 2026-06-18

This document captures the invariants that, when silently violated **by a merge**,
have broken (or will break) the `ai4s-platform` integration in ways that do not
fail loudly. Every break listed here was a *silent* regression: code imported
fine, started fine, and only failed at runtime far from the root cause.

Read this **before** and **after** every `git merge ejeannot/main`.

Related, do-not-duplicate references:
- [SYNC_CHECKLIST.md](SYNC_CHECKLIST.md) — branch topology, conflict-prone files, sync commands.
- [.github/instructions/umrs-branch-separation.instructions.md](../.github/instructions/umrs-branch-separation.instructions.md) — the config-shadowing case study and branch contract.
- Repo-root [CONSTITUTION.md](../CONSTITUTION.md) — platform-wide architecture/security invariants.

---

## 1. Branch Topology

| Branch | Role | Who changes it |
|--------|------|----------------|
| `main` (`ejeannot/main`) | Upstream UMRS core — ZMQ transport, task lifecycle, model registry, security primitives (incl. the CURVE session **infrastructure**) | UMRS team |
| `ai4s-platform` | UMRS core **plus** the ai4s integration layer (HTTP sidecar, gateway-facing routers, deployment profiles) | Both teams |

`ai4s-platform` is `main` + additive integration. As of 2026-06-18 it is
**6 commits ahead, 0 behind** `main`. Merges flow **`main` → `ai4s-platform`**.
Fixes to *shared* code should flow **back to `main` via PR** (see §6).

---

## 2. The Non-Negotiable Merge Invariants

These are the rules a merge must never silently violate. Each maps to a concrete
post-merge check in §5.

### 2.1 No module-level constant shadowing

A constant may be assigned **exactly once** at module scope. Never:

```python
DEFAULT_FOO = "value"
DEFAULT_FOO = None      # FORBIDDEN — silently nullifies the real value
```

Python reassigns without warning, so a merge that lands a second assignment
produces a `None` where a string is expected. If you need a "disabled" sentinel,
use a **different** name (`DEFAULT_FOO_OVERRIDE`).

> **Why this rule exists — the config-shadowing break (2026-06):** a merge added
> `DEFAULT_CURRENT_CONNECTION_PROFILE_FILE = None` and
> `DEFAULT_CURRENT_S3_CONFIG_FILE = None` directly below the correct string
> assignments in `src/umrs/shared/config.py`. Result: `umrs profile use`
> died with `[Errno 21] Is a directory: '.'`, `umrs info` showed
> `profile: (none)`, and `r4 auth login` skipped ZMQ bootstrap (503). Nothing
> failed at import time. See umrs-branch-separation.instructions.md.

### 2.2 Config resolution must never resolve to `None`

The chain `get_current_connection_profile_file()` →
`get_current_profile_path()` → profile read must always return a usable path.
A merge that changes any link in that chain must keep the resolved value
non-`None`.

### 2.3 Session-CURVE has two halves; a merge must keep BOTH wired

ZMQ session-CURVE is only functional when **both** of these are present. The
infrastructure (`session_curve_provider.py`, `zmq_sessions.py`,
`zmq_security.py`) existing is **not enough**:

1. **Server-side provider init** — at startup, when `UMRS_ZMQ_AUTH_TYPE=session`,
   the server must create one process-wide `SessionCredentialsProvider`,
   `start()` it, and register it via
   `session_curve_provider.set_shared_session_provider(...)`.
   Entry point: `server_umrs._init_session_curve_provider()`, called from
   `perform_initial_config()` **before** any socket bind.

2. **Socket-level CURVE application** — `MessageQueue` must apply CURVE to every
   socket it opens. Entry point: `MessageQueue._apply_socket_security(socket,
   server=...)`, called in `_initialize_control_queue`,
   `_initialize_data_queue`, and `reset_control_queue` **before** bind/connect.

If a merge reverts or bypasses either half, the HTTP `POST /umrs/zmq/session`
returns `503` (provider missing) **or** the client's CURVE handshake hangs
(server socket left plaintext). The client then silently falls back to
`auth=none`, so "it still works" — masking the regression.

> **Why this rule exists — the CURVE-inactive break:** both halves were absent
> on `main` *and* `ai4s-platform`; the CURVE files existed but were never
> wired in. `POST /umrs/zmq/session` returned
> `503 "...UMRS_ZMQ_AUTH_TYPE != session"` and CURVE never activated.

### 2.4 The single-provider rule (no cross-boundary import)

There is exactly **one** `SessionCredentialsProvider` per process, owned by the
neutral core module `umrs.core.session_curve_provider` via
`set_/get_shared_session_provider()`. The HTTP API
(`umrs.api.zmq_sessions`) and the ZAP authenticators (`umrs.core.MessageQueue`)
both read it from there. This preserves the boundary in §2.5: **never** make
`core` import `api`, or `api` hold the authoritative singleton.

### 2.5 Foundation modules never import `api`/`server`

`umrs.core`, `model_registry/`, `mcp/`, `rag/`, `pipeline_sdk/`,
`orchestrator/`, `security/`, and `shared/` must not import from `umrs.api` or
`umrs.server`. A merge that adds such an import is a hard boundary violation
(caught by `tools/import_boundary_validator.py` at the repo root).

### 2.6 Gated features must be strict no-ops when disabled

Any new security/transport feature wired in by ai4s must be a **no-op** unless
explicitly enabled by env. Session-CURVE does nothing unless
`UMRS_ZMQ_AUTH_TYPE=session`; `_apply_socket_security` returns immediately for
`auth_type == "none"`. This guarantees a merge can land the wiring without
changing the behavior of any deployment that has not opted in, and lets us
enable it deliberately per site.

### 2.7 Each `MessageQueue` owns its `zmq.Context`; track and stop authenticators

A ZAP authenticator is bound to a `zmq.Context`, and only one ZAP handler is
allowed per context. Each `MessageQueue` creates its own context, so per-socket
authenticators are safe — but they **must** be tracked
(`self._zmq_authenticators`) and `stop()`-ped in `close()`. A merge that drops
that bookkeeping leaks ZAP threads.

### 2.8 HTTP reload context must be wired in at startup

The HTTP reload endpoint (`POST /gw/ops/reload`) in `src/umrs/api/ops.py`
depends on module-level state populated by `set_reload_context()`. The endpoint
guards on this state:

```python
if _reload_configure_file is None or _reload_buckets_cfg is None:
    raise HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Reload context not initialised — server may still be starting",
    )
```

If `set_reload_context()` is **never called** during server startup, the
endpoint always returns `503`. This is a *silent* regression: the server starts
fine, ZMQ works, but `umrs reload` fails with:

```
✗ reload rejected: HTTP 503: Service Unavailable
```

Without a successful reload, the server cannot provision the S3 config
(`s3cfg_<user>`) needed by `upload_dir` / `upload_file`. Every demo that
uploads data then fails with "Server connection timeout".

**Wiring requirement:** `server_umrs.py` must call `set_reload_context()`
after `perform_initial_config()` (which sets up `server_cfg.buckets_cfg` and
`server_cfg.user_config_map`) and **before** starting the HTTP sidecar:

```python
from umrs.api.ops import set_reload_context

set_reload_context(
    configure_file,
    server_cfg.buckets_cfg,
    getattr(server_cfg, "user_config_map", None),
)
```

> **Why this rule exists — the reload-503 break (2026-06-18):**
> `set_reload_context()` was defined in `src/umrs/api/ops.py` but **never
> called** from `src/server_umrs.py`. The function existed, the endpoint
> existed, but the module-level `_reload_configure_file` and
> `_reload_buckets_cfg` remained `None`. Result: every `umrs reload`
> returned HTTP 503 with body `"Reload context not initialised — server may
> still be starting"`. Without reload, no `s3cfg` was generated, and all
> `upload_dir` / `upload_file` calls failed. The server appeared healthy
> (ping succeeded, ZMQ worked) — masking the regression.
>
> **Fix:** Added `set_reload_context()` call in `server_umrs.py` after
> `perform_initial_config()` and before `start_http_sidecar()`. Deployed via
> `./bootstrap.sh --profile umrs-front --ask-become-pass`. Verified:
> `umrs reload` now returns `✓ reload accepted`.

---

## 3. Conflict-Prone Files (verify after every merge)

In addition to the table in [SYNC_CHECKLIST.md](SYNC_CHECKLIST.md):

| File | What a merge can silently break |
|------|----------------------------------|
| `src/umrs/shared/config.py` | Constant shadowing (§2.1), resolution → `None` (§2.2) |
| `src/umrs/shared/connection_profiles.py` | Profile resolution chain (§2.2) |
| `src/umrs/core/MessageQueue.py` | CURVE socket application (§2.3), authenticator tracking (§2.7) |
| `src/server_umrs.py` | Provider init at startup (§2.3), reload context wiring (§2.8) |
| `src/umrs/core/session_curve_provider.py` | Shared singleton (§2.4) |
| `src/umrs/api/zmq_sessions.py` | Delegation to core singleton (§2.4) |
| `src/umrs/api/ops.py` | Reload endpoint 503 if `set_reload_context()` not called (§2.8) |

---

## 4. Sync Procedure

```bash
cd umrs/
git checkout ai4s-platform
git fetch ejeannot
git merge ejeannot/main          # resolve conflicts per §3 + SYNC_CHECKLIST.md
# → run the §5 post-merge validation BEFORE pushing
git push ejeannot ai4s-platform
```

When resolving a conflict in a §3 file, prefer **keeping both** the upstream
change and the ai4s wiring (additive). Never resolve by dropping the ai4s half
of a two-half invariant (§2.3).

---

## 5. Post-Merge Validation (MANDATORY, in order)

Run from the repo root unless noted. Use the repo venv
(`.venv/bin/python`) — base `python3` lacks `jsonschema`/`psycopg2`.

```bash
# (a) §2.1 / §2.2 — config resolution is non-None
cd umrs && PYTHONPATH=src ../.venv/bin/python -c "
from umrs.shared.config import (
    DEFAULT_CURRENT_CONNECTION_PROFILE_FILE as P,
    DEFAULT_CURRENT_S3_CONFIG_FILE as S,
    get_current_connection_profile_file as f,
)
assert P == 'current_connection_profile', P
assert S == 'current_s3_config', S
assert f() == 'current_connection_profile', f()
print('OK: config resolution')
"; cd ..

# (b) duplicate module-level assignment scan on conflict-prone files
cd umrs && for F in src/umrs/shared/config.py; do
  ../.venv/bin/python -c "
import re,sys
a=re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=', open('$F').read(), re.M)
d=sorted({x for x in a if a.count(x)>1})
sys.exit(f'DUPLICATE module-level assignments in $F: {d}') if d else print('OK: no dup assigns in $F')
"; done; cd ..

# (c) §2.3 / §2.4 — session-CURVE wiring regression tests
cd umrs && PYTHONPATH=src ../.venv/bin/python -m pytest \
  tests/test_zmq_sessions_api.py tests/test_zmq_security.py \
  tests/test_session_curve_wiring.py -q; cd ..

# (d) §2.5 — import boundaries (repo-root validator)
.venv/bin/python tools/import_boundary_validator.py

# (e) §2.8 — HTTP reload context is wired in at startup
cd umrs && ../.venv/bin/python -c "
import ast, sys
src = open('src/server_umrs.py').read()
tree = ast.parse(src)
found = any(
    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    and node.func.id == 'set_reload_context'
    for node in ast.walk(tree)
)
if not found:
    print('FAIL: set_reload_context() not called in server_umrs.py (§2.8)')
    sys.exit(1)
print('OK: set_reload_context() is wired in (§2.8)')
"; cd ..

# (f) full local CI gate (REQUIRED before any commit touching *.py)
bash scripts/ci-lint-local.sh
```

If any step fails, fix the merge before pushing. Do not `|| true` a failure.

---

## 6. Upstreaming Fixes to `main` (PR policy)

Because `ai4s-platform` = `main` + additive integration, a fix that touches
**shared** code (not an ai4s-only file from the SYNC_CHECKLIST list) usually
*belongs on `main`* so the next merge does not re-introduce the bug.

**Decision rule:** before fixing shared code on `ai4s-platform`, ask "is this
file byte-identical on `ejeannot/main`?"

```bash
cd umrs
for f in <files>; do
  m=$(git ls-tree ejeannot/main -- "$f" | awk '{print $3}')
  a=$(git ls-tree HEAD            -- "$f" | awk '{print $3}')
  echo "$f  $([ "$m" = "$a" ] && echo SAME→upstream-it || echo DIFFERENT→check-carefully)"
done
```

If `SAME`, the fix applies cleanly to `main` — open a PR to `main` and let it
flow back via the normal sync, rather than carrying an ai4s-only patch forever.

### Current upstream candidate: session-CURVE wiring fix (2026-06-18)

The session-CURVE wiring fix is a textbook upstream candidate. Verified state:

| File | vs `ejeannot/main` | Upstream-able |
|------|--------------------|---------------|
| `src/umrs/core/session_curve_provider.py` | SAME blob | yes — clean |
| `src/umrs/api/zmq_sessions.py` | SAME blob | yes — clean |
| `src/server_umrs.py` | SAME blob | yes — clean |
| `src/umrs/core/MessageQueue.py` | differs by 1 unrelated comment hunk (HANDSHAKE/HAND_CHECK) | yes — no overlap with CURVE hunks |
| `tests/test_session_curve_wiring.py` | new file | yes — new test |

The bug (both halves of §2.3 missing) exists on `main` too: the CURVE
infrastructure ships there but is never initialized or applied. **Recommended
PR to `main`:** "fix(zmq): initialize SessionCredentialsProvider at startup and
apply CURVE to MessageQueue sockets (gated on `UMRS_ZMQ_AUTH_TYPE=session`)".
The change is fully gated (§2.6), so it is safe for every `main` consumer that
has not enabled session auth.

### Current upstream candidate: HTTP reload context wiring (2026-06-18)

The `set_reload_context()` call in `src/server_umrs.py` is an upstream
candidate. Verified state:

| File | vs `ejeannot/main` | Upstream-able |
|------|--------------------|---------------|
| `src/server_umrs.py` | SAME blob | yes — clean |
| `src/umrs/api/ops.py` | SAME blob | yes — clean |

The bug exists on `main` too: `set_reload_context()` is defined in
`src/umrs/api/ops.py` but never called from `src/server_umrs.py`. Result: every
`umrs reload` returns HTTP 503 with body
`"Reload context not initialised — server may still be starting"`. Without
reload, no `s3cfg` is generated, and all `upload_dir` / `upload_file` calls
fail. The server appears healthy (ping succeeded, ZMQ worked) — masking the
regression.

**Fix:** Added `set_reload_context()` call in `src/server_umrs.py` after
`perform_initial_config()` and before `start_http_sidecar()`. Deployed via
`./bootstrap.sh --profile umrs-front --ask-become-pass`. Verified:
`umrs reload` now returns `✓ reload accepted`.

**Recommended PR to `main`:** "fix(ops): wire set_reload_context() at startup
so POST /gw/ops/reload does not return 503". The change is safe for every
`main` consumer: the endpoint is additive (only exists if the HTTP sidecar is
enabled), and the guard already returns 503 gracefully if the context is not
set.

---

## 7. Enabling Session-CURVE on a Deployment (operational note)

The wiring is inert until a site opts in. To enable end-to-end:

1. Set `umrs_zmq_auth_type: session` and a generated server CURVE keypair
   (`UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY` / `_SECRET_KEY`) in the Ansible
   group_vars / `site.yaml` for the target profile (e.g. `umrs-front`).
2. Redeploy via Ansible only — `cd deploy/ansible && ./bootstrap.sh --profile <profile>`.
   Never hand-edit or ad-hoc-SSH-restart (repo deployment-operations rules).

Fail-closed: if session mode is set but the server keypair is missing, the
server refuses to bind (loud failure) rather than silently downgrading.
