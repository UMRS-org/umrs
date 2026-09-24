"""Offline-first CRUD over ``user_files/<username>.json`` records.

This is the single read/write layer used by ``umrs-server admin`` subcommands.
It does NOT go over ZMQ; it manipulates the JSON files directly so the CLI
can be used for bootstrap, recovery, and maintenance with the daemon stopped.

Admin-only metadata (``role``, ``slurm_account``, ``slurm_accounts``,
``projects``, ``disabled``,
``email``, ``quota``) is stored as extra top-level keys on the same user JSON
file. The existing client schema does not enforce ``additionalProperties:
False`` at the top level, so the daemon continues to load these files
unchanged.
"""

from __future__ import annotations

import builtins
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umrs.shared.config import resolve_server_user_files_dir

logger = logging.getLogger(__name__)


_VALID_USERNAME_RE = re.compile(r"^[a-z][a-z0-9._-]{1,62}$")

VALID_ROLES = ("user", "admin", "service")


class UserStoreError(Exception):
    """Raised for user store failures the operator should see."""


@dataclass
class UserRecord:
    """In-memory view of a single ``user_files/<name>.json`` record."""

    username: str
    fullname: str = ""
    email: str = ""
    role: str = "user"
    slurm_account: str | None = None
    # Per-site Slurm account overrides keyed by site name (e.g.
    # ``{"rccs-front": "group-a", "ai4s": "group-b"}``).
    # When present, takes precedence over the flat ``slurm_account``.
    slurm_accounts: dict[str, str] = field(default_factory=dict)
    projects: list[str] = field(default_factory=list)
    disabled: bool = False
    quota: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    last_updated: str = ""
    # OIDC external identity subject (e.g. RIKEN ID "241723").  When set,
    # ``UserStore.resolve_user()`` can map an incoming OIDC ``sub`` claim
    # back to this UMRS username even though the subject value itself does
    # not satisfy the local username regex.
    oidc_subject: str = ""
    # Preserve any other top-level keys the daemon cares about
    # (s3_configuration_file, experiences, ...) verbatim.
    # D-03: long-lived per-user CURVE public key (Z85) — the account record
    # is the key authority; revocation timestamp empties/declares it dead.
    curve_public_key: str = ""
    curve_key_revoked_at: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "username": self.username,
            "fullname": self.fullname,
            "email": self.email,
            "role": self.role,
            "projects": list(self.projects),
            "disabled": bool(self.disabled),
            "quota": dict(self.quota),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }
        if self.oidc_subject:
            out["oidc_subject"] = self.oidc_subject
        if self.slurm_account:
            out["slurm_account"] = self.slurm_account
        if self.slurm_accounts:
            out["slurm_accounts"] = dict(self.slurm_accounts)
        if self.curve_public_key:
            out["curve_public_key"] = self.curve_public_key
        if self.curve_key_revoked_at:
            out["curve_key_revoked_at"] = self.curve_key_revoked_at
        # Extras (s3_configuration_file, experiences, ...) are restored last so
        # they never clobber the admin-managed fields above.
        for k, v in self.extras.items():
            out.setdefault(k, v)
        return out

    def resolve_slurm_account(self, site_name: str | None = None) -> str | None:
        """Return the Slurm account for *site_name*, falling back to the flat field."""
        if site_name and self.slurm_accounts:
            acct = self.slurm_accounts.get(site_name)
            if acct:
                return acct
        return self.slurm_account

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> UserRecord:
        known = {
            "username",
            "fullname",
            "email",
            "role",
            "slurm_account",
            "slurm_accounts",
            "projects",
            "disabled",
            "quota",
            "created_at",
            "last_updated",
            "oidc_subject",
            "curve_public_key",
            "curve_key_revoked_at",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        slurm_accounts = _parse_slurm_accounts(data.get("slurm_accounts"))
        return cls(
            username=str(data.get("username", "")),
            fullname=str(data.get("fullname", "")),
            email=str(data.get("email", "")),
            role=str(data.get("role", "user")),
            slurm_account=data.get("slurm_account") or None,
            slurm_accounts=slurm_accounts,
            projects=list(data.get("projects", []) or []),
            disabled=bool(data.get("disabled", False)),
            quota=dict(data.get("quota", {}) or {}),
            created_at=str(data.get("created_at", "")),
            last_updated=str(data.get("last_updated", "")),
            oidc_subject=str(data.get("oidc_subject", "")),
            curve_public_key=str(data.get("curve_public_key", "") or ""),
            curve_key_revoked_at=str(data.get("curve_key_revoked_at", "") or ""),
            extras=extras,
        )


def _parse_slurm_accounts(raw: Any) -> dict[str, str]:
    """Per-site account map; non-dict or empty entries read as absent."""
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if v}


def validate_username(username: str) -> None:
    """Reject names that would be unsafe as filenames or Slurm/SQL identifiers."""
    if not username or not _VALID_USERNAME_RE.match(username):
        raise UserStoreError(
            f"Invalid username {username!r}: must match {_VALID_USERNAME_RE.pattern}"
        )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class UserStore:
    """File-backed CRUD over ``<user_files>/<username>.json``."""

    def __init__(self, user_files_dir: str | Path | None = None):
        if user_files_dir is None:
            self._dir = Path(resolve_server_user_files_dir())
        else:
            self._dir = Path(user_files_dir).expanduser().resolve()

    @property
    def dir(self) -> Path:
        return self._dir

    # ----- low-level ------------------------------------------------------

    def _path(self, username: str) -> Path:
        validate_username(username)
        return self._dir / f"{username}.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _read(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, path: Path, data: dict[str, Any]) -> None:
        self._ensure_dir()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Best-effort on shared filesystems.
            pass
        self._pg_upsert(data)

    @staticmethod
    def _pg_upsert(data: dict[str, Any]) -> None:
        """D--06 dual-write: JSON stays the recovery lane; PG mirrors."""
        if _PG_POOL is None:
            return
        try:
            record = UserRecord.from_json(data)
            conn = _PG_POOL.getconn()
            try:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO users (venue, username, fullname, email, role, disabled,
                           slurm_account, slurm_accounts, projects, quota, oidc_subject,
                           curve_public_key, curve_key_revoked_at, extras,
                           created_at, last_updated)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (venue, username) DO UPDATE SET
                           fullname = EXCLUDED.fullname,
                           email = EXCLUDED.email,
                           role = EXCLUDED.role,
                           disabled = EXCLUDED.disabled,
                           slurm_account = EXCLUDED.slurm_account,
                           slurm_accounts = EXCLUDED.slurm_accounts,
                           projects = EXCLUDED.projects,
                           quota = EXCLUDED.quota,
                           oidc_subject = EXCLUDED.oidc_subject,
                           curve_public_key = EXCLUDED.curve_public_key,
                           curve_key_revoked_at = EXCLUDED.curve_key_revoked_at,
                           extras = EXCLUDED.extras,
                           last_updated = EXCLUDED.last_updated""",
                    _pg_row(record),
                )
                conn.commit()
            finally:
                _PG_POOL.putconn(conn)
        except Exception as exc:  # noqa: BLE001 — the JSON write already landed
            logger.warning(
                "user PG dual-write skipped for %s: %s", data.get("username"), exc
            )

    # ----- public API -----------------------------------------------------

    def exists(self, username: str) -> bool:
        return self._path(username).is_file()

    def get(self, username: str) -> UserRecord:
        # D-06: PG-first when the lane is wired; JSON stays the fallback
        # (identical behavior unwired — every pre-0110 test rides that).
        if pg_lane_active():
            record = pg_get(username)
            if record is not None:
                return record
            # fall through to JSON: a row not yet synced (e.g. created by
            # the offline CLI while the server is down) must still resolve.
        path = self._path(username)
        if not path.is_file():
            raise UserStoreError(f"User {username!r} not found at {path}")
        return UserRecord.from_json(self._read(path))

    def list(self) -> builtins.list[UserRecord]:
        if not self._dir.is_dir():
            return []
        records: list[UserRecord] = []
        for entry in sorted(self._dir.iterdir()):
            if not entry.is_file() or entry.suffix != ".json":
                continue
            if entry.name.startswith("_") or entry.name.startswith("."):
                continue
            # Skip auto-generated client config templates (*.client.json).
            if entry.name.endswith(".client.json"):
                continue
            try:
                records.append(UserRecord.from_json(self._read(entry)))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable user file %s: %s", entry, exc)
        return records

    def create(
        self,
        username: str,
        *,
        fullname: str = "",
        email: str = "",
        role: str = "user",
        slurm_account: str | None = None,
        projects: builtins.list[str] | None = None,
        quota: dict[str, Any] | None = None,
        oidc_subject: str = "",
        extras: dict[str, Any] | None = None,
    ) -> UserRecord:
        if role not in VALID_ROLES:
            raise UserStoreError(f"Invalid role {role!r}; must be one of {VALID_ROLES}")
        path = self._path(username)
        if path.exists():
            raise UserStoreError(f"User {username!r} already exists at {path}")
        # Reject duplicate OIDC subjects across different users.
        if oidc_subject:
            existing = self.lookup_by_oidc_subject(oidc_subject)
            if existing is not None:
                raise UserStoreError(
                    f"OIDC subject {oidc_subject!r} is already mapped to "
                    f"user {existing.username!r}"
                )
        now = _now_iso()
        record = UserRecord(
            username=username,
            fullname=fullname,
            email=email,
            role=role,
            slurm_account=slurm_account,
            projects=list(projects or []),
            disabled=False,
            quota=dict(quota or {}),
            created_at=now,
            last_updated=now,
            oidc_subject=oidc_subject,
            extras=dict(extras or {}),
        )
        self._write(path, record.to_json())
        logger.info("Created user %s at %s", username, path)
        return record

    def update(self, username: str, **changes: Any) -> UserRecord:
        """Merge ``changes`` into the record. Unknown fields go to ``extras``."""
        record = self.get(username)
        for key, value in changes.items():
            if key in {"username"}:
                raise UserStoreError("username is immutable; use rename()")
            if key == "role" and value not in VALID_ROLES:
                raise UserStoreError(
                    f"Invalid role {value!r}; must be one of {VALID_ROLES}"
                )
            if hasattr(record, key) and key != "extras":
                setattr(record, key, value)
            else:
                record.extras[key] = value
        record.last_updated = _now_iso()
        self._write(self._path(username), record.to_json())
        return record

    def set_disabled(self, username: str, disabled: bool) -> UserRecord:
        return self.update(username, disabled=bool(disabled))

    def set_role(self, username: str, role: str) -> UserRecord:
        return self.update(username, role=role)

    def set_slurm_account(self, username: str, slurm_account: str | None) -> UserRecord:
        return self.update(username, slurm_account=slurm_account or None)

    def set_oidc_subject(
        self,
        username: str,
        oidc_subject: str,
    ) -> UserRecord:
        """Set or clear the OIDC external identity subject for a user."""
        # Reject duplicate subjects when setting (not when clearing).
        if oidc_subject:
            existing = self.lookup_by_oidc_subject(oidc_subject)
            if existing is not None and existing.username != username:
                raise UserStoreError(
                    f"OIDC subject {oidc_subject!r} is already mapped to "
                    f"user {existing.username!r}"
                )
        return self.update(username, oidc_subject=oidc_subject)

    def lookup_by_oidc_subject(self, subject: str) -> UserRecord | None:
        """Return the user whose ``oidc_subject`` matches *subject*, or None."""
        if not subject:
            return None
        for rec in self.list():
            if rec.oidc_subject and rec.oidc_subject == subject:
                return rec
        return None

    def resolve_user(self, identity: str) -> UserRecord:
        """Look up a user by username first, then by OIDC subject.

        This is the primary entry point for resolving an incoming identity
        (JWT ``sub`` or ``X-User-ID`` header) to a local user record.
        It enables OIDC flows where the JWT ``sub`` is a numeric IdP
        identifier (e.g. RIKEN ID ``241723``) that does not pass the
        local username regex but has been registered as the user's
        ``oidc_subject``.
        """
        # Fast path: direct username lookup.
        try:
            return self.get(identity)
        except UserStoreError:
            pass
        # Slow path: scan for OIDC subject match.
        rec = self.lookup_by_oidc_subject(identity)
        if rec is not None:
            return rec
        raise UserStoreError(
            f"No user found for identity {identity!r} "
            "(checked username and oidc_subject)"
        )

    def set_slurm_account_for_site(
        self,
        username: str,
        site_name: str,
        slurm_account: str | None,
    ) -> UserRecord:
        """Set or clear the Slurm account for a specific site."""
        record = self.get(username)
        accounts = dict(record.slurm_accounts)
        if slurm_account:
            accounts[site_name] = slurm_account
        else:
            accounts.pop(site_name, None)
        return self.update(username, slurm_accounts=accounts)

    def add_project(self, username: str, project_id: str) -> UserRecord:
        record = self.get(username)
        if project_id not in record.projects:
            record.projects.append(project_id)
        return self.update(username, projects=record.projects)

    def remove_project(self, username: str, project_id: str) -> UserRecord:
        record = self.get(username)
        projects = [p for p in record.projects if p != project_id]
        return self.update(username, projects=projects)

    def delete(self, username: str) -> Path:
        """Hard delete the JSON file; returns the removed path."""
        path = self._path(username)
        if not path.is_file():
            raise UserStoreError(f"User {username!r} not found at {path}")
        path.unlink()
        logger.info("Deleted user %s (%s)", username, path)
        return path


__all__ = [
    "VALID_ROLES",
    "UserRecord",
    "UserStore",
    "UserStoreError",
    "validate_username",
]


# ── D-06 first leg: the Postgres-backed lane (migration 0110) ────────────────
#
# The JSON files remain the bootstrap/recovery lane (the offline-first CLI
# contract is unchanged). At boot the sidecar syncs JSON → Postgres; the
# runtime read seam goes PG-first once configured; writes dual-write so the
# JSON files never lag the truth the offline CLI needs. Unconfigured = the
# exact pre-0110 behavior (every existing test rides that path).

_PG_POOL: Any = None
# 0113: the site whose user directory this server owns. umrs-front's four
# servers share one umrs_db; every row read or written is scoped to it.
_PG_VENUE: str = ""


def configure_user_store_pg(
    pool: Any, store: UserStore | None = None, *, venue: str = ""
) -> int:
    """Wire the PG lane + sync JSON → Postgres; returns rows synced.

    ``venue`` is this server's ``site.name`` (0113): rows are keyed by
    (venue, username) so servers sharing one database never read or
    overwrite each other's accounts.

    Fail-open: a sync failure leaves the JSON lane authoritative (the
    pre-0110 behavior) — boot never dies on the accounts lane.
    """
    global _PG_POOL, _PG_VENUE
    _PG_POOL = pool
    _PG_VENUE = venue
    if pool is None:
        return 0
    synced = 0
    try:
        store = store or UserStore()
        conn = pool.getconn()
        try:
            cur = conn.cursor()
            for record in store.list():
                cur.execute(
                    """INSERT INTO users (venue, username, fullname, email, role, disabled,
                           slurm_account, slurm_accounts, projects, quota, oidc_subject,
                           curve_public_key, curve_key_revoked_at, extras,
                           created_at, last_updated)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (venue, username) DO UPDATE SET
                           fullname = EXCLUDED.fullname,
                           email = EXCLUDED.email,
                           role = EXCLUDED.role,
                           disabled = EXCLUDED.disabled,
                           slurm_account = EXCLUDED.slurm_account,
                           slurm_accounts = EXCLUDED.slurm_accounts,
                           projects = EXCLUDED.projects,
                           quota = EXCLUDED.quota,
                           oidc_subject = EXCLUDED.oidc_subject,
                           curve_public_key = EXCLUDED.curve_public_key,
                           curve_key_revoked_at = EXCLUDED.curve_key_revoked_at,
                           extras = EXCLUDED.extras,
                           last_updated = EXCLUDED.last_updated""",
                    _pg_row(record),
                )
                synced += 1
            conn.commit()
        finally:
            pool.putconn(conn)
        logger.info("user store PG lane synced (%d rows)", synced)
    except Exception as exc:  # noqa: BLE001 — fail-open by contract
        logger.warning("user store PG sync skipped (JSON stays authoritative): %s", exc)
        _PG_POOL = None
    return synced


def reset_user_store_pg() -> None:
    """Test/reset seam: unwire the PG lane."""
    global _PG_POOL, _PG_VENUE
    _PG_POOL = None
    _PG_VENUE = ""


def pg_venue() -> str:
    """The site scope of the wired PG lane (0113)."""
    return _PG_VENUE


def site_venue() -> str:
    """This deployment's ``site.name`` — the 0113 row scope ("" when the
    site config names no cluster, the JobService convention)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        return str(getattr(load_site_config(), "name", "") or "").strip()
    except Exception:  # noqa: BLE001 — sites without a config run unscoped
        return ""


def _pg_row(record: UserRecord) -> tuple:
    import json as _json

    return (
        _PG_VENUE,
        record.username,
        record.fullname,
        record.email,
        record.role,
        bool(record.disabled),
        record.slurm_account,
        _json.dumps(record.slurm_accounts or {}),
        _json.dumps(list(record.projects or [])),
        _json.dumps(record.quota or {}),
        record.oidc_subject,
        record.curve_public_key,
        record.curve_key_revoked_at,
        _json.dumps(record.extras or {}),
        record.created_at,
        record.last_updated,
    )


def pg_lane_active() -> bool:
    return _PG_POOL is not None


def _jsonb(value: Any, default: Any) -> Any:
    """A JSONB column as psycopg returns it: decoded already, or a JSON string."""
    if isinstance(value, str):
        import json as _json

        return _json.loads(value) if value else default
    return value or default


def pg_get(username: str) -> UserRecord | None:
    """PG-first read; None when absent/unwired (the caller falls back)."""
    if _PG_POOL is None:
        return None
    conn = _PG_POOL.getconn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT username, fullname, email, role, disabled, slurm_account,"
            " slurm_accounts, projects, quota, oidc_subject, curve_public_key,"
            " curve_key_revoked_at, extras, created_at, last_updated"
            " FROM users WHERE venue = %s AND username = %s",
            (_PG_VENUE, username),
        )
        row = cur.fetchone()
    finally:
        _PG_POOL.putconn(conn)
    if row is None:
        return None
    return UserRecord(
        username=row[0],
        fullname=row[1],
        email=row[2],
        role=row[3],
        slurm_account=row[5],
        slurm_accounts=_jsonb(row[6], {}),
        projects=list(_jsonb(row[7], [])),
        disabled=bool(row[4]),
        quota=_jsonb(row[8], {}),
        oidc_subject=row[9],
        curve_public_key=row[10],
        curve_key_revoked_at=row[11],
        extras=_jsonb(row[12], {}),
        created_at=row[13],
        last_updated=row[14],
    )
