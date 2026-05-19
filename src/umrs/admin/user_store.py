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
    # Preserve any other top-level keys the daemon cares about
    # (s3_configuration_file, experiences, ...) verbatim.
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
        if self.slurm_account:
            out["slurm_account"] = self.slurm_account
        if self.slurm_accounts:
            out["slurm_accounts"] = dict(self.slurm_accounts)
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
        }
        extras = {k: v for k, v in data.items() if k not in known}
        raw_accounts = data.get("slurm_accounts")
        slurm_accounts: dict[str, str] = {}
        if isinstance(raw_accounts, dict):
            slurm_accounts = {
                str(k): str(v) for k, v in raw_accounts.items() if v
            }
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
            extras=extras,
        )


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

    # ----- public API -----------------------------------------------------

    def exists(self, username: str) -> bool:
        return self._path(username).is_file()

    def get(self, username: str) -> UserRecord:
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
        extras: dict[str, Any] | None = None,
    ) -> UserRecord:
        if role not in VALID_ROLES:
            raise UserStoreError(
                f"Invalid role {role!r}; must be one of {VALID_ROLES}"
            )
        path = self._path(username)
        if path.exists():
            raise UserStoreError(f"User {username!r} already exists at {path}")
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

    def set_slurm_account(
        self, username: str, slurm_account: str | None
    ) -> UserRecord:
        return self.update(username, slurm_account=slurm_account or None)

    def set_slurm_account_for_site(
        self, username: str, site_name: str, slurm_account: str | None,
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
    "UserRecord",
    "UserStore",
    "UserStoreError",
    "VALID_ROLES",
    "validate_username",
]
