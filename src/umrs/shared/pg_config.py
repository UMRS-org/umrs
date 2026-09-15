"""Shared Postgres config reader (jobs-tier-safe home; ADR-053).

Rehomed from `umrs/server/config.py` — the package boundary (`umrs.jobs` may
not import `umrs.api`/`umrs.server`) needs this reader below `umrs.shared`.
Implementation is byte-exactly the original (env-first + optional INI file).
"""

from __future__ import annotations

import configparser
import os

from umrs.shared.config import get_resolved_umrs_subdir

_DB_HOST = "localhost"
_DB_PORT = "5432"
_DB_SUPERUSER = "postgres"
_UMRS_SERVER_DB_NAME = "umrs"
_DB_CREDENTIALS_FILENAME = os.path.join("credentials", "db_credentials.ini")


def get_db_config() -> dict:
    """Read DB connection config from env vars, with an optional creds file.

    Raises ValueError if no superuser password can be resolved. See the
    matching template in `umrs/server/config.py`'s docstring for the full
    env/INI matrix (identical semantics).
    """
    creds_file = get_resolved_umrs_subdir(_DB_CREDENTIALS_FILENAME)

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if os.path.isfile(creds_file):
        config.read(creds_file)

    def _get(env_key: str, ini_key: str, default: str | None = None) -> str | None:
        # Env var takes priority over credentials file
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        return config.get("postgres", ini_key, fallback=default)

    superuser_password = _get(
        "DB_SUPERUSER_PASSWORD",
        "superuser_password",
        os.environ.get("POSTGRES_PASSWORD"),
    )
    if not superuser_password:
        raise ValueError(
            "superuser_password is required: set DB_SUPERUSER_PASSWORD / "
            "POSTGRES_PASSWORD or the [postgres] section of the credentials file"
        )

    # umrs_server / per-experiment user passwords fall back to the superuser
    # password when not explicitly provided (env-only deploys).
    umrs_server_password = _get(
        "DB_UMRS_SERVER_PASSWORD",
        "umrs_server_password",
        superuser_password,
    )

    user_password = _get(
        "DB_USER_PASSWORD",
        "user_password",
        superuser_password,
    )

    return {
        "host": _get("DB_HOST", "host", _DB_HOST),
        "port": _get("DB_PORT", "port", _DB_PORT),
        "superuser": _get(
            "DB_SUPERUSER", "superuser", os.environ.get("POSTGRES_USER", _DB_SUPERUSER)
        ),
        "superuser_password": superuser_password,
        "umrs_server_password": umrs_server_password,
        "user_password": user_password,
        "db_name": _get(
            "DB_UMRS_NAME",
            "db_name",
            os.environ.get("DB_UMRS_NAME", _UMRS_SERVER_DB_NAME),
        ),
    }
