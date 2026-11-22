"""U4: the pre-merge-main upgrade check (scripts/migration_upgrade_check.py).

Runs the script as CI does (a subprocess; ADR-053: no spec loading) on a
scratch database. The "old" directory is synthetic: the current files up to
0054 without 0016/0018/0019, the shape of a pre-merge ``main`` database.
CI also runs the script on the real archived files (tag
``archive/main-pre-merge-2026-11``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "migration_upgrade_check.py"
GAPS = {"0016", "0018", "0019"}

pytestmark = pytest.mark.integration


@pytest.fixture
def scratch_db(db_url: str) -> Iterator[str]:
    psycopg2 = pytest.importorskip("psycopg2")
    name = f"umrs_upgrade_check_{abs(hash(db_url)) % 10**8}"
    admin = psycopg2.connect(db_url)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        cur.execute(f'CREATE DATABASE "{name}"')
    try:
        yield urlunsplit(urlsplit(db_url)._replace(path=f"/{name}"))
    finally:
        with admin.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


def _pre_merge_dir(tmp_path: Path) -> Path:
    from umrs.migrations import get_migration_files

    old = tmp_path / "old"
    old.mkdir()
    for version, path in get_migration_files():
        if version <= "0054" and version not in GAPS:
            shutil.copy(path, old / path.name)
    return old


def _run(db: str, old: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), db, str(old)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_pre_merge_database_upgrades_with_the_gaps_out_of_order(
    scratch_db: str, tmp_path: Path
) -> None:
    done = _run(scratch_db, _pre_merge_dir(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "out-of-order gaps ['0016', '0018', '0019']" in done.stdout


def test_a_version_the_current_line_lacks_is_reported(scratch_db: str, tmp_path: Path) -> None:
    old = _pre_merge_dir(tmp_path)
    (old / "0049_dropped_on_merge.sql").write_text("SELECT 1;\n")
    done = _run(scratch_db, old)
    assert done.returncode == 1
    assert "schema_migrations does not match the current files" in done.stdout
