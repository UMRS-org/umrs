"""Setup scripts install umrs-protocol and r4-contracts from local paths only.

Neither package (nor umrs) is published, so a package-index fallback installs
whatever someone registered under the name (dependency confusion). Since
U2.0 U-A0 (ADR-117 D-2) the server setup installs ``protocol/`` before
r4-contracts in the same pip command and fails when either checkout is
missing; the client setups and the umrs-front setup install both before
resolving umrs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

UMRS_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPTS = ("deploy/_common/setup_server.sh", "deploy/hpbd-atom/setup_server.sh")
CLIENT_SCRIPTS = ("deploy/_common/setup_client.sh", "deploy/hpbd-atom/setup_client.sh")
FRONT_SCRIPT = "deploy/umrs-front/setup.sh"


def _code(rel: str) -> list[str]:
    """Non-comment lines of a script."""
    lines = (UMRS_ROOT / rel).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.lstrip().startswith("#")]


def _first(lines: list[str], pattern: str) -> int:
    return next(i for i, line in enumerate(lines) if re.search(pattern, line))


@pytest.mark.parametrize("rel", SERVER_SCRIPTS)
def test_server_installs_protocol_first_in_one_pip_command(rel: str) -> None:
    lines = _code(rel)
    install = [line for line in lines if re.search(r"pip install .*R4_CONTRACTS_DIR", line)]
    assert install == ['pip install --quiet -e "${PROTOCOL_DIR}" -e "${R4_CONTRACTS_DIR}"']
    assert 'PROTOCOL_DIR="${REPO_ROOT}/protocol"' in lines
    assert _first(lines, r"pip install .*PROTOCOL_DIR") < _first(
        lines, r'pip install .*"\$\{SRC_DIR\}"'
    )


@pytest.mark.parametrize("rel", SERVER_SCRIPTS)
def test_server_has_no_package_index_fallback(rel: str) -> None:
    text = "\n".join(_code(rel))
    assert "package index" not in text
    assert re.search(r'\|\| die "umrs-protocol not found', text)
    assert re.search(r'\|\| die "r4-contracts not found', text)


@pytest.mark.parametrize("rel", SERVER_SCRIPTS)
def test_server_verifies_umrs_protocol_import(rel: str) -> None:
    lines = _code(rel)
    assert _first(lines, r"^import umrs_protocol;") < _first(lines, r"^import r4_contracts;")


@pytest.mark.parametrize("rel", CLIENT_SCRIPTS)
def test_client_installs_local_contracts_before_umrs(rel: str) -> None:
    lines = _code(rel)
    local = _first(lines, r'uv pip install -e "\$\{PROTOCOL_DIR\}" -e "\$\{R4_CONTRACTS_DIR\}"')
    assert local < _first(lines, r'uv pip install -e "\$\{SRC_DIR\}"')
    assert 'PROTOCOL_DIR="${SCRIPT_DIR}/../../protocol"' in [line.strip() for line in lines]
    assert any("exit 1" in line for line in lines[:local])


def test_front_installs_local_contracts_before_umrs() -> None:
    lines = _code(FRONT_SCRIPT)
    local = _first(lines, r'uv pip install -e "\$\{PROTOCOL_DIR\}" -e "\$\{R4_CONTRACTS_DIR\}"')
    assert local < _first(lines, r'uv pip install -e "\$\{SRC_DIR\}\[all\]"')
    assert 'PROTOCOL_DIR="${REPO_ROOT}/protocol"' in lines
    text = "\n".join(lines)
    assert re.search(r'\|\| die "umrs-protocol not found', text)
    assert re.search(r'\|\| die "r4-contracts not found', text)


def test_protocol_dir_the_scripts_install_exists() -> None:
    assert (UMRS_ROOT / "protocol" / "pyproject.toml").is_file()
