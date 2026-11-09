"""SSH transport identity hygiene (ADR-112 A.3.6, wave A).

Two pieces make one login's connection unusable by another: a key-pinned
connection offers only that key (``IdentitiesOnly=yes``), and the automatic
ControlMaster socket names the login, so a second login's ``ssh`` never
attaches to the first login's master and runs as it.
"""

from __future__ import annotations

import os

from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager


def _options(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "-o"]


def test_identity_file_pins_identities_only():
    mgr = SSHControlManager(
        SSHControlConfig(ssh_target="cluster", ssh_user="op", identity_file="/k/id")
    )
    argv = mgr.command_argv("true")
    assert "IdentitiesOnly=yes" in _options(argv)
    assert argv[argv.index("-i") + 1] == "/k/id"


def test_no_identity_file_leaves_ssh_defaults():
    argv = SSHControlManager(SSHControlConfig(ssh_target="cluster")).command_argv("true")
    assert "-i" not in argv
    assert not any(o.startswith("IdentitiesOnly") for o in _options(argv))


def test_automatic_control_path_names_the_login(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    alice = SSHControlManager(SSHControlConfig(ssh_target="cluster", ssh_user="alice"))
    bob = SSHControlManager(SSHControlConfig(ssh_target="cluster", ssh_user="bob"))
    assert alice.control_path == os.path.join(str(tmp_path), "umrs-ssh-cluster-alice-%p")
    assert alice.control_path != bob.control_path


def test_automatic_control_path_without_a_login(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    mgr = SSHControlManager(SSHControlConfig(ssh_target="cluster"))
    assert mgr.control_path == os.path.join(str(tmp_path), "umrs-ssh-cluster-%p")


def test_an_explicit_control_path_stands():
    mgr = SSHControlManager(
        SSHControlConfig(ssh_target="cluster", ssh_user="op", control_path="/run/umrs-ssh/c-%p")
    )
    assert mgr.control_path == "/run/umrs-ssh/c-%p"
    assert "ControlPath=/run/umrs-ssh/c-%p" in _options(mgr.command_argv("true"))
