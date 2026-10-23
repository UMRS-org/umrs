"""Tests for ``umrs-server admin`` CLI + user_store + Slurm account plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from umrs.admin.user_store import (
    VALID_ROLES,
    UserStore,
    UserStoreError,
    validate_username,
)
from umrs.cli.admin.main import build_parser
from umrs.cli.admin.main import main as admin_main
from umrs.cli.completion import completion_candidates

# ---------------------------------------------------------------------------
# user_store
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> UserStore:
    return UserStore(user_files_dir=tmp_path / "user_files")


class TestUsernameValidation:
    @pytest.mark.parametrize("name", ["alice", "alice-bob", "user_1", "a0"])
    def test_valid(self, name: str) -> None:
        validate_username(name)

    @pytest.mark.parametrize(
        "name",
        ["", "Alice", "1user", "../etc/shadow", "user name", "user@host", "a" * 100],
    )
    def test_invalid(self, name: str) -> None:
        with pytest.raises(UserStoreError):
            validate_username(name)


class TestUserStoreCRUD:
    def test_create_round_trip(self, store: UserStore) -> None:
        rec = store.create(
            "alice",
            fullname="Alice",
            email="a@example.org",
            role="user",
            slurm_account="lab-alpha",
            projects=["proj-a"],
        )
        assert rec.username == "alice"
        assert rec.slurm_account == "lab-alpha"
        assert (store.dir / "alice.json").exists()

        loaded = store.get("alice")
        assert loaded.fullname == "Alice"
        assert loaded.email == "a@example.org"
        assert loaded.projects == ["proj-a"]
        assert loaded.role == "user"
        assert loaded.disabled is False
        assert loaded.created_at and loaded.last_updated

    def test_create_rejects_duplicate(self, store: UserStore) -> None:
        store.create("alice")
        with pytest.raises(UserStoreError):
            store.create("alice")

    def test_create_rejects_invalid_role(self, store: UserStore) -> None:
        with pytest.raises(UserStoreError):
            store.create("alice", role="root")

    def test_list_sorted_and_skips_hidden(self, store: UserStore) -> None:
        store.create("bob")
        store.create("alice")
        # Files that must be ignored by list().
        (store.dir / "_notes.json").write_text("{}")
        (store.dir / ".hidden.json").write_text("{}")
        (store.dir / "readme.txt").write_text("ignore me")
        names = [r.username for r in store.list()]
        assert names == ["alice", "bob"]

    def test_disable_enable_roundtrip(self, store: UserStore) -> None:
        store.create("alice")
        assert store.set_disabled("alice", True).disabled is True
        assert store.set_disabled("alice", False).disabled is False

    def test_set_role(self, store: UserStore) -> None:
        store.create("alice")
        rec = store.set_role("alice", "admin")
        assert rec.role == "admin"
        with pytest.raises(UserStoreError):
            store.set_role("alice", "root")

    def test_set_slurm_account(self, store: UserStore) -> None:
        store.create("alice")
        rec = store.set_slurm_account("alice", "lab-alpha")
        assert rec.slurm_account == "lab-alpha"
        rec = store.set_slurm_account("alice", None)
        assert rec.slurm_account is None
        # Cleared field must not linger in the JSON payload.
        raw = json.loads((store.dir / "alice.json").read_text())
        assert "slurm_account" not in raw

    def test_set_slurm_account_for_site(self, store: UserStore) -> None:
        store.create("alice")
        rec = store.set_slurm_account_for_site("alice", "rccs-front", "group-a")
        assert rec.slurm_accounts == {"rccs-front": "group-a"}
        rec = store.set_slurm_account_for_site("alice", "ai4s", "group-b")
        assert rec.slurm_accounts == {"rccs-front": "group-a", "ai4s": "group-b"}
        # Clear one site
        rec = store.set_slurm_account_for_site("alice", "rccs-front", None)
        assert rec.slurm_accounts == {"ai4s": "group-b"}
        # Verify JSON round-trip
        raw = json.loads((store.dir / "alice.json").read_text())
        assert raw["slurm_accounts"] == {"ai4s": "group-b"}

    def test_resolve_slurm_account_per_site(self, store: UserStore) -> None:
        store.create("alice", slurm_account="default-group")
        store.set_slurm_account_for_site("alice", "rccs-front", "group-a")
        rec = store.get("alice")
        # Per-site takes precedence
        assert rec.resolve_slurm_account("rccs-front") == "group-a"
        # Unknown site falls back to flat field
        assert rec.resolve_slurm_account("unknown-site") == "default-group"
        # No site name falls back to flat field
        assert rec.resolve_slurm_account(None) == "default-group"
        assert rec.resolve_slurm_account() == "default-group"

    def test_project_membership(self, store: UserStore) -> None:
        store.create("alice")
        store.add_project("alice", "proj-a")
        store.add_project("alice", "proj-a")  # idempotent
        store.add_project("alice", "proj-b")
        rec = store.get("alice")
        assert rec.projects == ["proj-a", "proj-b"]
        store.remove_project("alice", "proj-a")
        assert store.get("alice").projects == ["proj-b"]

    def test_update_preserves_extras(self, store: UserStore) -> None:
        store.create(
            "alice",
            extras={
                "s3_configuration_file": "~/.s3cfg",
                "experiences": [{"xp_name": "x"}],
            },
        )
        # Update a managed field; extras must survive.
        store.set_role("alice", "admin")
        raw = json.loads((store.dir / "alice.json").read_text())
        assert raw["s3_configuration_file"] == "~/.s3cfg"
        assert raw["experiences"] == [{"xp_name": "x"}]
        assert raw["role"] == "admin"

    def test_delete_hard(self, store: UserStore) -> None:
        store.create("alice")
        store.delete("alice")
        assert not (store.dir / "alice.json").exists()
        with pytest.raises(UserStoreError):
            store.get("alice")

    def test_get_missing_raises(self, store: UserStore) -> None:
        with pytest.raises(UserStoreError):
            store.get("ghost")

    def test_valid_roles_contract(self) -> None:
        assert set(VALID_ROLES) == {"user", "admin", "service"}


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


class TestAdminParser:
    def test_user_add_parses_all_flags(self) -> None:
        p = build_parser()
        args = p.parse_args(
            [
                "user",
                "add",
                "alice",
                "--fullname",
                "Alice A",
                "--email",
                "a@example.org",
                "--role",
                "admin",
                "--slurm-account",
                "lab-alpha",
                "--project",
                "proj-a",
                "--project",
                "proj-b",
                "--json",
            ]
        )
        assert args.group == "user"
        assert args.action == "add"
        assert args.username == "alice"
        assert args.fullname == "Alice A"
        assert args.role == "admin"
        assert args.slurm_account == "lab-alpha"
        assert args.project == ["proj-a", "proj-b"]
        assert args.json is True

    def test_user_delete_requires_confirm_flag(self) -> None:
        p = build_parser()
        args = p.parse_args(["user", "delete", "alice", "--purge", "-y"])
        assert args.purge is True
        assert args.yes is True

    def test_user_set_slurm_account_accepts_optional_arg(self) -> None:
        p = build_parser()
        args = p.parse_args(["user", "set-slurm-account", "alice", "lab-alpha"])
        assert args.account == "lab-alpha"
        args = p.parse_args(["user", "set-slurm-account", "alice"])
        assert args.account is None

    def test_token_subcommands(self) -> None:
        p = build_parser()
        assert p.parse_args(["token", "generate", "boot"]).action == "generate"
        assert p.parse_args(["token", "list"]).action == "list"
        assert p.parse_args(["token", "revoke", "boot", "-y"]).yes is True

    def test_bootstrap_defaults(self) -> None:
        p = build_parser()
        args = p.parse_args(["bootstrap"])
        assert args.admin_user == "admin"
        assert args.token_name == "bootstrap"

    def test_completion_includes_nested_user_actions(self) -> None:
        p = build_parser()
        assert "user" in completion_candidates(p, [""], 0)
        assert "add" in completion_candidates(p, ["user", ""], 1)
        assert "list" in completion_candidates(p, ["user", ""], 1)

    def test_generate_jwt_uses_persisted_gateway_secret(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from argparse import Namespace

        from umrs.cli.admin.main import _resolve_jwt_secret

        gateway_dir = tmp_path / "xdg" / "umrs" / "gateway"
        gateway_dir.mkdir(parents=True)
        (gateway_dir / "jwt.secret").write_text("gateway-secret\n")
        monkeypatch.delenv("JWT_SECRET", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        assert _resolve_jwt_secret(Namespace(secret=None)) == "gateway-secret"


# ---------------------------------------------------------------------------
# end-to-end CLI invocations against a temp user_files dir
# ---------------------------------------------------------------------------


def _run(tmp: Path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp), *argv])


class TestAdminCLI:
    def test_user_add_list_show_disable_delete(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        d = tmp_path / "ufiles"
        assert _run(d, "user", "add", "alice", "--slurm-account", "lab-alpha") == 0
        assert _run(d, "user", "list") == 0
        out = capsys.readouterr().out
        assert "alice" in out and "lab-alpha" in out

        assert _run(d, "user", "show", "alice") == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown["username"] == "alice"
        assert shown["slurm_account"] == "lab-alpha"

        assert _run(d, "user", "disable", "alice") == 0
        assert UserStore(d).get("alice").disabled is True

        # Soft delete == disable; file stays.
        assert _run(d, "user", "delete", "alice") == 0
        assert (d / "alice.json").exists()

        # Purge requires explicit --yes (or interactive confirm).
        assert _run(d, "user", "delete", "alice", "--purge", "-y") == 0
        assert not (d / "alice.json").exists()

    def test_user_delete_deprovision_uses_persisted_extras(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import importlib

        admin_main_module = importlib.import_module("umrs.cli.admin.main")

        captured: dict[str, Any] = {}

        def fake_deprovision_user(user_cfg: dict[str, Any], admin_s3_cfg: str) -> None:
            captured["user_cfg"] = user_cfg
            captured["admin_s3_cfg"] = admin_s3_cfg

        d = tmp_path / "ufiles"
        store = UserStore(d)
        store.create(
            "alice",
            extras={
                "experiences": [
                    {
                        "xp_name": "omics",
                        "sql_username": "alice_db",
                        "modalities": [
                            {
                                "s3": {
                                    "bucket_alias": "input",
                                    "need_output_bucket": True,
                                }
                            }
                        ],
                    }
                ],
                "s3_configuration_file": "~/.s3cfg_alice",
            },
        )

        monkeypatch.setitem(
            sys.modules,
            "umrs.server.provisioning",
            type(
                "_M",
                (),
                {
                    "deprovision_user": staticmethod(fake_deprovision_user),
                },
            )(),
        )
        monkeypatch.setattr(admin_main_module, "_maybe_signal_reload", lambda _args: None)

        assert _run(d, "user", "delete", "alice", "--purge", "--deprovision", "-y") == 0
        out = capsys.readouterr().out

        assert captured["user_cfg"]["username"] == "alice"
        assert captured["user_cfg"]["experiences"][0]["xp_name"] == "omics"
        assert captured["user_cfg"]["s3_configuration_file"] == "~/.s3cfg_alice"
        assert captured["admin_s3_cfg"] == ""
        assert "deprovisioned and deleted" in out
        assert not (d / "alice.json").exists()

    def test_user_add_duplicate_returns_error(self, tmp_path: Path) -> None:
        d = tmp_path / "ufiles"
        assert _run(d, "user", "add", "alice") == 0
        assert _run(d, "user", "add", "alice") == 2

    def test_token_list_shows_expiration_metadata(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import importlib

        class FakeManager:
            def list_tokens(self) -> list[dict[str, str]]:
                return [
                    {
                        "name": "bootstrap",
                        "description": "Bootstrap token",
                        "created_at": "2026-05-08T00:00:00+00:00",
                        "expires_at": "never",
                    },
                    {
                        "name": "temporary",
                        "description": "Temporary access",
                        "created_at": "2026-05-08T00:00:00+00:00",
                        "expires_at": "2026-06-07T00:00:00+00:00",
                    },
                ]

        admin_main_module = importlib.import_module("umrs.cli.admin.main")
        monkeypatch.setattr(admin_main_module, "_token_manager", lambda: FakeManager())

        d = tmp_path / "ufiles"
        assert _run(d, "token", "list") == 0
        out = capsys.readouterr().out

        assert "expires_at" in out
        assert "bootstrap" in out
        assert "never" in out
        assert "2026-06-07T00:00:00+00:00" in out


# ---------------------------------------------------------------------------
# Phase 4: Slurm account plumbing
# ---------------------------------------------------------------------------


class TestSlurmAccountDirective:
    """Per-submission Slurm account must reach the sbatch script."""

    def test_slurm_target_prefers_spec_account(self) -> None:
        from umrs.orchestrator.hybrid.targets import JobSpec, SlurmTarget

        target = SlurmTarget(partition="gpu", account="site-default")
        spec = JobSpec(
            name="test",
            execution_authority="slurm",
            gpus=1,
            cpus=4,
            memory_mb=8192,
            slurm_account="alice-proj",
        )
        script = target._build_allocation_script(spec)
        assert "#SBATCH --account=alice-proj" in script
        assert "#SBATCH --account=site-default" not in script

    def test_slurm_target_falls_back_to_instance_account(self) -> None:
        from umrs.orchestrator.hybrid.targets import JobSpec, SlurmTarget

        target = SlurmTarget(partition="gpu", account="site-default")
        spec = JobSpec(name="test", execution_authority="slurm")
        script = target._build_allocation_script(spec)
        assert "#SBATCH --account=site-default" in script

    def test_slurm_target_omits_when_unset(self) -> None:
        from umrs.orchestrator.hybrid.targets import JobSpec, SlurmTarget

        target = SlurmTarget(partition="gpu", account="")
        spec = JobSpec(name="test", execution_authority="slurm")
        script = target._build_allocation_script(spec)
        assert "#SBATCH --account" not in script

    def test_k8s_translator_emits_account_when_default_set(self) -> None:
        from umrs.orchestrator.hybrid.k8s_slurm_scheduler import (
            K8sResourceSpec,
            K8sToSlurmTranslator,
        )

        tr = K8sToSlurmTranslator(default_account="alice-proj")
        pod = K8sResourceSpec(
            api_version="v1",
            kind="Pod",
            metadata={"name": "hello"},
            spec={
                "containers": [
                    {
                        "name": "c",
                        "image": "busybox",
                        "resources": {"limits": {"cpu": "1", "memory": "1G"}},
                    }
                ]
            },
        )
        translated = tr.translate(pod)
        assert "#SBATCH --account=alice-proj" in translated.slurm_script

    def test_k8s_translator_omits_account_by_default(self) -> None:
        from umrs.orchestrator.hybrid.k8s_slurm_scheduler import (
            K8sResourceSpec,
            K8sToSlurmTranslator,
        )

        tr = K8sToSlurmTranslator()
        pod = K8sResourceSpec(
            api_version="v1",
            kind="Pod",
            metadata={"name": "hello"},
            spec={"containers": [{"name": "c", "image": "busybox"}]},
        )
        translated = tr.translate(pod)
        assert "#SBATCH --account" not in translated.slurm_script


class TestUserContextSlurmAccount:
    def test_from_headers_parses_slurm_account(self) -> None:
        from umrs.shared.isolation import UserContext

        ctx = UserContext.from_headers(
            x_user_id="alice",
            x_project_id="proj-a",
            x_user_role="user",
            x_slurm_account="lab-alpha",
        )
        assert ctx.slurm_account == "lab-alpha"

    def test_from_headers_empty_slurm_account_becomes_none(self) -> None:
        from umrs.shared.isolation import UserContext

        ctx = UserContext.from_headers(
            x_user_id="alice",
            x_project_id="proj-a",
            x_slurm_account="   ",
        )
        assert ctx.slurm_account is None


def test_cli_refuses_a_subject_that_is_another_username(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-097 Amendment D-8 (2026-10-23): one identity may not name two
    accounts; both CLI paths print the store's refusal and exit 2."""
    d = tmp_path / "ufiles"
    assert _run(d, "user", "add", "alice") == 0
    assert _run(d, "user", "add", "bob") == 0
    capsys.readouterr()
    assert _run(d, "user", "set-oidc-subject", "bob", "alice") == 2
    assert "username of another user" in capsys.readouterr().err
    assert _run(d, "user", "add", "carol", "--oidc-subject", "alice") == 2
    assert "username of another user" in capsys.readouterr().err
    assert not (d / "carol.json").exists()
    assert UserStore(d).get("bob").oidc_subject == ""
