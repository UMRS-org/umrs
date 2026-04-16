import json
import os
import sys
from argparse import Namespace

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from client_umrs import _handle_config_command
from umrs.shared.json_config import load_client_cfg_file
from umrs.shared.loaded_config_profiles import (
    apply_s3_config_override,
    discover_loaded_config_profiles,
    resolve_loaded_config,
    set_current_s3_config,
    set_current_server_config,
    set_current_user_config,
)


def _write_s3cfg(path, access_key="user1"):
    path.write_text(
        "[default]\n"
        "host_base=127.0.0.1:9000\n"
        "use_https=false\n"
        "check_ssl_certificate=false\n"
        f"access_key={access_key}\n"
        "secret_key=test-secret\n"
        "bucket_location=us-east-1\n",
        encoding="utf-8",
    )


def _write_user_config(path, username, s3cfg_path):
    path.write_text(
        json.dumps(
            {
                "username": username,
                "fullname": f"{username} Full",
                "s3_configuration_file": str(s3cfg_path),
                "experiences": [
                    {
                        "xp_name": "xp1",
                        "sql_username": f"sql_{username}",
                        "modalities": [
                            {
                                "modality_name": "text",
                                "s3": {
                                    "bucket_alias": "bucket-1",
                                    "data": {
                                        "type": "inline",
                                        "mime_type": "text/plain",
                                    },
                                },
                                "model": {
                                    "model_name": "intfloat/e5-base-v2",
                                    "vector_size": "768",
                                    "model_type": "TransformerEncoder",
                                    "spacy_model": "en_core_web_sm",
                                    "return_type": None,
                                },
                                "sql": {
                                    "store_embeddings": "yes",
                                    "table_name": "text_embeddings",
                                    "meta_data": [
                                        {"name": "keywords", "type": "TEXT"}
                                    ],
                                },
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_server_config(path, admin_s3_path, user_paths):
    path.write_text(
        json.dumps(
            {
                "admin_s3_configuration_file": str(admin_s3_path),
                "users": [str(user_path) for user_path in user_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def config_workspace(tmp_path, monkeypatch):
    cfg_root = tmp_path / "umrs_cfg"
    users_dir = cfg_root / "users"
    project_dir = tmp_path / "project"
    users_dir.mkdir(parents=True)
    project_dir.mkdir()

    admin_s3cfg = cfg_root / "s3cfg-admin"
    user_s3cfg = cfg_root / "s3cfg-user1"
    alt_s3cfg = project_dir / "s3cfg-alt"
    _write_s3cfg(admin_s3cfg, access_key="admin")
    _write_s3cfg(user_s3cfg, access_key="user1")
    _write_s3cfg(alt_s3cfg, access_key="override")

    user_cfg = users_dir / "user1.json"
    alt_user_cfg = project_dir / "user_alt.json"
    _write_user_config(user_cfg, "user1", user_s3cfg)
    _write_user_config(alt_user_cfg, "user_alt", alt_s3cfg)

    server_cfg = cfg_root / "conf_umrs.json"
    alt_server_cfg = project_dir / "conf_umrs.json"
    _write_server_config(server_cfg, admin_s3cfg, [user_cfg])
    _write_server_config(alt_server_cfg, admin_s3cfg, [alt_user_cfg])

    monkeypatch.setenv("UMRS_CONFIG", str(cfg_root))
    monkeypatch.chdir(project_dir)

    return {
        "cfg_root": cfg_root,
        "project_dir": project_dir,
        "admin_s3cfg": admin_s3cfg,
        "user_s3cfg": user_s3cfg,
        "alt_s3cfg": alt_s3cfg,
        "user_cfg": user_cfg,
        "alt_user_cfg": alt_user_cfg,
        "server_cfg": server_cfg,
        "alt_server_cfg": alt_server_cfg,
    }


def test_resolve_loaded_config_prefers_current_s3_selection(config_workspace):
    set_current_s3_config(str(config_workspace["alt_s3cfg"]))

    resolved = resolve_loaded_config()

    assert resolved.server.valid
    assert resolved.user.valid
    assert resolved.s3.valid
    assert resolved.user.path == str(config_workspace["alt_user_cfg"])
    assert resolved.s3.path == str(config_workspace["alt_s3cfg"])
    assert resolved.s3.source.startswith("current:")

    user_cfg = load_client_cfg_file(str(config_workspace["alt_user_cfg"]))
    overridden = apply_s3_config_override(user_cfg, resolved.s3.path)
    assert overridden["s3_configuration_file"] == str(config_workspace["alt_s3cfg"])


def test_discover_loaded_config_profiles_marks_current_selection(config_workspace):
    set_current_server_config(str(config_workspace["server_cfg"]))
    set_current_user_config(str(config_workspace["user_cfg"]))

    discovered = discover_loaded_config_profiles()

    server_candidates = [candidate for candidate in discovered["server"] if candidate.current]
    user_candidates = [candidate for candidate in discovered["user"] if candidate.current]

    assert len(server_candidates) == 1
    assert server_candidates[0].path == str(config_workspace["server_cfg"])
    assert len(user_candidates) == 1
    assert user_candidates[0].path == str(config_workspace["user_cfg"])


def test_config_subcommand_use_and_show(config_workspace, capsys):
    use_args = Namespace(
        action="config",
        config_action="use",
        server=str(config_workspace["alt_server_cfg"]),
        user=str(config_workspace["alt_user_cfg"]),
        s3=str(config_workspace["alt_s3cfg"]),
        output="json",
        conf=None,
        server_conf=None,
        s3cfg=None,
    )
    assert _handle_config_command(use_args) == 0
    use_payload = json.loads(capsys.readouterr().out)
    assert use_payload["status"] == "ok"
    assert "server" in use_payload["updates"]
    assert "user" in use_payload["updates"]
    assert "s3" in use_payload["updates"]

    show_args = Namespace(
        action="config",
        config_action="show",
        output="json",
        conf=None,
        server_conf=None,
        s3cfg=None,
        server=None,
        user=None,
        s3=None,
    )
    assert _handle_config_command(show_args) == 0
    show_payload = json.loads(capsys.readouterr().out)
    config_payload = show_payload["config"]

    assert config_payload["server"]["path"] == str(config_workspace["alt_server_cfg"])
    assert config_payload["user"]["path"] == str(config_workspace["alt_user_cfg"])
    assert config_payload["user"]["details"]["username"] == "user_alt"
    assert config_payload["s3"]["path"] == str(config_workspace["alt_s3cfg"])