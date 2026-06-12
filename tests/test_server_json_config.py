import json
import os

from umrs.server.json_config import load_server_cfg_file, load_server_cfg_json
from umrs.shared.config import get_resolved_umrs_cfg_dir, resolve_server_user_files_dir


def _write_client_config(path, username="user1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "username": username,
        "fullname": f"{username} Example",
        "s3_configuration_file": "~/.s3cfg",
        "experiences": [
            {
                "xp_name": "xp1",
                "sql_username": f"sql_{username}",
                "modalities": [
                    {
                        "modality": "text",
                        "task": "embedding",
                        "s3": {
                            "bucket_alias": "bucket1",
                            "data": {
                                "type": "file",
                                "mime_type": "text/plain",
                            },
                        },
                        "model": {
                            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                            "vector_size": "384",
                            "model_type": "sentence_transformer",
                            "return_type": None,
                        },
                        "sql": {
                            "store_embeddings": "yes",
                            "table_name": "documents",
                            "meta_data": [
                                {"name": "filename", "type": "text"},
                            ],
                        },
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_get_resolved_umrs_cfg_dir_uses_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.delenv("UMRS_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    resolved = get_resolved_umrs_cfg_dir()

    assert resolved == os.path.realpath(tmp_path / "xdg" / "umrs")


def test_load_server_cfg_file_resolves_relative_paths(monkeypatch, tmp_path):
    conf_dir = tmp_path / "deploy"
    user_dir = conf_dir / "user_files"
    user_cfg = user_dir / "user1.json"
    conf_path = conf_dir / "conf_umrs.json"

    _write_client_config(user_cfg)
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(
        json.dumps(
            {
                "admin_s3_configuration_file": "./admin.s3cfg",
                "user_files_dir": "./user_files",
                "users": ["./user_files/user1.json"],
            }
        ),
        encoding="utf-8",
    )

    other_dir = tmp_path / "other-cwd"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    admin_cfg, users, server_cfg = load_server_cfg_file(str(conf_path))

    assert admin_cfg == os.path.realpath(conf_dir / "admin.s3cfg")
    assert server_cfg["user_files_dir"] == os.path.realpath(user_dir)
    assert server_cfg["users"] == [os.path.realpath(user_cfg)]
    assert users[0]["config_file_path"] == os.path.realpath(user_cfg)


def test_resolve_server_user_files_dir_prefers_env_override(monkeypatch, tmp_path):
    conf_path = tmp_path / "conf_umrs.json"
    override_dir = tmp_path / "override-users"

    monkeypatch.setenv("UMRS_USER_FILES_DIR", str(override_dir))

    resolved = resolve_server_user_files_dir(
        str(conf_path),
        server_cfg={"user_files_dir": "./configured-users"},
        user_config_map={"user1": str(tmp_path / "existing" / "user1.json")},
    )

    assert resolved == os.path.realpath(override_dir)


def test_resolve_server_user_files_dir_uses_existing_user_config_parent(tmp_path):
    conf_path = tmp_path / "conf_umrs.json"
    existing_dir = tmp_path / "deploy" / "user_files"
    existing_dir.mkdir(parents=True)

    resolved = resolve_server_user_files_dir(
        str(conf_path),
        user_config_map={
            "user1": str(existing_dir / "user1.json"),
            "user2": str(existing_dir / "user2.json"),
        },
    )

    assert resolved == os.path.realpath(existing_dir)


def test_load_server_cfg_json_does_not_require_user_files_to_exist(tmp_path):
    conf_path = tmp_path / "conf_umrs.json"
    conf_path.write_text(
        json.dumps(
            {
                "admin_s3_configuration_file": "./admin.s3cfg",
                "user_files_dir": "./user_files",
                "users": ["./user_files/missing-user.json"],
            }
        ),
        encoding="utf-8",
    )

    server_cfg = load_server_cfg_json(str(conf_path))

    assert server_cfg["user_files_dir"] == os.path.realpath(tmp_path / "user_files")
    assert server_cfg["users"] == [os.path.realpath(tmp_path / "user_files" / "missing-user.json")]
