import json
import os

from jsonschema import ValidationError, validate

from umrs.shared.config import resolve_config_path
from umrs.shared.modality import LEGACY_MAPPING, VALID_MODALITIES

_ALL_ACCEPTED_MODALITY_NAMES = sorted(
    set(VALID_MODALITIES) | set(LEGACY_MAPPING.keys())
)

client_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["username", "fullname", "s3_configuration_file", "experiences"],
    "properties": {
        "username": {"type": "string"},
        "fullname": {"type": "string"},
        "s3_configuration_file": {"type": "string"},
        "experiences": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["xp_name", "sql_username", "modalities"],
                "properties": {
                    "xp_name": {"type": "string"},
                    "sql_username": {"type": "string"},
                    "modalities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["s3", "model", "sql"],
                            "oneOf": [
                                {"required": ["modality_name"]},
                                {"required": ["modality"]},
                            ],
                            "properties": {
                                "modality_name": {
                                    "type": "string",
                                    "enum": _ALL_ACCEPTED_MODALITY_NAMES,
                                },
                                "modality": {
                                    "type": "string",
                                    "enum": VALID_MODALITIES,
                                },
                                "task": {"type": "string"},
                                "s3": {
                                    "type": "object",
                                    "required": ["bucket_alias", "data"],
                                    "properties": {
                                        "bucket_alias": {"type": "string"},
                                        "data": {
                                            "type": "object",
                                            "required": ["type", "mime_type"],
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": ["file", "inline"],
                                                },
                                                "mime_type": {"type": "string"},
                                            },
                                            "need_output_bucket": {
                                                "type": "boolean",
                                                "default": False,
                                            },
                                        },
                                    },
                                },
                                "model": {
                                    "type": "object",
                                    "required": [
                                        "model_name",
                                        "vector_size",
                                        "model_type",
                                        "return_type",
                                    ],
                                    "properties": {
                                        "model_name": {"type": "string"},
                                        "vector_size": {"type": "string"},
                                        "model_type": {"type": "string"},
                                        "server_port": {"type": "string"},
                                        "model_cfg_file": {"type": "string"},
                                        "checkpoint_path": {"type": "string"},
                                        "spacy_model": {"type": "string"},
                                        "vision_model_type": {"type": "string"},
                                        "vision_model_config": {"type": "object"},
                                        "return_type": {
                                            "type": ["string", "null"],
                                            "enum": ["file", None],
                                        },
                                    },
                                    "additionalProperties": False,
                                },
                                "sql": {
                                    "type": "object",
                                    "required": [
                                        "store_embeddings",
                                        "table_name",
                                        "meta_data",
                                    ],
                                    "properties": {
                                        "store_embeddings": {
                                            "type": "string",
                                            "enum": ["yes", "no"],
                                        },
                                        "table_name": {"type": "string"},
                                        "meta_data": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "required": ["name", "type"],
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "type": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def load_client_cfg_file(json_filename, username_list=None):
    if username_list is None:
        username_list = []

    json_filename = os.path.expanduser(json_filename)
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"Client JSON file {json_filename} does not exist")

    try:
        with open(json_filename, encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON file {json_filename}: {e.msg} (char {e.pos})"
        ) from e

    try:
        validate(instance=data, schema=client_schema)
        if data["username"] in username_list:
            raise ValueError(
                f"{json_filename} uses the same username as another client configuration file. "
                "Merge the two configuration files or change the username."
            )
        return data
    except ValidationError as e:
        raise ValueError(
            f"Invalid client JSON file {json_filename}: {e.message}"
        ) from e


__all__ = ["client_schema", "load_client_cfg_file"]


# ---------------------------------------------------------------------------
# Server configuration (moved from umrs.server.json_config for boundary
# compliance — shared/ must not import from server/).
# ---------------------------------------------------------------------------

server_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["admin_s3_configuration_file", "users"],
    "properties": {
        "admin_s3_configuration_file": {"type": "string"},
        "user_files_dir": {"type": "string"},
        "users": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "additionalProperties": False,
}


def load_server_cfg_json(json_filename):
    json_filename = os.path.abspath(os.path.expanduser(json_filename))
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"Server JSON file {json_filename} does not exist")

    try:
        with open(json_filename, encoding="utf-8") as file:
            content = json.load(file)
    except json.decoder.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON file {json_filename}: {e.msg} (char {e.pos})"
        ) from e

    try:
        validate(instance=content, schema=server_schema)
        content = dict(content)
        content["admin_s3_configuration_file"] = resolve_config_path(
            content["admin_s3_configuration_file"], config_file=json_filename
        )
        if content.get("user_files_dir"):
            content["user_files_dir"] = resolve_config_path(
                content["user_files_dir"], config_file=json_filename
            )
        content["users"] = [
            resolve_config_path(user_conf_file, config_file=json_filename)
            for user_conf_file in content["users"]
        ]
        return content
    except ValidationError as e:
        raise ValueError(
            f"Invalid server JSON file {json_filename}: {e.message}"
        ) from e


def load_server_cfg_file(json_filename):
    users = []
    username_list = []
    content = load_server_cfg_json(json_filename)

    # Deduplicate user entries that resolve to the same absolute path
    # (e.g. "./user_files/x.json" and "/etc/umrs/user_files/x.json").
    seen_paths: set[str] = set()
    deduped_users: list[str] = []
    for user_conf_file in content["users"]:
        real = os.path.realpath(user_conf_file)
        if real in seen_paths:
            continue
        seen_paths.add(real)
        deduped_users.append(user_conf_file)
    content["users"] = deduped_users

    try:
        admin_s3_configuration_file = content["admin_s3_configuration_file"]
        for user_conf_file in content["users"]:
            user_content = load_client_cfg_file(
                user_conf_file, username_list=username_list
            )
            username_list.append(user_content["username"])
            user_content["config_file_path"] = user_conf_file
            users.append(user_content)
        return admin_s3_configuration_file, users, content
    except ValidationError as e:
        raise ValueError(
            f"Invalid server JSON file {json_filename}: {e.message}"
        ) from e
