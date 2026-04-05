import json
import os
import sys

from jsonschema import validate, ValidationError
from umrs.shared.json_config import load_client_cfg_file

server_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["admin_s3_configuration_file", "users"],
  "properties": {
    "admin_s3_configuration_file": {
      "type": "string"
    },
    "users": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1
    }
  },
  "additionalProperties": False
}


def load_server_cfg_file(json_filename):
    json_filename = os.path.expanduser(json_filename)
    users = []
    username_list = []
    if not os.path.exists(json_filename):
      sys.exit(f"Server JSON file {json_filename} does not exist. Exiting...")

    try:
      with open(json_filename, "r", encoding="utf-8") as file:
        content = json.load(file)
    except json.decoder.JSONDecodeError as e:
        sys.exit(f"Error: Failed to parse JSON file {json_filename}: {e.msg} (char {e.pos}).\nExiting...")

    try:
      validate(instance=content, schema=server_schema)
      admin_s3_configuration_file = content["admin_s3_configuration_file"]
      for user_conf_file in content["users"]:
        user_content = load_client_cfg_file(user_conf_file, username_list=username_list)
        username_list.append(user_content["username"])
        user_content['config_file_path'] = user_conf_file
        users.append(user_content)
      return admin_s3_configuration_file, users, content
    except ValidationError as e:
      sys.exit(f"Invalid server JSON file {json_filename}: {e.message}.\nExiting...")


__all__ = ["server_schema", "load_server_cfg_file", "load_client_cfg_file"]