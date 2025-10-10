import sys
import json
import os

from jsonschema import validate, ValidationError
client_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["username", "fullname", "s3_configuration_file", "experiences"],
  "properties": {
    "username": { "type": "string" },
    "fullname": { "type": "string" },
    "s3_configuration_file": { "type": "string" },
    "experiences": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["xp_name", "sql_username", "modalities"],
        "properties": {
          "xp_name": { "type": "string" },
          "sql_username": { "type": "string" },
          "modalities": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["modality_name", "s3", "model", "sql"],
              "properties": {
                "modality_name": {
                  "type": "string",
                  "enum": ["image", "text", "text file", 'tomography']
                },
                "s3": {
                  "type": "object",
                  "required": ["bucket_alias", "data"],
                  "properties": {
                    "bucket_alias": { "type": "string" },
                    "data": {
                      "type": "object",
                      "required": ["type", "mime_type"],
                      "properties": {
                        "type": { "type": "string", "enum": ["file", "inline"] },
                        "mime_type": { "type": "string" }
                      }
                    }
                  }
                },
                "model": {
                  "type": "object",
                  "required": ["model_name", "vector_size", "model_type"],
                  "properties": {
                    "model_name": { "type": "string" },
                    "vector_size": { "type": "string" },
                    "model_type": { "type": "string" },
                    "model_cfg_file" : { "type": "string" },
                    "checkpoint_path": { "type": "string" },
                    "spacy_model": { "type": "string" }
                  },
                  "additionalProperties": False
                },
                "sql": {
                  "type": "object",
                  "required": ["store_embeddings", "table_name", "meta_data"],
                  "properties": {
                    "store_embeddings": {
                      "type": "string",
                      "enum": ["yes", "no"]
                    },
                    "table_name": { "type": "string" },
                    "meta_data": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                          "name": { "type": "string" },
                          "type": { "type": "string" }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

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

def load_client_cfg_file(json_filename):
    json_filename = os.path.expanduser(json_filename)
    if not os.path.exists(json_filename):
      sys.exit(f"Client JSON file {json_filename} does not exist. Exiting...")

    with open(json_filename, encoding="utf-8") as f:
      data = json.load(f)

      # Validate using the in-memory variable
      try:
          validate(instance=data, schema=client_schema)
          return data
      except ValidationError as e:
          sys.exit(f"Invalid client JSON file {json_filename}: {e.message}.\nExiting...")


def load_server_cfg_file(json_filename):
    json_filename = os.path.expanduser(json_filename)
    users = []
    if not os.path.exists(json_filename):
      sys.exit(f"Server JSON file {json_filename} does not exist. Exiting...")
    with open(json_filename, "r", encoding="utf-8") as file:
        content = json.load(file)
        try:
            validate(instance=content, schema=server_schema)
            admin_s3_configuration_file = content["admin_s3_configuration_file"]
            for user_conf_file in content["users"]:
                user_content = load_client_cfg_file(user_conf_file)
                user_content['config_file_path'] = user_conf_file
                users.append(user_content)
            return admin_s3_configuration_file, users
        except ValidationError as e:
            sys.exit(f"Invalid server JSON file {json_filename}: {e.message}.\nExiting...")
