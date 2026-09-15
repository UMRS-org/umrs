#!/usr/bin/python3
import json
import os
import sys
from umrs.server.sql import *
from umrs.shared.storage.s3 import *
from umrs.inference.embeddings import *
import time
from collections import defaultdict
import signal

# Path to the configuration file


def store_db_model(
    db_model_store, bucket_name, db_pool, table_name, model, modality_name, s3_connection
):
    """Stores db_pool and model for a given (user, bucket_name)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry

    db_model_store[bucket_name] = {
        "db_pool": db_pool,
        "table_name": table_name,
        "model": model,
        "modality_name": modality_name,
        "s3_connection": s3_connection,
    }


def query_db_model(db_model_store, bucket_name):
    """Retrieves db_pool and model for a given (user, bucket_name)."""
    entry = db_model_store.get(bucket_name, None)
    if not entry:
        return None, None, None
    else:
        return (
            entry["db_pool"],
            entry["table_name"],
            entry["model"],
            entry["modality_name"],
            entry["s3_connection"],
        )


def configure_user(user_cfg, user_md5):
    print(f"######################\nConfiguring {fullname}\n######################")

    db_model_store = defaultdict(dict)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\tExperience: {xp_name}")
        sql_username = xp["sql_username"]
        db_pool = create_db_and_get_pool(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            print(f"\t\tModality: {modality_name}")
            bucket_name, s3_connection = configure_s3(modality["s3"], user_md5)
            table_name, model_name = configure_sql(db_pool, modality["sql"], modality_name)
            model = load_embedding_model(model_name)
            store_db_model(
                db_model_store,
                bucket_name,
                db_pool,
                table_name,
                model,
                modality_name,
                s3_connection,
            )

    return db_model_store


def compute_and_store_embedding(
    bucket_name, s3_key, db_pool, table_name, model, modality_name, s3_connection
):
    data = load_data_from_s3(s3_connection, bucket_name, s3_key, modality_name)
    if "text" in modality_name:
        text = data
    else:
        text = None
    embedding = model.encode(data, convert_to_tensor=False).tolist()
    store_embeddings_in_db(db_pool, embedding, table_name, bucket_name, s3_key, text)


if len(sys.argv) != 2:
    sys.exit(f"Usage: {sys.argv[0]} <JSON config file>")

configure_file = sys.argv[1]

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f'Configuration file "{configure_file}" not found.')

# Load JSON content from the file
with open(configure_file, "r", encoding="utf-8") as file:
    content = json.load(file)

# Print loaded content (optional)
print("Configuration loaded successfully.")

# Pretty-print the JSON content
# print(json.dumps(content, indent=4, sort_keys=True, ensure_ascii=False))


for user in content["users"]:
    fullname = user["fullname"]
    user_md5 = user["md5"]
    check_md5 = (user_md5, user["username"])
    db_model_store = configure_user(user, user_md5)

print(f"\n######################\nServer configuration done!\n######################\n")


try:
    # loop for new new object to be created
    while True:
        print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

        bucket_name, s3_key = wait_for_s3_event()
        print(f"\t-->Detected new file: s3://{bucket_name}/{s3_key}")
        db_pool, table_name, model, modality_name, s3_connection = query_db_model(
            db_model_store, bucket_name
        )
        if model:
            print(f"\t--> Storing to data base in {table_name}")
            compute_and_store_embedding(
                bucket_name, s3_key, db_pool, table_name, model, modality_name, s3_connection
            )
        else:
            print(f"No model associated with this bucket")

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")
    sys.exit(0)  # ✅ Ensures a clean exit
