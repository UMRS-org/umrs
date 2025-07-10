#!/usr/bin/python3

import argparse
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.models import load_model
from umrs.perf_test import run_perf_test
from  umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg
from collections import defaultdict
import signal
import torch

def store_db_model(db_model_store, path_to_bucket, bucket_alias, username, db_pool, table_name, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name):
    """Stores data base and model for a given (user, bucket_alias)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry

    db_model_store[path_to_bucket] = {"bucket_alias": bucket_alias,
                                      "username": username,
                                      "data_base": db_pool,
                                      "table_name": table_name,
                                      "model": model,
                                      "store_embeddings_flag": store_embeddings_flag,
                                      "modality_name": modality_name,
                                      "s3_connection": s3_connection,
                                      "metadata": metadata,
                                      "xp_name": xp_name}


def query_db_model(db_model_store, path_to_bucket):
    """Retrieves database and model for a given (user, bucket_alias)."""
    entry = db_model_store.get(path_to_bucket, None)
    if not entry:
        return None, None, None, None, None, None
    else:
        return entry["username"], entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"]


def configure_user(user_cfg, db_model_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection = connect_to_s3(user_s3_configuration_file)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base = create_db(sql_username, xp_name)
        if data_base is None:
            print(f"\t\t-->  Error: Failed to create database for user '{sql_username}' and experience '{xp_name}'. Skipping this experience.")
            continue
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = configure_bucket(s3_connection, bucket_alias, xp_name, username)
            model = load_model(modality['model'], modality_name)
            table_name, store_embeddings_flag, metadata = configure_table(data_base, modality["sql"], modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")


def compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection):
    """
    Loads data from S3, computes embeddings **in batch**, and stores them in the database.

    Args:
        bucket_alias (str): The name of the S3 bucket.
        s3_keys (list): List of S3 object keys.
        db_pool: Database connection pool.
        table_name (str): Target table for storing embeddings.
        model (SentenceTransformer): Model used for embedding computation.
        device (str): "cpu" or "cuda".
        modality_name (str): "text" or "image".
        s3_connection: Boto3 S3 client.
    """
    objects, metadata = load_data_from_s3(s3_connection, path_to_bucket, s3_keys, modality_name)

    embeddings = model.compute_embeddings(objects)

    # Store **all** embeddings in batch
    store_embeddings_in_db(data_base, embeddings, table_name, path_to_bucket, s3_keys, modality_name, metadata_list=metadata)


def initialize_umrs_system(configure_file, queue_type='zeromq'):
    """
    Initialize the UMRS system: configure users, databases, and buckets.

    Args:
        configure_file (str): Path to the configuration file
        queue_type (str): Type of message queue (for compatibility)

    Returns:
        tuple: (db_model_store, device) if successful, (None, None) if failed
    """
    # Check if the file exists
    if not os.path.exists(configure_file):
        print(f"Configuration file '{configure_file}' not found.")
        return None, None

    # Load JSON content from the file
    with open(configure_file, "r", encoding="utf-8") as file:
        content = json.load(file)

    print("Configuration loaded successfully.")

    db_model_store = defaultdict(dict)
    for user in content["users"]:
        configure_user(user, db_model_store)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    print(f"\n######################\nServer configuration done!\n######################\n")
    pretty_print_cfg(db_model_store)

    return db_model_store, device


def run_server_loop(db_model_store, device, queue_type):
    """
    Run the main server loop that waits for S3 events and processes them.

    Args:
        db_model_store: Dictionary containing database and model configurations
        device (str): Device to run models on ("cpu" or "cuda")
        queue_type (str): Type of message queue to use
    """
    # Create the message queue
    mq = MessageQueue(queue_type=queue_type)
    print(f"Subscribed to message queue '{queue_type}'")

    try:
        # loop for new objects to be created
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

            s3_objects = mq.wait_for_s3_events()

            keys_per_bucket = defaultdict(list)  # Create a dictionary with lists
            print(f"\t-->Detected {len(s3_objects)} new file(s):")

            for s3_object in s3_objects:
                print(f"\t\t-->{s3_object}")
                path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)

                # Append s3_key to the list associated with the bucket
                if s3_key not in keys_per_bucket[path_to_bucket]:  # Ensure no duplicates
                    keys_per_bucket[path_to_bucket].append(s3_key)

            for path_to_bucket in keys_per_bucket.keys():
                username, data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
                s3_keys = keys_per_bucket[path_to_bucket]
                if strore_embeddings_flag:
                    print(f"\t--> Storing to data base in {table_name}")
                    compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection)
                else:
                    print(f"We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully...")
        mq.close()
        sys.exit(0)  # Ensures a clean exit


parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument('--init-only', action='store_true', help='Initialize system only (users, databases, buckets) without starting the server loop')


args = parser.parse_args()

# Initialize the system
db_model_store, device = initialize_umrs_system(args.conf, args.queue_type)

if db_model_store is None:
    sys.exit(1)

# If init-only mode, exit after initialization
if args.init_only:
    print("Initialization completed. Exiting (--init-only mode).")
    sys.exit(0)

# Otherwise, run the full server loop
run_server_loop(db_model_store, device, args.queue_type)
