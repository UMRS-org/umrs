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
from collections import defaultdict
import signal
import torch

def store_db_model(db_model_store, bucket_name, db_pool, table_name, model, store_embeddings_flag, modality_name, s3_connection):
    """Stores data base and model for a given (user, bucket_name)."""
    # if user not in db_model_store:
    #     storage[user] = {}  # Initialize user entry
    
    db_model_store[bucket_name] = {"data_base": db_pool, "table_name": table_name,  "model": model, "store_embeddings_flag": store_embeddings_flag, "modality_name": modality_name, "s3_connection": s3_connection}


def query_db_model(db_model_store, bucket_name):
    """Retrieves database and model for a given (user, bucket_name)."""
    entry = db_model_store.get(bucket_name, None)
    if not entry:
        return None, None, None, None, None, None
    else:
        return entry["data_base"], entry["table_name"], entry["model"], entry["store_embeddings_flag"], entry["modality_name"], entry["s3_connection"]

    
def configure_user(user_cfg):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    db_model_store = defaultdict(dict)
    s3_connection = connect_to_s3(user_s3_configuration_file)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            print(f"\t\tModality: {modality_name}")
            bucket_name = configure_bucket(s3_connection, modality["s3"]["bucket_name"], xp_name, username)
            model = load_model(modality['model'], modality_name)
            table_name, store_embeddings_flag = configure_table(data_base, modality["sql"], modality_name, model)
            store_db_model(db_model_store, bucket_name, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection)

    return db_model_store


def compute_and_store_embedding(bucket_name, s3_keys, data_base, table_name, model, device, modality_name, s3_connection):
    """
    Loads data from S3, computes embeddings **in batch**, and stores them in the database.

    Args:
        bucket_name (str): The name of the S3 bucket.
        s3_keys (list): List of S3 object keys.
        db_pool: Database connection pool.
        table_name (str): Target table for storing embeddings.
        model (SentenceTransformer): Model used for embedding computation.
        device (str): "cpu" or "cuda".
        modality_name (str): "text" or "image".
        s3_connection: Boto3 S3 client.
    """
    objects = load_data_from_s3(s3_connection, bucket_name, s3_keys, modality_name)

    embeddings = model.compute_embeddings(objects)

    # Store **all** embeddings in batch
    store_embeddings_in_db(data_base, embeddings, table_name, bucket_name, s3_keys, objects)


parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')


args = parser.parse_args()

configure_file = args.conf

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file '{configure_file}' not found.")


# Load JSON content from the file
with open(configure_file, "r", encoding="utf-8") as file:
    content = json.load(file)

# Print loaded content (optional)
print("Configuration loaded successfully.")

# create the message queue:
mq = MessageQueue(queue_type = args.queue_type);
print (f"Subscribed to message queue '{args.queue_type}'")

# Pretty-print the JSON content
# print(json.dumps(content, indent=4, sort_keys=True, ensure_ascii=False))

for user in content["users"]:
    db_model_store = configure_user(user)




device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"-->  Models will be run on device: {device}")





print(f"\n######################\nServer configuration done!\n######################\n")

# run_perf_test()

try:
    # loop for new new object to be created
    while True:
        print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

        s3_objects = mq.wait_for_s3_events()

        keys_per_bucket = defaultdict(list)  # Create a dictionary with lists
        print(f"\t-->Detected {len(s3_objects)} new file(s):")
        
        for s3_object in s3_objects:
            print(f"\t\t-->{s3_object}")
            bucket_name, s3_key = s3_object.replace("s3://", "").split("/", 1)
    
            # Append s3_key to the list associated with the bucket
            if s3_key not in keys_per_bucket[bucket_name]:  # Ensure no duplicates
                keys_per_bucket[bucket_name].append(s3_key)


        for bucket_name in keys_per_bucket.keys():
            data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, bucket_name)
            s3_keys = keys_per_bucket[bucket_name]
            if strore_embeddings_flag:
                print(f"\t--> Storing to data base in {table_name}")
                compute_and_store_embedding(bucket_name, s3_keys, data_base, table_name, model, device, modality_name, s3_connection) 
            else:
                print(f"We do not compute embeddings for this bucket \"{bucket_name}\" (modality = \"{modality_name}\")")
        
except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully...")
    mq.close()
    sys.exit(0)  # ✅ Ensures a clean exit
