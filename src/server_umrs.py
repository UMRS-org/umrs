#!/usr/bin/python3
import threading
import argparse
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.models import load_model
from umrs.perf_test import run_perf_test
from umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check
from umrs.modality import *
from umrs.db_store import *
from umrs.utils import zero_out_embeddings_for_empty_objects
from umrs.json_utils import load_server_cfg_file
from collections import defaultdict
import signal
import torch
import time

    
def create_s3_sql_users(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        sql_username  = xp["sql_username"]
        create_db_user(sql_username)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            # print(f"{bucket_alias} -> {path_to_bucket}")
            bucket_list.append(path_to_bucket)

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, _ = connect_to_s3(admin_s3_configuration_file)
    s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list)
    s3_connection.close()

def configure_user(user_cfg, db_model_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection, bucket_location = connect_to_s3(user_s3_configuration_file)
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base, db_name = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = configure_bucket(s3_connection, bucket_alias, xp_name, username, bucket_location)
            model = load_model(modality['model'], modality_name)
            sql_cfg = modality["sql"]
            table_name, store_embeddings_flag, metadata = configure_table(data_base, sql_cfg, modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")

def load_json_conf_file(configure_file):
# Load JSON content from the file
    users = []
    with open(configure_file, "r", encoding="utf-8") as file:
        content = json.load(file)
        admin_s3_configuration_file = content["admin_s3_configuration_file"]
        for user_conf_file in content["users"]:
            with open(user_conf_file, "r", encoding="utf-8") as file:
                user_content = json.load(file)
                users.append(user_content)

    return admin_s3_configuration_file, users

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
    objects, metadata, checksums, sizes, has_at_least_a_zero = load_data_from_s3(s3_connection, path_to_bucket, s3_keys, modality_name)

    # text objects of size zeo are replaced by string " ". Because some models cannot compute embedding on empty strings
    embeddings = model.compute_embeddings(objects)

    # once all the embeddings are computed we zero the embeddings for which the the size is zero
    if has_at_least_a_zero:
        zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name)

    # Store **all** embeddings in batch
    store_embeddings_in_db(data_base, embeddings, table_name, path_to_bucket, s3_keys, modality_name, checksums, model.model_name, metadata_list=metadata)

def process_s3_objects(s3_objects, db_model_store, device):

    nb_objects = len(s3_objects)
    keys_per_bucket = defaultdict(list)  # Create a dictionary with lists
    print(f"\t-->Detected {nb_objects} new file(s):")
        
    for s3_object in s3_objects:
        print(f"\t\t-->{s3_object}")
        path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
    
        # Append s3_key to the list associated with the bucket
        if s3_key not in keys_per_bucket[path_to_bucket]:  # Ensure no duplicates
            keys_per_bucket[path_to_bucket].append(s3_key)

    # Now that keys are sorted per bucket, process each bucket one at a time
    for path_to_bucket in keys_per_bucket.keys():
        username, data_base, table_name, model, strore_embeddings_flag, modality_name, s3_connection = query_db_model(db_model_store, path_to_bucket)
        s3_keys = keys_per_bucket[path_to_bucket]
        if strore_embeddings_flag:
            print(f"\t--> Storing to data base in {table_name}")
            compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection) 
        else:
            print(f"Error: We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")\n\tUser is probably not configured")

    print(f"\t--> Processing {nb_objects} file(s) done!")
    print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")
    
def launch_s3_thread(s3_objects, db_model_store, device):
    thread = threading.Thread(
        target=process_s3_objects,
        args=(s3_objects, db_model_store, device)
    )
    thread.start()
    return thread

def clean_finished_threads(threads):
    new_threads = []
    for thread in threads:
        if not thread.is_alive():
            thread.join()  # join is instant if already finished
        else:
            new_threads.append(thread)
    print (f"==> Running threads: {len(new_threads)}")
    return new_threads

# No dynamic column addition; metadata columns are strictly defined by config

def recompute_all_objects(db_model_store, device):
    print(f"\n######################\nRecomputing all objects in all buckets\n######################\n")

    recompute_dict = {}

    for path_to_bucket, entry in db_model_store.items():
        s3_objects, total_size, _ = list_s3_bucket(entry["s3_connection"], path_to_bucket)

        if len(s3_objects) > 0:
            start_time = time.perf_counter()  # More precise than time.time()
            process_s3_objects(s3_objects, db_model_store, device)
            duration = round((time.perf_counter() - start_time),3)            
        else:
            duration = 0

        recompute_dict[path_to_bucket] = {
            "bucket_alias": entry["bucket_alias"],
            "duration": duration,
            "nb_objects": len(s3_objects),
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"]
        }

    pretty_print_recompute(recompute_dict)

def update_objects(db_model_store, device, verbose=False):
    print(f"\n######################\nUpdating missing/mismatched objects in all buckets\n######################\n")

    update_dict = {}  

    for path_to_bucket, entry in db_model_store.items():
        s3_objects, total_size, checksums = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, db_model_store)

        to_update_objects = []
        for i, s3_object in enumerate(s3_objects):
            s = status[i]['status']
            if verbose:
                print(f"{s3_object}: {s}")
            if s != "match":
                to_update_objects.append(s3_object)

        if len(to_update_objects) > 0:
            start_time = time.perf_counter()
            process_s3_objects(to_update_objects, db_model_store, device)
            duration = round(time.perf_counter() - start_time, 3)
        else:
            duration = 0.0

        update_dict[path_to_bucket] = {
            "bucket_alias": entry["bucket_alias"],
            "duration": duration,
            "nb_objects": f"{len(to_update_objects)}/{len(s3_objects)}",
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"]
        }

    pretty_print_recompute(update_dict, update=True) 

def check_objects(db_model_store, verbose= False):
    print(f"\n######################\nChecking all objects in all buckets\n######################\n")

    perform_update = False
    check_dict = {}
    for path_to_bucket, entry in db_model_store.items():
        nb_not_found, nb_match, nb_mismatch, nb_error = 0, 0, 0, 0
        s3_objects, total_size, checksums = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, db_model_store)
        for i, s3_object in enumerate(s3_objects):
            s = status[i]['status']
            if verbose: 
                print(f"{s3_object}: {s}")
            if s == "match":
                nb_match += 1
            elif s == "mismatch":
                nb_mismatch += 1
            elif s == "not_found":
                nb_not_found += 1
            else:
                print(f"Error: {status[i]['message']}")
                nb_error += 1

        if len(s3_objects) != nb_match:
            perform_update = True
                
        check_dict[path_to_bucket] = {
            "bucket_alias": entry["bucket_alias"],
            "nb_objects": len(s3_objects),
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"],
            "matches":  nb_match,
            "Mismatches": nb_mismatch,
            "Not_found": nb_not_found,
            "Errors": nb_error
        }
    pretty_print_check(check_dict)
    if perform_update:
        print("\n⚠️ Warning: The database is not fully consistent with the objects stored in S3.")
        print("→ To recompute and update the missing or mismatched entries, restart the server with the `--force-update` option.\n")
    else:
        print("\n ✅ No problem: The database is fully consistent with the objects stored in S3.")

def main():
    parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue 'zeromq' or 'posix_ipc', or 'sysv_ipc'")
    parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
    parser.add_argument('--force-recompute', action='store_true', help='Recompute all embeddings by scanning all S3 directories. Existing entries will be fully overwritten.')
    parser.add_argument('--force-update', action='store_true', help='Recompute embeddings only for missing or mismatched entries. Only those entries will be updated in the database.')
    parser.add_argument('--check-objects', action='store_true', help='Check all the buckets to see if the corresponding objects are stored in the database')
    # no auto add columns flag; metadata columns must be declared in config


    args = parser.parse_args()

    configure_file = args.conf

    # Check if the file exists
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")


    admin_s3_configuration_file, users = load_server_cfg_file(configure_file)
    # Print loaded content (optional)
    print("Configuration loaded successfully.")

    # create the control message queue:
    control_mq = MessageQueue(role='server', queue_type='control', timeout_ms=1000) # This timeout is only the sending. Receiving is always blocking for the server
    print ("Subscribed to zeroMQ message queue")

    # create S3/SQL users if necessary
    for user in users:
        create_s3_sql_users(user, admin_s3_configuration_file)


    #configure server for each users
    db_model_store = defaultdict(dict)
    for user in users:
        configure_user(user, db_model_store)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    print(f"\n######################\nServer configuration done!\n######################\n")



    # run_perf_test()

    if args.force_recompute:
        recompute_all_objects(db_model_store, device)

    if args.force_update:
        update_objects(db_model_store, device)

    if args.check_objects:
        check_objects(db_model_store, verbose=False)


    pretty_print_cfg(db_model_store)

    active_threads = []
        
    try:

        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")
            # Server loop, e.g. polling for S3 objects, etc.
            result = control_mq.server_control_queue_loop()
            if not result:
                continue
            cmd, payload = result

            if cmd == "OBJECT": # Launch worker
                thread = launch_s3_thread(payload, db_model_store, device)
                active_threads.append(thread)
            elif cmd == "FILE":
                if payload:
                    for file in payload:
                        print(f"Received file: {file}")
                else:
                    print("Warning: FILE command received with no payload")
            elif cmd == "RECOMPUTE":
                # Blocking recomputation across all buckets
                recompute_all_objects(db_model_store, device)
            elif cmd == "UPDATE":
                # Update only missing/mismatched entries
                update_objects(db_model_store, device)
            elif cmd == "CHECK":
                # Check consistency and print a summary
                check_objects(db_model_store, verbose=False)
            
            else:
                print(f"====> Error: unknown command  '{cmd}' returned by server loop")

            # Clean up finished threads (non-blocking)
            active_threads = clean_finished_threads(active_threads)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for running threads to finish...")

        # Join all remaining threads
        for thread in active_threads:
            thread.join()

        print("All threads finished. Exiting now.")
        control_mq.close()
        sys.exit(0)  # Ensures a clean exit


if __name__ == "__main__":
    main()
