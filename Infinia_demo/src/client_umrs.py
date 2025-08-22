#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from umrs.MessageQueue import MessageQueue
from umrs.metadata import check_and_build_metadata
from umrs.modality import *
from umrs.json_utils import load_client_cfg_file 
from umrs.pretty_print import pretty_print_user_cfg
from glob import glob
import argparse
from collections import defaultdict

def is_server_ready(mq, timeout_ms=5000):
    print("Checking for server availability...")
    if timeout_ms < 1500:
        timeout_ms = 1500

    mq.set_timeout(timeout_ms)
    try: 
        try:
            mq.send_string("READY?")
            answer = mq.recv_string()
            if answer == "READY!":
                return True
        except zmq.Again:
            # Retry once on timeout
            try:
                mq.send_string("READY?")
                answer = mq.recv_string()
                if answer == "READY!":
                    return True
            except zmq.Again:
                return False
        return False
    finally:
        mq.set_timeout(-1) # reset timeout before exiting
    


def store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata, sql_username, db_name):
    if(bucket_store[bucket_alias]):
        sys.exit(f"Error: duplicate bucket_alias {bucket_alias}")
    bucket_store[bucket_alias]= {"path_to_bucket": path_to_bucket,
                                "username": username,
                                "xp_name": xp_name,
                                "modality_name": modality_name,
                                 "s3_configuration_file": s3_configuration_file,
                                 "metadata": metadata,
                                 "sql_username": sql_username,
                                 "db_name": db_name}


def configure_bucket(user_cfg, bucket_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        sql_username  = xp["sql_username"]
        xp_name = xp["xp_name"]
        db_name = create_db_name(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias =  modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            # print(bucket_alias, username, xp_name, modality_name)
            metadata = metadata_dict(modality["sql"])
            store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata, sql_username, db_name)



parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue bewteen 'zeromq', 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument("-k", action="append", help="Metadatata key:value", required=False)
parser.add_argument('--no-check-server', action='store_true', help='Do not check if the server is ready. Just copy data in the S3 server if no ready')
parser.add_argument("data", type=str, nargs="?", help="input data", default = None)
parser.add_argument("bucket", type=str, nargs="?", help="target bucket alias", default = None)
args = parser.parse_args()

# Connect to message queue
control_mq = MessageQueue(role = "client", queue_type = 'control')

if not args.no_check_server:
    if not is_server_ready(control_mq, timeout_ms=5000):
        sys.exit("Server connection timeout...\n"
                 "Cannot continue!"
                 )

print (f"Server ready.\nConnected to message queue of type '{args.queue_type}'")

# Path to the configuration file
configure_file = args.conf

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

# Load JSON content from the file
user = load_client_cfg_file(configure_file)

bucket_store = defaultdict(dict)
configure_bucket(user, bucket_store)

print("Configuration loaded successfully.")


if args.data == None or args.bucket == None :
    print("Misssing data and/or bucket.")
    pretty_print_user_cfg(bucket_store)
    sys.exit(
        "Error: You must provide both the data and the bucket alias.\n"
        "These are required to upload data to the S3 server according to the configuration above."
    )

if args.verbose:
    pretty_print_user_cfg(bucket_store)

    
bucket_alias  = args.bucket

if not bucket_store[bucket_alias]:
    sys.exit(f"Error: unknown bucket name '{bucket_alias}' in configuration")

entry = bucket_store[bucket_alias];

# print(entry)

path_to_bucket        = entry['path_to_bucket']
username              = entry['username']
xp_name               = entry['xp_name']
modality              = entry['modality_name']
s3_configuration_file = entry['s3_configuration_file']
metadata              = entry['metadata']


data_mq = control_mq.get_data_queue(username)

s3, bucket_location = connect_to_s3(s3_configuration_file)

full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username)

if not success or full_bucket_name != path_to_bucket:
    sys.exit(f"No bucket {bucket_alias} for  modality {modality} and for user {username}")

print(f"Bucket '{bucket_alias}' for  modality '{modality}' and for user '{username}': OK!")

metadata_args = check_and_build_metadata(metadata, args.k)

print(f"Storing data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

server_notify_result = None
if modality == "text" :
    if os.path.isfile(args.data): #if its a file store each line individually
        try:
            with open(args.data,"r") as file:
                nb_objects = 0
                for line in file.readlines():
                    line = line.strip()
                    s3_key = s3_store_txt(s3, full_bucket_name, line, data_mq, metadata_args)
                    nb_objects += 1
                    print(f"'{line}' saved as {s3_key} on: {full_bucket_name}")
                server_notify_result = control_mq.notify_server(username, nb_objets) 
        except UnicodeDecodeError as e:
            print(f"Unicode decoding error: {e}")
            print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file. Try an other bucket.")
    else:
        s3_key = s3_store_txt(s3, full_bucket_name, args.data, data_mq, metadata_args)
        print(f"'{args.data}' saved as {s3_key} on: {full_bucket_name}")
        server_notify_result = control_mq.notify_server(username, 1) 
elif modality == "image" :
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_image(s3, full_bucket_name, args.data, data_mq, metadata_args)
        if s3_key:
            print(f"'{s3_key}' saved on: {full_bucket_name}")
            server_notify_result = control_mq.notify_server(username, 1) 
    elif os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        nb_objects = 0
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_image(s3, full_bucket_name, file, data_mq, metadata_args)
                if s3_key:
                    nb_objects += 1
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
        server_notify_result = control_mq.notify_server(username, nb_objets) 

    else :
        print(f"Error: do not know what to do with '{args.data}'...")
        print("Provide a path to an image or to a directory of images")

elif modality == "text file" :
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_text_file(s3, full_bucket_name, args.data, data_mq, metadata_args)
        if s3_key:
            print(f"'{s3_key}' saved on: {full_bucket_name}")
            server_notify_result = control_mq.notify_server(username, 1) 

    elif os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        nb_objects = 0
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_text_file(s3, full_bucket_name, file, data_mq, metadata_args)
                if s3_key:
                    nb_objects += 1
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
        server_notify_result = control_mq.notify_server(username, nb_objets) 
    else :
        print(f"Error: do not know what to do with '{args.data}'...")
        print("Provide a path to a text file or to a directory of text_files")

else:
    print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")

if server_notify_result != None:
    if server_notify_result == True:
        print("Server received data")
    else:
        print("Error: Server did not data")
# close queues before exiting
control_mq.close()
data_mq.close()
