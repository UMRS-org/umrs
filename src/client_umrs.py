#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from  umrs.MessageQueue import MessageQueue
from umrs.metadata import check_and_build_metadata
from glob import glob
import argparse
from collections import defaultdict

def store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata):
    if(bucket_store[bucket_alias]):
        sys.exit(f"Error: duplicate bucket_alias {bucket_alias}")
    bucket_store[bucket_alias]= {"path_to_bucket": path_to_bucket,
                                "username": username,
                                "xp_name": xp_name,
                                "modality_name": modality_name,
                                 "s3_configuration_file": s3_configuration_file,
                                 "metadata": metadata}


def configure_bucket(user_cfg, bucket_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        # sql_username  = xp["sql_username"]
        # db_name = create_db_name(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            bucket_alias =  modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            # print(cfg_bucket_alias, bucket_alias, username, xp_name, modality_name)
            metadata = metadata_dict(modality["sql"])
            store_bucket(bucket_store, bucket_alias, path_to_bucket, username, xp_name, modality_name, user, s3_configuration_file, metadata)



parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue bewteen 'zeromq', 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument("-k", action="append", help="Metadatata key:value", required=False)
parser.add_argument("data", type=str, help="input data")
parser.add_argument("bucket", type=str, help="target bucket")
args = parser.parse_args()

# Connect to message queue
mq = MessageQueue(queue_type= args.queue_type, role = "client")
print (f"Connected to message queue of type '{args.queue_type}'")

# Path to the configuration file
configure_file = args.conf

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

# Load JSON content from the file
with open(configure_file, "r", encoding="utf-8") as file:
    content = json.load(file)


bucket_store = defaultdict(dict)
for user in content["users"]:
    configure_bucket(user, bucket_store)
if args.verbose:
    for bucket_alias, entry in bucket_store.items():
        print(f"{bucket_alias}: {entry['path_to_bucket']}, {entry['username']}, {entry['xp_name']}, {entry['modality_name']}")

print("Configuration loaded successfully.")

bucket_alias  = args.bucket

if not bucket_store[bucket_alias]:
    sys.exit(f"Error: unknown bucket name '{bucket_alias}' in configuration")

entry = bucket_store[bucket_alias];

path_to_bucket        = entry['path_to_bucket']
username              = entry['username']
xp_name               = entry['xp_name']
modality              = entry['modality_name']
s3_configuration_file = entry['s3_configuration_file']
metadata              = entry['metadata']

s3 = connect_to_s3(s3_configuration_file)

full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username)

if not success or full_bucket_name != path_to_bucket:
    sys.exit(f"No bucket {bucket_alias} for  modality {modality} and for user {username}")

print(f"Bucket '{bucket_alias}' for  modality '{modality}' and for user '{username}': OK!")

metadata_args = check_and_build_metadata(metadata, args.k)

print(f"Storing data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

def is_likely_path(data_str):
    """
    Heuristic to determine if a string is likely a file/directory path.
    Returns True if it looks like a path, False if it looks like text content.
    """
    # Check for path-like characteristics
    if ('/' in data_str or '\\' in data_str or
        data_str.startswith('.') or data_str.startswith('~') or
        data_str.startswith('/') or
        (len(data_str.split()) == 1 and '.' in data_str)):  # single word with extension
        return True

    # If it contains spaces and multiple words, likely text content
    if ' ' in data_str and len(data_str.split()) > 2:
        return False

    # If it's a single word without spaces, could be either - default to text
    return False

if modality == "text":
    if os.path.exists(args.data):
        if os.path.isfile(args.data):
            # It's an existing file - store each line individually
            try:
                with open(args.data, "r", encoding="utf-8") as file:
                    for line in file.readlines():
                        line = line.strip()
                        if line:  # Skip empty lines
                            s3_key = s3_store_txt(s3, full_bucket_name, line, mq, metadata_args)
                            print(f"'{line}' saved as {s3_key} on: {full_bucket_name}")
            except UnicodeDecodeError as e:
                print(f"Unicode decoding error: {e}")
                print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file. Try another bucket.")
                sys.exit(1)
            except IOError as e:
                print(f"Error reading file {args.data}: {e}")
                sys.exit(1)
        elif os.path.isdir(args.data):
            print(f"Error: '{args.data}' is a directory. For text modality, provide a file path or text content.")
            sys.exit(1)
    else:
        # Path doesn't exist - check if it looks like a path or text content
        if is_likely_path(args.data):
            print(f"Error: File or directory '{args.data}' does not exist.")
            sys.exit(1)
        else:
            # Treat as text content
            try:
                s3_key = s3_store_txt(s3, full_bucket_name, args.data, mq, metadata_args)
                print(f"'{args.data}' saved as {s3_key} on: {full_bucket_name}")
            except Exception as e:
                print(f"Error storing text content: {e}")
                sys.exit(1)

elif modality == "image":
    if os.path.exists(args.data):
        if os.path.isfile(args.data):
            # It's an existing file - store it
            try:
                s3_key = s3_store_image(s3, full_bucket_name, args.data, mq, metadata_args)
                if s3_key:
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
                else:
                    print(f"Error: Failed to store image '{args.data}'")
                    sys.exit(1)
            except Exception as e:
                print(f"Error storing image {args.data}: {e}")
                sys.exit(1)
        elif os.path.isdir(args.data):
            # It's a directory - store each file in it
            image_files = []
            for file in glob(os.path.join(args.data, "*")):
                if os.path.isfile(file):
                    image_files.append(file)

            if not image_files:
                print(f"Error: No files found in directory '{args.data}'")
                sys.exit(1)

            for file in image_files:
                try:
                    s3_key = s3_store_image(s3, full_bucket_name, file, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
                    else:
                        print(f"Warning: Failed to store image '{file}' - skipping")
                except Exception as e:
                    print(f"Error storing image {file}: {e} - skipping")
    else:
        print(f"Error: Image file or directory '{args.data}' does not exist.")
        print("Provide a path to an existing image file or directory containing images.")
        sys.exit(1)

elif modality == "text_file":
    if os.path.exists(args.data):
        if os.path.isfile(args.data):
            # It's an existing file - store it
            try:
                s3_key = s3_store_text_file(s3, full_bucket_name, args.data, mq, metadata_args)
                if s3_key:
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
                else:
                    print(f"Error: Failed to store text file '{args.data}'")
                    sys.exit(1)
            except Exception as e:
                print(f"Error storing text file {args.data}: {e}")
                sys.exit(1)
        elif os.path.isdir(args.data):
            # It's a directory - store each file in it
            text_files = []
            for file in glob(os.path.join(args.data, "*")):
                if os.path.isfile(file):
                    text_files.append(file)

            if not text_files:
                print(f"Error: No files found in directory '{args.data}'")
                sys.exit(1)

            for file in text_files:
                try:
                    s3_key = s3_store_text_file(s3, full_bucket_name, file, mq, metadata_args)
                    if s3_key:
                        print(f"'{s3_key}' saved on: {full_bucket_name}")
                    else:
                        print(f"Warning: Failed to store text file '{file}' - skipping")
                except Exception as e:
                    print(f"Error storing text file {file}: {e} - skipping")
    else:
        print(f"Error: Text file or directory '{args.data}' does not exist.")
        print("Provide a path to an existing text file or directory containing text files.")
        sys.exit(1)

else:
    print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")
