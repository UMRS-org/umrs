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

if modality == "text" :
    if os.path.isfile(args.data): #if its a file store each line individually
        try:
            with open(args.data,"r") as file:
                for line in file.readlines():
                    line = line.strip()
                    s3_key = s3_store_txt(s3, full_bucket_name, line, mq, metadata_args)
                    print(f"'{line}' saved as {s3_key} on: {full_bucket_name}")
        except UnicodeDecodeError as e:
            print(f"Unicode decoding error: {e}")
            print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file. Try an other bucket.")
    else:
        s3_key = s3_store_txt(s3, full_bucket_name, args.data, mq, metadata_args)
        print(f"'{args.data}' saved as {s3_key} on: {full_bucket_name}")
elif modality == "image" :
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_image(s3, full_bucket_name, args.data, mq, metadata_args)
        if s3_key:
            print(f"'{s3_key}' saved on: {full_bucket_name}")
    elif os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_image(s3, full_bucket_name, file, mq, metadata_args)
                if s3_key:
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
    else :
        print(f"Error: do not know what to do with '{args.data}'...")
        print("Provide a path to an image or to a directory of images")
            
elif modality == "text_file" :
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_text_file(s3, full_bucket_name, args.data, mq, metadata_args)
        if s3_key:
            print(f"'{s3_key}' saved on: {full_bucket_name}")
    elif os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_text_file(s3, full_bucket_name, file, mq, metadata_args)
                if s3_key:
                    print(f"'{s3_key}' saved on: {full_bucket_name}")
    else :
        print(f"Error: do not know what to do with '{args.data}'...")
        print("Provide a path to a text file or to a directory of text_files")
            
else:
    print(f"Error: Unknown modality '{modality}'. Nothing has been sent to the server")
