#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
import time
from glob import glob
import argparse
import posix_ipc


parser = argparse.ArgumentParser(description="Parse mandatory arguments.")

parser.add_argument("-u", "--user", required=True, help="User")
parser.add_argument("-x", "--xp", required=True, help="Experience")
parser.add_argument("-m", "--modality", required=True, help="Modality")
parser.add_argument("-k", "--keywords", required=True, help="Keywords")
parser.add_argument("-t", "--type", required=True, help="Type")
parser.add_argument("-d", "--data", required=True, help="Data")

args = parser.parse_args()

# Path to the configuration file
configure_file = "./conf_umrs.json"

# Check if the file exists
if not os.path.exists(configure_file):
    sys.exit(f"Configuration file \"{configure_file}\" not found.")

# Load JSON content from the file
with open(configure_file, "r", encoding="utf-8") as file:
    content = json.load(file)

# Print loaded content (optional)
print("Configuration loaded successfully.")


# Find user in the list
users = content["users"]
user = next((u for u in users if u["username"] == args.user), None)

if user is None:
    sys.exit(f"User {args.user} unknown")

# Find experience in the user's experiences listb
user_s3_configuration_file = user["s3_configuration_file"]
xps = user["experiences"]
xp = next((xp for xp in xps if xp["xp_name"] == args.xp), None)

if xp is None:
    sys.exit(f"Experience {args.xp} unknown")

xp_name = xp["xp_name"]
# Find modality in the experience's modalities list
modalities = xp["modalities"]
modality = next((m for m in modalities if m["modality_name"] == args.modality), None)

if modality is None:
    sys.exit(f"Modality {args.modality} unknown")


print(user["username"], xp["xp_name"], modality["modality_name"])

bucket_name = modality["s3"]["bucket_name"]
user_md5 = user["md5"]


s3 = connect_to_s3(user_s3_configuration_file)

bucket_name, success = check_bucket_name(s3, xp_name, modality["s3"]["bucket_name"], args.user)

if not success:
    sys.exit(f"No bucket {bucket_name}  this modality  for user {username}")
    

if(modality["modality_name"] == "text"):
    if os.path.isfile(args.data): #if its a file store each line individually
        with open(args.data,"r") as file:
            for line in file.readlines():
                line = line.strip()
                s3_key = s3_store_txt(s3, bucket_name, line)
                print(f"'{line}' saved as {s3_key} on: {bucket_name}")
    else:
        s3_key = s3_store_txt(s3, bucket_name, args.data)
        print(f"'{args.data}' saved as {s3_key} on: {bucket_name}")


    
if(modality["modality_name"] == "image"):
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_image(s3, bucket_name, args.data)
        print(f"'{s3_key}' saved on: {bucket_name}")
    if os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_image(s3, bucket_name, file)
                print(f"'{s3_key}' saved on: {bucket_name}")

            

