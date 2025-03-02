#!/usr/bin/python3
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *
from  umrs.MessageQueue import MessageQueue
import time
from glob import glob
import argparse
import posix_ipc

parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-q', '--queue_type', default='zeromq', help="Message Queue bewteen 'zeromq', 'posix_ipc', or 'sysv_ipc'")
parser.add_argument('-c', '--conf', default='./conf_umrs.json', help='Config file path')
parser.add_argument("-u", "--user", required=True, help="User")
parser.add_argument("-x", "--xp", required=True, help="Experience")
parser.add_argument("-m", "--modality", required=True, help="Modality")
parser.add_argument("-k", "--keywords", required=True, help="Keywords")
parser.add_argument("-t", "--type", required=True, help="Type")
parser.add_argument("-d", "--data", required=True, help="Data")

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

# Print loaded content (optional)
print("Configuration loaded successfully.")

# Find user in the list
users = content["users"]
user = next((u for u in users if u["username"] == args.user), None)

if user is None:
    sys.exit(f"User {args.user} unknown")

# Find experience in the user's experiences list
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

# Connect to S3 and check bucket for the user/experience/modality 
user_s3_configuration_file = user["s3_configuration_file"]
s3 = connect_to_s3(user_s3_configuration_file)
modality_bucket_name = modality["s3"]["bucket_name"]
full_bucket_name, success = check_bucket_name(s3, xp_name, modality_bucket_name, args.user)

if not success:
    sys.exit(f"No bucket {modality_bucket_name} for  modality {args.modality} and  for user {args.user}")
    

if(modality["modality_name"] == "text"):
    if os.path.isfile(args.data): #if its a file store each line individually
        with open(args.data,"r") as file:
            for line in file.readlines():
                line = line.strip()
                s3_key = s3_store_txt(s3, full_bucket_name, line, mq)
                print(f"'{line}' saved as {s3_key} on: {full_bucket_name}")
    else:
        s3_key = s3_store_txt(s3, full_bucket_name, args.data, mq)
        print(f"'{args.data}' saved as {s3_key} on: {full_bucket_name}")


    
if(modality["modality_name"] == "image"):
    if os.path.isfile(args.data): #if it is a file, store it
        s3_key = s3_store_image(s3, full_bucket_name, args.data, mq)
        print(f"'{s3_key}' saved on: {full_bucket_name}")
    if os.path.isdir(args.data): #if it is a path, store each file in the corresponding directory
        for file in glob(os.path.join(args.data, "*")):
            if os.path.isfile(file):
                s3_key = s3_store_image(s3, full_bucket_name, file, mq)
                print(f"'{s3_key}' saved on: {full_bucket_name}")

            

