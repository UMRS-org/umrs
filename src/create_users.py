#!/usr/bin/python3

import argparse
import json
import os
import sys
from umrs.sql import *
from umrs.s3 import *

def create_s3_user(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")


    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection = connect_to_s3(admin_s3_configuration_file)

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        # sql_username  = xp["sql_username"]
        # data_base = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality.get("modality_name") or modality.get("modality")
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            bucket_list.append(path_to_bucket)
            # model = load_model(modality['model'], modality_name)
            # table_name, store_embeddings_flag, metadata = configure_table(data_base, modality["sql"], modality_name, model)
            # store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")

    s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list)
    s3_connection.close()

def create_sql_users(user_cfg, sql_admin):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")


    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base = create_db(sql_username, xp_name)

parser = argparse.ArgumentParser(description="UMRS create user utility", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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


# Pretty-print the JSON content
# print(json.dumps(content, indent=4, sort_keys=True, ensure_ascii=False))
sql_admin = content["sql_admin"]
for user in content["users"]:
    create_sql_users(user, sql_admin)

print(f"\n######################\nUser creation done!\n######################\n")
