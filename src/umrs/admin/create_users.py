#!/usr/bin/python3

import argparse
import json
import os
import sys

from umrs.server.sql import create_db
from umrs.shared.storage.s3 import (
    connect_to_s3,
    make_path_to_bucket_string,
    s3_create_users_and_buckets,
)


def create_s3_user(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(
        f"######################\nCreating {fullname} ({username})\n######################"
    )

    user_s3_configuration_file = user_cfg["s3_configuration_file"]

    s3_connection = connect_to_s3(admin_s3_configuration_file)

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        for modality in xp["modalities"]:
            modality_name = modality.get("modality_name") or modality.get("modality")
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            bucket_list.append(path_to_bucket)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")

    s3_create_users_and_buckets(
        username,
        s3_connection,
        user_s3_configuration_file,
        admin_s3_configuration_file,
        bucket_list,
    )
    s3_connection.close()


def create_sql_users(user_cfg, sql_admin):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(
        f"######################\nCreating {fullname} ({username})\n######################"
    )

    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username = xp["sql_username"]
        create_db(sql_username, xp_name)


def build_parser():
    parser = argparse.ArgumentParser(
        description="UMRS create user utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--conf", default="./conf_umrs.json", help="Config file path"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_file = args.conf
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")

    with open(configure_file, encoding="utf-8") as file:
        content = json.load(file)

    print("Configuration loaded successfully.")

    sql_admin = content["sql_admin"]
    for user in content["users"]:
        create_sql_users(user, sql_admin)

    print("\n######################\nUser creation done!\n######################\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
