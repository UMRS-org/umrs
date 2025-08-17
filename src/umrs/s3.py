#!/usr/bin/python3
import os
import sys
import boto3
from botocore.exceptions import ClientError
import json
import warnings
from PIL import Image
import mimetypes
from io import BytesIO
import configparser
from charset_normalizer import from_path, from_bytes
from typing import List, Dict
from umrs.utils import compute_md5

import secrets
import string
import subprocess


import urllib3
# Suppress only SSL warnings from urllib3
warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)


def get_s3_server_name(s3):

    response = s3.list_buckets()
    headers = response['ResponseMetadata']['HTTPHeaders']


    server= headers['server']

    if "minio" in server.lower():
        return "MinIO"
    else:
        sys.exit(f"Do not know how to handle S3 server {server}")


def s3_strong_password(length=24):
    chars = string.ascii_letters + string.digits + "-_@"
    return ''.join(secrets.choice(chars) for _ in range(length))

def s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list):


    server = get_s3_server_name(s3_connection)
    config = {}

    config_path = os.path.expanduser(admin_s3_configuration_file)

    # Initialize parser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)
    bucket_location = config.get("default", "bucket_location", fallback = None)
    if not bucket_list:
        sys.exit(f"file {admin_s3_configuration_file} must have a bucket_location entry for [deafult]")

    for bucket in bucket_list:
        if not check_bucket_path(s3_connection, bucket, indent=False):
            s3_connection.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={
                    'LocationConstraint':   bucket_location # Match your desired region
                }
            )

    if server == "MinIO":
        from umrs.minio import minio_s3_create_user, minio_grant_user_access
        alias = minio_s3_create_user(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file)
        if alias:
            for bucket in bucket_list:
                minio_grant_user_access(username, bucket, alias)

    else:
        sys.exit("Do not know how to create user and buckets for server '{server}'")


def s3_create_buckets_only(s3_connection, bucket_list):
    """
    Create only the buckets without creating individual users.
    Uses admin credentials for all operations.
    """
    # Get bucket location from the existing s3_connection configuration
    # We'll try to get it from the s3_connection metadata, or use a default
    try:
        # Try to get bucket location from any existing bucket
        response = s3_connection.list_buckets()
        if response['Buckets']:
            # Use the location of the first bucket as reference
            bucket_location = s3_connection.get_bucket_location(Bucket=response['Buckets'][0]['Name'])['LocationConstraint']
        else:
            bucket_location = 'us-east-1'  # Default location
    except:
        bucket_location = 'us-east-1'  # Fallback default

    # Handle MinIO case where LocationConstraint might be None
    if bucket_location is None:
        bucket_location = 'us-east-1'

    print(f"--> Creating buckets with admin credentials (location: {bucket_location})")

    for bucket in bucket_list:
        if not check_bucket_path(s3_connection, bucket, indent=False):
            try:
                if bucket_location == 'us-east-1':
                    # For us-east-1, don't specify CreateBucketConfiguration
                    s3_connection.create_bucket(Bucket=bucket)
                else:
                    s3_connection.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={
                            'LocationConstraint': bucket_location
                        }
                    )
                print(f"--> Created bucket: {bucket}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                    print(f"--> Bucket {bucket} already exists and is owned by you")
                else:
                    print(f"--> Error creating bucket {bucket}: {e}")
        else:
            print(f"--> Bucket {bucket} already exists")


def connect_to_s3(config_file):
    """
    Parses an S3 configuration file and returns a boto3 S3 client.

    Args:
        file_path (str): Path to the S3 config file.

    Returns:
        boto3.client: Configured S3 client.
    """
    config = {}

    config_path = os.path.expanduser(config_file)

    # Initialize parser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)

    # Extract values with defaults
    s3_host = config.get("default", "host_base")
    use_https = config.getboolean("default", "use_https", fallback=False)
    verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
    access_key = config.get("default", "access_key")
    secret_key = config.get("default", "secret_key")
    bucket_location = config.get("default", "bucket_location")

    # Ensure correct protocol (http or https)
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"

    print(f"--> Connecting to S3 service on: {endpoint_url}...")

    # Create the S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=verify_ssl  # Disables SSL verification if False
    )


    # Test connection
    try:
        response = s3.list_buckets()
        print("--> Succesfully connected to S3 service.")
    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}")


    return s3, bucket_location

def create_path_to_bucket(xp_name, bucket_alias, username):
    path = username+"."+xp_name+"."+bucket_alias
    return path.lower().replace(" ","-").replace("_","-")



def check_bucket_path(s3, path_to_bucket, indent = True):
    if indent:
        print(f"\t\t\t--> Checking Bucket '{path_to_bucket}'...")
    else:
        print(f"--> Checking Bucket '{path_to_bucket}'...")

    try:
        s3.head_bucket(Bucket = path_to_bucket)

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False #Not found
        else:
            sys.exit(f"S3 Error: {e}") #Un-managed head_bucket exception

    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}") #Un-managed S3 error

    return True


def check_bucket_name(s3, xp_name, bucket_alias, username):
    path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
    return path_to_bucket, check_bucket_path(s3, path_to_bucket)

def configure_bucket(s3, bucket_alias, xp_name, username, bucket_location):

    path_to_bucket, success = check_bucket_name(s3, xp_name, bucket_alias, username)
    if success:
        print(f"\t\t\t--> Bucket '{path_to_bucket}' already exists!")
        # add_bucket_sqs_queue(s3, path_to_bucket)
    else:
        print(f"\t\t\t--> Bucket '{path_to_bucket}' does not exists...")
        s3.create_bucket(
            Bucket=path_to_bucket,
            CreateBucketConfiguration={
                'LocationConstraint': bucket_location  # Match your desired region
            }
        )
        print(f"\t\t\t--> Bucket {path_to_bucket} created");

    print("\t\t\t--> S3 Bucket Configuration Successful")
    return path_to_bucket




def list_s3_bucket(s3, path_to_bucket):
    print(f"Listing objects in bucket: {path_to_bucket}")
    objects_list = []
    checksums = []
    total_size = 0

    try:
        response = s3.list_objects_v2(Bucket=path_to_bucket)
        for obj in response.get('Contents', []):
            objects_list.append(f"s3://{path_to_bucket}/{obj['Key']}")
            checksums.append(obj['ETag'].strip('"'))
            total_size += obj['Size']
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket not found: {path_to_bucket}")
    except Exception as e:
        print(f"Error listing bucket '{path_to_bucket}': {e}")

    return objects_list, total_size, checksums



def load_data_from_s3(s3, path_to_bucket, s3_keys, modality_name):
    objects = []
    metadata = []
    checksums = []
    sizes = []
    has_at_least_a_zero = False;
    for s3_key in s3_keys:
        obj = s3.get_object(Bucket=path_to_bucket, Key=s3_key)
        checksums.append(obj['ETag'].strip('"'))
        metadata.append(obj.get("Metadata", {}))
        obj_size = obj['ContentLength']
        sizes.append(obj_size)

        if modality_name == "text":  # Handle text stored in file
            if obj_size:
                text = obj["Body"].read().decode("utf-8")
            else :
                has_at_least_a_zero = True
                text = "A"
            objects.append(text)

        elif modality_name == "image":  # Handle Image Files
            image = Image.open(BytesIO(obj["Body"].read()))
            objects.append(image)

        elif modality_name == "text file":  # Handle Text Files
            if obj_size:
                text = str(from_bytes(obj["Body"].read()).best())
            else :
                has_at_least_a_zero = True
                text = "A"
            objects.append(text)

        else:
            raise ValueError(f"Unsupported modality '{modality_name}'while loading data '{path_to_bucket}/{s3_key}' from S3")

    return objects, metadata, checksums, sizes, has_at_least_a_zero




def s3_store_txt(s3, path_to_bucket, text_data, mq, metadata_args):

    print(f"text_data={text_data}|")

    s3_key = compute_md5(text_data)+".txt"

    s3.put_object(Bucket=path_to_bucket, Key=s3_key, Body=text_data, Metadata=metadata_args["Metadata"], ContentType="text/plain")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    if mq:  # Only notify if message queue is provided
        mq.notify_s3(s3_object)

    return s3_key;

def is_image_by_extension(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type is not None and mime_type.startswith("image")

def is_image_by_opening(filename):
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify if it's a valid image
        return True
    except (IOError, SyntaxError):
        return False

def is_an_image(filename):
    # Check by extension first
    if not is_image_by_extension(filename):
        return False

    # If extension is valid, confirm by opening
    return is_image_by_opening(filename)

def s3_store_image(s3, path_to_bucket, in_path, mq, metadata_args):
    if not is_an_image(in_path):
        print(f"Error: '{in_path}' is not an image. Skipping...")
        return None

    s3_key = os.path.basename(in_path)

    if os.path.exists(in_path):
        s3.upload_file(in_path, path_to_bucket, s3_key, ExtraArgs=metadata_args)
    else:
        sys.exit(f"Error: {path} not found")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key;

def is_a_text_file(in_path):
    try:
        match = from_path(in_path).best()
        if match is None:
            return False  # Unable to detect encoding
        _ = match.output()  # Try decoding
        return True
    except Exception as e:
        print(f"[DEBUG] Failed to decode '{in_path}': {e}")
        return False



def s3_store_text_file(s3, path_to_bucket, in_path, mq, metadata_args):
    if not is_a_text_file(in_path):
        print(f"Error: '{in_path}' is not a txt file. Skipping...")
        return None

    s3_key = os.path.basename(in_path)

    if os.path.exists(in_path):
        s3.upload_file(in_path, path_to_bucket, s3_key, ExtraArgs=metadata_args)
    else:
        sys.exit(f"Error: {path} not found")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key;

