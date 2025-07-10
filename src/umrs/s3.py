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

import urllib3
# Suppress only SSL warnings from urllib3
warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)




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

    # Check if the s3cfg file exists
    if not os.path.exists(config_path):
        sys.exit(f"ERROR: S3 configuration file not found: {config_path}\n"
                 f"This file is required for S3 connectivity. Please ensure the s3cfg file exists.")

    # Initialize parser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)

    # Extract values with defaults
    s3_host = config.get("default", "host_base")
    use_https = config.getboolean("default", "use_https", fallback=False)
    verify_ssl = config.getboolean("default", "check_ssl_certificate", fallback=True)
    access_key = config.get("default", "access_key")
    secret_key = config.get("default", "secret_key")

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

    return s3

def create_path_to_bucket(xp_name, bucket_alias, username):
    path = username+"."+xp_name+"."+bucket_alias
    return path.lower().replace(" ","-")

def check_bucket_name(s3, xp_name, bucket_alias, username):
    path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
    print(f"\t\t\t--> Checking Bucket '{path_to_bucket}'...")
    try:
        s3.head_bucket(Bucket = path_to_bucket)

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return path_to_bucket, False #Not found
        else:
            sys.exit(f"S3 Error: {e}") #Un-managed head_bucket exception

    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}") #Un-managed S3 error

    return path_to_bucket, True

def configure_bucket(s3, bucket_alias, xp_name, username):

    path_to_bucket, success = check_bucket_name(s3, xp_name, bucket_alias, username)
    if success:
        print(f"\t\t\t--> Bucket '{path_to_bucket}' already exists!")
        # add_bucket_sqs_queue(s3, path_to_bucket)
    else:
        print(f"\t\t\t--> Bucket '{path_to_bucket}' does not exists...")
        s3.create_bucket(
            Bucket=path_to_bucket,
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-northeast-1'  # Match your desired region
            }
        )
        print(f"\t\t\t--> Bucket {path_to_bucket} created");

    print("\t\t\t--> S3 Bucket Configuration Successful")
    return path_to_bucket




def load_data_from_s3(s3, path_to_bucket, s3_keys, modality_name):
    objects = []
    metadata = []

    for s3_key in s3_keys:
        obj = s3.get_object(Bucket=path_to_bucket, Key=s3_key)
        metadata.append(obj.get("Metadata", {}))

        if modality_name == "text":  # Handle text stored in file
            text = obj["Body"].read().decode("utf-8")
            objects.append(text)

        elif modality_name == "image":  # Handle Image Files
            image = Image.open(BytesIO(obj["Body"].read()))
            objects.append(image)

        elif modality_name == "text_file":  # Handle Text Files
            text = str(from_bytes(obj["Body"].read()).best())
            objects.append(text)

        else:
            raise ValueError(f"Unsupported modality '{modality_name}'while loading data '{path_to_bucket}/{s3_key}' from S3")

    return objects, metadata



def s3_store_txt(s3, path_to_bucket, text_data, mq, metadata_args):

    s3_key = compute_md5(text_data)+".txt"

    s3.put_object(Bucket=path_to_bucket, Key=s3_key, Body=text_data, Metadata=metadata_args["Metadata"], ContentType="text/plain")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
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
        sys.exit(f"Error: {in_path} not found")

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
        sys.exit(f"Error: {in_path} not found")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key;

