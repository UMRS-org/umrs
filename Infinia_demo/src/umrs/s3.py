#!/usr/bin/python3
import os
import sys
import boto3
from botocore.exceptions import ClientError
import hashlib
import json
import warnings
import urllib3
import posix_ipc

from PIL import Image
from io import BytesIO

mq = posix_ipc.MessageQueue("/my_queue", posix_ipc.O_CREAT)


# ✅ Suppress only SSL warnings from urllib3
warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)


def wait_for_s3_event(debug = False):
    """Blocks execution until a new object is added to the S3 bucket."""

    mq = posix_ipc.MessageQueue("/my_queue")
    message, _ = mq.receive()
    print("Received:", message.decode())

    return message.decode();


def connect_to_s3(config_file):
    """
    Parses an S3 configuration file and returns a boto3 S3 client.

    Args:
        file_path (str): Path to the S3 config file.

    Returns:
        boto3.client: Configured S3 client.
    """
    config = {}

    # Read and parse the file line by line
    with open(config_file, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("["):  # Ignore empty lines and section headers
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()

    # Extract values with defaults
    s3_host = config.get("host_base")
    use_https = config.get("use_https", "True").lower() == "true"
    verify_ssl = config.get("check_ssl_certificate", "True").lower() == "true"
    access_key = config.get("access_key")
    secret_key = config.get("secret_key")

    # Ensure correct protocol (http or https)
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"

    # Create the S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=verify_ssl  # Disables SSL verification if False
    )

    return s3


def check_bucket_name(s3, xp_name, modality_bucket_name, username):
    cfg_bucket_name = username+"-"+xp_name+"-"+modality_bucket_name
    # cfg_bucket_name = modality_bucket_name
    bucket_name = cfg_bucket_name #+ "-" + user_md5
    bucket_name = bucket_name.lower().replace(" ","-")
    print(f"\t\t\t--> Checking Bucket '{bucket_name}'...")
    try:
        s3.head_bucket(Bucket = bucket_name)
            
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return bucket_name, False #Not found 
        else:
            sys.exit(f"S3 Error: {e}") #Un-managed head_bucket exception
            
    except Exception as e:        
        sys.exit(f"S3 Connection Failed: {e}") #Un-managed S3 error

    return bucket_name, True

    

def configure_s3(s3_cfg, user_s3_conf_file, xp_name, username):
    s3 = connect_to_s3(user_s3_conf_file)

    bucket_name, success = check_bucket_name(s3, xp_name, s3_cfg["bucket_name"], username)
    if success:    
        print(f"\t\t\t--> Bucket '{bucket_name}' already exists!")
        # add_bucket_sqs_queue(s3, bucket_name)
    else:
        print(f"\t\t\t--> Bucket '{bucket_name}' does not exists...")
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-northeast-1'  # Match your desired region
            }
        )
        print(f"\t\t\t--> Bucket {bucket_name} created");


        
    print("\t\t\t--> S3 Configuration Successful")
    return bucket_name, s3


def load_data_from_s3(s3, bucket_name, s3_key, modality_name):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)

    if modality_name == "text":  # ✅ Handle Text Files
        text = obj["Body"].read().decode("utf-8")
        return text

    elif modality_name == "image":  # ✅ Handle Image Files
        image = Image.open(BytesIO(obj["Body"].read()))
        return image

    else:
        raise ValueError(f"Unsupported modality while loading data {bucket_name/s3_key} from s3: {modality_name}")



def compute_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()

def notify_s3(s3_object):
    print(f"Notyfying {s3_object}");
    mq.send(s3_object)



def s3_store_txt(s3, bucket_name, text_data):

    s3_key = compute_md5(text_data)+".txt"
    
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=text_data, ContentType="text/plain")

    s3_object = f"s3://{bucket_name}/{s3_key}"
    notify_s3(s3_object)
    
    return s3_key;


def s3_store_image(s3, bucket_name, in_path):
    s3_key = os.path.basename(in_path)

    if os.path.exists(in_path):
        s3.upload_file(in_path, bucket_name, s3_key) 
    else:
        sys.exit(f"Error: {path} not found")
    
    s3_object = f"s3://{bucket_name}/{s3_key}"
    notify_s3(s3_object)

    return s3_key;

