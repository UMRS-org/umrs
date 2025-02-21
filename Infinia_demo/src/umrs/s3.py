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
from time import sleep
from PIL import Image
from io import BytesIO
import posix_ipc


## Global configuration of the message queue
target_nb_messages = 10
target_message_size = 4096

mq = posix_ipc.MessageQueue(
    "/BIG_Queue",
    posix_ipc.O_CREAT,    # Create the queue if it doesn't exist
    max_messages=target_nb_messages,      # Limit to 100 messages in the queue
    max_message_size=target_message_size  # Each message can be up to 8192 bytes
)

if (mq.max_messages < target_nb_messages or mq.max_message_size < target_message_size):
    print("############# Warning #############")
    print("Configuration of the IPC Message Queue did not match the requested values")
    print(f"\tQueue size (nb of msg) = {mq.max_messages} (requested_value = {target_nb_messages})")
    print(f"\tMessage size = {mq.max_message_size} (requested_value = {target_message_size})")
    print("\tConsider changing the values of /proc/sys/fs/mqueue/msg_max and /proc/sys/fs/mqueue/msgsize_max...\n")


# Suppress only SSL warnings from urllib3
warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)

def close_message_queue():
    mq.close()
    mq.unlink()



def notify_s3(s3_object):
    print(f"Notyfying {s3_object}");
    mq.send(s3_object)


def wait_for_s3_events(debug=False):
    """Blocks execution until a new message arrives, then reads all available messages."""
    
    # BLOCKING: Wait for the first message
    print(f"Listening the queue ({mq.current_messages} pending msg)...")
    message, _ = mq.receive()
    print("Received (blocking):", message.decode())

    messages = [message.decode()]  # Store messages

    if 0 == mq.current_messages:
        return messages
    
    print("More to come...")
    # NON-BLOCKING: Read remaining messages without blocking
    while True:
        try:
            message, _ = mq.receive(timeout=0)  # Non-blocking read
            messages.append(message.decode())
            print("Received (Non blocking):", message.decode())

        except posix_ipc.BusyError:
            if debug:
                print("No more messages in queue.")
            break  # Exit loop if queue is empty

    return messages


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
    
    # Read and parse the file line by line
    with open(config_path, "r") as file:
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

def load_data_from_s3(s3, bucket_name, s3_keys, modality_name):
    objects = []  
    
    for s3_key in s3_keys: 
        obj = s3.get_object(Bucket=bucket_name, Key=s3_key)

        if modality_name == "text":  # Handle Text Files
            text = obj["Body"].read().decode("utf-8")
            objects.append(text)

        elif modality_name == "image":  # Handle Image Files
            image = Image.open(BytesIO(obj["Body"].read()))
            objects.append(image)

        else:
            raise ValueError(f"Unsupported modality while loading data {bucket_name}/{s3_key} from S3: {modality_name}")

    return objects

def compute_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()



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

