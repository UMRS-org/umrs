#!/usr/bin/python3
import hashlib
import json
import os
import sys
from io import BytesIO

import boto3
from botocore.exceptions import ClientError
from PIL import Image

sqs = boto3.client("sqs")
queue_name = "umrs-s3-queue"

import boto3

sqs = boto3.client("sqs")
queue_name = "umrs-s3-queue"


def wait_for_s3_event(debug=False):
    """Blocks execution until a new object is added to the S3 bucket."""
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    queue_url = f"https://sqs.ap-northeast-1.amazonaws.com/{account_id}/{queue_name}"

    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,  # Long polling
            MessageAttributeNames=["All"],
        )

        if "Messages" in response:
            for message in response["Messages"]:
                body = json.loads(message["Body"])

                # 🔹 DEBUG: Print the raw message to analyze unexpected formats
                if debug:
                    print("Received Raw SQS Message:", json.dumps(body, indent=4))

                # ✅ Check if "Records" key exists
                if "Records" in body:
                    print("\t-->New S3 Object Detected!")

                    for record in body["Records"]:
                        bucket = record["s3"]["bucket"]["name"]
                        key = record["s3"]["object"]["key"]
                        print(f"\t--> File: s3://{bucket}/{key}")

                    # Delete message from queue (so we don’t process it again)
                    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])

                    return bucket, key  # Exit after detecting a new object

                else:
                    print("\t-->Warning: Received a message without 'Records'. Skipping.")
                    if debug:
                        print(json.dumps(body, indent=4))

                # 🔹 Always delete the message, even if not an S3 event
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message["ReceiptHandle"])


def configure_s3(s3_cfg, user_md5):
    # Initialize S3 client
    access_key = open(s3_cfg["s3_access_key_path"]).read().strip()
    secret_key = open(s3_cfg["s3_secret_key_path"]).read().strip()
    try:
        s3 = boto3.client(
            "s3",
            region_name="ap-northeast-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        cfg_bucket_name = s3_cfg["bucket_name"]
        bucket_name = cfg_bucket_name + "-" + user_md5
        print(f"\t\t\t--> Configuring Bucket '{cfg_bucket_name}'...")

        s3.head_bucket(Bucket=bucket_name)
        print(f"\t\t\t--> Bucket '{cfg_bucket_name}' already exists!")
        add_bucket_sqs_queue(s3, bucket_name)

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    "LocationConstraint": "ap-northeast-1"  # Match your desired region
                },
            )
            print(f"\t\t\t--> Bucket {bucket_name} created")
            add_bucket_sqs_queue(s3, bucket_name)
        else:
            sys.exit(f"S3 Error: {e}")

    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}")

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
        raise ValueError(
            f"Unsupported modality while loading data {bucket_name / s3_key} from s3: {modality_name}"
        )


def compute_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()


def s3_store_txt(text_data, s3_cfg, user_md5):
    # Initialize S3 client
    access_key = open(s3_cfg["s3_access_key_path"]).read().strip()
    secret_key = open(s3_cfg["s3_secret_key_path"]).read().strip()
    try:
        s3 = boto3.client(
            "s3",
            region_name="ap-northeast-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        cfg_bucket_name = s3_cfg["bucket_name"]
        bucket_name = cfg_bucket_name + "-" + user_md5

        s3.head_bucket(Bucket=bucket_name)

    except ClientError:
        print(f"Bucket {cfg_bucket_name} Does not exists. Have it created.")
    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}")

    s3_key = compute_md5(text_data) + ".txt"

    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=text_data, ContentType="text/plain")

    return s3_key


def s3_store_image(in_path, s3_cfg, user_md5):
    # Initialize S3 client
    access_key = open(s3_cfg["s3_access_key_path"]).read().strip()
    secret_key = open(s3_cfg["s3_secret_key_path"]).read().strip()
    try:
        s3 = boto3.client(
            "s3",
            region_name="ap-northeast-1",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        cfg_bucket_name = s3_cfg["bucket_name"]
        bucket_name = cfg_bucket_name + "-" + user_md5

        s3.head_bucket(Bucket=bucket_name)

    except ClientError:
        print(f"Bucket {bucket_name} Does not exists. Have it created.")
    except Exception as e:
        sys.exit(f"S3 Connection Failed: {e}")

    s3_key = os.path.basename(in_path)

    if os.path.exists(in_path):
        s3.upload_file(in_path, bucket_name, s3_key)
    else:
        sys.exit(f"Error: {path} not found")

    return s3_key


queue_name = "umrs-s3-queue"


def add_bucket_sqs_queue(s3, bucket_name):
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    queue_arn = f"arn:aws:sqs:ap-northeast-1:{account_id}:{queue_name}"
    # Define the event notification configuration
    notification_config = {
        "QueueConfigurations": [{"QueueArn": queue_arn, "Events": ["s3:ObjectCreated:*"]}]
    }

    # Apply the event notification configuration to the S3 bucket
    s3.put_bucket_notification_configuration(
        Bucket=bucket_name, NotificationConfiguration=notification_config
    )

    print(f"\t\t\t--> S3 bucket is now sending events to '{queue_name}'")
