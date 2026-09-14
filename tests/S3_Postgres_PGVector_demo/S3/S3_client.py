#!/usr/bin/python3
import hashlib
import os

import boto3
from botocore.exceptions import ClientError


def check_s3_free_tier_usage():
    # Initialize the S3 and CloudWatch clients
    s3 = boto3.client("s3")
    cloudwatch = boto3.client("cloudwatch")

    try:
        # Get the total size of objects in S3 buckets
        total_storage_bytes = 0
        buckets = s3.list_buckets().get("Buckets", [])

        for bucket in buckets:
            bucket_name = bucket["Name"]
            response = s3.list_objects_v2(Bucket=bucket_name)
            if "Contents" in response:
                total_storage_bytes += sum(obj["Size"] for obj in response["Contents"])

        total_storage_gb = total_storage_bytes / (1024**2)  # Convert to MB

        get_requests = 0
        put_requests = 0

        # Compare with Free Tier limits
        under_storage_limit = total_storage_gb <= 5 * 1024  # 5 GB limit
        under_get_limit = get_requests <= 20000  # 20,000 GET requests
        under_put_limit = put_requests <= 2000  # 2,000 PUT requests

        print(f"Total Storage: {total_storage_gb:.2f} MB")
        print(f"GET Requests: {get_requests}")
        print(f"PUT Requests: {put_requests}")

        if under_storage_limit and under_get_limit and under_put_limit:
            print("You are within the AWS Free Tier limits.")
        else:
            print("You have exceeded one or more AWS Free Tier limits.")
            if not under_storage_limit:
                print(" - Storage limit exceeded.")
            if not under_get_limit:
                print(" - GET requests limit exceeded.")
            if not under_put_limit:
                print(" - PUT requests limit exceeded.")
    except ClientError as e:
        print(f"Error: {e}")


def compute_md5(input_string):
    return hashlib.md5(input_string.encode()).hexdigest()


def progress_img_callback(bytes_transferred):
    print(f"Image uploaded: {bytes_transferred} bytes")


def upload_image(in_path, bucket_name, s3_key, desc="", author="Unknown"):

    response = s3.head_object(Bucket=bucket_name, Key=s3_key)
    if response:
        s3_size = response["ContentLength"]  # Get object size in bytes
        local_size = os.path.getsize(in_path)
        if s3_size == local_size:
            print(f"{in_path} already exist with same size. Skipping")
            return

    # Upload file with ContentType and custom Metadata
    s3.upload_file(
        in_path,
        bucket_name,
        s3_key,
        ExtraArgs={
            "ContentType": "text/plain",  # Specify the MIME type
            "Metadata": {  # Custom metadata
                "content-summary": f"{desc}",
                "author": f"{author}",
            },
        },
        Callback=progress_img_callback,
    )


def upload_all_images():
    upload_image("./img/Cat_1.jpeg", bucket_name, "./img/Cat_1.jpeg", "cat,mammal")
    upload_image("./img/Cat_2.jpeg", bucket_name, "./img/Cat_2.jpeg", "cat,mammal")
    upload_image("./img/Cat_3.jpeg", bucket_name, "./img/Cat_3.jpeg", "cat,mammal", "EJ")
    upload_image("./img/Cat_4.jpeg", bucket_name, "./img/Cat_3.jpeg", "cat,mammal", "EJ")
    upload_image("./img/Dog_1.jpeg", bucket_name, "./img/Dog_1.jpeg", "dog,mammal")
    upload_image("./img/Dog_2.jpeg", bucket_name, "./img/Dog_2.jpeg", "dog,mammal", "EJ")
    upload_image("./img/Dog_3.jpeg", bucket_name, "./img/Dog_3.jpeg", "dog,mammal")
    upload_image("./img/Eagle_1.jpeg", bucket_name, "./img/Eagle_1.jpeg", "eagle,bird")
    upload_image("./img/Eagle_2.jpeg", bucket_name, "./img/Eagle_2.jpeg", "eagle,bird")
    upload_image("./img/Eagle_3.jpeg", bucket_name, "./img/Eagle_3.jpeg", "eagle,bird")


def list_objects(bucket_name, desc=None, author=None, content_type=None):
    response = s3.list_objects_v2(Bucket=bucket_name)

    for obj in response.get("Contents", []):
        obj_key = obj["Key"]
        obj_data = s3.head_object(Bucket=bucket_name, Key=obj_key)  # Single API call
        obj_type = obj_data.get("ContentType", {})
        obj_metadata = obj_data.get("Metadata", {})
        obj_author = obj_metadata.get("author")
        obj_desc = obj_metadata.get("content-summary")
        if author != None and author != obj_author:
            continue
        if desc != None and desc not in obj_desc:
            continue
        if content_type != None and content_type != obj_type:
            continue
        print(obj_key)


# Initialize S3 client
s3 = boto3.client("s3", region_name="ap-northeast-1")  # Specify the region here

# Generate a unique bucket name
str = "Emmanuel Jeannot"
bucket_name = f"ej-bucket2-{compute_md5(str)}"

if not s3.head_bucket(Bucket=bucket_name):
    # Create a new S3 bucket with a region configuration
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={
            "LocationConstraint": "ap-northeast-1"  # Match your desired region
        },
    )

    print(f"Bucket '{bucket_name}' created!")
else:
    print(f"Bucket '{bucket_name}' already exists!")


upload_all_images()

list_objects(bucket_name, desc="mammal", author="EJ")


# Run the function
check_s3_free_tier_usage()
