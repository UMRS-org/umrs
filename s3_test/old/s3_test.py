#!/usr/bin/env python3
import boto3
import argparse
import os, sys
import warnings
import urllib3
import subprocess
import json
from botocore.exceptions import ClientError, EndpointConnectionError

# ✅ Suppress only SSL warnings from urllib3
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
    region = config.get("bucket_location")
    
    # Ensure correct protocol (http or https)
    protocol = "https" if use_https else "http"
    endpoint_url = f"{protocol}://{s3_host}"

    # Create the S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=verify_ssl,  # Disables SSL verification if False
        region_name = region
    )

    # Test connection: try listing buckets
    try:
        s3.list_buckets()
    except EndpointConnectionError as e:
        raise RuntimeError(f"Failed to connect to the S3 endpoint: {e}")
    except ClientError as e:
        # If access denied or other client errors, it still means connection succeeded
        error_code = e.response['Error']['Code']
        if error_code in ['AccessDenied', 'Unauthorized']:
            print("Connection successful but access denied for listing buckets.")
        else:
            raise RuntimeError(f"S3 ClientError: {e}")

    print("S3 connection test passed.")

    
    return s3, access_key

def delete_all_files_in_s3_bucket(s3_client, bucket_name):
    """Deletes all files in an S3 bucket."""
    objects = s3_client.list_objects_v2(Bucket=bucket_name).get("Contents", [])
    if objects:
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]}
        )
        
def upload_file_to_s3(s3_client, bucket_name, file_path, s3_key):
    """Uploads a file to an S3 bucket."""
    s3_client.upload_file(file_path, bucket_name, s3_key)

def list_s3_bucket(s3_client, bucket_name):
    """Lists all files in an S3 bucket."""
    return [obj["Key"] for obj in s3_client.list_objects_v2(Bucket=bucket_name).get("Contents", [])]

def get_file_from_s3(s3_client, bucket_name, s3_key, download_path):
    """Downloads a file from an S3 bucket."""
    s3_client.download_file(bucket_name, s3_key, download_path)

# def set_bucket_to_users(s3, bucket_name, access_key_list):
#     """
#     Set a bucket policy that *allows* access only to the specified users and *denies* everyone else.

#     Args:
#         s3 (boto3.client): Boto3 S3 client
#         bucket_name (str): Bucket name
#         access_key_list (list): List of allowed principals (strings of ARNs or MinIO access keys)
#     """
#     policy = {
#         "Version": "2012-10-17",
#         "Statement": [
#             # Allow the specified users full access
#             {
#                 "Sid": "AllowSpecificUsers",
#                 "Effect": "Allow",
#                 "Principal": {
#                     "AWS": access_key_list
#                 },
#                 "Action": "s3:*",
#                 "Resource": [
#                     f"arn:aws:s3:::{bucket_name}",
#                     f"arn:aws:s3:::{bucket_name}/*"
#                 ]
#             },
#             # Deny everyone else access
#             {
#                 "Sid": "DenyAllExceptSpecifiedUsers",
#                 "Effect": "Deny",
#                 "NotPrincipal": {
#                     "AWS": access_key_list
#                 },
#                 "Action": "s3:*",
#                 "Resource": [
#                     f"arn:aws:s3:::{bucket_name}",
#                     f"arn:aws:s3:::{bucket_name}/*"
#                 ]
#             }
#         ]
#     }

#     policy_json = json.dumps(policy)
#     s3.put_bucket_policy(Bucket=bucket_name, Policy=policy_json)
#     print(f"Bucket policy applied to bucket '{bucket_name}', restricted to specified users only.")


# Example call (replace with actual ARNs or for MinIO access keys)
# For AWS IAM users:
# allowed_arns = ["arn:aws:iam::123456789012:user/YourUsername"]
# For MinIO, use access keys like: ["minio-access-key"]

# s3 = boto3.client('s3', ...)
# set_bucket_to_users(s3, "test-ej", allowed_arns)

def set_bucket_to_users(s3, bucket_name, access_key_list_allow, access_key_list_deny):
    """
    Set a bucket policy that *allows* access to the specified users and *denies* everyone else.

    Args:
        s3 (boto3.client): Boto3 S3 client
        bucket_name (str): Bucket name
        access_key_list (list): List of allowed principals (IAM ARNs or MinIO access keys)
    """
    
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            # First Deny everyone else aaccess to bucket and objects
            {
                "Effect": "Deny",
                "Principal": {
                    "AWS": access_key_list_deny
                },
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            },
            # Then allow specified users full access
            {
                "Effect": "Allow",
                "Principal": {
                    "AWS": access_key_list_allow
                },
                "Action": "s3:*",
                "Resource": [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy)
    print (policy_json)
    s3.put_bucket_policy(Bucket=bucket_name, Policy=policy_json)
    print(f"Bucket policy applied to bucket '{bucket_name}', allowing only the specified users access and denying everyone else.")


def readonly_user(user):

    policy_name = "readonly"
    target = "myminio"
    # Attach policy to user
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True
        )
        print(f"Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")

    
def grant_access(user, bucket):
    """background
    Create and attach a deny-write policy for the specified bucket to a MinIO user.

    Args:
        user (str): MinIO username (access key).
        bucket (str): Bucket name to restrict.
    """

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:DeleteObject",
                    "s3:ListAllMyBuckets",
                    "s3:ListBucket",
                    "s3:PutObject"
                ],
                "Resource": [
                    f"arn:aws:s3:::{bucket}/*"
                ]
            }
        ]
    }

    policy_json = json.dumps(policy, indent=2)

    policy_file = "/tmp/policy.json"
    # Save policy JSON to file
    with open(policy_file, 'w') as f:
        f.write(policy_json)

    policy_name = f"deny-{bucket}"
    target = "myminio"  # mc alias name (must be configured)

    # Create policy
    try:
        subprocess.run(
            ["mc", "admin", "policy", "create", target, policy_name, policy_file, "--insecure"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error creating policy: {e}")
        return

    # Attach policy to user
    try:
        subprocess.run(
            ["mc", "admin", "policy", "attach", target, policy_name, "--user", user, "--insecure"],
            check=True
        )
        print(f"Policy '{policy_name}' attached to user '{user}'.")
    except subprocess.CalledProcessError as e:
        print(f"Error attaching policy to user: {e}")


default_config_file = os.path.expanduser("~/.s3cfg")
parser = argparse.ArgumentParser(description="S3 Client demo")
parser.add_argument("-c", "--config", default=default_config_file, help="Path to the S3 config file (default: ~/.s3cfg)")

args = parser.parse_args()


s3, access_key = connect_to_s3(args.config)

readonly_user("Badguy")
grant_access("Badguy", "test-ej")

print (f"acces_key: {access_key}")

# Test: List Buckets

print ("Listing my buckets...");
try:
    buckets = s3.list_buckets()
    print("Bucket exists and access granted!")
    for bucket in buckets["Buckets"]:
        print("\t"+bucket["Name"])
except ClientError as e:
    # error_code = int(e.response['Error']['Code'])
    print(f"Error: {e}")#. Error code;{error_code}")



try:
    s3.head_bucket(Bucket="test-ej")
except ClientError as e:
    error_code = int(e.response['Error']['Code'])
    if error_code == 404:
        print("Bucket does not exist.")
        s3.create_bucket(
            Bucket="test-ej",
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-northeast-1'  # Match your desired region
            }
        )
    elif error_code == 403:
        print(f"{e}: Bucket exists but you do not have access.")
        sys.exit()
    else:
        print(f"Error: {e}")


delete_all_files_in_s3_bucket(s3, "test-ej")

print("Listing test-ej:")
keys = list_s3_bucket(s3, "test-ej")
if len(keys)==0:
   print("\t<empty>")
for key in keys:
    print ("\t"+key)

os.makedirs('../../data/tmp/', exist_ok=True)
inpath = "../../data/img/Cat_4.jpg"
outpath = "../../data/tmp/test.jpg"

print(f"Addding {inpath} to test-ej...")
upload_file_to_s3(s3, "test-ej", inpath, "img/data-1")

print("Listing test-ej:")
keys = list_s3_bucket(s3, "test-ej")
if len(keys)==0:
   print("\t<empty>")
for key in keys:
    print ("\t"+key)

print(f"Pulling file to test-ej into {outpath}")
get_file_from_s3(s3, "test-ej", "img/data-1", outpath)

print("Checking md5")

cmd = ["md5sum", inpath, outpath]
subprocess.run(cmd, check=True)
