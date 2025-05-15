import boto3
import argparse
import os
import warnings
import urllib3
import subprocess

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


default_config_file = os.path.expanduser("~/.s3cfg")
parser = argparse.ArgumentParser(description="S3 Client demo")
parser.add_argument("-c", "--config", default=default_config_file, help="Path to the S3 config file (default: ~/.s3cfg)")

args = parser.parse_args()

s3 = connect_to_s3(args.config)
# 🔹 Test: List Buckets

print ("Listing my buckets...");
buckets = s3.list_buckets()
for bucket in buckets["Buckets"]:
    print("\t"+bucket["Name"])

s3.head_bucket(Bucket = "test-ej")

delete_all_files_in_s3_bucket(s3, "test-ej")

print("Listing test-ej:")
keys = list_s3_bucket(s3, "test-ej")
if len(keys)==0:
   print("\t<empty>")
for key in keys:
    print ("\t"+key)

   

    
print("Adding /home/ejeannot/data/Cat_4.jpg  to test-ej...")
upload_file_to_s3(s3, "test-ej", "/home/ejeannot/data/Cat_4.jpg", "img/data-1")

print("Listing test-ej:")
keys = list_s3_bucket(s3, "test-ej")
if len(keys)==0:
   print("\t<empty>")
for key in keys:
    print ("\t"+key)



print("Pulling file to test-ej into /home/ejeannot/tmp/test.jpg")
get_file_from_s3(s3, "test-ej", "img/data-1", "/home/ejeannot/tmp/test.jpg")


print("Checking md5")

cmd = ["md5sum", "/home/ejeannot/tmp/test.jpg", "/home/ejeannot/data/Cat_4.jpg"]
subprocess.run(cmd, check=True)
