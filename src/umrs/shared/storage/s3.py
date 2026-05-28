#!/usr/bin/python3
import configparser
import mimetypes
import os, sys
import secrets
import string
import warnings
from io import BytesIO

import boto3
import urllib3
from botocore.exceptions import ClientError
from charset_normalizer import from_bytes, from_path
from PIL import Image

from umrs.shared.utils import compute_md5, md5_of_file

# Suppress only SSL warnings from urllib3
warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)


def get_s3_server_name(s3):

    response = s3.list_buckets()
    headers = response['ResponseMetadata']['HTTPHeaders']


    server= headers['server']

    if "minio" in server.lower():
        return "MinIO"
    else:
        raise RuntimeError(f"Do not know how to handle S3 server {server}")


def s3_strong_password(length=24):
    if length <= 0:
        raise ValueError("length must be positive")

    chars = string.ascii_letters + string.digits + "-_@"
    leading_chars = string.ascii_letters + string.digits + "_@"
    return secrets.choice(leading_chars) + ''.join(
        secrets.choice(chars) for _ in range(length - 1)
    )


def s3_key_exists(s3, bucket_name, key):
    """
    Check if a specific key exists in the given S3 bucket.

    Returns True if the key exists, False otherwise.
    """
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            # Something else went wrong, re-raise
            raise


def s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list):


    server_name = get_s3_server_name(s3_connection)
    config = {}

    config_path = os.path.expanduser(admin_s3_configuration_file)

    # Initialize parser
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read(config_path)
    bucket_location = config.get("default", "bucket_location", fallback = None)
    if bucket_location is None:
        raise RuntimeError(f"file {admin_s3_configuration_file} must have a bucket_location entry for [default]")

    for bucket in bucket_list:
        if not check_bucket_path(s3_connection, bucket, indent=False):
            print("\t\t\t--> Creating bucket {bucket}")
            s3_connection.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={
                    'LocationConstraint':   bucket_location # Match your desired region
                }
            )

    if server_name == "MinIO":
        from umrs.shared.storage.minio import minio_grant_user_access, minio_s3_create_user
        alias, s3cfg_content = minio_s3_create_user(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file)
        if alias:
            for bucket in bucket_list:
                minio_grant_user_access(username, bucket, alias)
        return s3cfg_content

    else:
        raise RuntimeError(f"Do not know how to create user and buckets for server '{server_name}'")



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

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"S3 Config file '{config_file}' not found")

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
        s3.list_buckets()
        print("--> Succesfully connected to S3 service.")
    except Exception as e:
        # raise RuntimeError(f"S3 Connection Failed: {e}\nCheck that the S3 service is up and running.") from e
        print(f"S3 Connection Failed: {e}\nCheck that the S3 service is up and running.\nCannot continue.")
        sys.exit(-1)

    return s3, bucket_location

def make_path_to_bucket_string(xp_name, bucket_alias, username, output=False):
    path = username+"."+xp_name+"."+bucket_alias
    if output:
        path += "-output"
    return path.lower().replace(" ","-").replace("_","-")



def check_bucket_path(s3, path_to_bucket, indent = True, exit_on_error = True):
    if indent:
        print(f"\t\t\t--> Checking bucket '{path_to_bucket}'... ", end="")
    else:
        print(f"--> Checking bucket '{path_to_bucket}'... ", end="")

    try:
        s3.head_bucket(Bucket = path_to_bucket)

    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("Not Found")
            return False #Not found
        else:
            if exit_on_error:
                raise RuntimeError(f"S3 Error: {e}") from e
            else:
                return False
    except Exception as e:
        raise RuntimeError(f"S3 Connection Failed: {e}") from e

    print("OK")
    return True


def decompose_bucket_alias_arg(bucket_alias_arg):
    parts = bucket_alias_arg.split("/", 1)
    first_subdir = parts[0]
    remaining = parts[1] if len(parts) > 1 else None
    return first_subdir, remaining

def check_bucket_name(s3, xp_name, bucket_alias, username, output=False, indent=False):
    path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username, output=output)
    return path_to_bucket, check_bucket_path(s3, path_to_bucket, indent=indent)

def configure_bucket(s3, bucket_alias, xp_name, username, bucket_location, output=False):

    path_to_bucket, success = check_bucket_name(s3, xp_name, bucket_alias, username, output=output, indent=True)
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
        print(f"\t\t\t--> Bucket {path_to_bucket} created")

    print("\t\t\t--> S3 Bucket Configuration Successful")
    return path_to_bucket


def count_objects_in_bucket(s3, bucket_name, verbose=False):
    """
    Counts the total number of objects in an S3 bucket.

    Args:
        s3 (boto3.client): A boto3 S3 client object.
        bucket_name (str): Name of the S3 bucket.
        verbose (bool): If True, prints progress for each paginated request.

    Returns:
        int: Total number of objects in the bucket.
    """
    total_objects = 0

    try:
        paginator = s3.get_paginator("list_objects_v2")
        page_iter = paginator.paginate(Bucket=bucket_name)

        for i, page in enumerate(page_iter, start=1):
            count = page.get("KeyCount", 0)
            total_objects += count
            if verbose:
                print(f"Page {i}: {count} objects, cumulative total: {total_objects}")

        if verbose:
            print(f"Total objects in bucket '{bucket_name}': {total_objects}")

        return total_objects

    except ClientError as e:
        raise RuntimeError(f"Error accessing bucket '{bucket_name}': {e}") from e


def list_s3_bucket(s3, bucket_name, prefix=""):
    """
    List all objects in an S3 bucket, optionally restricted to one subdirectory.

    Parameters
    ----------
    s3 : boto3 client
    bucket_name : str
    prefix : str
        e.g. "folder/subfolder/" — only files under this path are listed.

    Returns
    -------
    objects_list : list[str]
        Full S3 URIs
    total_size : int
        Sum of object sizes in bytes
    checksums : list[str]
        ETag checksums
    """
    # print(f"Listing objects in bucket: {bucket_name} (prefix='{prefix}')")

    objects_list = []
    checksums = []
    total_size = 0

    prefix = "" if prefix is None else prefix

    try:
        continuation_token = None

        while True:
            kwargs = {
                "Bucket": bucket_name,
                "Prefix": prefix        # <── FILTER HERE
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = s3.list_objects_v2(**kwargs)

            for obj in response.get('Contents', []):
                objects_list.append(f"s3://{bucket_name}/{obj['Key']}")
                checksums.append(obj['ETag'].strip('"'))
                total_size += obj['Size']

            if response.get('IsTruncated'):
                continuation_token = response.get('NextContinuationToken')
            else:
                break

    except s3.exceptions.NoSuchBucket:
        print(f"Bucket not found: {bucket_name}")
    except Exception as e:
        print(f"Error listing bucket '{bucket_name}': {e}")

    return objects_list, total_size, checksums

def load_data_from_s3(s3, path_to_bucket, s3_keys, modality_name):
    from umrs.shared.modality import modality_category
    category = modality_category(modality_name)
    objects = []
    metadata = []
    checksums = []
    sizes = []
    has_at_least_a_zero = False
    for s3_key in s3_keys:
        obj = s3.get_object(Bucket=path_to_bucket, Key=s3_key)
        checksums.append(obj['ETag'].strip('"'))
        metadata.append(obj.get("Metadata", {}))
        obj_size = obj['ContentLength']
        sizes.append(obj_size)

        if category == "text":  # Handle text stored in file
            if obj_size:
                text = obj["Body"].read().decode("utf-8")
            else :
                has_at_least_a_zero = True
                text = "A"
            objects.append(text)

        elif category == "image":  # Handle Image Files
            image = Image.open(BytesIO(obj["Body"].read()))
            objects.append(image)

        elif category in ("text_file", "text file"):  # Handle Text Files
            if obj_size:
                text = str(from_bytes(obj["Body"].read()).best())
            else :
                has_at_least_a_zero = True
                text = "A"
            objects.append(text)

        else:
            raise ValueError(f"Unsupported modality '{modality_name}' (category: '{category}') while loading data '{path_to_bucket}/{s3_key}' from S3")

    return objects, metadata, checksums, sizes, has_at_least_a_zero



def download_objects(s3, objects_list, checksums, local_path, files_allready_received=None, force_overwrite=False, verbose=False, print_only_new=False):
    """
    Download S3 objects into a local directory ONLY if the directory already exists.

    objects_list : ["s3://bucket/key1", ...]
    checksums    : ["etag1", "etag2", ...]
    local_path   : local folder (must already exist)
    """
    downloaded_files = set()
    skipped_files = set()
    # --- 1. Local directory must exist ---
    if not os.path.isdir(local_path):
        print(f"Local directory does not exist: {local_path}. Nothing done.")
        return

    # print(f"Downloading {len(objects_list)} objects into {local_path}")

    for s3_uri, s3_etag in zip(objects_list, checksums, strict=False):

        # Extract bucket + key
        # e.g. "s3://my-bucket/path/to/file" → bucket="my-bucket", key="path/to/file"
        assert s3_uri.startswith("s3://")
        _, _, bucket_key = s3_uri.partition("s3://")
        bucket, _, key = bucket_key.partition("/")

        # Local destination path
        filename = os.path.basename(key)
        local_file = os.path.join(local_path, filename)

        # --- 2. If the file has already been received previously continue
        if files_allready_received and local_file in files_allready_received:
            continue

        # --- 3. If the file already exists → use checksum to skip or overwrite ---
        if os.path.exists(local_file):
            local_md5 = md5_of_file(local_file)
            # print(f"{local_md5=} <=> {s3_etag=}")
            if local_md5 == s3_etag:
                if not print_only_new and verbose:
                    print(f"[SKIP] {filename} (checksum match)")
                skipped_files.add(f"{local_file}: checksum match")
                downloaded_files.add(local_file)
                continue
            else:
                if  force_overwrite :
                    if verbose:
                        print(f"[OVERWRITE] {filename} (checksum mismatch)")
                else :
                    if verbose:
                        print(f"[NO OVERWRITE - SKIP] {filename} (checksum mismatch) -")
                    skipped_files.add(f"{local_file}: No overwrite")
                    continue
        # --- 4. Download the file ---
        try:
            if verbose:
                print(f"[DOWNLOAD] {s3_uri} → {local_file}")
            s3.download_file(bucket, key, local_file)
        except Exception as e:
            print(f"Error downloading {s3_uri}: {e}")

        downloaded_files.add(local_file)

    return downloaded_files, skipped_files


def s3_store_txt(s3, path_to_bucket, subdir, text_data, mq, metadata_args):
    # Store inline text as an S3 object and notify server

    s3_key = compute_md5(text_data)+".txt"
    if subdir is not None:
        s3_key = f"{subdir}/{s3_key}"

    s3.put_object(Bucket=path_to_bucket, Key=s3_key, Body=text_data, Metadata=metadata_args["Metadata"], ContentType="text/plain")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key

def is_image_by_extension(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type is not None and mime_type.startswith("image")

def is_image_by_opening(filename):
    try:
        with Image.open(filename) as img:
            img.verify()  # Verify if it's a valid image
        return True
    except (OSError, SyntaxError):
        return False

def is_an_image(filename):
    # Check by extension first
    if not is_image_by_extension(filename):
        return False

    # If extension is valid, confirm by opening
    return is_image_by_opening(filename)

def s3_store_image(s3, path_to_bucket, subdir, in_path, mq, metadata_args, check = True):
    if check and not is_an_image(in_path):
        print(f"Error: '{in_path}' is not an image. Skipping...")
        return None

    s3_key = os.path.basename(in_path)
    if subdir is not None:
        s3_key = f"{subdir}/{s3_key}"

    if os.path.exists(in_path):
        s3.upload_file(in_path, path_to_bucket, s3_key, ExtraArgs=metadata_args)
    else:
        raise FileNotFoundError(f"Error: {in_path} not found")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key

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


def save_tiff_image_to_s3(s3, img_array, path_to_bucket, s3_key):
    """
    Save a NumPy image array to S3 as a TIFF file.

    img_array: HWC or CHW uint8 NumPy array
    bucket_name: str, name of the bucket
    s3_key: str, full key/path in the bucket
    """
    # Convert CHW -> HWC if needed
    if img_array.shape[0] in [1, 3]:  # CHW
        img_array = img_array.transpose(1, 2, 0)

    # Convert to PIL Image
    img = Image.fromarray(img_array)

    # Save to in-memory buffer
    buffer = BytesIO()
    img.save(buffer, format="TIFF")
    buffer.seek(0)

    # Upload to S3
    return s3.put_object(Bucket=path_to_bucket, Key=s3_key, Body=buffer.getvalue())

def s3_store_text_file(s3, path_to_bucket, subdir, in_path, mq, metadata_args):
    if not is_a_text_file(in_path):
        print(f"Error: '{in_path}' is not a txt file. Skipping...")
        return None

    s3_key = os.path.basename(in_path)
    if subdir is not None:
        s3_key = f"{subdir}/{s3_key}"

    if os.path.exists(in_path):
        s3.upload_file(in_path, path_to_bucket, s3_key, ExtraArgs=metadata_args)
    else:
        raise FileNotFoundError(f"Error: {in_path} not found")

    s3_object = f"s3://{path_to_bucket}/{s3_key}"
    mq.notify_s3(s3_object)

    return s3_key

