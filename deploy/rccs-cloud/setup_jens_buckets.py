#!/usr/bin/env python3
"""Create MinIO buckets and public policies for Jens.

Run on rccs-cloud (login node) where MinIO is reachable at localhost:9000.

Creates:
  - 'jens'        bucket with anonymous GET  (input images)
  - 'jens-output' bucket with anonymous GET + PUT (results)

After running, Jens can use plain HTTP URLs (no signature, no expiry):
  Download: curl http://localhost:9000/jens/<key>
  Upload:   curl -X PUT --upload-file result.png http://localhost:9000/jens-output/<key>
"""

import json
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ENDPOINT = "http://localhost:9000"
ACCESS_KEY = os.environ["MINIO_ROOT_USER"]
SECRET_KEY = os.environ["MINIO_ROOT_PASSWORD"]

BUCKETS = {
    "jens": {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": ["arn:aws:s3:::jens/*"],
            },
        ],
    },
    "jens-output": {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": ["arn:aws:s3:::jens-output/*"],
            },
        ],
    },
}


def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    for bucket_name, policy in BUCKETS.items():
        # Create bucket (idempotent)
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"  Bucket '{bucket_name}' already exists.")
        except ClientError:
            s3.create_bucket(Bucket=bucket_name)
            print(f"  Created bucket '{bucket_name}'.")

        # Apply public policy
        s3.put_bucket_policy(
            Bucket=bucket_name,
            Policy=json.dumps(policy),
        )
        print(f"  Applied public policy to '{bucket_name}'.")

    # Verify
    print("\n--- Verification ---")
    response = s3.list_buckets()
    existing = {b["Name"] for b in response["Buckets"]}
    all_ok = True
    for bucket_name in BUCKETS:
        if bucket_name in existing:
            print(f"  [OK] {bucket_name}")
        else:
            print(f"  [FAIL] {bucket_name} not found!")
            all_ok = False

    print("\n--- URLs for Jens ---")
    print(f"  Input  (GET): http://localhost:9000/jens/<key>")
    print(f"  Output (PUT): http://localhost:9000/jens-output/<key>")
    print(f"\nExample:")
    print(f"  curl http://localhost:9000/jens/test/image.png -o image.png")
    print(
        f"  curl -X PUT --upload-file result.png http://localhost:9000/jens-output/test/result.png"
    )

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
