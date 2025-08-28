import boto3
import json

bucket_name = "test-ej"
endpoint_url = "https://127.0.0.1:9000"  # or your MinIO endpoint

# Admin credentials for performing policy changes
MINIO_USER="ejeannot"                      # MinIO Root User
MINIO_PASS="jemdaz-satwyz-xojtA6"          # MinIO Root Password

client = boto3.client(
    "s3",
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASS,
    verify=False,
    endpoint_url=endpoint_url
)

deny_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyUser",
            "Effect": "Deny",
            "Principal": {"AWS": ["ssD4YLf53VDMiVUy2LMe"]},
            "Action": "s3:*",
            "Resource": [
                f"arn:aws:s3:::{bucket_name}",
                f"arn:aws:s3:::{bucket_name}/*"
            ]
        }
    ]
}

policy_json = json.dumps(deny_policy)
r = client.put_bucket_policy(Bucket=bucket_name, Policy=policy_json)
print(f"Policy applied: user access blocked: {r}")
