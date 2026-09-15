import boto3
import json

sqs = boto3.client("sqs")
sts = boto3.client("sts")

queue_name = "umrs-s3-queue"
account_id = sts.get_caller_identity()["Account"]
queue_url = f"https://sqs.ap-northeast-1.amazonaws.com/{account_id}/{queue_name}"
queue_arn = f"arn:aws:sqs:ap-northeast-1:{account_id}:{queue_name}"

# 🔹 List of multiple S3 buckets that should send events
buckets = [
    "ej-bucket-1-c4e3b8ce14815fbc1b26b8677f0b2c14",
    "ej-bucket-2-c4e3b8ce14815fbc1b26b8677f0b2c14",
]  # Add all your bucket names

# 🔹 Create a list of allowed bucket ARNs
source_arns = [f"arn:aws:s3:::{bucket}" for bucket in buckets]

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "s3.amazonaws.com"},
            "Action": "sqs:SendMessage",
            "Resource": queue_arn,
            "Condition": {
                "ArnLike": {"aws:SourceArn": source_arns}  # 🔹 Allow multiple buckets
            },
        }
    ],
}

# Apply the updated policy to the SQS queue
sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": json.dumps(policy)})

print(f"✅ Updated SQS queue policy to allow events from: {', '.join(buckets)}")
