import boto3
import json

sqs = boto3.client("sqs")
sts = boto3.client("sts")

queue_name = "umrs-s3-queue"
account_id = sts.get_caller_identity()["Account"]

queue_url = f"https://sqs.ap-northeast-1.amazonaws.com/{account_id}/{queue_name}"
queue_arn = f"arn:aws:sqs:ap-northeast-1:{account_id}:{queue_name}"

# Allow all buckets in this AWS account to send events to SQS
policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "s3.amazonaws.com"},
            "Action": "sqs:SendMessage",
            "Resource": queue_arn,
            "Condition": {"StringLike": {"aws:SourceArn": f"arn:aws:s3:::{account_id}-*"}},
        }
    ],
}

# Apply the updated policy to the SQS queue
sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": json.dumps(policy)})

print(f"✅ Updated SQS queue policy to allow events from all S3 buckets in account {account_id}.")
