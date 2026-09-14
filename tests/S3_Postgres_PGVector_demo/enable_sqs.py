import json

import boto3

sqs = boto3.client("sqs")
sts = boto3.client("sts")

queue_name = "umrs-s3-queue"
account_id = sts.get_caller_identity()["Account"]
bucket_name = "my-bucket-name"  # Replace with your bucket name

queue_url = f"https://sqs.ap-northeast-1.amazonaws.com/{account_id}/{queue_name}"
queue_arn = f"arn:aws:sqs:ap-northeast-1:{account_id}:{queue_name}"

policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "sqs:SendMessage",
            "Resource": queue_arn,
            "Condition": {"ArnLike": {"aws:SourceArn": f"arn:aws:s3:::{bucket_name}"}},
        }
    ],
}

# Apply the updated policy to the SQS queue
sqs.set_queue_attributes(QueueUrl=queue_url, Attributes={"Policy": json.dumps(policy)})

print(f"✅ S3 is now allowed to send events to '{queue_name}'")
