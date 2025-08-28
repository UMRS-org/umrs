import boto3
import json

s3 = boto3.client("s3")
sts = boto3.client("sts")

queue_name = "umrs-s3-queue"
account_id = sts.get_caller_identity()["Account"]
queue_arn = f"arn:aws:sqs:ap-northeast-1:{account_id}:{queue_name}"

# Get all buckets
buckets = s3.list_buckets()["Buckets"]

attached_buckets = []

# Check each bucket for SQS event notifications
for bucket in buckets:
    bucket_name = bucket["Name"]
    
    try:
        response = s3.get_bucket_notification_configuration(Bucket=bucket_name)
        queue_configs = response.get("QueueConfigurations", [])

        # Check if any notification is set for this SQS queue
        for config in queue_configs:
            if config.get("QueueArn") == queue_arn:
                attached_buckets.append(bucket_name)

    except Exception as e:
        print(f"⚠️ Could not retrieve notifications for {bucket_name}: {e}")

# Print the list of attached buckets
if attached_buckets:
    print("✅ Buckets sending events to", queue_name, ":")
    for bucket in attached_buckets:
        print(f"- {bucket}")
else:
    print("⚠️ No buckets are currently sending events to", queue_name)
