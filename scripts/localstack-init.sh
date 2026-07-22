#!/bin/bash
# Runs inside the LocalStack container on startup.
# Creates the bucket, the report-processing queue, and its dead-letter queue with a
# redrive policy — mirroring what must exist in real AWS before cutover.
set -euo pipefail

BUCKET="${S3_BUCKET:-mhn-reports-local}"
QUEUE="${QUEUE_NAME:-report-processing}"
DLQ="${QUEUE_NAME:-report-processing}-dlq"
MAX_RECEIVE="${SQS_MAX_RECEIVE_COUNT:-5}"

awslocal s3api create-bucket \
  --bucket "$BUCKET" \
  --create-bucket-configuration LocationConstraint="${AWS_DEFAULT_REGION:-ap-south-1}" \
  2>/dev/null || awslocal s3api create-bucket --bucket "$BUCKET"

# Source files stay private. No public access, ever.
awslocal s3api put-public-access-block \
  --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

DLQ_URL=$(awslocal sqs create-queue --queue-name "$DLQ" --output text --query QueueUrl)
DLQ_ARN=$(awslocal sqs get-queue-attributes \
  --queue-url "$DLQ_URL" --attribute-names QueueArn --output text --query 'Attributes.QueueArn')

# Visibility timeout is generous: an AI stage can legitimately run for minutes, and
# the worker extends it via heartbeat for anything longer.
awslocal sqs create-queue --queue-name "$QUEUE" --attributes "$(cat <<JSON
{
  "VisibilityTimeout": "300",
  "MessageRetentionPeriod": "345600",
  "RedrivePolicy": "{\"deadLetterTargetArn\":\"${DLQ_ARN}\",\"maxReceiveCount\":\"${MAX_RECEIVE}\"}"
}
JSON
)" >/dev/null

echo "localstack-init: bucket=${BUCKET} queue=${QUEUE} dlq=${DLQ} maxReceiveCount=${MAX_RECEIVE}"
