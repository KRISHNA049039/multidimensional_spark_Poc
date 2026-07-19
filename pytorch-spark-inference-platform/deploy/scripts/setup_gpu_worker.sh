#!/bin/bash
set -e
export ARTIFACTS_BUCKET=sparkinferenceclusterstack-artifactsbucket2aac5544-zlcd7fgifndh

echo "=== Pulling code from S3 ==="
cd /opt/spark-inference
aws s3 cp s3://$ARTIFACTS_BUCKET/inference/project.zip /opt/spark-inference/project.zip
rm -rf /opt/spark-inference/app/*
unzip -o /opt/spark-inference/project.zip -d /opt/spark-inference/app

echo "=== Building Docker image ==="
cd /opt/spark-inference/app
docker rm -f spark-gpu-worker 2>/dev/null || true
docker build --network host -t multi-model-inference:latest -f deploy/Dockerfile .

echo "=== Starting GPU Worker ==="
# Get master private IP from first argument or discover it
MASTER_IP=${1:-$(aws ec2 describe-instances --filters "Name=tag:Name,Values=spark-master" "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].PrivateIpAddress" --output text --region ap-south-1)}
echo "Connecting to master at: $MASTER_IP"

docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
sleep 5

docker logs spark-gpu-worker --tail 5
echo "=== GPU Worker deploy complete ==="
