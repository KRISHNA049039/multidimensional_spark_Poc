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
ls deploy/Dockerfile || echo "ERROR: Dockerfile not found!"
docker rm -f spark-master spark-cpu-worker 2>/dev/null || true
docker build --network host -t multi-model-inference:latest -f deploy/Dockerfile .

echo "=== Starting Spark Master ==="
MASTER_IP=$(hostname -I | awk '{print $1}')
docker run -d --name spark-master --network host multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
sleep 10

echo "=== Starting CPU Worker ==="
docker run -d --name spark-cpu-worker --network host multi-model-inference:latest \
  bash -c "start-worker.sh spark://${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
sleep 5

echo "=== Flushing iptables ==="
iptables -F || true

docker logs spark-master --tail 3
echo ""
echo "MASTER_PRIVATE_IP=${MASTER_IP}"
echo "=== Master deploy complete ==="
