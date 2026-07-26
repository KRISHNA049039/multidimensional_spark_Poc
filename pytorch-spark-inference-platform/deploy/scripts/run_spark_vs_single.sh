#!/bin/bash
# =============================================================================
# run_spark_vs_single.sh — Spark vs Single GPU proof benchmark
# Run this inside the GPU EC2 instance (SSM session)
# =============================================================================

set +e
cd /opt/benchmark/app
MASTER_IP=$(hostname -I | awk '{print $1}')
BUCKET=$(grep BUCKET /etc/environment | cut -d= -f2)
REGION=${AWS_DEFAULT_REGION:-us-east-1}

echo "============================================================"
echo "  SPARK vs SINGLE GPU — SCALABILITY PROOF"
echo "  $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# Ensure containers are running
docker ps | grep spark-master > /dev/null || {
    echo "=== Starting Spark cluster ==="
    docker rm -f spark-master spark-gpu-worker 2>/dev/null
    docker run -d --name spark-master --network host --gpus all --shm-size=4g \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -v /opt/benchmark/app/results:/app/results \
      multi-model-inference:latest \
      bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"
    sleep 10
    docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -v /opt/benchmark/app/results:/app/results \
      multi-model-inference:latest \
      bash -c "start-worker.sh spark://${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"
    sleep 10
}

echo ""
echo "=== TEST 1: Main comparison (5K signals) ==="
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/spark_vs_single_gpu.py --signals 5000 --images 200 --detections 50"

echo ""
echo "=== TEST 2: Large scale (20K signals) ==="
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/spark_vs_single_gpu.py --signals 20000 --images 500 --detections 100"

echo ""
echo "=== TEST 3: Full scaling test (find crossover) ==="
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/spark_vs_single_gpu.py --scaling-test --signals 5000 --images 200 --detections 50"

echo ""
echo "=== Syncing results ==="
docker cp spark-master:/app/results/. /opt/benchmark/app/results/ 2>/dev/null || true
aws s3 sync /opt/benchmark/app/results/ s3://${BUCKET}/results/ --region $REGION 2>/dev/null

echo ""
echo "============================================================"
echo "  ALL TESTS COMPLETE — $(date)"
echo "============================================================"
