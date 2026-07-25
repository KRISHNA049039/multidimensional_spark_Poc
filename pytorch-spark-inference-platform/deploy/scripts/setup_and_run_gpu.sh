#!/bin/bash
# =============================================================================
# setup_and_run_gpu.sh — Full automated GPU benchmark on EC2
# Upload this to S3 and run via SSM. Logs to /opt/benchmark/benchmark.log
#
# Expects: BUCKET env var set in /etc/environment (done by CDK UserData)
# =============================================================================
set +e  # Don't exit on errors (Spark logs non-fatal errors to stderr)
exec > >(tee -a /opt/benchmark/benchmark.log) 2>&1

echo "============================================================"
echo "  GPU BENCHMARK — AUTOMATED SETUP AND RUN"
echo "  Started: $(date)"
echo "============================================================"

source /etc/environment
REGION=${AWS_DEFAULT_REGION:-us-east-1}
# BUCKET might be set by CDK UserData
if [ -z "$BUCKET" ]; then
    BUCKET=$(grep BUCKET /etc/environment 2>/dev/null | cut -d= -f2)
fi

# =============================================================================
# STEP 1: Setup Docker + GPU runtime
# =============================================================================
echo ""
echo "=== STEP 1: Setting up Docker and GPU ==="

# Deep Learning AMI already has docker-ce + nvidia drivers
# Just make sure Docker is running
systemctl start docker 2>/dev/null || service docker start 2>/dev/null || true
docker --version || { echo "FATAL: Docker not available"; exit 1; }

# Install missing utils
apt-get install -y unzip jq 2>/dev/null || true

# Ensure nvidia-container-toolkit is configured
if ! docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi 2>/dev/null; then
    echo "  Configuring nvidia-container-toolkit..."
    nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
    systemctl restart docker 2>/dev/null || true
    sleep 3
fi

echo ""
echo "=== STEP 1b: Pulling code from S3 ==="
echo "  Bucket: $BUCKET | Region: $REGION"
mkdir -p /opt/benchmark/app
aws s3 cp s3://$BUCKET/project.zip /opt/benchmark/project.zip --region $REGION
cd /opt/benchmark && rm -rf app/* && unzip -o project.zip -d app
cd /opt/benchmark/app
echo "  Code extracted: $(ls | wc -l) top-level items"

# =============================================================================
# STEP 2: Verify GPU
# =============================================================================
echo ""
echo "=== STEP 2: Verifying GPU ==="
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed! GPU not available."
    echo "Attempting driver install..."
    apt-get install -y linux-headers-$(uname -r) || true
    nvidia-smi || { echo "FATAL: No GPU. Aborting."; exit 1; }
fi
echo "  GPU verified OK"


# =============================================================================
# STEP 3: Build Docker image
# =============================================================================
echo ""
echo "=== STEP 3: Building Docker image ==="
docker build --network host -t multi-model-inference:latest -f deploy/Dockerfile .
if [ $? -ne 0 ]; then
    echo "FATAL: Docker build failed!"
    exit 1
fi
echo "  Docker image built successfully"

# Verify GPU inside Docker
echo "  Testing GPU inside container..."
GPU_TEST=$(docker run --rm --gpus all multi-model-inference:latest python -c "import torch; print(f'CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')")
echo "  $GPU_TEST"
if echo "$GPU_TEST" | grep -q "CUDA:False"; then
    echo "WARNING: CUDA not available inside Docker! Running CPU-only mode."
fi

# =============================================================================
# STEP 4: Start Spark cluster
# =============================================================================
echo ""
echo "=== STEP 4: Starting Spark cluster ==="
docker rm -f spark-master spark-gpu-worker 2>/dev/null || true
MASTER_IP=$(hostname -I | awk '{print $1}')
echo "  Master IP: $MASTER_IP"

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

# Verify cluster
WORKERS=$(docker exec spark-master curl -s http://localhost:8080/json/ | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('workers',[])))" 2>/dev/null || echo "0")
echo "  Workers registered: $WORKERS"

# Final GPU check inside spark-master
docker exec spark-master python -c "import torch; print(f'  Container CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"


# =============================================================================
# STEP 5: Run ALL benchmark phases
# =============================================================================
echo ""
echo "=== STEP 5: Running benchmarks ==="
echo "  Start time: $(date)"

run_bench() {
    local desc="$1"
    shift
    echo ""
    echo "--- $desc ---"
    echo "  Command: docker exec spark-master bash -c \"$*\""
    local start=$(date +%s)
    docker exec spark-master bash -c "$*"
    local end=$(date +%s)
    echo "  Duration: $((end-start))s"
    sleep 3
}

# Phase 6: CPU baseline
run_bench "P6.1 cpu_only 4-part 3K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 3000 --batch-size 256"

# Phase 7: GPU tests
run_bench "P7.1 gpu_only 2-part 1K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000 --image-samples 50 --detection-samples 20 --batch-size 256"

run_bench "P7.2 gpu_only 4-part 3K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 256"

run_bench "P7.3 gpu_only 4-part 5K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256"

# Phase 7b: Hybrid
run_bench "P7b.1 hybrid 4-part 3K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size 256"

run_bench "P7b.2 hybrid 4-part 5K" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256"

# Phase 8: GPU batch size scaling
for bs in 64 128 256 512; do
  run_bench "P8 gpu_only batch=$bs" \
    "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size $bs"
done

# Phase 9: Incremental load test
run_bench "P9 incremental_load_test" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/incremental_load_test.py"

# Phase 10: Full incremental (all modes x 3 loads)
run_bench "P10 full_incremental" \
  "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --incremental"

# =============================================================================
# STEP 6: Sync results to S3
# =============================================================================
echo ""
echo "=== STEP 6: Syncing results ==="
RESULT_COUNT=$(ls /opt/benchmark/app/results/*.json 2>/dev/null | wc -l)
echo "  Result files on host: $RESULT_COUNT"

# Also grab from inside container
docker cp spark-master:/app/results/. /opt/benchmark/app/results/ 2>/dev/null || true
RESULT_COUNT=$(ls /opt/benchmark/app/results/*.json 2>/dev/null | wc -l)
echo "  Total result files: $RESULT_COUNT"

aws s3 sync /opt/benchmark/app/results/ s3://$BUCKET/results/ --region $REGION
# Also upload the log
aws s3 cp /opt/benchmark/benchmark.log s3://$BUCKET/benchmark.log --region $REGION

echo ""
echo "============================================================"
echo "  ALL BENCHMARKS COMPLETE"
echo "  Results: $RESULT_COUNT files synced to s3://$BUCKET/results/"
echo "  Finished: $(date)"
echo "============================================================"
