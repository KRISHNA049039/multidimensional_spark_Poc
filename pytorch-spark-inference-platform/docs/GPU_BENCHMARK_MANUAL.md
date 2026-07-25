# GPU Benchmark — Manual Deployment Guide

## Step 1: Deploy the GPU Instance (from your Windows machine)

```powershell
cd C:\multidim_spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform\deploy\aws-cdk

# Install CDK dependencies (one-time)
pip install -r requirements.txt
npm install -g aws-cdk    # if not installed

# Deploy the GPU benchmark stack
cdk deploy GpuBenchmarkStack --context region=us-east-1 --require-approval never
```

This creates:
- 1× g4dn.xlarge (4 vCPU, 16GB RAM, NVIDIA T4 GPU)
- Deep Learning AMI (drivers pre-installed)
- Docker + nvidia-container-toolkit ready
- S3 bucket for code/results
- Auto-shutdown after 4 hours

## Step 2: Upload your code

```powershell
# Get the bucket name from CDK output
$bucket = aws cloudformation describe-stacks --stack-name GpuBenchmarkStack --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text

# Zip and upload
cd C:\multidim_spark_poc\multidimensional_spark_Poc\pytorch-spark-inference-platform
python -c "
import zipfile, os
exclude = {'.git','.venv','cdk.out','node_modules','__pycache__','project.zip','results'}
with zipfile.ZipFile('project.zip','w',zipfile.ZIP_DEFLATED) as zf:
    for r,d,files in os.walk('.'):
        d[:] = [x for x in d if x not in exclude]
        for f in files:
            if f=='project.zip': continue
            p=os.path.join(r,f); a=p.replace('\\\\','/').lstrip('./')
            zf.write(p,a)
    print(f'Zipped {len(zf.namelist())} files')
"
aws s3 cp project.zip "s3://$bucket/project.zip" --region us-east-1
```

## Step 3: SSM into the instance

```powershell
$instanceId = aws cloudformation describe-stacks --stack-name GpuBenchmarkStack --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text
```

Then go to AWS Console → Systems Manager → Session Manager → Start Session → select `gpu-benchmark-instance`.

## Step 4: Verify GPU (run on EC2)

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker can see GPU
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

You should see the T4 GPU with 16GB VRAM.


## Step 5: Pull code and build Docker image (run on EC2)

```bash
# Load bucket name
source /etc/environment
aws s3 cp s3://$BUCKET/project.zip /opt/benchmark/project.zip --region us-east-1
cd /opt/benchmark
rm -rf app/*
unzip -o project.zip -d app
cd app

# Build the Docker image (takes ~8 min first time)
docker build --network host -t multi-model-inference:latest -f deploy/Dockerfile .
```

## Step 6: Start Spark cluster (single-node with GPU)

```bash
MASTER_IP=$(hostname -I | awk '{print $1}')

# Start Spark Master
docker run -d --name spark-master --network host --gpus all --shm-size=4g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /opt/benchmark/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-master.sh && tail -f /opt/spark/logs/*master*"

sleep 10

# Start GPU Worker (same machine)
docker run -d --name spark-gpu-worker --network host --gpus all --shm-size=4g \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -v /opt/benchmark/app/results:/app/results \
  multi-model-inference:latest \
  bash -c "start-worker.sh spark://${MASTER_IP}:7077 -c 4 -m 12g && tail -f /opt/spark/logs/*worker*"

sleep 5

# Verify cluster
docker exec spark-master curl -s http://localhost:8080/json/ | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'Workers: {len(d[\"workers\"])}')
for w in d['workers']:
    print(f'  {w[\"host\"]}:{w[\"port\"]} cores={w[\"cores\"]} state={w[\"state\"]}')
"
```

## Step 7: Verify GPU inside Docker

```bash
docker exec spark-master python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output: `CUDA: True, GPU: Tesla T4`

## Step 8: Run benchmarks one by one

```bash
MASTER_IP=$(hostname -I | awk '{print $1}')

# Phase 6: CPU baseline
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 CUDA_VISIBLE_DEVICES='' python benchmark/cluster_benchmark.py --device-mode cpu_only --partitions 4 --signal-samples 3000 --batch-size 256"

# Phase 7: GPU only
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 2 --signal-samples 1000 --image-samples 50 --detection-samples 20 --batch-size 256"

docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256"

# Phase 7b: Hybrid
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode hybrid --partitions 4 --signal-samples 5000 --image-samples 200 --detection-samples 50 --batch-size 256"

# Phase 8: GPU batch sizes
for bs in 64 128 256 512; do
  echo "=== Batch size: $bs ==="
  docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --device-mode gpu_only --partitions 4 --signal-samples 3000 --image-samples 100 --detection-samples 30 --batch-size $bs"
  sleep 3
done

# Phase 9: Incremental
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/incremental_load_test.py"

# Phase 10: Full incremental (all modes × 3 loads)
docker exec spark-master bash -c "SPARK_MASTER_URL=spark://${MASTER_IP}:7077 python benchmark/cluster_benchmark.py --incremental"
```

## Step 9: Download results

```bash
# Check results
docker exec spark-master ls /app/results/ | wc -l

# Sync to S3
source /etc/environment
docker cp spark-master:/app/results/. /opt/benchmark/results/
aws s3 sync /opt/benchmark/results/ s3://$BUCKET/results/ --region us-east-1
```

Then on your Windows machine:

```powershell
$bucket = aws cloudformation describe-stacks --stack-name GpuBenchmarkStack --region us-east-1 --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text
aws s3 sync "s3://$bucket/results/" .\results\gpu_benchmark_real\ --region us-east-1
```

## Step 10: Tear down

```powershell
cdk destroy GpuBenchmarkStack --context region=us-east-1 --force
```

---

## Expected GPU vs CPU Results

With a real T4 GPU, you should see:

| Model | CPU (samples/sec) | GPU (samples/sec) | Speedup |
|-------|-------------------|-------------------|---------|
| signal_denoiser | 175,000 | ~180,000 | 1× (no benefit) |
| resnet18 | 21 | ~200-400 | 10-20× |
| efficientnet_b0 | 25 | ~250-500 | 10-20× |
| yolov8_small | 14 | ~100-200 | 7-14× |
| mobilenetv3 | 170 | ~1,000-2,000 | 6-12× |

The hybrid mode should significantly outperform both pure modes at scale.
